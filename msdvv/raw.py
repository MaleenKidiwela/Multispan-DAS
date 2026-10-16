"""Reader for the raw Nokia multispan binary files (OFDR1MAGIC + msgpack header).

Port of ``open_array_from_file`` / ``get_metadata_from_chunk_meta`` from the official notebook
(github.com/uwfiberlab/OOI_DAS_2025, multispan_code/read_multispan_das_data.ipynb, Z. Krauss).
Data are stored as chunks of shape ``array_info['shape'] = [1, nch, m]``; stacking the chunks
along time gives (nch, nchunks * m).
"""
from __future__ import annotations

import mmap
import os
import struct

import msgpack
import numpy as np
from scipy.constants import c as C_LIGHT

MAGIC = b"OFDR1MAGIC"


def _decoder(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def is_raw(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(len(MAGIC)) == MAGIC


def is_complete(path: str) -> bool:
    """A complete file ends with the magic bytes (truncated downloads / writes do not)."""
    try:
        with open(path, "rb") as f:
            f.seek(-len(MAGIC), os.SEEK_END)
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def fs_dx(cm: dict):
    """Sampling rate [Hz] and channel spacing [m] from chunk metadata (official formulas)."""
    fs = 1 / cm["Tp"] * cm["gauge"] / (cm["D"] * cm["D2"])
    dx = cm["gauge"] * (0.250 / 2**24) / cm["gamma"] * (C_LIGHT / 2 / 1.45) * 1e-9 * 4
    return fs, dx


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not an OFDR1MAGIC file")
        _version, headerbytes, arrstart = struct.unpack("<BII", f.read(struct.calcsize("<BII")))
        u = msgpack.Unpacker(raw=False, object_hook=_decoder)
        u.feed(f.read(headerbytes))
        ainfo = u.unpack()
        u.unpack()
        cm = u.unpack()
        f.seek(-len(MAGIC), os.SEEK_END)
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: incomplete file (no trailing magic)")
        f.seek(-(len(MAGIC) + 4), os.SEEK_END)
        (back,) = struct.unpack("<I", f.read(4))
        f.seek(-(len(MAGIC) + 4 + back), os.SEEK_END)
        u = msgpack.Unpacker(raw=False, object_hook=_decoder)
        u.feed(f.read(back))
        bytes_written = u.unpack()["bytes_written"]
    shape = ainfo["shape"]  # [1, nch, m]
    itemsize = np.dtype(ainfo["dtype"]).itemsize
    nchunk = bytes_written // (int(np.prod(shape)) * itemsize)
    fs, dx = fs_dx(cm)
    attrs = dict(cm)
    attrs.setdefault("dx", dx)
    attrs["new_fs"] = fs  # same key as the downsampled h5 files
    attrs["manually_downsampled"] = False
    return dict(attrs=attrs, arrstart=arrstart, bytes_written=bytes_written, dtype=ainfo["dtype"],
                nch=shape[-2], m=shape[-1], nchunk=nchunk, nt=nchunk * shape[-1])


def read_data(path: str, hdr: dict | None = None, tslice=slice(None)) -> np.ndarray:
    """(nch, nt) float32 raw phase data."""
    hdr = hdr or read_header(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), access=mmap.ACCESS_READ, length=hdr["bytes_written"] + hdr["arrstart"])
        try:
            a = np.frombuffer(mm, dtype=hdr["dtype"], offset=hdr["arrstart"],
                              count=hdr["nchunk"] * hdr["nch"] * hdr["m"]).reshape(hdr["nchunk"], hdr["nch"], hdr["m"])
            out = np.ascontiguousarray(np.transpose(a, (1, 0, 2)).reshape(hdr["nch"], -1)[:, tslice], dtype=np.float32)
        finally:
            del a
            mm.close()
    return out


def read_chunk_meta(path: str) -> tuple[list, list]:
    """Per-chunk write info and chunk metadata from the trailer: (write_info, chunk_meta), one entry per chunk.
    chunk_meta[k] holds 'stream.proc_start' / 'stream.gpu_idx' of chunk k (empty for k = 0: use the header's)."""
    with open(path, "rb") as f:
        f.seek(-(len(MAGIC) + 4), os.SEEK_END)
        (back,) = struct.unpack("<I", f.read(4))
        f.seek(-(len(MAGIC) + 4 + back), os.SEEK_END)
        buf = f.read(back)
    u = msgpack.Unpacker(raw=False, object_hook=_decoder, max_buffer_size=max(10**8, 2 * back))
    u.feed(buf)
    objs = list(u)
    if len(objs) < 3:
        return [], []
    return objs[1], objs[2]


def chunk_start_times(path: str, hdr: dict | None = None) -> np.ndarray:
    """True start time (POSIX s, same latency as stream.proc_start) of every chunk.

    The acquisition occasionally drops whole chunks without leaving a gap in the array (seen as gpu_idx steps of
    2 x 256 and a 2-chunk proc_start step); the per-chunk proc_start in the trailer records the real time, so sample
    j of chunk k is at chunk_start_times[k] + j / fs. Falls back to continuous sampling if the trailer has no
    per-chunk metadata."""
    hdr = hdr or read_header(path)
    t0, fs, m, n = float(hdr["attrs"]["stream.proc_start"]), hdr["attrs"]["new_fs"], hdr["m"], hdr["nchunk"]
    nominal = t0 + np.arange(n) * m / fs
    _, cm = read_chunk_meta(path)
    if len(cm) < n:
        return nominal
    ps = np.array([t0] + [c.get("stream.proc_start", np.nan) for c in cm[1:n]], dtype=float)
    bad = ~np.isfinite(ps)
    ps[bad] = nominal[bad]
    return ps


def sample_times(path: str, hdr: dict | None = None) -> np.ndarray:
    """(nt,) true time of every sample (see chunk_start_times)."""
    hdr = hdr or read_header(path)
    fs, m = hdr["attrs"]["new_fs"], hdr["m"]
    return (chunk_start_times(path, hdr)[:, None] + np.arange(m)[None] / fs).ravel()


def chunk_start_times_counter(path: str, hdr: dict | None = None) -> np.ndarray:
    """Chunk start times from the sample counter: t_k = t_0 + (gpu_idx_k - gpu_idx_0) / step * chunk_s, anchored at the
    file's first proc_start. Dropped chunks appear as counter jumps (2 x step); unlike the per-chunk proc_start this is
    immune to slowly growing processing latency (proc_start creeps 3-10 s/h during the Jan 7-13 and 22-28 drop periods
    while the counter shows no missing samples)."""
    hdr = hdr or read_header(path)
    t0, fs, m, n = float(hdr["attrs"]["stream.proc_start"]), hdr["attrs"]["new_fs"], hdr["m"], hdr["nchunk"]
    step_s = m / fs
    w, cm = read_chunk_meta(path)
    if len(cm) < n:
        return t0 + np.arange(n) * step_s
    gi = np.array([float(w[0].get("idx", np.nan))] + [c.get("stream.gpu_idx", np.nan) for c in cm[1:n]], float)
    if not np.isfinite(gi).all():
        return chunk_start_times(path, hdr)
    st = np.median(np.diff(gi))
    return t0 + (gi - gi[0]) / st * step_s
