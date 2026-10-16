"""Reading the downsampled Nokia multispan h5 files and building the inventory.

Conventions follow the official reader (github.com/uwfiberlab/OOI_DAS_2025,
multispan_code/read_multispan_das_data.ipynb):
  * ``stream.proc_start`` is the data start time (filename time is unreliable)
  * ``new_fs`` is the true sampling rate of the downsampled files
  * ``masked`` lists the *kept* optical-distance intervals [m]; kept channels are
    concatenated in the array, block i spanning int(mask/dx) channel indices.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from scipy.constants import c as C_LIGHT
from scipy.signal import decimate

from .raw import is_complete, is_raw, read_data, read_header, sample_times

RUN_GAP_S = 120.0  # files closer than this are grouped into one run


def strain_rate_factor(attrs) -> float:
    """Raw phase -> strain rate factor (official notebook)."""
    rads_to_strain = 1.55e-6 / 4 / np.pi / 1.4677 / 0.78
    dxf = C_LIGHT / 250e6 / 2 / 1.4677
    gauge_m = attrs["gauge"] * dxf
    tp = attrs["Tp"] / attrs["Npulses"]
    return rads_to_strain / tp / gauge_m


def gauge_length_m(attrs) -> float:
    return float(attrs["gauge"] * C_LIGHT / 250e6 / 2 / 1.4677)


def cable_from_name(path: str) -> str:
    name = os.path.basename(path)
    for cab in ("north", "south"):
        if f"_{cab}_" in name or f"_{cab}-" in name:
            return cab
    raise ValueError(f"cannot infer cable from {name}")


def block_table(attrs, nch: int) -> pd.DataFrame:
    """One row per kept mask block: array index range and optical index range."""
    dx = float(attrs["dx"])
    mask = np.asarray(attrs["masked"], dtype=float)
    imask = (mask / dx).astype(np.int32)
    lengths = imask[:, 1] - imask[:, 0]
    if lengths.sum() != nch:
        raise ValueError(f"mask lengths {lengths.sum()} != nch {nch}")
    a0 = np.r_[0, np.cumsum(lengths)[:-1]]
    return pd.DataFrame(
        dict(
            block=list("ABCDEFG"[: len(lengths)]),
            arr_start=a0,
            arr_end=a0 + lengths,
            opt_start_idx=imask[:, 0],
            opt_start_km=mask[:, 0] / 1e3,
            opt_end_km=mask[:, 1] / 1e3,
            nch=lengths,
        )
    )


def channel_table(attrs, nch: int, cable: str) -> pd.DataFrame:
    """Per-channel metadata: block id and optical distance along the fiber."""
    bt = block_table(attrs, nch)
    dx = float(attrs["dx"])
    rows = []
    for b in bt.itertuples():
        j = np.arange(b.nch)
        rows.append(
            pd.DataFrame(
                dict(
                    cable=cable,
                    ch=b.arr_start + j,
                    block=b.block,
                    opt_idx=b.opt_start_idx + j,
                    opt_km=(b.opt_start_idx + j) * dx / 1e3,
                    pos_in_block=j,
                    block_len=b.nch,
                )
            )
        )
    return pd.concat(rows, ignore_index=True)


@dataclass
class DasFile:
    path: str
    cable: str
    t0: float  # POSIX seconds (proc_start)
    fs: float
    dx: float
    nch: int
    nt: int
    attrs: dict = field(repr=False)
    data: np.ndarray | None = field(default=None, repr=False)
    times: np.ndarray | None = field(default=None, repr=False)  # true per-sample times (raw files, chunk proc_start)
    valid: np.ndarray | None = field(default=None, repr=False)  # per-sample data present (False in dropped-chunk gaps, regrid)

    @property
    def duration(self) -> float:
        return self.nt / self.fs

    @property
    def t1(self) -> float:
        return self.t0 + self.duration

    @property
    def setup(self) -> str:
        a = self.attrs
        s = f"D{int(a['D'])}_g{int(a['gauge'])}_{'ds' if bool(a.get('manually_downsampled', False)) else 'native'}"
        return s + ("_vm" if bool(a.get("virtual_mask", False)) else "")

    def channels(self) -> pd.DataFrame:
        return channel_table(self.attrs, self.nch, self.cable)

    def blocks(self) -> pd.DataFrame:
        return block_table(self.attrs, self.nch)


def read_meta(path: str, dx_default: float | None = None) -> DasFile:
    """dx_default: channel spacing [m] for files without a ``dx`` attribute (the November downsampled h5 files)."""
    if is_raw(path):
        hdr = read_header(path)
        attrs, nch, nt = hdr["attrs"], hdr["nch"], hdr["nt"]
    else:
        with h5py.File(path, "r") as h:
            d = h["data"]
            attrs = {k: v for k, v in d.attrs.items()}
            nch, nt = d.shape
    if "dx" not in attrs and dx_default:
        attrs["dx"] = float(dx_default)
    return DasFile(
        path=path,
        cable=cable_from_name(path),
        t0=float(attrs["stream.proc_start"]),
        fs=float(attrs["new_fs"]),
        dx=float(attrs.get("dx", np.nan)),
        nch=nch,
        nt=nt,
        attrs=attrs,
    )


def spatial_filter_decimate(data: np.ndarray, attrs: dict, nch_blocks, q: int, k_lowpass: float | None,
                            k_notch=(), notch_w: float = 0.5):
    """Per mask block: zero-phase f-k style filter along the cable (low-pass |k| <= k_lowpass, notches at
    +-k_notch, all in 1/km), then keep every q-th channel. Blocks are trimmed to int(mask/(q*dx)) channels so
    the result matches files natively recorded at q*dx. Returns (data, new_attrs)."""
    dx_km = float(attrs["dx"]) / 1e3
    out = []
    a0 = 0
    mask = np.asarray(attrs["masked"], dtype=float)
    for nb, (m0, m1) in zip(nch_blocks, mask):
        x = data[a0 : a0 + nb]
        a0 += nb
        k = np.fft.rfftfreq(2 * nb, dx_km)  # even extension avoids wrap-around edge artefacts
        keep = np.ones_like(k)
        if k_lowpass:
            keep[k > k_lowpass] = 0.0
        for kn in k_notch:
            keep[np.abs(k - kn) < notch_w] = 0.0
        xe = np.concatenate([x, x[::-1]], axis=0)
        X = np.fft.rfft(xe, axis=0)
        X *= keep[:, None]
        x = np.fft.irfft(X, n=2 * nb, axis=0)[:nb].astype(np.float32)
        dxn = q * float(attrs["dx"])
        n_new = int(m1 / dxn) - int(m0 / dxn)  # same rule as block_table()
        xd = x[::q][:n_new]
        if len(xd) < n_new:  # pad by repeating the edge channel (never more than one)
            xd = np.concatenate([xd, xd[-1:].repeat(n_new - len(xd), 0)], axis=0)
        out.append(xd)
    new = dict(attrs, dx=q * float(attrs["dx"]), native_dx=float(attrs["dx"]))
    return np.concatenate(out, axis=0), new


def is_unmasked(attrs, nch: int) -> bool:
    """True if the array holds the full fibre (no mask applied), e.g. Nov 2025 files and Nov 27 - Dec 2 binaries
    (which carry a ``masked`` attribute that was not yet applied)."""
    if "masked" not in attrs:
        return True
    imask = (np.asarray(attrs["masked"], dtype=float) / float(attrs["dx"])).astype(np.int64)
    return int((imask[:, 1] - imask[:, 0]).sum()) != nch


def apply_mask(data: np.ndarray, attrs: dict, mask_m) -> tuple[np.ndarray, dict]:
    """Cut an unmasked array (channel i at optical i*dx) to the kept intervals ``mask_m`` [[m0, m1], ...] [m],
    concatenated exactly like the acquisition mask (block i spans int(m0/dx):int(m1/dx))."""
    dx = float(attrs["dx"])
    mask = np.asarray(mask_m, dtype=float)
    idx = np.concatenate([np.arange(int(m0 / dx), int(m1 / dx)) for m0, m1 in mask])
    if idx.max() >= data.shape[0]:
        raise ValueError(f"mask reaches channel {idx.max()} but the file has {data.shape[0]}")
    return data[idx], dict(attrs, masked=mask, virtual_mask=True)


def load_file(path: str, strain_rate: bool = True, tslice=slice(None), target_fs: float | None = None,
              target_dx: float | None = None, k_lowpass: float | None = None, k_notch=(), notch_w: float = 0.5,
              virtual_mask=None, dx_default: float | None = None, native_filter: bool = True, regrid: bool = False) -> DasFile:
    """Read a file. Returns float32 data (strain rate if requested), shape (nch, nt).

    target_fs: if the file is sampled at an integer multiple of target_fs, decimate to it
    (zero-phase FIR anti-alias filter); ``tslice`` refers to the native samples.
    virtual_mask: {cable: [[m0, m1], ...]} applied to unmasked files (same layout as masked recordings).
    native_filter: apply k_lowpass / k_notch also to files already at target_dx.
    regrid: raw files only (whole file): put the samples on a regular grid of true time (per-chunk proc_start); dropped
    chunks become zero-filled gaps flagged in DasFile.valid instead of silently spliced records.
    Reduced files (written by d05_stream_reduce.py: float16 + per-channel scale, already strain rate) are restored.
    """
    f = read_meta(path, dx_default)
    times = None
    if is_raw(path):
        data = read_data(path, tslice=tslice)
        times = sample_times(path)[tslice]
        if regrid and tslice == slice(None):
            fsn = f.fs
            pos = np.round((times - times[0]) * fsn).astype(np.int64)
            n = int(pos[-1]) + 1
            if n > data.shape[1]:  # at least one dropped chunk
                full = np.zeros((data.shape[0], n), np.float32)
                full[:, pos] = data
                valid = np.zeros(n, bool)
                valid[pos] = True
                data, times = full, times[0] + np.arange(n) / fsn
                f.valid = valid
    else:
        with h5py.File(path, "r") as h:
            data = h["data"][:, tslice].astype(np.float32)
            if "scale" in h:
                data *= h["scale"][:][:, None]
            if "valid" in h:  # reduced, regridded file: dropped-chunk gaps (zero-filled) flagged
                f.valid = h["valid"][:][tslice]
    if f.attrs.get("reduced", False):
        strain_rate = False  # stored as strain rate
    if virtual_mask and is_unmasked(f.attrs, data.shape[0]):
        data, f.attrs = apply_mask(data, f.attrs, virtual_mask[f.cable])
        f.nch = data.shape[0]
    if strain_rate:
        data *= np.float32(strain_rate_factor(f.attrs))
    if target_fs and f.fs > 1.01 * target_fs:
        q = int(round(f.fs / target_fs))
        if abs(f.fs / q - target_fs) > 1e-6 * target_fs:
            raise ValueError(f"{path}: fs {f.fs} is not an integer multiple of {target_fs}")
        data = decimate(data, q, ftype="fir", zero_phase=True, axis=1).astype(np.float32)
        f.attrs = dict(f.attrs, native_fs=f.fs)
        f.fs = f.fs / q
        if times is not None:
            times = times[::q][: data.shape[1]]
        if f.valid is not None:
            f.valid = f.valid[::q][: data.shape[1]]
    if target_dx and f.dx < 0.75 * target_dx:
        q = int(round(target_dx / f.dx))
        nchb = block_table(f.attrs, f.nch).nch.tolist()
        data, f.attrs = spatial_filter_decimate(data, f.attrs, nchb, q, k_lowpass, k_notch, notch_w)
        f.dx, f.nch = f.attrs["dx"], data.shape[0]
    elif target_dx and native_filter and (k_lowpass or k_notch):  # native spacing: still remove instrument stripes
        nchb = block_table(f.attrs, f.nch).nch.tolist()
        data, _ = spatial_filter_decimate(data, f.attrs, nchb, 1, k_lowpass, k_notch, notch_w)
    f.data = data
    f.nt = data.shape[1]
    f.times = times
    return f


def load_cfg(path: str, cfg: dict, tslice=slice(None), strain_rate: bool = True) -> DasFile:
    """load_file with the config's resampling options (target_fs, spatial filter/decimation)."""
    sp = cfg.get("spatial", {})
    return load_file(path, strain_rate, tslice, target_fs=cfg.get("target_fs"), target_dx=sp.get("target_dx"),
                     k_lowpass=sp.get("k_lowpass"), k_notch=sp.get("k_notch", ()), notch_w=sp.get("notch_w", 0.5),
                     virtual_mask=cfg.get("virtual_mask"), dx_default=cfg.get("dx_default"),
                     native_filter=sp.get("native_filter", True), regrid=cfg.get("regrid_drops", False))


def build_inventory(data_dirs, date_range=None, file_glob="*.h5", require=None) -> pd.DataFrame:
    """All matching files (h5 or raw binary), de-duplicated on (cable, proc_start, nt), grouped into runs.

    require: optional dict of metadata that every file must match (e.g. dx / fs of one acquisition setup).
    """
    paths = []
    for d in data_dirs:
        paths += glob.glob(os.path.join(d, "**", file_glob), recursive=True)
    rows = []
    for p in sorted(set(paths)):
        if is_raw(p) and not is_complete(p):
            continue  # truncated download / write
        f = read_meta(p)
        if require and not all(
            any(np.isclose(float(getattr(f, k)), vv, rtol=1e-4) for vv in (v if isinstance(v, list) else [v]))
            for k, v in require.items()
        ):
            continue
        if "dx" not in f.attrs or "masked" not in f.attrs:
            continue  # November-style files: no mask / spacing metadata
        rows.append(
            dict(
                path=p,
                cable=f.cable,
                t0=f.t0,
                t1=f.t1,
                start=pd.Timestamp(f.t0, unit="s", tz="UTC"),
                end=pd.Timestamp(f.t1, unit="s", tz="UTC"),
                duration_h=f.duration / 3600,
                fs=f.fs,
                dx=f.dx,
                nch=f.nch,
                nt=f.nt,
                gauge=int(f.attrs["gauge"]),
                D=int(f.attrs["D"]),
                setup=f.setup,
                dtype="float32-raw" if is_raw(p) else str(h5py.File(p, "r")["data"].dtype),
                size=os.path.getsize(p),
            )
        )
    inv = pd.DataFrame(rows)
    if date_range is not None:
        lo, hi = (pd.Timestamp(x, tz="UTC") for x in date_range)
        inv = inv[(inv.start >= lo) & (inv.start < hi + pd.Timedelta(days=1))]
    inv = inv.sort_values(["cable", "t0", "path"])
    # keep the top-level copy when the same data exist in two folders
    inv["depth"] = inv.path.str.count("/")
    inv = inv.sort_values(["cable", "t0", "depth"]).drop_duplicates(["cable", "t0", "nt"], keep="first")
    inv = inv.drop(columns="depth").reset_index(drop=True)
    # runs: files separated by < RUN_GAP_S
    run_ids = []
    for cab, g in inv.groupby("cable", sort=False):
        rid, prev_end = 0, None
        for r in g.itertuples():
            if prev_end is not None and r.t0 - prev_end > RUN_GAP_S:
                rid += 1
            run_ids.append((r.Index, f"{cab[0].upper()}{rid:02d}"))
            prev_end = r.t1
    inv["run"] = pd.Series(dict(run_ids))
    inv["gap_before_s"] = inv.t0 - inv.groupby("run").t1.shift()
    return inv
