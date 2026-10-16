"""Stream the archive and keep a reduced local copy: download -> load_cfg (virtual mask, 105.9 m, 1.86 Hz,
strain rate) -> float16 h5 with per-channel scale -> delete the raw file.

usage: MSDVV_CONFIG=config_nov.yaml python scripts/d05_stream_reduce.py [START END]
       MSDVV_CONFIG=config_nov.yaml python scripts/d05_stream_reduce.py local <src_dir> <out_dir>   # test
Output: <data_dirs[0]>/<YYYY>/<MM>/<DD>/<stream>/<name>.h5, readable by msdvv.io like any h5 file (attrs reduced=True).
Resumable (existing outputs are skipped); manifest in <data_dirs[0]>/reduce_manifest.csv.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.download import crawl, fetch
from msdvv.io import load_cfg, read_meta

cfg = load_config()
dl = cfg["download"]
dest = Path(cfg["data_dirs"][0])
dest.mkdir(parents=True, exist_ok=True)
req = cfg["stream_require"]
KEEP = ("D", "D2", "gauge", "Tp", "Nfft", "Npulses", "F", "gamma", "manually_downsampled", "time_decimation_factor",
        "stream.proc_start", "stream.module_name", "shm")


CUT = cfg.get("crossing_cutout")  # item 1: native-resolution cutouts around the fault crossings (unmasked raw south files)


def crossing_cutout(path, m):
    """Save opt CUT['opt_km'] (native dx, decimated to target_fs, true times, gaps zero-filled) for unmasked raw files of
    CUT['cable']; returns the output name or ''."""
    from msdvv.io import is_unmasked, load_file
    from msdvv.raw import is_raw
    if not CUT or m.cable != CUT["cable"] or not is_raw(str(path)) or not is_unmasked(m.attrs, m.nch):
        return ""
    o = Path(CUT["dir"]) / (path.name + ".h5")
    if o.exists():
        return o.name
    f = load_file(str(path), target_fs=cfg["target_fs"], regrid=True)
    i0, i1 = int(CUT["opt_km"][0] * 1e3 / f.dx), int(CUT["opt_km"][1] * 1e3 / f.dx)
    x = f.data[i0:i1]
    sc = x.std(axis=1).astype(np.float32)
    sc[~np.isfinite(sc) | (sc == 0)] = 1.0
    o.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(o.with_name(o.name + ".part"), "w") as h:
        d = h.create_dataset("data", data=(x / sc[:, None]).astype(np.float16), compression="lzf")
        h["scale"] = sc
        if f.valid is not None:
            h["valid"] = np.asarray(f.valid[: f.nt], bool)
        d.attrs.update({"dx": f.dx, "new_fs": f.fs, "opt_km0": i0 * f.dx / 1e3, "stream.proc_start": float(f.times[0]) if f.times is not None else f.t0,
                        "gauge": int(f.attrs["gauge"]), "D": int(f.attrs["D"]), "reduced": True, "source": path.name})
    o.with_name(o.name + ".part").rename(o)
    return o.name


def reduce_one(path, out):
    try:
        m = read_meta(str(path), cfg.get("dx_default"))
        if not any(abs(m.dx - v) < 1e-3 for v in req["dx"]) or abs(m.fs - req["fs"]) > 1e-3:
            return dict(status=f"skipped setup dx={m.dx:.3f} fs={m.fs:.4f}")
        f = load_cfg(str(path), cfg)  # regrid_drops: dropped raw chunks -> zero gaps on the true time grid (f.valid)
        x = f.data
        cut = crossing_cutout(path, m)
        sc = x.std(axis=1).astype(np.float32)
        sc[~np.isfinite(sc) | (sc == 0)] = 1.0
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        with h5py.File(part, "w") as h:
            d = h.create_dataset("data", data=(x / sc[:, None]).astype(np.float16), compression="lzf", chunks=(min(256, f.nch), f.nt))
            h["scale"] = sc
            if f.valid is not None:  # samples present (False = dropped-chunk gap), on the reduced time grid
                h["valid"] = np.asarray(f.valid[: f.nt], bool)
            a = {k: f.attrs[k] for k in KEEP if k in f.attrs}
            a["stream.proc_start"] = float(f.times[0]) if f.times is not None else float(f.t0)  # true start (regridded)
            a.update(dx=f.dx, new_fs=f.fs, regridded=f.valid is not None, masked=np.asarray(f.attrs["masked"], float), reduced=True,
                     native_fs=float(f.attrs.get("native_fs", m.fs)), native_dx=float(f.attrs.get("native_dx", m.dx)),
                     virtual_mask=bool(f.attrs.get("virtual_mask", False)), source=path.name)
            for k, v in a.items():
                d.attrs[k] = v
        part.rename(out)
        return dict(status="ok", t0=f.t0, t1=f.t1, nch=f.nch, nt=f.nt, setup=f.setup, native_dx=m.dx, unmasked=bool(f.attrs.get("virtual_mask", False)),
                    n_gap=int((~f.valid).sum()) if f.valid is not None else 0, crossing_cutout=cut)
    except Exception as e:  # noqa: BLE001 - keep streaming, report in manifest
        return dict(status=f"failed: {type(e).__name__}: {e}")


if len(sys.argv) == 4 and sys.argv[1] == "local":  # test: reduce local files  (local <src_dir> <out_dir>)
    for p in sorted(Path(sys.argv[2]).glob("*")):
        print(p.name, reduce_one(p, Path(sys.argv[3]) / (p.name if p.name.endswith(".h5") else p.name + ".h5")), flush=True)
    sys.exit(0)
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else cfg["date_range"]

files = crawl(start, end, dl["streams"]).sort_values(["day", "stream", "name"]).reset_index(drop=True)
files["out"] = [dest / f"{pd.Timestamp(d):%Y/%m/%d}" / s / (n if n.endswith(".h5") else n + ".h5")
                for d, s, n in zip(files.day, files.stream, files.name)]
todo = files[[not p.exists() for p in files.out]].reset_index(drop=True)
print(f"{len(files)} files listed {start}..{end}; {len(todo)} to process", flush=True)
tmp = dest.parent / "tmp_reduce"
tmp.mkdir(exist_ok=True)
B = dl.get("batch_files", 36)
batches = [todo.iloc[i : i + B] for i in range(0, len(todo), B)]
mpath = dest / "reduce_manifest.csv"
def download_batch(b):
    return list(ThreadPoolExecutor(dl["parallel"]).map(fetch, b.url, [tmp / n for n in b.name]))


t_start = time.time()
with ThreadPoolExecutor(1) as pre, Parallel(n_jobs=12) as par:
    nxt = pre.submit(download_batch, batches[0]) if batches else None
    for i, b in enumerate(batches):
        dres = nxt.result()
        nxt = pre.submit(download_batch, batches[i + 1]) if i + 1 < len(batches) else None  # prefetch
        ok = [(Path(d["path"]), o) for d, o in zip(dres, b.out) if d["status"] in ("ok", "exists")]
        res = par(delayed(reduce_one)(p, o) for p, o in ok)
        for p, _ in ok:
            p.unlink(missing_ok=True)
        r = pd.DataFrame(res)
        r["out"] = [str(o) for _, o in ok]
        man = b[["day", "stream", "name", "url"]].assign(out=b.out.astype(str)).merge(r, on="out", how="left")
        man.loc[man.status.isna(), "status"] = "download failed"
        man.to_csv(mpath, mode="a", header=not mpath.exists(), index=False)
        el = time.time() - t_start
        gb = sum(d["bytes"] for d in dres) / 1e9
        print(f"batch {i + 1}/{len(batches)} ({b.day.iloc[0]}): {man.status.str.split(':').str[0].value_counts().to_dict()} "
              f"{gb:.0f} GB, elapsed {el / 60:.1f} min, ETA {el / (i + 1) * (len(batches) - i - 1) / 60:.0f} min", flush=True)
