"""Stream the archive and compute inter-span bin-to-bin correlations per file (raw files deleted after use).

usage: MSDVV_CONFIG=config_hf.yaml python scripts/d03_stream_interspan.py [START END]
Bins/pairs from <output_dir>/channel_quality_<cable>.csv (run s05 first). Output: <output_dir>/<interspan.out_subdir>/<file>.h5
with the same layout as s20, so s21-s24 run unchanged. Resumable (existing outputs are skipped).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import pandas as pd
from joblib import Parallel, delayed

from msdvv import ROOT, interspan_cfg, load_config
from msdvv.download import crawl, fetch
from msdvv.interspan import correlate_file_interspan, load_features, make_bins, pair_table
from msdvv.io import read_meta

cfg = load_config()
xs = interspan_cfg(cfg)
cfg["interspan"] = xs
out = Path(cfg["output_dir"])
od = out / xs.get("out_subdir", "interspan")
od.mkdir(exist_ok=True)
cab = xs["cable"]
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else cfg["date_range"]
q = pd.read_csv(out / f"channel_quality_{cab}.csv")
bins, bid = make_bins(q, xs["bin_km"], xs["min_good"])
lines, points = load_features(ROOT / xs["features_geojson"], {k: tuple(v) for k, v in xs["points"].items()})
P = pair_table(bins, lines, points, min_km=xs["min_km"])
bins.to_csv(od / "bins.csv", index=False)
P.to_csv(od / "pairs.csv", index=False)
pairs = P[["bin_i", "bin_j"]].values
ev = pd.read_csv(out / "events.csv", parse_dates=["win0", "win1"])
files = crawl(start, end, [cab])
files = files[[not (od / (n + ".h5")).exists() for n in files.name]].reset_index(drop=True)
print(f"{len(bins)} bins, {len(P)} pairs; {len(files)} files to process ({start}..{end}, {cab})", flush=True)
tmp = Path(cfg["data_dirs"][0]).parent / "tmp_interspan"
tmp.mkdir(parents=True, exist_ok=True)
B = cfg["download"].get("batch_files", 36)
batches = [files.iloc[i : i + B] for i in range(0, len(files), B)]
dx_ok = cfg["require"]["dx"]


def dl(b):
    return list(ThreadPoolExecutor(cfg["download"]["parallel"]).map(fetch, b.url, [tmp / n for n in b.name]))


def one(path):
    try:
        f = read_meta(str(path))
        if not any(abs(f.dx - v) < 1e-3 for v in dx_ok) or abs(f.fs - cfg["require"]["fs"]) > 1e-3:
            return "skipped setup"
        r = correlate_file_interspan(str(path), cfg, bid, pairs, ev, workers=2)
        if len(r["t_sub"]) == 0:
            return "no substacks"
        with h5py.File(od / (path.name + ".h5"), "w") as h:
            h.attrs.update(dict(cable=cab, fs=r["fs"], nfft=r["nfft"], maxlag_s=xs["maxlag_s"], substack_s=xs["substack_s"], setup=r["setup"]))
            h["t_sub"], h["nwin"] = r["t_sub"], r["nwin"]
            for bnd in r["spec"]:
                g = h.create_group(bnd)
                g["bins"] = r["bins"][bnd]
                g.create_dataset("spec", data=r["spec"][bnd], compression="lzf")
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"failed: {type(e).__name__}: {e}"


t0 = time.time()
with ThreadPoolExecutor(1) as pre, Parallel(n_jobs=16) as par:
    nxt = pre.submit(dl, batches[0]) if batches else None
    for i, b in enumerate(batches):
        res = nxt.result()
        nxt = pre.submit(dl, batches[i + 1]) if i + 1 < len(batches) else None
        paths = [Path(r["path"]) for r in res if r["status"] in ("ok", "exists")]
        st = par(delayed(one)(p) for p in paths)
        for p in paths:
            p.unlink(missing_ok=True)
        el = time.time() - t0
        print(f"batch {i + 1}/{len(batches)}: {pd.Series(st).str.split(':').str[0].value_counts().to_dict()} "
              f"elapsed {el / 60:.1f} min, ETA {el / (i + 1) * (len(batches) - i - 1) / 60:.0f} min", flush=True)
