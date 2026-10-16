"""Inter-span bin-to-bin correlations across features (see msdvv/interspan.py).
Writes <output_dir>/interspan/{bins,pairs}.csv and one h5 per run with hourly bin-pair substacks."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from msdvv import ROOT, interspan_cfg, load_config
from msdvv.interspan import correlate_file_interspan, load_features, make_bins, pair_table

cfg = load_config()
xs = interspan_cfg(cfg)
cfg["interspan"] = xs  # correlate_file_interspan reads cfg["interspan"]
out = Path(cfg["output_dir"])
od = out / xs.get("out_subdir", "interspan")
od.mkdir(exist_ok=True)
cab = xs["cable"]
q = pd.read_csv(out / f"channel_quality_{cab}.csv")
bins, bid = make_bins(q, xs["bin_km"], xs["min_good"])
lines, points = load_features(ROOT / xs["features_geojson"], {k: tuple(v) for k, v in xs["points"].items()})
P = pair_table(bins, lines, points, min_km=xs["min_km"])
bins.to_csv(od / "bins.csv", index=False)
P.to_csv(od / "pairs.csv", index=False)
pairs = P[["bin_i", "bin_j"]].values
print(f"{len(bins)} bins, {len(P)} pairs", flush=True)
inv = pd.read_csv(out / "inventory.csv")
inv = inv[inv.cable == cab]
ev = pd.read_csv(out / "events.csv", parse_dates=["win0", "win1"])
t = time.time()
nj = xs.get("n_jobs", 8)


COMPACTED = set((od / "compacted.txt").read_text().split()) if (od / "compacted.txt").exists() else set()


def one(path):
    name = od / (Path(path).name + ".h5")
    if os.environ.get("RESUME") == "1" and ((name.exists() and name.stat().st_size > 0) or name.name in COMPACTED):
        return 0, 0  # resume an interrupted run (also skips outputs already merged by s20b_compact_interspan.py)
    try:
        r = correlate_file_interspan(path, cfg, bid, pairs, ev, workers=2)
    except ValueError as e:  # file without any usable window (e.g. short Nov files, all rejected): skip, reported
        print("skipped (no usable window):", Path(path).name, str(e)[:60], flush=True)
        return 0, 0
    with h5py.File(name, "w") as h:
        h.attrs.update(dict(cable=cab, fs=r["fs"], nfft=r["nfft"], maxlag_s=xs["maxlag_s"], substack_s=xs["substack_s"], setup=r["setup"]))
        h["t_sub"], h["nwin"] = r["t_sub"], r["nwin"]
        for b in r["spec"]:
            g = h.create_group(b)
            g["bins"] = r["bins"][b]
            g.create_dataset("spec", data=r["spec"][b], compression="lzf")
    return len(r["t_sub"]), r["n_rejected"]


res = Parallel(n_jobs=nj, verbose=5)(delayed(one)(p) for p in inv.path)
print(f"done {len(res)} files, {sum(r[0] for r in res)} substacks, {time.time() - t:.0f} s")
