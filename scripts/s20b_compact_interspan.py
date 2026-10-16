"""Compact per-file cross-span outputs (s20) into one file per UTC day, summing substacks of the same hour.

Why: the Nov 6-26 input files are 17 min long, so s20 writes one small h5 per input file with partial hourly
substacks, ~3x the storage of hourly data. s43 sums same-hour substacks across files when it loads them
(np.add.at over t_sub), so summing them here is lossless for s43 (spec is a sum over windows; nwin is summed).
Each day's sources are checked (sum of the compact file == sum of the sources, per band) before they are deleted; their
names go to compacted.txt so that s20 RESUME=1 does not recompute them.
usage: MSDVV_CONFIG=config_nov_ram.yaml MSDVV_INTERSPAN=interspan_v4_ram python scripts/s20b_compact_interspan.py
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from msdvv import load_config

cfg = load_config()
xs = cfg[os.environ.get("MSDVV_INTERSPAN", "interspan")]
od = Path(cfg["output_dir"]) / xs.get("out_subdir", "interspan")
done_list = od / "compacted.txt"
src = sorted(p for p in od.glob("*.h5") if not p.name.startswith("compact_"))
print(f"{od.name}: {len(src)} per-file outputs to compact", flush=True)
by_day = defaultdict(list)
for p in src:
    try:
        with h5py.File(p, "r") as h:
            t = h["t_sub"][:]
    except OSError:
        print("unreadable (left in place):", p.name, flush=True)
        continue
    if len(t) == 0:
        by_day["empty"].append(p)
        continue
    by_day[int(t.min() // 86400)].append(p)
n_del, freed = 0, 0
for day, files in sorted(by_day.items(), key=lambda kv: str(kv[0])):
    if day == "empty":
        continue
    T, NW, SP, attrs, bins = [], [], defaultdict(list), None, {}
    for p in files:
        with h5py.File(p, "r") as h:
            if attrs is None:
                attrs = dict(h.attrs)
                bands = [k for k in h.keys() if isinstance(h[k], h5py.Group)]
            T.append(h["t_sub"][:])
            NW.append(h["nwin"][:])
            for b in bands:
                SP[b].append(h[b]["spec"][:])
                bins[b] = h[b]["bins"][:]
    t_all = np.concatenate(T)
    ut, inv = np.unique(t_all, return_inverse=True)
    k = 0
    while (od / f"compact_{day}_{k}.h5").exists():
        k += 1
    out = od / f"compact_{day}_{k}.h5"
    sums = {}
    with h5py.File(out, "w") as h:
        h.attrs.update(attrs)
        h["t_sub"] = ut
        h["nwin"] = np.bincount(inv, weights=np.concatenate(NW), minlength=len(ut)).astype(np.int32)
        for b in bands:
            S = np.concatenate(SP[b])
            M = np.zeros((len(ut),) + S.shape[1:], S.dtype)
            np.add.at(M, inv, S)
            g = h.create_group(b)
            g["bins"] = bins[b]
            g.create_dataset("spec", data=M, compression="lzf")
            sums[b] = (np.abs(S.sum()), np.abs(M.sum()))
    with h5py.File(out, "r") as h:  # verify what was written
        ok = all(np.isclose(np.abs(h[b]["spec"][:].sum()), sums[b][0], rtol=1e-4) for b in bands) and len(h["t_sub"]) == len(ut)
    if not ok:
        print(f"day {day}: verification FAILED, sources kept, {out.name} removed", flush=True)
        out.unlink()
        continue
    with open(done_list, "a") as fl:
        for p in files:
            fl.write(p.name + "\n")
            freed += p.stat().st_size
            p.unlink()
            n_del += 1
    print(f"day {day}: {len(files)} files -> {out.name} ({len(ut)} hours, {out.stat().st_size / 1e9:.2f} GB)", flush=True)
for p in by_day.get("empty", []):  # files without substacks carry no data
    with open(done_list, "a") as fl:
        fl.write(p.name + "\n")
    p.unlink()
print(f"compacted {n_del} files, freed {freed / 1e9:.1f} GB before the compact files", flush=True)
