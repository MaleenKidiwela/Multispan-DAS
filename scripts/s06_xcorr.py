"""Step 2: cross-correlate all January files (parallel over files) -> outputs/ccf/<run>.h5

usage: python scripts/s06_xcorr.py [RUN ...]     (default: all runs)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.xcorr import correlate_file, write_run

cfg = load_config()
out = Path(cfg["output_dir"])
(out / "ccf").mkdir(parents=True, exist_ok=True)
inv = pd.read_csv(out / "inventory.csv")
ev = pd.read_csv(out / "events.csv", parse_dates=["win0", "win1"])
runs = sys.argv[1:] or sorted(inv.run.unique())
inv = inv[inv.run.isin(runs)]

N_CORES = 48
n_jobs = min(len(inv), cfg["xcorr"].get("n_jobs", 12))
workers = max(1, N_CORES // n_jobs)
good = {}
if cfg["xcorr"].get("good_pairs_only", False):
    good = {c: pd.read_csv(out / f"channel_quality_{c}.csv").good.values for c in inv.cable.unique()}
t = time.time()
# process in batches of runs so that only ~one batch of results is held in memory at a time
BATCH_FILES = cfg["xcorr"].get("batch_files", 48)
batches, cur = [], []
for run, g in inv.groupby("run"):
    cur.append(run)
    if inv.run.isin(cur).sum() >= BATCH_FILES:
        batches.append(cur)
        cur = []
if cur:
    batches.append(cur)
with Parallel(n_jobs=n_jobs, verbose=0) as par:
    for bi, bruns in enumerate(batches):
        sub = inv[inv.run.isin(bruns)]
        res = par(delayed(correlate_file)(p, cfg, ev, workers, good.get(c)) for p, c in zip(sub.path, sub.cable))
        by_path = {r["path"]: r for r in res}
        for run, g in sub.groupby("run"):
            rr = [by_path[p] for p in g.path]
            write_run(out / "ccf" / f"{run}.h5", rr, cfg, run)
            print(run, f"substacks={sum(len(r['t_sub']) for r in rr)}",
                  f"windows used={sum(r['nwin'].sum() for r in rr)}/{sum(r['n_windows'] for r in rr)}",
                  f"rejected={sum(r['n_rejected'] for r in rr)}", flush=True)
        del res, by_path
        print(f"batch {bi + 1}/{len(batches)} done, {time.time() - t:.0f} s", flush=True)
