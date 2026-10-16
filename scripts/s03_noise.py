"""Step 1: PSD / coherence-vs-offset / f-k per cable; saves arrays to outputs/noise/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.io import load_cfg
from msdvv.noise import coherence_vs_offset, fk_power, psd_from_fft, remove_common_mode, segment_fft

cfg = load_config()
out = Path(cfg["output_dir"])
nd = out / "noise"
nd.mkdir(parents=True, exist_ok=True)
inv = pd.read_csv(out / "inventory.csv")
ev = pd.read_csv(out / "events.csv", parse_dates=["win0", "win1"])
SEG_S = cfg["noise"]["win_s"]
KS = [1, 2, 3, 4, 6, 10, 20, 50]
MAX_H = 2.0
FS = dict(zip(inv.path, inv.fs))
N_FILES = cfg["noise"].get("n_files_per_cable")  # long deployments: a sample of files is enough


def bad_segments(t0, fs, n, nseg):
    """Boolean per segment: overlaps an earthquake window."""
    s0 = t0 + np.arange(nseg) * n / fs
    s1 = s0 + n / fs
    bad = np.zeros(nseg, bool)
    for w0, w1 in zip(ev.win0.astype("int64") / 1e9, ev.win1.astype("int64") / 1e9):
        bad |= (s0 < w1) & (s1 > w0)
    return bad


def one_file(path):
    f = load_cfg(path, cfg, tslice=slice(0, int(MAX_H * 3600 * FS[path])))
    chans = f.channels()
    res = {}
    for tag, data in (("raw", f.data), ("cm", remove_common_mode(f.data, chans))):
        X, fr = segment_fft(data, f.fs, SEG_S, workers=4)
        n = int(round(SEG_S * f.fs))
        X = X[:, ~bad_segments(f.t0, f.fs, n, X.shape[1])]
        res[f"psd_{tag}"] = psd_from_fft(X, f.fs, n)
        res[f"coh_{tag}"] = coherence_vs_offset(X, chans, KS)
    res["f"] = fr
    return path, res


if N_FILES:
    inv = inv.groupby("cable", group_keys=False).apply(lambda g: g.iloc[np.linspace(0, len(g) - 1, min(N_FILES, len(g))).astype(int)])
jobs = Parallel(n_jobs=min(len(inv), 24), verbose=5)(delayed(one_file)(p) for p in inv.path)
res = dict(jobs)
for cab, g in inv.groupby("cable"):
    arrs = [res[p] for p in g.path]
    np.savez_compressed(
        nd / f"spectra_{cab}.npz",
        f=arrs[0]["f"],
        ks=np.array(KS),
        files=np.array(g.path),
        **{k: np.stack([a[k] for a in arrs]) for k in ("psd_raw", "psd_cm", "coh_raw", "coh_cm")},
    )


def fk_file(path):
    f = load_cfg(path, cfg, tslice=slice(0, int(3600 * FS[path])))
    chans = f.channels()
    outd = {}
    for tag, data in (("raw", f.data), ("cm", remove_common_mode(f.data, chans))):
        for b, gb in chans.groupby("block"):
            sl = slice(gb.ch.min() + 3, gb.ch.max() - 2)
            P, k, fr = fk_power(data[sl], f.fs, f.dx, workers=4)
            outd[f"{tag}_{b}"] = P
            outd[f"k_{b}"] = k
    outd["f"] = fr
    return path, outd


first = inv.groupby("cable").path.first()
fks = dict(Parallel(n_jobs=2)(delayed(fk_file)(p) for p in first))
for cab, p in first.items():
    np.savez_compressed(nd / f"fk_{cab}.npz", src=p, **fks[p])
print("done")
