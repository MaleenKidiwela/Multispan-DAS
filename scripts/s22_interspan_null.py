"""Null test for inter-span feature dv/v: each real feature's single-feature regression term vs the same
statistic for random pseudo-faults (60-km lines striking N50-80W through the bin cloud), per band and stack
length. Only pairs whose stacks converge (median CC >= CC_MIN) within MAX_KM, and no tonal pairs.
Output: <interspan dir>/null_test.csv.   Run with MSDVV_INTERSPAN=<variant> for config variants."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from msdvv import interspan_cfg, load_config
from msdvv.interspan import _local_xy

cfg = load_config()
xs = interspan_cfg(cfg)
od = Path(cfg["output_dir"]) / xs.get("out_subdir", "interspan")
CC_MIN, MAX_KM, N_PSEUDO = 0.3, 70.0, 200
P = pd.read_csv(od / "pairs_with_stats.csv")
B = pd.read_csv(od / "bins.csv").set_index("bin")
DA = pd.read_csv(od / "pair_dvv.csv.gz")
if "stack_h" not in DA:
    DA["stack_h"] = 24
feats = [c for c in P if c.startswith("x_")]
bx, by = _local_xy(B.lat, B.lon, B.lat.mean(), B.lon.mean())
bx, by = pd.Series(bx, index=B.index), pd.Series(by, index=B.index)
p1 = np.c_[bx[P.bin_i].values, by[P.bin_i].values]
p2 = np.c_[bx[P.bin_j].values, by[P.bin_j].values]


def orient(a, b, c):
    return np.sign((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


rng = np.random.default_rng(1)
lines = []
for _ in range(N_PSEUDO):
    c = np.array([rng.uniform(bx.min(), bx.max()), rng.uniform(by.min(), by.max())])
    ang = np.radians(90 + rng.uniform(50, 80))
    d = np.array([np.cos(ang), np.sin(ang)]) * 30
    Q1, Q2 = np.broadcast_to(c - d, p1.shape), np.broadcast_to(c + d, p1.shape)
    lines.append((orient(p1, p2, Q1) != orient(p1, p2, Q2)) & (orient(Q1, Q2, p1) != orient(Q1, Q2, p2)))

rows = []
for (band, SH), D in DA.groupby(["band", "stack_h"]):
    conv = P.get(f"conv_cc_{band}_{SH}h", P.get(f"conv_cc_{band}"))
    tonal = P.get(f"tonal_{band}", pd.Series(False, index=P.index)).astype(bool)
    good = P[(conv >= CC_MIN) & (P.dist_km <= MAX_KM) & ~tonal].index
    D = D[D.pair.isin(good)]
    if len(good) < 50 or D.t.nunique() < 3:
        continue

    def fit(x):
        out = []
        for t, g in D.groupby("t"):
            xx = x[g.pair.values].astype(float)
            if xx.sum() < 5 or (1 - xx).sum() < 5:
                continue
            X = np.c_[np.ones(len(g)), xx]
            W = 1 / np.maximum(g.err.values, 1e-5) ** 2
            cov = np.linalg.pinv((X.T * W) @ X)
            beta = cov @ ((X.T * W) @ g.dv.values)
            r = g.dv.values - X @ beta
            s2 = np.sum(W * r**2) / max(len(g) - 2, 1)
            out.append((beta[1], np.sqrt(cov[1, 1] * s2)))
        a = np.array(out)
        return (a[:, 0].std(), np.sqrt(np.mean((a[:, 0] / a[:, 1]) ** 2))) if len(a) >= 3 else (np.nan, np.nan)

    null = np.array([fit(x) for x in lines if x[good].sum() >= 30 and (~x[good]).sum() >= 30])
    null = null[np.isfinite(null[:, 0])]
    for f in feats:
        s, z = fit(P[f].values)
        rows.append(dict(band=band, stack_h=SH, n_pairs_conv=len(good), feature=f[2:], n_cross=int(P.loc[good, f].sum()),
                         std_pct=100 * s, rms_z=z, null_med_pct=100 * np.median(null[:, 0]), null_p95_pct=100 * np.percentile(null[:, 0], 95),
                         pct_std=float(np.mean(null[:, 0] < s)), pct_rmsz=float(np.mean(null[:, 1] < z)), n_null=len(null)))
R = pd.DataFrame(rows)
R.to_csv(od / "null_test.csv", index=False)
print(R.round(3).to_string(index=False))
