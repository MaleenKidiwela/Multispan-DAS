"""Correlation functions of the seismometer-DAS path clusters used for dv/v (s37 pairs.csv 'cluster' column).

Figure 1 (obsdas_cluster_ccfs_ref.png): per cluster (rows) and component Z/R/T (columns), the all-period reference CCF
of every pair in the cluster (neighbour-stacked, latency-corrected, b1 0.12-0.3 Hz) as wiggles sorted by distance,
the cluster average on top (thick) and the early dv/v window shaded.
Figure 2 (obsdas_cluster_ccfs_24h.png): per cluster and component, the cluster-average 24-h back-stacked CCF
(step 1 h) against time, every 3 h, with CC of each 24-h stack to the reference (early window).
Output CCFs: outputs_100m/obsdas_combined/cluster_ccfs.h5
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv.obsdas import OUT, early_window, load_hourly, moving, moving_windows, neighbour_stack

warnings.filterwarnings("ignore", category=RuntimeWarning)
COMPS = ["Z", "R", "T"]
import os
PREP = os.environ.get("OBSDAS_PREP", "ram")      # production: running-absolute-mean
STACK_H = int(os.environ.get("STACK_H", "48"))   # production: 48-h back-stacked, step 1 h
CL = pd.read_csv(OUT / "obsdas_combined" / "pairs.csv")
CLUSTERS = ["red: HYS14 x ACF -> SB3", "red: HYS14 x ACF -> SC4 west", "red: HYS14 x ACF -> SC4 east",
            "blue: HYSB1 control -> SB3", "blue: HYSB1 control -> SC4", "orange: HYS14 x Daisy Bank"]
COL = dict(zip(CLUSTERS, ["#8b0000", "#e31a1c", "#f768a1", "#08306b", "#4292c6", "tab:orange"]))
XL = 290.0  # lag range shown (s)
PK = pd.read_csv(OUT / "obsdas_coda" / "ballistic_picks.csv")  # s40: ballistic picks per pair / component / side


def coda_edges(g, comp, u_floor=0.40, ml=290.0):
    """Cluster-median ballistic pick per side and coda window as used by s40: c0 = max(pick + 15, d_med / 0.40 + 20),
    75 s long."""
    q = PK[(PK.cluster == g) & (PK.comp == comp)]
    bp, bn = float(q.pick_pos_s.median()), float(q.pick_neg_s.median())
    dmed = float(q.dist_km.median())
    cp, cn = max(bp + 15, dmed / u_floor + 20), max(bn + 15, dmed / u_floor + 20)
    return bp, bn, ((cp, min(cp + 75, ml)), (-min(cn + 75, ml), -cn))
fd = OUT / "figs" / "obsdas"

refs, moves = {}, {}
for comp in COMPS:
    t, H, lags, P = load_hourly(comp, prep=PREP, maxlag=300.0)
    dt = lags[1] - lags[0]
    H = neighbour_stack(H, P, dt, 2)
    tend, i0, i1 = moving_windows(t, STACK_H, int(round(STACK_H * 5 / 6)))
    ref = H.sum(0)
    C = moving(H, i0, i1)
    del H
    for g in CLUSTERS:
        idx = np.where(CL.cluster.values == g)[0]
        refs[(comp, g)] = ref[idx]
        norm = np.abs(ref[idx]).max(1, keepdims=True) + 1e-30  # each pair weighted by its reference amplitude
        moves[(comp, g)] = (C[:, idx] / norm[None]).mean(1)
    del C
    print(f"[{comp}] done", flush=True)

with h5py.File(OUT / "obsdas_combined" / "cluster_ccfs.h5", "w") as h:
    h["lags"], h["t_end"] = lags, tend
    h.attrs.update(dict(band="b1 0.12-0.30 Hz", note="neighbour-stacked, latency-corrected; positive lag = DAS after OBS"))
    for (comp, g), r in refs.items():
        k = g.replace(":", "").replace(" ", "_").replace(">", "")
        h[f"{comp}/{k}/ref_pairs"] = r
        h[f"{comp}/{k}/dist_km"] = CL.dist_km.values[CL.cluster.values == g]
        h[f"{comp}/{k}/stack_24h"] = moves[(comp, g)].astype(np.float32)

# ---- figure 1: reference CCFs per pair + cluster average
lm = np.abs(lags) <= XL
fig, axs = plt.subplots(len(CLUSTERS), 3, figsize=(17, 3.2 * len(CLUSTERS)), sharex=True, squeeze=False)
for i, g in enumerate(CLUSTERS):
    d = CL.dist_km.values[CL.cluster.values == g]
    o = np.argsort(d)
    for j, comp in enumerate(COMPS):
        ax = axs[i, j]
        bp, bn, wins = coda_edges(g, comp)
        r = refs[(comp, g)][o]
        r = r / (np.abs(r[:, lm]).max(1, keepdims=True) + 1e-30)
        for k, (y, dk) in enumerate(zip(r, d[o])):
            ax.plot(lags[lm], k + 0.45 * y[lm], color=COL[g], lw=0.5)
        avg = r.mean(0)
        avg = avg / (np.abs(avg[lm]).max() + 1e-30)
        ax.plot(lags[lm], len(r) + 0.8 + 0.9 * avg[lm], color="k", lw=1.4)
        for a_, b_ in wins:
            ax.axvspan(a_, b_, color="0.85", zorder=0)
        qq = PK[(PK.cluster == g) & (PK.comp == comp)].set_index("bin")
        for k, p in enumerate(np.where(CL.cluster.values == g)[0][o]):
            ax.plot([qq.loc[CL.bin[p], "pick_pos_s"], -qq.loc[CL.bin[p], "pick_neg_s"]], [k, k], "kv", ms=3)
        ax.axvline(0, color="0.5", lw=0.5)
        ax.set_yticks([0, len(r) - 1, len(r) + 0.8])
        ax.set_yticklabels([f"{d[o][0]:.1f} km", f"{d[o][-1]:.1f} km", "average"], fontsize=7)
        ax.set_xlim(-XL, XL)
        if i == 0:
            ax.set_title(f"{comp} component", fontsize=11)
        if j == 0:
            ax.set_ylabel(g.replace(": ", ":\n"), fontsize=8, color=COL[g])
axs[-1, 1].set_xlabel("lag (s), latency-corrected (+ = DAS after seismometer)")
fig.suptitle(f"Seismometer-DAS reference correlation functions per cluster ({PREP}, clock-fixed DAS times, Dec 23 - Jan 28, b1 0.12-0.3 Hz, neighbour-stacked);\n"
             "pairs sorted by distance, cluster average on top; triangles = ballistic picks; grey = coda window used for dv/v (pick + 15 to + 90 s)", fontsize=11)
fig.tight_layout()
fig.savefig(fd / "obsdas_cluster_ccfs_ref.png", dpi=110)
plt.close(fig)

# ---- figure 2: 24-h back-stacked cluster CCFs vs time
sel = np.arange(0, len(tend), 3)
td = pd.to_datetime(tend[sel], unit="s", utc=True)
fig, axs = plt.subplots(len(CLUSTERS), 3, figsize=(17, 3.0 * len(CLUSTERS)), sharex=True, squeeze=False)
for i, g in enumerate(CLUSTERS):
    d = CL.dist_km.values[CL.cluster.values == g]
    for j, comp in enumerate(COMPS):
        ax = axs[i, j]
        bp, bn, wins = coda_edges(g, comp)
        wm = np.zeros(len(lags), bool)
        for a_, b_ in wins:
            wm |= (lags >= a_) & (lags <= b_)
        M = moves[(comp, g)][sel]
        rr = refs[(comp, g)] / (np.abs(refs[(comp, g)]).max(1, keepdims=True) + 1e-30)
        rr = rr.mean(0)
        cc = (M[:, wm] @ rr[wm]) / (np.linalg.norm(M[:, wm], axis=1) * np.linalg.norm(rr[wm]) + 1e-30)
        Mn = M / (np.abs(M[:, lm]).max(1, keepdims=True) + 1e-30)
        ax.pcolormesh(lags[lm], td, Mn[:, lm], cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
        for a_, b_ in wins:
            ax.axvline(a_, color="k", ls=":", lw=0.8)
            ax.axvline(b_, color="k", ls=":", lw=0.8)
        ax.axvline(bp, color="g", ls="--", lw=0.8)
        ax.axvline(-bn, color="g", ls="--", lw=0.8)
        ax.set_xlim(-XL, XL)
        ax.set_title(f"{comp}: coda-window CC of {STACK_H}-h stacks to reference, median {np.nanmedian(cc):.2f}", fontsize=8)
        if j == 0:
            ax.set_ylabel(g.replace(": ", ":\n"), fontsize=8, color=COL[g])
axs[-1, 1].set_xlabel("lag (s), latency-corrected (+ = DAS after seismometer)")
fig.suptitle(f"Cluster-average {STACK_H}-h back-stacked correlation functions ({PREP}, clock-fixed; step 1 h, shown every 3 h); green dashed = ballistic, dotted = coda window", fontsize=11)
fig.tight_layout()
fig.savefig(fd / "obsdas_cluster_ccfs_24h.png", dpi=100)
print("done")
