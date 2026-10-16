"""Within-span (s45) figures: per cable & span, the offset gather of one representative 5-km bin with the coda windows
used, the best unit's daily stacks over its reference, and the dv/v series (24 h and 48 h) with the even/odd floor.

env: MSDVV_CONFIG (config_100m_v2ram.yaml), NORM (ram), BAND (b2), LEVEL (5km).
Figures: <output_dir>/within_coda/figs/within_windows_<band>.png, within_series_<band>.png
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv import load_config
from msdvv.stack import distance_bins, load_aggregated

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
out = Path(cfg["output_dir"])
NORM, BAND, LEVEL = os.environ.get("NORM", "ram"), os.environ.get("BAND", "b2"), os.environ.get("LEVEL", "5km")
wd = out / "within_coda"
fd = wd / "figs"
fd.mkdir(exist_ok=True)
U = pd.read_csv(wd / f"{NORM}_{BAND}_24h" / f"units_{BAND}_24h.csv")
ML = 150.0
units_used = U[U.scc.notna()] if "scc" in U else U.iloc[:0]
spans = units_used.groupby(["cable", "span"]).size().index.tolist()
fig, axs = plt.subplots(len(spans), 2, figsize=(16, 3.0 * len(spans)), squeeze=False, gridspec_kw=dict(width_ratios=[1, 1.3]))
row = 0
for cab in ("north", "south"):
    geo = pd.read_csv(out / f"channel_quality_{cab}.csv")
    bins_ch = distance_bins(geo, float(LEVEL[:-2]))
    bins_ch[~geo.good.values] = -1
    nbin = int(bins_ch.max()) + 1
    R = load_aggregated(out / "ccf", cab, BAND, geo.good.values, bins_ch, nbin)
    R.maxlag_s = ML
    t = R.t_sub
    ref, lags = R.ccf(R.spec.sum(0), workers=8)  # (nk, nbin, nlag)
    ks, dx = R.ks, R.dx_km
    for sp in [s for c, s in spans if c == cab]:
        uu = units_used[(units_used.cable == cab) & (units_used.span == sp)]
        b = int(uu.groupby("bin").conv.median().idxmax())  # bin with the best convergence
        ax = axs[row, 0]
        for kk, k in enumerate(ks):
            y = ref[kk, b] / (np.abs(ref[kk, b]).max() + 1e-30)
            ax.plot(lags, k * dx + 0.09 * y, "k", lw=0.5)
            for _, r in U[(U.cable == cab) & (U.span == sp) & (U.bin == b) & (U.k == k)].iterrows():
                sg = 1 if r.side == "+" else -1
                used = np.isfinite(r.get("scc", np.nan))
                ax.plot([sg * r.c0, sg * r.c1], [k * dx - 0.05] * 2, color="tab:green" if used else "0.7", lw=2)
        ax.set(xlim=(-ML, ML), ylabel="offset (km)", title=f"{cab} {sp}: reference CCFs of bin {b} (best convergence); green = coda windows used", xlabel="lag (s)")
        ax.title.set_fontsize(8)
        # best unit: daily 24-h stacks over the reference
        best = uu.sort_values("conv").iloc[-1]
        kk = int(np.where(ks == best.k)[0][0])
        bb = int(best.bin)
        cs = np.concatenate([np.zeros((1,) + R.spec.shape[2:], R.spec.dtype)[:, bb][:, None] * 0, np.cumsum(R.spec[:, kk, bb][:, None], 0)])[:, 0]
        days = np.arange(np.ceil(t[0] / 86400) * 86400 + 86400, t[-1], 86400.0)
        i1 = np.searchsorted(t, days)
        i0 = np.searchsorted(t, days - 86400)
        okd = (i1 - i0) >= 20
        C, _ = R.ccf((cs[i1] - cs[i0])[okd], workers=8)
        ax = axs[row, 1]
        rn = ref[kk, bb] / np.abs(ref[kk, bb]).max()
        for i, c in enumerate(C):
            ax.plot(lags, -i * 0.5 + 0.4 * rn, color="0.7", lw=0.5)
            ax.plot(lags, -i * 0.5 + 0.4 * c / (np.abs(c).max() + 1e-30), "k", lw=0.5)
        sg = 1 if best.side == "+" else -1
        ax.axvspan(*sorted((sg * best.c0, sg * best.c1)), color="tab:green", alpha=0.15)
        ax.set(xlim=(-ML, ML), yticks=[], xlabel="lag (s)")
        ax.set_title(f"best unit: k={int(best.k)} ({best.offset_km:.1f} km), bin {bb}, side {best.side}: daily 24-h stacks (black) over reference (grey); "
                     f"conv {best.conv:.2f}, stretch CC {best.scc:.2f}", fontsize=8)
        row += 1
    del R
fig.suptitle(f"Within-span coda windows and stability ({NORM}, {BAND} {cfg['xcorr']['bands'][BAND]} Hz, {LEVEL} bins)", fontsize=11)
fig.tight_layout()
fig.savefig(fd / f"within_windows_{BAND}.png", dpi=95)
plt.close(fig)

# ---- dv/v series, 24 h and 48 h
S24 = pd.read_csv(wd / f"{NORM}_{BAND}_24h" / f"series_{BAND}_24h.csv")
S48 = pd.read_csv(wd / f"{NORM}_{BAND}_48h" / f"series_{BAND}_48h.csv")
keys = [k for k in S24.groupby(["cable", "span"]).size().index if k[1] != "all"] + [("north", "all"), ("south", "all")]
fig, axs = plt.subplots(len(keys), 1, figsize=(14, 1.9 * len(keys)), sharex=True)
for ax, (cab, sp) in zip(axs, keys):
    for S, col, lab in ((S24, "0.55", "24 h"), (S48, "C3", "48 h")):
        s = S[(S.cable == cab) & (S.span == sp)].set_index("t")
        if s.empty:
            continue
        full = np.arange(s.index.min(), s.index.max() + 1, 3600.0)
        s = s.reindex(full)
        td = pd.to_datetime(full, unit="s", utc=True)
        fl = float(s.floor.dropna().iloc[0])
        ax.fill_between(td, 100 * (s.dv - fl), 100 * (s.dv + fl), color=col, alpha=0.2)
        ax.plot(td, 100 * s.dv, color=col, lw=0.9, label=f"{lab}: floor {100 * fl:.3f}%, std {100 * np.nanstd(s.dv):.3f}%")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylabel(f"{cab}\n{sp}\ndv/v %", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
fig.suptitle(f"Within-span coda dv/v per span ({NORM}, {BAND}), back-stacked 24 h (grey) and 48 h (red), step 1 h; shaded = even/odd floor", fontsize=10)
fig.tight_layout()
fig.savefig(fd / f"within_series_{BAND}.png", dpi=105)
print("done", fd)
