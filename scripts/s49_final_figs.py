"""Final dv/v figures of the three coda-only products, 24 h and 48 h (DAS-DAS products uncorrected: after the dropped-chunk
gap fix the D128 setup steps are insignificant, s48; DAS-station: D128 period excluded, s48),
back-stacked (step 1 h), with the even/odd floor; plus Hs. Figures in outputs_100m/figs/final/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

B = Path("/data/wsd02/multispan_dvv")
fd = B / "outputs_100m/figs/final"
fd.mkdir(parents=True, exist_ok=True)
env = pd.read_csv(B / "outputs_100m/dvv_24h/environment_timeseries.csv", index_col=0)
T1, T2 = pd.Timestamp("2026-01-13 18:18", tz="UTC"), pd.Timestamp("2026-01-21 18:22", tz="UTC")


def panel(ax, s24, s48, title, ylim=None):
    for s, col, lab in ((s24, "0.55", "24 h"), (s48, "C3", "48 h")):
        if s is None or s.empty:
            continue
        s = s.set_index("t").sort_index()
        full = np.arange(s.index.min(), s.index.max() + 1, 3600.0)
        s = s.reindex(full)
        td = pd.to_datetime(full, unit="s", utc=True)
        fl = float(s.floor.dropna().iloc[0])
        ax.fill_between(td, 100 * (s.dv - fl), 100 * (s.dv + fl), color=col, alpha=0.2)
        ax.plot(td, 100 * s.dv, color=col, lw=0.9, label=f"{lab}: floor {100 * fl:.3f}%, std {100 * np.nanstd(s.dv):.3f}%")
    ax.axvspan(T1, T2, color="tab:blue", alpha=0.04)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_ylabel("dv/v %", fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)


def hs(ax):
    ax.plot(pd.to_datetime(env.index.values, unit="s", utc=True), env.Hs, "k", lw=0.8)
    ax.axvspan(T1, T2, color="tab:blue", alpha=0.04)
    ax.set_ylabel("Hs (m)", fontsize=8)


# ---- within-span (RAM, b2 and b1)
for band in ("b2", "b1"):
    root = B / "outputs_100m_v2ram/within_coda"
    S24 = pd.read_csv(root / f"ram_{band}_24h/series_{band}_24h.csv")  # no setup correction needed after the gap fix (s48 steps)
    S48 = pd.read_csv(root / f"ram_{band}_48h/series_{band}_48h.csv")
    keys = [k for k in S24.groupby(["cable", "span"]).size().index if k[1] != "all"] + [("north", "all"), ("south", "all")]
    fig, axs = plt.subplots(len(keys) + 1, 1, figsize=(14, 1.8 * (len(keys) + 1)), sharex=True)
    for ax, (cab, sp) in zip(axs, keys):
        panel(ax, S24[(S24.cable == cab) & (S24.span == sp)], S48[(S48.cable == cab) & (S48.span == sp)], f"within-span {cab} {sp}")
    hs(axs[-1])
    fig.suptitle(f"Within-span coda dv/v (running-absolute-mean, {band} {'0.3-0.7' if band == 'b2' else '0.12-0.3'} Hz, 5-km bins, offsets 0.2-3.2 km)\n"
                 "24 h (grey) / 48 h (red) back-stacked, step 1 h; blue = D128 acquisition period (no step after the gap fix); shaded = even/odd floor", fontsize=9)
    fig.tight_layout()
    fig.savefig(fd / f"final_within_{band}.png", dpi=100)
    plt.close(fig)

# ---- cross-span (RAM, b1)
root = B / "outputs_100m_v2/interspan_coda"
S24 = pd.read_csv(root / "ram_b1_24h/series_24h.csv")  # no setup correction needed after the gap fix (s48 steps ~0)
S48 = pd.read_csv(root / "ram_b1_48h/series_48h.csv")
groups = ["control (no feature)", "Alvin Canyon fault", "Daisy Bank fault", "SHR", "all paths"]
groups = [g for g in groups if g in set(S24.group)]
fig, axs = plt.subplots(len(groups) + 1, 1, figsize=(14, 2.0 * (len(groups) + 1)), sharex=True)
for ax, g in zip(axs, groups):
    panel(ax, S24[S24.group == g], S48[S48.group == g], f"cross-span: {g}")
hs(axs[-1])
fig.suptitle("Cross-span coda dv/v (south cable, running-absolute-mean, 0.12-0.3 Hz; coda after the slowest coherent ballistic packet)\n"
             "24 h (grey) / 48 h (red) back-stacked, step 1 h; blue = D128 period (no step after the gap fix); shaded = even/odd floor", fontsize=9)
fig.tight_layout()
fig.savefig(fd / "final_crossspan_b1.png", dpi=100)
plt.close(fig)

# ---- DAS-station (RAM, Z+R+T)
S24 = pd.read_csv(B / "outputs_100m/obsdas_coda/variants/final_ram_24h/series_24h.csv")  # gap-fixed: no systematic D128 step (s48)
S48 = pd.read_csv(B / "outputs_100m/obsdas_coda/series_48h.csv")
cl = list(S48.cluster.unique())
fig, axs = plt.subplots(len(cl) + 1, 1, figsize=(14, 2.0 * (len(cl) + 1)), sharex=True)
for ax, c in zip(axs, cl):
    panel(ax, S24[S24.cluster == c], S48[S48.cluster == c], f"DAS-station: {c}")
hs(axs[-1])
fig.suptitle("DAS-station coda dv/v (HYSB1/HYS14 x south-cable clusters, Z+R+T, running-absolute-mean, 0.12-0.3 Hz; clock-fixed)\n"
             "24 h (grey) / 48 h (red) back-stacked, step 1 h; blue = D128 period (no systematic step after the gap fix); shaded = even/odd floor", fontsize=9)
fig.tight_layout()
fig.savefig(fd / "final_das_station.png", dpi=100)
print("done", fd)
