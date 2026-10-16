"""Span-to-span (cross-span) figures in one common layout for each period (item 7):
  - dv/v per path group (south: Alvin Canyon fault, Daisy Bank fault, Southern Hydrate Ridge, control, all paths;
    north: all paths), 24 h and 48 h back-stacked, labelled with stretching CC, even/odd floor and number of pairs;
  - map of the pairs with a dv/v measurement (48 h), coloured by group, same pair filtering as s43 (exclude_segs,
    <= 70 km) and the pairs.csv / bins.csv of the correlation run.
Also the November-vs-January side-by-side figure (nov_crossspan_dvv.png).
Inputs: Nov outputs_nov_ram/{interspan_coda, interspan_v4_ram, interspan_v6_north};
        Jan outputs_100m_v2/{interspan_coda (ram_*, north_ram_*), interspan_v4_ram, interspan_v6_north}.
Outputs: figs/{nov,jan}_crossspan_dvv_groups.png, figs/{nov,jan}_crossspan_pairs_map.png, figs/nov_crossspan_dvv.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv.geometry import read_route

B = Path("/data/wsd02/multispan_dvv")
FD = B / "outputs_100m/figs"
PER = {
    "nov": dict(label="Nov 6 - Dec 17, 2025", coda=B / "outputs_nov_ram/interspan_coda", corr=B / "outputs_nov_ram", south="ram", north="north"),
    "jan": dict(label="Dec 23, 2025 - Jan 28, 2026", coda=B / "outputs_100m_v2/interspan_coda", corr=B / "outputs_100m_v2", south="ram", north="north_ram"),
}
GROUPS = ["Alvin Canyon fault", "Daisy Bank fault", "SHR", "control (no feature)", "all paths"]
COL = {"Alvin Canyon fault": "tab:red", "Daisy Bank fault": "tab:orange", "SHR": "tab:purple", "control (no feature)": "tab:blue", "all paths": "k"}
NAME = {"SHR": "Southern Hydrate Ridge"}


def series(d, sh, g):
    s = pd.read_csv(d / f"series_{sh}h.csv")
    s = s[s.group == g]
    x = pd.Series(s.dv.values * 100, index=pd.to_datetime(s.t, unit="s", utc=True)).sort_index()
    gap = x.index.to_series().diff() > pd.Timedelta("3h")  # break the line at outages / data gaps
    return pd.concat([x, pd.Series(np.nan, index=x.index[gap.values] - pd.Timedelta("1min"))]).sort_index()


def summ(d, g):
    S = pd.read_csv(d / "summary.csv").set_index("group")
    return S.loc[g] if g in S.index else None


def panel(ax, P, sub, g, cab):
    for sh, lw, a in ((24, 0.7, 0.5), (48, 1.5, 1.0)):
        d = P["coda"] / f"{sub}_b1_{sh}h"
        if not (d / f"series_{sh}h.csv").exists():
            continue
        x = series(d, sh, g)
        m = summ(d, g)
        lab = f"{sh} h" + (f": CC {m.stretch_cc:.2f}, floor {m.floor_pct:.3f} %, {int(m.n_pairs)} pairs" if m is not None else "")
        ax.plot(x.index, x, color=COL[g], lw=lw, alpha=a, label=lab)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_ylim(-0.12, 0.12)
    ax.set_title(f"{cab} cable, {NAME.get(g, g)} — {P['label']}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


ROWS = [(g, "south") for g in GROUPS] + [("all paths", "north")]
# ---- one figure per period, common layout
for key, P in PER.items():
    fig, axs = plt.subplots(len(ROWS), 1, figsize=(13, 2.3 * len(ROWS)), sharex=True)
    for ax, (g, cab) in zip(axs, ROWS):
        panel(ax, P, P[cab], g, cab)
        ax.set_ylabel("dv/v (%)")
    fig.suptitle(f"Span-to-span (cross-span) coda dv/v per path group, {P['label']}: b1 0.12-0.3 Hz, RAM, back-stacked (thin 24 h, thick 48 h)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FD / f"{key}_crossspan_dvv_groups.png", dpi=95)
# ---- side by side (November vs January)
fig, axs = plt.subplots(len(ROWS), 2, figsize=(18, 2.5 * len(ROWS)))
for r, (g, cab) in enumerate(ROWS):
    for c, key in enumerate(("nov", "jan")):
        panel(axs[r, c], PER[key], PER[key][cab], g, cab)
        if c == 0:
            axs[r, c].set_ylabel("dv/v (%)")
fig.suptitle("Span-to-span (cross-span) coda dv/v per path group, b1 0.12-0.3 Hz, back-stacked: November (left) vs January (right), same settings")
fig.tight_layout()
fig.savefig(FD / "nov_crossspan_dvv.png", dpi=95)

# ---- maps of the pairs used, per period
for key, P in PER.items():
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))
    for ax, (cab, vdir) in zip(axs, (("south", "interspan_v4_ram"), ("north", "interspan_v6_north"))):
        bins = pd.read_csv(P["corr"] / vdir / "bins.csv").set_index("bin")
        Pp = pd.read_csv(P["corr"] / vdir / "pairs.csv")
        excl = {"interspan_v4_ram": {"C3"}, "interspan_v6_north": set()}[vdir]
        Pp = Pp[~Pp.seg_i.isin(excl) & ~Pp.seg_j.isin(excl) & (Pp.dist_km <= 70.0)].reset_index(drop=True)
        S = pd.read_csv(P["coda"] / f"{P[cab]}_b1_48h" / "pair_sides.csv")
        used = S[S.scc.notna()].pair.unique()
        for c in ("north", "south"):
            rt = read_route(c)
            ax.plot(rt.lon, rt.lat, "-", color="0.8", lw=1)
        feats = [c for c in Pp if c.startswith("x_")]
        for g in (["control (no feature)", "Daisy Bank fault", "Alvin Canyon fault", "SHR"] if cab == "south" else ["control (no feature)"]):
            sel = ~Pp[feats].any(axis=1) if g == "control (no feature)" else Pp[f"x_{g}"]
            idx = [p for p in used if bool(sel.iloc[p])]
            for p in idx:
                a, b_ = Pp.bin_i[p], Pp.bin_j[p]
                ax.plot([bins.lon[a], bins.lon[b_]], [bins.lat[a], bins.lat[b_]], "-", color=COL[g], lw=0.3, alpha=0.35)
            ax.plot([], [], "-", color=COL[g], lw=2, label=f"{NAME.get(g, g)}: {len(idx)} pairs used")
        ax.plot(bins.lon, bins.lat, "k.", ms=2)
        ax.set(xlim=(bins.lon.min() - 0.15, bins.lon.max() + 0.15), ylim=(bins.lat.min() - 0.1, bins.lat.max() + 0.1),
               xlabel="longitude", ylabel="latitude", title=f"{P['label']}, {cab} cable: span-to-span pairs with a dv/v measurement (48 h)")
        ax.set_aspect(1 / np.cos(np.radians(45)))
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FD / f"{key}_crossspan_pairs_map.png", dpi=95)
print("done")
