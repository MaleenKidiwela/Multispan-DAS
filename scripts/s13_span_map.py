"""Span overview for planning cross-span work: map + along-route depth profiles with kept blocks,
repeater spans, and the usable channels at 0.12-0.7 Hz (100-m stream) and 1-2 Hz (fine-spacing stream).

usage: python scripts/s13_span_map.py   (reads outputs_100m and, if present, outputs_hf)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv.geometry import read_route, repeaters

LF = Path("/data/wsd02/multispan_dvv/outputs_100m")
HF = Path("/data/wsd02/multispan_dvv/outputs_hf")
OUT = LF / "figs" / "span_map.png"
SPAN_COL = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}

q_lf = {c: pd.read_csv(LF / f"channel_quality_{c}.csv") for c in ("north", "south")}
q_hf = {c: pd.read_csv(HF / f"channel_quality_{c}.csv") for c in ("north", "south") if (HF / f"channel_quality_{c}.csv").exists()}

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1], width_ratios=[1, 1], hspace=0.35)
ax_map = fig.add_subplot(gs[0, :])
rows = []
for cab, mk in (("north", "o"), ("south", "s")):
    route, rep = read_route(cab), repeaters(cab)
    g = q_lf[cab]
    ax_map.plot(route.lon, route.lat, "-", color="0.75", lw=1.2, zorder=1)
    for sp, gg in g.groupby("span"):
        ax_map.scatter(gg.lon, gg.lat, s=4, color=SPAN_COL[sp], zorder=2)
    ax_map.plot(rep.lon, rep.lat, "k^", ms=7, zorder=3)
    for r in rep.itertuples():
        ax_map.annotate(f"R{r.rep_id}", (r.lon, r.lat), xytext=(4, 4 if cab == "north" else -10), textcoords="offset points", fontsize=7)
    for (blk, sp), gg in g.groupby(["block", "span"]):
        if len(gg) < 15:
            continue
        m = gg.iloc[len(gg) // 2]
        ax_map.annotate(f"{cab[0].upper()}-{blk}{sp}", (m.lon, m.lat), xytext=(0, 9 if cab == "north" else -14),
                        textcoords="offset points", fontsize=8, fontweight="bold", color=SPAN_COL[sp], ha="center")
        h = q_hf.get(cab)
        rows.append(dict(span=f"{cab[0].upper()}-{blk}{sp}", route_km=f"{gg.route_km.min():.0f}-{gg.route_km.max():.0f}",
                         depth_m=f"{gg.depth.min():.0f}-{gg.depth.max():.0f}",
                         good_lf=f"{gg.good.mean():.0%}",
                         good_hf=(f"{h[(h.opt_km >= gg.opt_km.min()) & (h.opt_km <= gg.opt_km.max())].good.mean():.0%}" if h is not None else "n/a")))
    ax_map.text(route.lon.iloc[-1], route.lat.iloc[-1], f"  {cab} cable end", fontsize=7, color="0.4", va="center")
ax_map.plot(-123.967, 45.202, "k*", ms=12)
ax_map.annotate("Pacific City\n(shore station)", (-123.967, 45.202), xytext=(6, -2), textcoords="offset points", fontsize=8)
ax_map.set(xlabel="longitude", ylabel="latitude", title="Kept DAS data by repeater span (colour = span, ▲ = repeater); grey = full cable route")
ax_map.set_aspect(1 / np.cos(np.radians(45)))
for i, c in SPAN_COL.items():
    ax_map.scatter([], [], color=c, s=20, label=f"span {i}")
ax_map.legend(loc="lower left", fontsize=8, ncol=4)

for row, cab in enumerate(("north", "south"), start=1):
    ax = fig.add_subplot(gs[row, :])
    route, rep = read_route(cab), repeaters(cab)
    g = q_lf[cab]
    ax.fill_between(route.route_km, route.depth, 3200, color="0.92", zorder=0)
    ax.plot(route.route_km, route.depth, "-", color="0.4", lw=1)
    for (blk, sp), gg in g.groupby(["block", "span"]):
        ax.axvspan(gg.route_km.min(), gg.route_km.max(), color=SPAN_COL[sp], alpha=0.18, lw=0)
        if len(gg) >= 15:
            ax.text((gg.route_km.min() + gg.route_km.max()) / 2, 150, f"{blk}{sp}", ha="center", fontsize=9, fontweight="bold", color=SPAN_COL[sp])
    gd = g[g.good]
    ax.plot(gd.route_km, np.full(len(gd), 3350), "|", color="C0", ms=7, label="usable 0.12-0.7 Hz (100-m stream)")
    if cab in q_hf:
        h = q_hf[cab]
        h = h[h.good]
        ax.plot(h.route_km, np.full(len(h), 3550), "|", color="C3", ms=7, label="usable 1-2 Hz (fine-spacing stream)")
    for r in rep.itertuples():
        ax.axvline(r.route_km, color="k", ls="--", lw=0.8)
        ax.text(r.route_km, 3100, f" R{r.rep_id}", fontsize=7, va="bottom")
    ax.set(xlim=(0, route.route_km.max()), ylim=(3700, 0), ylabel="water depth (m)", title=f"{cab} cable: depth along route, kept blocks shaded by span",
           xlabel="distance along route (km)" if row == 2 else None)
    ax.legend(loc="lower right", fontsize=7)
fig.savefig(OUT, dpi=125, bbox_inches="tight")
t = pd.DataFrame(rows)
t.to_csv(LF / "span_table.csv", index=False)
print(t.to_string(index=False))
print(OUT)
