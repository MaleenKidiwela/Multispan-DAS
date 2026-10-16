"""Map of all seismometer-DAS pairs (HYSB1/HYS14 x south-cable 1-km bins): every pair drawn, the converging ones
(24-h convergence CC >= 0.3, b1 early window; used for dv/v) coloured by convergence CC; plus CC vs distance."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from obspy import read_inventory

from msdvv import ROOT
from msdvv.geometry import read_route, repeaters

od = Path("/data/wsd02/multispan_dvv/outputs_100m/obsdas")
fd = Path("/data/wsd02/multispan_dvv/outputs_100m/figs/obsdas")
P = pd.read_csv(od / "paths_with_stats.csv")
B = pd.read_csv(od / "bins.csv")
inv = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
CONV_MIN = 0.3
OBS = ["HYSB1", "HYS14"]
cmap, norm = plt.get_cmap("plasma_r"), plt.Normalize(0.3, 0.9)
fc = json.load(open(ROOT / "resources/qfaults_oregon_margin_strikeslip.geojson"))
rt = read_route("south")
rp = repeaters("south")

fig = plt.figure(figsize=(20, 12), layout="constrained")
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8])
LEG = []
ZOOM = ((-125.55, -124.75), (44.30, 44.66))
map_axes = []
for k, o in enumerate(OBS):
    for row, (xl, yl) in enumerate((((-125.8, -123.85), (44.25, 45.3)), ZOOM)):
        ax = fig.add_subplot(gs[row, k])
        map_axes.append(ax)
        ax.plot(rt.lon, rt.lat, "-", color="0.8", lw=2, zorder=1)
        for f in fc["features"]:
            c = np.array(f["geometry"]["coordinates"])
            ax.plot(c[:, 0], c[:, 1], color="saddlebrown", lw=1.8, zorder=2)
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(-125.148 + 5 / (111.2 * np.cos(np.radians(44.569))) * np.cos(th), 44.569 + 5 / 111.2 * np.sin(th), "--", color="goldenrod", zorder=2)
        g = P[P.obs == o]
        seg = lambda d: [[(crd[o][1], crd[o][0]), (r.bin_lon, r.bin_lat)] for r in d.itertuples()]  # noqa: E731
        off = g[~(g.conv_b1 >= CONV_MIN)]
        on = g[g.conv_b1 >= CONV_MIN].sort_values("conv_b1")
        ax.add_collection(LineCollection(seg(off), colors="0.6", linewidths=0.4, alpha=0.5, zorder=3))
        ax.add_collection(LineCollection(seg(on), colors=cmap(norm(on.conv_b1.values)), linewidths=1.8 if row else 1.4, zorder=4))
        ax.scatter(B.lon, B.lat, s=10 if row == 0 else 18, color="k", zorder=5)
        ax.scatter(on.bin_lon, on.bin_lat, s=40 if row else 22, facecolor=cmap(norm(on.conv_b1.values)), edgecolor="k", lw=0.6, zorder=6)
        ax.plot(rp.lon, rp.lat, "k^", ms=9, zorder=6)
        for s_ in OBS:
            ax.plot(crd[s_][1], crd[s_][0], "v", color="tab:blue" if s_ == "HYSB1" else "tab:red", mec="k", ms=15 if s_ == o else 10, zorder=7)
            ax.annotate(s_, (crd[s_][1], crd[s_][0]), xytext=(7, 7), textcoords="offset points", fontsize=10, fontweight="bold")
        ax.set(xlim=xl, ylim=yl, xlabel="longitude", ylabel="latitude")
        ax.set_aspect(1 / np.cos(np.radians(44.6)))
        if row == 0:
            ax.set_title(f"{o} x south-cable bins: {len(g)} pairs, {len(on)} used (coloured)", fontsize=11)
            ax.add_patch(plt.Rectangle((ZOOM[0][0], ZOOM[1][0]), ZOOM[0][1] - ZOOM[0][0], ZOOM[1][1] - ZOOM[1][0], fill=False, ls=":", lw=1))
            from matplotlib.lines import Line2D
            LEG[:] = [Line2D([], [], color="0.6", lw=0.8, label="pairs not converging (grey)"),
                               Line2D([], [], color=cmap(0.6), lw=2, label="pairs used for dv/v (coloured by CC)"),
                               Line2D([], [], color="k", marker="o", ls="", ms=4, label="DAS 1-km bins"),
                               Line2D([], [], color="k", marker="^", ls="", ms=7, label="repeaters"),
                               Line2D([], [], color="saddlebrown", lw=2, label="faults (Qfaults)"),
                               Line2D([], [], color="goldenrod", ls="--", label="S. Hydrate Ridge (5 km)"),
                      Line2D([], [], color="k", ls=":", label="zoom area")]
        else:
            ax.set_title(f"{o}: zoom on Hydrate Ridge", fontsize=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
lax = fig.add_subplot(gs[1, 2])
lax.axis("off")
lax.legend(handles=LEG, loc="upper left", fontsize=10, frameon=False)
fig.colorbar(sm, ax=lax, location="bottom", shrink=0.9, label="24-h convergence CC (b1 0.12-0.3 Hz, early window)")
ax = fig.add_subplot(gs[0, 2])
for o, col in (("HYSB1", "tab:blue"), ("HYS14", "tab:red")):
    g = P[P.obs == o]
    ax.scatter(g.dist_km, g.conv_b1, s=18, color=col, alpha=0.8, label=o)
ax.axhline(CONV_MIN, color="k", ls="--", lw=0.8)
ax.text(140, CONV_MIN + 0.01, "used ≥ 0.3", ha="right", fontsize=8)
ax.set(xlabel="seismometer-bin distance (km)", ylabel="24-h convergence CC", title="convergence vs distance")
ax.legend(fontsize=8)
fig.suptitle("Seismometer-DAS pairs, south cable (gap-filled OBS data, 24-h stacks moving 1 h, back-stacked)", fontsize=12)
fig.savefig(fd / "obsdas_all_pairs_map.png", dpi=110, bbox_inches="tight")
print(P.groupby("obs").apply(lambda g: pd.Series(dict(pairs=len(g), used=int((g.conv_b1 >= CONV_MIN).sum()),
                                                      used_dist=f"{g[g.conv_b1 >= CONV_MIN].dist_km.min():.1f}-{g[g.conv_b1 >= CONV_MIN].dist_km.max():.1f} km",
                                                      used_segs=",".join(sorted(g[g.conv_b1 >= CONV_MIN].seg.unique()))))).to_string())
