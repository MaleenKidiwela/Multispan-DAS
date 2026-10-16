"""Final maps of what each coda-only dv/v product used (outputs_100m/figs/final/final_maps.png):
(a) within-span: 5-km bins with converging (offset, side) units, coloured by span, size = number of units;
(b) cross-span: bin-pair paths with converging coda sides, coloured by group;
(c) DAS-station: seismometer-bin pairs of the six clusters.
Faults (Qfaults), Southern Hydrate Ridge (5 km), repeaters and stations on every panel.
env: WITHIN (ram_b2_48h), CROSS (ram_b1_48h).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from obspy import read_inventory

from msdvv import ROOT
from msdvv.geometry import read_route, repeaters
from msdvv.stack import bin_table, distance_bins

B = Path("/data/wsd02/multispan_dvv")
WITHIN, CROSS = os.environ.get("WITHIN", "ram_b2_48h"), os.environ.get("CROSS", "ram_b1_48h")
fc = json.load(open(ROOT / "resources/qfaults_oregon_margin_strikeslip.geojson"))
inv = read_inventory(str(B / "obs/continuous/OO_response.xml"))
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n if s.code in ("HYSB1", "HYS14")}


def base(ax, cables=("north", "south"), xl=(-125.85, -123.85), yl=(44.25, 45.75)):
    for c in cables:
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.82", lw=2, zorder=1)
        rp = repeaters(c)
        ax.plot(rp.lon, rp.lat, "k^", ms=7, zorder=6)
    for f in fc["features"]:
        cc = np.array(f["geometry"]["coordinates"])
        ax.plot(cc[:, 0], cc[:, 1], color="saddlebrown", lw=1.6, zorder=2)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(-125.148 + 5 / (111.2 * np.cos(np.radians(44.569))) * np.cos(th), 44.569 + 5 / 111.2 * np.sin(th), "--", color="goldenrod", zorder=2)
    for s_, (la, lo) in crd.items():
        ax.plot(lo, la, "v", color="w", mec="k", ms=10, zorder=7)
        ax.annotate(s_, (lo, la), xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold")
    ax.set(xlim=xl, ylim=yl, xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1 / np.cos(np.radians(44.8)))


fig, axs = plt.subplots(1, 3, figsize=(24, 9.5), layout="constrained")
# (a) within-span
ax = axs[0]
base(ax)
SPC = {}
cols = plt.get_cmap("tab10").colors
handles = []
for cab, root in (("north", B / "outputs_100m_v2ram"), ("south", B / "outputs_100m_v2ram")):
    U = pd.read_csv(root / "within_coda" / WITHIN / f"units_{WITHIN.split('_')[1]}_{WITHIN.split('_')[2]}.csv")
    U = U[(U.cable == cab) & U.scc.notna()]
    geo = pd.read_csv(root / f"channel_quality_{cab}.csv")
    bins_ch = distance_bins(geo, 5.0)
    bins_ch[~geo.good.values] = -1
    BT = bin_table(geo, bins_ch).set_index("bin")
    n = U.groupby("bin").size()
    for sp, g in U.groupby("span"):
        key = f"{cab} {sp}"
        SPC.setdefault(key, cols[len(SPC) % 10])
        bb = sorted(g.bin.unique())
        ax.scatter(BT.loc[bb, "lon"], BT.loc[bb, "lat"], s=8 + 1.2 * n.loc[bb].values, color=SPC[key], edgecolor="k", lw=0.4, zorder=5)
        handles.append(Line2D([], [], marker="o", ls="", color=SPC[key], mec="k", label=f"{key}: {len(bb)} bins, {len(g)} units"))
ax.legend(handles=handles, fontsize=7, loc="upper left")
ax.set_title(f"(a) within-span ({WITHIN}): 5-km bins with converging coda units (offsets 0.2-3.2 km)", fontsize=10)
# (b) cross-span
ax = axs[1]
base(ax, cables=("south",), xl=(-125.8, -123.9), yl=(44.25, 45.3))
src = B / "outputs_100m_v2" / ("interspan_v4_ram" if CROSS.startswith("ram") else "interspan_v3")
bins = pd.read_csv(src / "bins.csv").set_index("bin")
T = pd.read_csv(B / "outputs_100m_v2/interspan_coda" / CROSS / "pair_sides.csv")
T = T[T.scc.notna()].drop_duplicates("pair")
P = pd.read_csv(src / "pairs.csv")
P = P.merge(T[["bin_i", "bin_j"]], on=["bin_i", "bin_j"])
GC = [("control (no feature)", "tab:blue"), ("Daisy Bank fault", "tab:orange"), ("Alvin Canyon fault", "tab:red"), ("SHR", "goldenrod")]
feat = [c for c in P if c.startswith("x_")]
for g, col in GC:
    m = (~P[feat].any(axis=1)) if g.startswith("control") else P[f"x_{g}"].astype(bool)
    q = P[m]
    seg = [[(bins.lon[r.bin_i], bins.lat[r.bin_i]), (bins.lon[r.bin_j], bins.lat[r.bin_j])] for r in q.itertuples()]
    ax.add_collection(LineCollection(seg, colors=col, linewidths=0.35, alpha=0.35, zorder=3))
ax.scatter(bins.lon, bins.lat, s=6, color="k", zorder=5)
ax.legend(handles=[Line2D([], [], color=c, lw=2, label=f"{g} ({int(((~P[feat].any(axis=1)) if g.startswith('control') else P[f'x_{g}'].astype(bool)).sum())} pairs)") for g, c in GC],
          fontsize=8, loc="lower left")
ax.set_title(f"(b) cross-span ({CROSS}): bin-pair paths with converging coda ({len(P)} pairs)", fontsize=10)
# (c) DAS-station
ax = axs[2]
base(ax, cables=("south",), xl=(-125.55, -124.5), yl=(44.3, 44.8))
CL = pd.read_csv(B / "outputs_100m/obsdas_combined/pairs.csv")
CC = dict(zip(["red: HYS14 x ACF -> SB3", "red: HYS14 x ACF -> SC4 west", "red: HYS14 x ACF -> SC4 east", "blue: HYSB1 control -> SB3",
               "blue: HYSB1 control -> SC4", "orange: HYS14 x Daisy Bank"], ["#8b0000", "#e31a1c", "#fb9a99", "#08306b", "#6baed6", "tab:orange"]))
for g, col in CC.items():
    q = CL[CL.cluster == g]
    ax.add_collection(LineCollection([[(crd[r.obs][1], crd[r.obs][0]), (r.bin_lon, r.bin_lat)] for r in q.itertuples()], colors=col, linewidths=1.0, zorder=4))
ax.legend(handles=[Line2D([], [], color=c, lw=2.5, label=f"{g} ({int((CL.cluster == g).sum())})") for g, c in CC.items()], fontsize=8, loc="upper right")
ax.set_title("(c) DAS-station: HYSB1/HYS14 x south-cable bins, six clusters (Z+R+T)", fontsize=10)
fig.suptitle("Pairs and bins used by the final coda-only dv/v products (Dec 23 2025 - Jan 28 2026)", fontsize=12)
fd = B / "outputs_100m/figs/final"
fd.mkdir(parents=True, exist_ok=True)
fig.savefig(fd / "final_maps.png", dpi=100)
print("done", fd / "final_maps.png")
