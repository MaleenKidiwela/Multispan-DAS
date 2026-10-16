"""Map of seismometer-seismometer pairs and seismometer-DAS candidate paths (before convergence selection)."""
import itertools
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
P = pd.read_csv(od / ("paths_with_stats.csv" if (od / "paths_with_stats.csv").exists() else "paths.csv"))
B = pd.read_csv(od / "bins.csv")
inv = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
fig, axs = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
rt = read_route("south")
ax.plot(rt.lon, rt.lat, "-", color="0.75", lw=2, zorder=1)
SPC = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}
for sp, g in B.groupby("span"):
    ax.scatter(g.lon, g.lat, s=14, color=SPC.get(sp, "k"), edgecolor="k", lw=0.3, zorder=4)
for seg, g in B.groupby("seg"):
    m = g.iloc[len(g) // 2]
    ax.annotate(f"S-{seg}", (m.lon, m.lat), xytext=(6, 6), textcoords="offset points", fontsize=9, fontweight="bold")
feat = [c for c in P if c.startswith("x_")]
style = {"HYSB1": ("tab:blue", "HYSB1 (slope base)"), "HYS14": ("tab:red", "HYS14 (summit)")}
for o, (col, lab) in style.items():
    g = P[P.obs == o]
    cross = g[feat].any(axis=1) if feat else pd.Series(False, index=g.index)
    for sub, ls, lw, a, extra in ((g[~cross], ":", 0.6, 0.45, "no feature"), (g[cross], "-", 0.8, 0.6, "crossing a feature")):
        if len(sub):
            ax.add_collection(LineCollection([[(crd[o][1], crd[o][0]), (r.bin_lon, r.bin_lat)] for r in sub.itertuples()],
                                             colors=[col], linestyles=ls, linewidths=lw, alpha=a, zorder=2,
                                             label=f"{lab}: {len(sub)} paths, {extra}"))
for o in crd:
    ax.plot(crd[o][1], crd[o][0], "v", color="gold" if o.startswith("HYS1") and o != "HYS14" else style.get(o, ("k",))[0],
            mec="k", ms=11 if o in style else 8, zorder=6)
ax.annotate("HYS14", (crd["HYS14"][1], crd["HYS14"][0]), xytext=(8, 8), textcoords="offset points", fontsize=10, fontweight="bold")
ax.annotate("HYSB1", (crd["HYSB1"][1], crd["HYSB1"][0]), xytext=(-48, -14), textcoords="offset points", fontsize=10, fontweight="bold")
fc = json.load(open(ROOT / "resources" / "qfaults_oregon_margin_strikeslip.geojson"))
for f in fc["features"]:
    c = np.array(f["geometry"]["coordinates"])
    ax.plot(c[:, 0], c[:, 1], color="saddlebrown", lw=2.2, zorder=3)
th = np.linspace(0, 2 * np.pi, 100)
ax.plot(-125.148 + 5 / (111.2 * np.cos(np.radians(44.569))) * np.cos(th), 44.569 + 5 / 111.2 * np.sin(th), "--", color="goldenrod", zorder=3)
rp = repeaters("south")
ax.plot(rp.lon, rp.lat, "k^", ms=8, zorder=5)
for nm, (la, lo) in {"Pacific City": (45.202, -123.967), "Newport": (44.637, -124.053)}.items():
    ax.plot(lo, la, "ks", ms=4)
    ax.annotate(nm, (lo, la), xytext=(4, -3), textcoords="offset points", fontsize=8)
ax.set(xlim=(-125.8, -123.85), ylim=(44.25, 45.3), xlabel="longitude", ylabel="latitude",
       title=f"Seismometer-DAS candidate paths: {len(P)} (HYSB1/HYS14 x {len(B)} south-cable bins); brown = faults, dashed = SHR 5 km")
ax.set_aspect(1 / np.cos(np.radians(44.8)))
ax.legend(fontsize=8, loc="lower left")
# zoom: seismometer-seismometer pairs at Hydrate Ridge
ax = axs[1]
STA = ["HYS14", "HYSB1", "HYS11", "HYS12", "HYS13"]
for a, b in itertools.combinations(STA, 2):
    long = "HYSB1" in (a, b)
    ax.plot([crd[a][1], crd[b][1]], [crd[a][0], crd[b][0]], "-", color="tab:purple" if long else "tab:green", lw=1.5 if long else 2, alpha=0.8, zorder=2)
for s in STA:
    ax.plot(crd[s][1], crd[s][0], "v", color=style.get(s, ("gold",))[0], mec="k", ms=12, zorder=5)
    ax.annotate(s, (crd[s][1], crd[s][0]), xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.plot(rt.lon, rt.lat, "-", color="0.75", lw=2, zorder=1)
ins = ax.inset_axes([0.55, 0.05, 0.42, 0.42])
for a, b in itertools.combinations(["HYS14", "HYS11", "HYS12", "HYS13"], 2):
    ins.plot([crd[a][1], crd[b][1]], [crd[a][0], crd[b][0]], "-", color="tab:green", lw=2)
for s in ["HYS14", "HYS11", "HYS12", "HYS13"]:
    ins.plot(crd[s][1], crd[s][0], "v", color=style.get(s, ("gold",))[0], mec="k", ms=10)
    ins.annotate(s, (crd[s][1], crd[s][0]), xytext=(4, 3), textcoords="offset points", fontsize=7)
ins.set_title("summit cluster (0.4-0.9 km)", fontsize=8)
ins.tick_params(labelsize=6)
ax.plot([], [], "-", color="tab:purple", label="HYSB1 - summit stations (~21 km, 4 pairs)")
ax.plot([], [], "-", color="tab:green", label="summit pairs (0.4-0.9 km, 6 pairs)")
ax.set(xlim=(-125.5, -125.05), ylim=(44.45, 44.65), xlabel="longitude", title="Seismometer-seismometer pairs (10)")
ax.set_aspect(1 / np.cos(np.radians(44.55)))
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fn = Path("/data/wsd02/multispan_dvv/outputs_100m/figs/obsdas/obs_pairs_map.png")
fn.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fn, dpi=115)
print(fn)
print(P.groupby("obs").apply(lambda g: pd.Series(dict(n=len(g), dmin=g.dist_km.min(), dmed=g.dist_km.median(), **{c[2:]: int(g[c].sum()) for c in feat}))).to_string())
