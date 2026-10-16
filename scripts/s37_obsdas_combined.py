"""Production seismometer-DAS dv/v: Z + R + T, along-fibre neighbour stacking (+-2 bins), b1 early window,
24-h back-stacked stacks moving 1 h, components combined per pair by 1/err^2, path groups with even/odd floor.

Groups (same feature definitions as the inter-span work):
  HYSB1 control      HYSB1 paths crossing no mapped feature
  HYS14 x ACF        HYS14 paths crossing the Alvin Canyon fault
  HYS14 no ACF       HYS14 paths not crossing it (same station: the ACF contrast cancels station/source effects)
  all HYSB1 / all HYS14 / all
INJECT=<eps> (INJECT_AT, default 2026-01-01): synthetic stretch in every hourly CCF; outputs get suffix _inj<eps>.
Output: outputs_100m/obsdas_combined/ (series_24h*.csv, pairs*.csv, summary*.csv) and figs/obsdas/*combined*.png
"""
import json
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
from joblib import Parallel, delayed
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from obspy import read_inventory

from msdvv import ROOT
from msdvv.geometry import read_route, repeaters
from msdvv.obsdas import OUT, convergence, early_window, load_hourly, moving, moving_windows, neighbour_stack, stretch_series

warnings.filterwarnings("ignore", category=RuntimeWarning)
COMPS = ["Z", "R", "T"]
BAND, RNG = "b1", (0.12, 0.30)
CONV_MIN, NBR = 0.3, 2
INJECT = float(os.environ.get("INJECT", "0"))
INJ_AT = os.environ.get("INJECT_AT", "2026-01-01")
SUF = f"_inj{INJECT:g}" if INJECT else ""
od = OUT / "obsdas_combined"
od.mkdir(exist_ok=True)
fd = OUT / "figs" / "obsdas"

per = {}  # comp -> dict(dv=(3, n_stack, npath), W=..., conv=(npath,))
for comp in COMPS:
    t, H, lags, P = load_hourly(comp, BAND, INJECT, INJ_AT)
    dt = lags[1] - lags[0]
    H = neighbour_stack(H, P, dt, NBR)
    tend, i0, i1 = moving_windows(t)
    hour = np.round(t / 3600).astype(np.int64)
    ref = H.sum(0)
    t1, t2 = early_window(P.dist_km.values, lags[-1])
    WM = [(np.abs(lags) >= a) & (np.abs(lags) <= b) for a, b in zip(t1, t2)]
    stacks = [moving(H, i0, i1), moving(H, i0, i1, (hour % 2 == 0).astype(np.float32)), moving(H, i0, i1, (hour % 2 == 1).astype(np.float32))]
    del H

    def one(p):
        cv = convergence(stacks[0][:, p], ref[p], WM[p])
        if cv < CONV_MIN:
            return p, cv, None
        return p, cv, [stretch_series(C[:, p], ref[p], lags, t1[p], t2[p], RNG) for C in stacks]

    res = Parallel(n_jobs=24, prefer="threads")(delayed(one)(p) for p in range(len(P)))
    ns, npath = len(tend), len(P)
    dv = np.full((3, ns, npath), np.nan)
    W = np.zeros((3, ns, npath))
    conv = np.full(npath, np.nan)
    for p, cv, ser in res:
        conv[p] = cv
        if ser is not None:
            for k, (d_, w_) in enumerate(ser):
                dv[k, :, p], W[k, :, p] = d_, w_
    per[comp] = dict(t=tend, dv=dv, W=W, conv=conv)
    print(f"[{comp}] converging pairs {int((conv >= CONV_MIN).sum())} of {npath}", flush=True)
    del stacks

# ---- align components on a common time axis and combine per pair (1/err^2)
tt = np.unique(np.concatenate([per[c]["t"] for c in COMPS]))
npath = len(P)
num = np.zeros((3, len(tt), npath))
den = np.zeros((3, len(tt), npath))
for c in COMPS:
    j = np.searchsorted(tt, per[c]["t"])
    num[:, j] += np.nan_to_num(per[c]["dv"]) * per[c]["W"]
    den[:, j] += per[c]["W"]
pair_dv = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)  # (3, nt, npath)
pairs = P[["obs", "bin", "seg", "dist_km", "bin_lat", "bin_lon", "x_Alvin Canyon fault", "x_Daisy Bank fault", "x_SHR"]].copy()
for c in COMPS:
    pairs[f"conv_{c}"] = per[c]["conv"]
pairs["n_comp"] = sum((pairs[f"conv_{c}"] >= CONV_MIN).astype(int) for c in COMPS)
pairs["used"] = pairs.n_comp > 0
acf = pairs["x_Alvin Canyon fault"].values
dbf = pairs["x_Daisy Bank fault"].values
groups = {
    "HYSB1 control (no feature)": (pairs.obs == "HYSB1").values & ~acf & ~dbf & ~pairs.x_SHR.values,
    "HYS14 x Alvin Canyon fault": (pairs.obs == "HYS14").values & acf,
    "HYS14 x Daisy Bank fault": (pairs.obs == "HYS14").values & ~acf & dbf,
    "all HYSB1": (pairs.obs == "HYSB1").values,
    "all HYS14": (pairs.obs == "HYS14").values,
    "all pairs": np.ones(npath, bool),
}
# split series (map clusters, separated at the gaps between used bins along the cable):
#   red = HYS14 x ACF -> SB3, SC4 west (lon < -124.885), SC4 east;  blue = HYSB1 control -> SB3, SC4;  orange = HYS14 x Daisy Bank
C4W = pairs.bin_lon.values < -124.885
red, blue = groups["HYS14 x Alvin Canyon fault"], groups["HYSB1 control (no feature)"]
seg = pairs.seg.values
SPLIT = {
    "red: HYS14 x ACF -> SB3": red & (seg == "B3"),
    "red: HYS14 x ACF -> SC4 west": red & (seg == "C4") & C4W,
    "red: HYS14 x ACF -> SC4 east": red & (seg == "C4") & ~C4W,
    "blue: HYSB1 control -> SB3": blue & (seg == "B3"),
    "blue: HYSB1 control -> SC4": blue & (seg == "C4"),
    "orange: HYS14 x Daisy Bank": groups["HYS14 x Daisy Bank fault"],
}
SEG_GROUPS = list(SPLIT)
groups.update(SPLIT)
pairs["cluster"] = ""
for g, m in SPLIT.items():
    pairs.loc[m & pairs.used.values, "cluster"] = g
early = tt < pd.Timestamp("2026-01-08 12:00", tz="UTC").timestamp()
rows, ser = [], []
for g, m in groups.items():
    m = m & pairs.used.values
    if m.sum() < (3 if g in SEG_GROUPS else 2):
        continue
    gs = []
    for k in range(3):
        w = den[k][:, m]
        gs.append((np.where(w.sum(1) > 0, (num[k][:, m]).sum(1) / np.maximum(w.sum(1), 1e-30), np.nan), (w > 0).sum(1)))
    d = gs[1][0] - gs[2][0]
    d = d[np.isfinite(d)]
    floor = 1.4826 * np.median(np.abs(d - np.median(d))) / 2
    row = dict(group=g, n_pairs=int(m.sum()), floor_pct=100 * floor)
    for lab, mm in (("Dec23-Jan8", early), ("Jan9-28", ~early)):
        row[f"std_{lab}"] = 100 * np.nanstd(gs[0][0][mm])
        row[f"pairs_per_stack_{lab}"] = float(np.median(gs[0][1][mm]))
    rows.append(row)
    ser.append(pd.DataFrame(dict(group=g, t=tt, dv=gs[0][0], n=gs[0][1], floor=floor)))
SUM = pd.DataFrame(rows)
SER = pd.concat(ser, ignore_index=True)
SUM.to_csv(od / f"summary{SUF}.csv", index=False)
SER.to_csv(od / f"series_24h{SUF}.csv", index=False)
pairs.to_csv(od / f"pairs{SUF}.csv", index=False)
pd.set_option("display.width", 220)
print(SUM.round(3).to_string(index=False))
if INJECT:
    sys.exit(0)

# ---- figures: series (+ Hs) and map of the pairs used
_envf = OUT / "dvv_24h" / "environment_timeseries.csv"  # wave height for the figure (January only; optional)
env = pd.read_csv(_envf, index_col=0) if _envf.exists() else pd.DataFrame()
show = ["HYSB1 control (no feature)", "HYS14 x Alvin Canyon fault", "HYS14 x Daisy Bank fault", "all pairs"]
show = [g for g in show if g in set(SER.group)]
fig, axs = plt.subplots(len(show) + 1, 1, figsize=(14, 2.2 * (len(show) + 1)), sharex=True)
for ax, g in zip(axs, show):
    s = SER[SER.group == g].set_index("t").reindex(np.arange(tt[0], tt[-1] + 1, 3600.0)).reset_index()
    s["floor"] = s.floor.ffill().bfill()
    td = pd.to_datetime(s.t, unit="s", utc=True)
    ax.fill_between(td, 100 * (s.dv - s.floor), 100 * (s.dv + s.floor), color="C0", alpha=0.25)
    ax.plot(td, 100 * s.dv, "C0", lw=0.9)
    r = SUM[SUM.group == g].iloc[0]
    ax.set_title(f"{g}: {r.n_pairs} pairs, floor {r.floor_pct:.3f}%, ~{r['pairs_per_stack_Dec23-Jan8']:.0f} / {r['pairs_per_stack_Jan9-28']:.0f} pairs per stack (Dec 23-Jan 8 / Jan 9-28)", fontsize=9, loc="left")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylabel("dv/v %")
    ax.set_ylim(-0.8, 0.8)
if "Hs" in env:
    axs[-1].plot(pd.to_datetime(env.index.values, unit="s", utc=True), env.Hs, "k", lw=0.8)
axs[-1].set_ylabel("Hs (m)")
fig.suptitle("Seismometer-DAS dv/v, Z+R+T combined, neighbour-stacked, b1 0.12-0.3 Hz early window, 24-h back-stacked step 1 h (shaded = even/odd floor)", fontsize=10)
fig.tight_layout()
fig.savefig(fd / "obsdas_combined_series_24h.png", dpi=110)
plt.close(fig)

segs = [g for g in SEG_GROUPS if g in set(SER.group)]
SCOL = dict(zip(SEG_GROUPS, ["#8b0000", "#e31a1c", "#fb9a99", "#08306b", "#6baed6", "tab:orange"]))
fig, axs = plt.subplots(len(segs) + 1, 1, figsize=(14, 2.0 * (len(segs) + 1)), sharex=True)
for ax, g in zip(axs, segs):
    s = SER[SER.group == g].set_index("t").reindex(np.arange(tt[0], tt[-1] + 1, 3600.0)).reset_index()
    s["floor"] = s.floor.ffill().bfill()
    td = pd.to_datetime(s.t, unit="s", utc=True)
    ax.fill_between(td, 100 * (s.dv - s.floor), 100 * (s.dv + s.floor), color=SCOL[g], alpha=0.25)
    ax.plot(td, 100 * s.dv, color=SCOL[g], lw=0.9)
    r = SUM[SUM.group == g].iloc[0]
    dk = pairs[groups[g] & pairs.used.values].dist_km
    ax.set_title(f"{g}: {r.n_pairs} pairs ({dk.min():.0f}-{dk.max():.0f} km), floor {r.floor_pct:.3f}%, "
                 f"~{r['pairs_per_stack_Dec23-Jan8']:.0f} / {r['pairs_per_stack_Jan9-28']:.0f} pairs per stack (Dec 23-Jan 8 / Jan 9-28)", fontsize=9, loc="left")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylabel("dv/v %")
    ax.set_ylim(-0.8, 0.8)
if "Hs" in env:
    axs[-1].plot(pd.to_datetime(env.index.values, unit="s", utc=True), env.Hs, "k", lw=0.8)
axs[-1].set_ylabel("Hs (m)")
fig.suptitle("Seismometer-DAS dv/v per path cluster (Z+R+T, neighbour-stacked, b1 early window), 24-h back-stacked step 1 h; shaded = even/odd floor", fontsize=10)
fig.tight_layout()
fig.savefig(fd / "obsdas_split_series_24h.png", dpi=110)
plt.close(fig)

inv = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
fc = json.load(open(ROOT / "resources/qfaults_oregon_margin_strikeslip.geojson"))
rt, rp = read_route("south"), repeaters("south")
gcol = {"HYSB1 control (no feature)": "tab:blue", "HYS14 x Alvin Canyon fault": "tab:red", "HYS14 x Daisy Bank fault": "tab:orange"}
fig, axs = plt.subplots(1, 2, figsize=(19, 8.5), layout="constrained")
for ax, (xl, yl) in zip(axs, (((-125.8, -123.85), (44.25, 45.3)), ((-125.55, -124.75), (44.30, 44.66)))):
    ax.plot(rt.lon, rt.lat, "-", color="0.8", lw=2)
    for f in fc["features"]:
        c = np.array(f["geometry"]["coordinates"])
        ax.plot(c[:, 0], c[:, 1], color="saddlebrown", lw=1.8)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(-125.148 + 5 / (111.2 * np.cos(np.radians(44.569))) * np.cos(th), 44.569 + 5 / 111.2 * np.sin(th), "--", color="goldenrod")
    un = pairs[~pairs.used]
    ax.add_collection(LineCollection([[(crd[r.obs][1], crd[r.obs][0]), (r.bin_lon, r.bin_lat)] for r in un.itertuples()], colors="0.75", linewidths=0.35, alpha=0.5))
    for g, col in gcol.items():
        u = pairs[groups[g] & pairs.used.values]
        ax.add_collection(LineCollection([[(crd[r.obs][1], crd[r.obs][0]), (r.bin_lon, r.bin_lat)] for r in u.itertuples()], colors=col,
                                         linewidths=[0.6 + 0.6 * n for n in u.n_comp], alpha=0.85))
    B = pd.read_csv(OUT / "obsdas" / "bins.csv")
    ax.scatter(B.lon, B.lat, s=10, color="k", zorder=5)
    ax.plot(rp.lon, rp.lat, "k^", ms=9, zorder=6)
    for s_ in ("HYSB1", "HYS14"):
        ax.plot(crd[s_][1], crd[s_][0], "v", color="tab:blue" if s_ == "HYSB1" else "tab:red", mec="k", ms=14, zorder=7)
        ax.annotate(s_, (crd[s_][1], crd[s_][0]), xytext=(7, 7), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set(xlim=xl, ylim=yl, xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1 / np.cos(np.radians(44.6)))
axs[0].set_title(f"Seismometer-DAS pairs used ({int(pairs.used.sum())} of {npath}); line width = number of components converging (Z/R/T)", fontsize=10)
axs[1].set_title("zoom on Hydrate Ridge", fontsize=10)
axs[0].legend(handles=[Line2D([], [], color=c, lw=2, label=f"{g} ({int((groups[g] & pairs.used.values).sum())})") for g, c in gcol.items()]
              + [Line2D([], [], color="0.75", lw=0.8, label="not converging"), Line2D([], [], color="saddlebrown", lw=2, label="faults (Qfaults)"),
                 Line2D([], [], color="goldenrod", ls="--", label="S. Hydrate Ridge (5 km)")], fontsize=9, loc="lower left")
fig.savefig(fd / "obsdas_combined_pairs_map.png", dpi=110)
plt.close(fig)
fig, ax = plt.subplots(figsize=(13, 8.5), layout="constrained")
ax.plot(rt.lon, rt.lat, "-", color="0.8", lw=2)
for f in fc["features"]:
    c = np.array(f["geometry"]["coordinates"])
    ax.plot(c[:, 0], c[:, 1], color="saddlebrown", lw=1.8)
th = np.linspace(0, 2 * np.pi, 100)
ax.plot(-125.148 + 5 / (111.2 * np.cos(np.radians(44.569))) * np.cos(th), 44.569 + 5 / 111.2 * np.sin(th), "--", color="goldenrod")
for g in segs:
    u = pairs[groups[g] & pairs.used.values]
    ax.add_collection(LineCollection([[(crd[r.obs][1], crd[r.obs][0]), (r.bin_lon, r.bin_lat)] for r in u.itertuples()], colors=[SCOL[g]], linewidths=1.2, alpha=0.85))
B = pd.read_csv(OUT / "obsdas" / "bins.csv")
for sg, gb in B.groupby("seg"):
    ax.scatter(gb.lon, gb.lat, s=10, color="k", zorder=5)
    ax.annotate(f"S{sg}", (gb.lon.mean(), gb.lat.mean()), xytext=(8, -12), textcoords="offset points", fontsize=11, fontweight="bold", color="0.25")
ax.plot(rp.lon, rp.lat, "k^", ms=9, zorder=6)
for s_ in ("HYSB1", "HYS14"):
    ax.plot(crd[s_][1], crd[s_][0], "v", color="w", mec="k", ms=14, zorder=7)
    ax.annotate(s_, (crd[s_][1], crd[s_][0]), xytext=(7, 7), textcoords="offset points", fontsize=10, fontweight="bold")
ax.set(xlim=(-125.55, -124.5), ylim=(44.3, 44.8), xlabel="longitude", ylabel="latitude",
       title="Seismometer-DAS pairs used, split into clusters (one dv/v series each)")
ax.set_aspect(1 / np.cos(np.radians(44.6)))
ax.legend(handles=[Line2D([], [], color=SCOL[g], lw=2.5, label=f"{g} ({int((groups[g] & pairs.used.values).sum())})") for g in segs], fontsize=9, loc="upper right")
fig.savefig(fd / "obsdas_split_pairs_map.png", dpi=110)
print("done")
