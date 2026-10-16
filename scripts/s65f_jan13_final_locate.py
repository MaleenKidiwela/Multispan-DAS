"""Item 8b (final): Jan 13 2026 ML 2.0 located with multi-span DAS wavefronts + Hydrate Ridge OBS, absolute timing.

DAS: per-channel S times from s65e (das_location_jan13_channel_times.csv): stretch-stack onset (20 % of the stack
maximum) + the channel's envelope-cross-correlation delay within its stretch (north 67-100, south 35-46,
south 98-115 km). DAS time -> UTC with the 8-Hz-stream latency 5.80 s (s67; same on both cables, measured against the
simultaneous 100-m stream, no earthquake used). OBS: PhaseNet-obs P/S picks (s61).
Model: homogeneous Vs, Vp/Vs, depth; one origin time shared by all data (median of all residuals).
Misfit: L1, each DAS stretch and the OBS set weighted equally (4 groups). Runs: DAS only (absolute), OBS only, joint;
and joint with the DAS latency +-2 s (sensitivity to the D256 calibration uncertainty).
Uncertainty: misfit <= min + 0.3 s. PNSN (onshore only, gap 250 deg) shown for comparison only.
Outputs: outputs_100m/das_location_jan13_final.json, figs/das_location_jan13_final.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read_inventory
from pyproj import Geod

from msdvv.geometry import read_route

G = Geod(ellps="WGS84")
T0 = pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp()
CAT = dict(lat=45.1618, lon=-124.390)
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
LAT_D512, DMIS = 5.80, 0.3
D = pd.read_csv(f"{OUT}/das_location_jan13_channel_times.csv")
inv = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
O = pd.read_csv("/data/wsd02/multispan_dvv/obs/dl_scan_picks_v2.csv")
O = O[(O.t > T0 - 30) & (O.t < T0 + 120)].copy()
O["tt"] = O.t - T0
O["lat"] = [crd[s][0] for s in O.sta]
O["lon"] = [crd[s][1] for s in O.sta]
isP = (O.phase == "P").values
LATG = np.arange(44.80, 45.60, 0.01)
LONG = np.arange(-125.30, -123.85, 0.01)
ZS = [1, 3, 5, 8, 10, 12, 15, 20, 25, 30]
VS = np.arange(2.6, 4.01, 0.1)
KPS = (1.73, 1.8, 1.9, 2.0)
groups = [D.stretch.values == k for k in D.stretch.unique()]


def solve(use_das, use_obs, lat_das=LAT_D512):
    mis = np.full((len(LATG), len(LONG)), np.inf)
    best = (np.inf,)
    for ia, la in enumerate(LATG):
        def dist(df):
            _, _, d = G.inv(np.tile(df.lon.values, len(LONG)), np.tile(df.lat.values, len(LONG)), np.repeat(LONG, len(df)), np.full(len(LONG) * len(df), la))
            return d.reshape(len(LONG), len(df)) / 1e3
        dD, dO = dist(D), dist(O)
        for z in ZS:
            rD, rO = np.hypot(dD, z), np.hypot(dO, z)
            for v in VS:
                for kps in (KPS if use_obs else (1.8,)):
                    parts = []
                    if use_das:
                        parts.append(D.t_abs_das.values[None] - lat_das - rD / v)
                    if use_obs:
                        parts.append(O.tt.values[None] - rO / np.where(isP, v * kps, v)[None])
                    allr = np.concatenate(parts, axis=1)
                    t0 = np.median(allr, axis=1, keepdims=True)
                    terms = []
                    if use_das:
                        rd = parts[0] - t0
                        terms += [np.mean(np.abs(rd[:, gm]), axis=1) for gm in groups]
                    if use_obs:
                        terms.append(np.mean(np.abs(parts[-1] - t0), axis=1))
                    tot = np.mean(terms, axis=0)
                    k = int(np.argmin(tot))
                    mis[ia] = np.minimum(mis[ia], tot)
                    if tot[k] < best[0]:
                        best = (float(tot[k]), dict(misfit=float(tot[k]), lat=float(la), lon=float(LONG[k]), depth_km=z, vs=round(float(v), 2),
                                                    vp_vs=kps if use_obs else None, origin_minus_PNSN_s=float(t0[k, 0]),
                                                    group_L1_s=[round(float(t[k]), 2) for t in terms]))
    return best[1], mis


res, maps = {}, {}
RUNS = [("DAS only (3 stretches, latency 5.8 s)", True, False, LAT_D512), ("OBS only", False, True, LAT_D512),
        ("joint DAS + OBS", True, True, LAT_D512), ("joint, latency 3.8 s", True, True, LAT_D512 - 2), ("joint, latency 7.8 s", True, True, LAT_D512 + 2)]
for name, ud, uo, lat_ in RUNS:
    b, mis = solve(ud, uo, lat_)
    ia, io = np.where(mis <= b["misfit"] + DMIS)
    _, _, ds = G.inv(CAT["lon"], CAT["lat"], b["lon"], b["lat"])
    b.update(dist_to_PNSN_km=round(ds / 1e3, 1), region=dict(lat=(round(LATG[ia].min(), 2), round(LATG[ia].max(), 2)),
                                                            lon=(round(LONG[io].min(), 2), round(LONG[io].max(), 2))),
             region_area_km2=round(len(ia) * 1.112 * 1.112 * np.cos(np.radians(45.2)), 0))
    res[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()}
    maps[name] = mis
    print(name, res[name], flush=True)
json.dump(res, open(f"{OUT}/das_location_jan13_final.json", "w"), indent=1)

show = ["DAS only (3 stretches, latency 5.8 s)", "OBS only", "joint DAS + OBS"]
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
for ax, name in zip(axs, show):
    r_ = res[name]
    pc = ax.pcolormesh(LONG, LATG, maps[name] - r_["misfit"], vmin=0, vmax=2, cmap="viridis_r", shading="auto")
    ax.contour(LONG, LATG, maps[name] - r_["misfit"], levels=[DMIS], colors="w", linewidths=1.3)
    for c in ("north", "south"):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.6", lw=1)
    if "DAS" in name:
        sc = ax.scatter(D.lon, D.lat, c=D.t_abs_das - LAT_D512, s=8, cmap="plasma", vmin=0, vmax=25, zorder=4)
    oo = O.drop_duplicates("sta")
    ax.plot(oo.lon, oo.lat, "^", ms=9, color="w" if "OBS" in name else "0.5", mec="k", label="Hydrate Ridge OBS")
    ax.plot(CAT["lon"], CAT["lat"], "*", ms=16, color="0.85", mec="k", label="PNSN (onshore only; comparison)")
    ax.plot(r_["lon"], r_["lat"], "*", ms=18, color="tab:red", mec="k", label=f"this solution ({r_['dist_to_PNSN_km']} km from PNSN)")
    ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude")
    ax.set_title(f"{name}\nmisfit {r_['misfit']:.2f} s, z {r_['depth_km']} km, Vs {r_['vs']}, origin {r_['origin_minus_PNSN_s']:+.1f} s; "
                 f"region {r_['region_area_km2']:.0f} km$^2$", fontsize=9)
    ax.set_aspect(1 / np.cos(np.radians(45.2)))
    ax.legend(fontsize=7, loc="lower left")
axs[0].set_ylabel("latitude")
fig.colorbar(pc, ax=axs, shrink=0.7, label=f"misfit - minimum (s); white: +{DMIS} s")
fig.suptitle("Jan 13 2026 ML 2.0: multi-span DAS S wavefronts (UTC via measured 5.8 s latency) + Hydrate Ridge OBS P/S")
fig.savefig(f"{OUT}/figs/das_location_jan13_final.png", dpi=95, bbox_inches="tight")
