"""Item 8b (v4): joint location of the Jan 13 2026 ML 2.0 from multi-span DAS S onsets + Hydrate Ridge OBS P/S picks.

DAS: s65c onset bins (spatially coherent stretches with event SNR >= 33x), onset re-picked with an AIC change-point
(Maeda 1985) on the bin's 2-3.6 Hz energy in [t_half - 25 s, t_half + 3 s] (t_half = s65c half-max onset), which suits
the emergent onsets at long range better than the half-max. DAS times are relative: one free offset per cable
(= origin time + that cable's latency).
OBS: PhaseNet-obs picks from the s61 scan (dl_scan_picks_v2.csv): HYS11 P, HYS11/12/13 and HYSB1 S; absolute UTC.
DAS timing modes: one free offset per cable; one common offset (s67 shows both cables share the same latency); or
absolute (latency 5.80 s from s67, measured against the simultaneous 100-m stream - no earthquake used).
Model: homogeneous Vs (2.6-4.2 km/s) and Vp/Vs (1.73, 1.8, 1.9), depth grid, lat/lon 0.01 deg.
Misfit = mean |res| over DAS + mean |res| over OBS (each data class weighted equally). Origin time from the OBS
residual median; per-cable DAS offsets from the DAS residual medians -> DAS latency per cable = offset - origin
(now independent of the PNSN origin time). Runs: DAS only (no absolute time), OBS only, joint.
Uncertainty: misfit <= min + DMIS (0.5 s). The PNSN location (onshore only, azimuthal gap 250 deg) is a comparison only.
Outputs: outputs_100m/das_location_jan13_joint.json, figs/das_location_jan13_joint.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read_inventory
from pyproj import Geod
from scipy.signal import butter, sosfiltfilt

from msdvv.geometry import read_route
from msdvv.io import load_file

G = Geod(ellps="WGS84")
T0 = pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp()
CAT = dict(lat=45.1618, lon=-124.390)
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
DMIS = 0.5
LAT_D512 = 5.80  # s67: 8-Hz stream latency, both cables (100-m stream D256 25.2 s - 19.40 s), no earthquake used

# ---- DAS: AIC re-pick on the s65c bins
P = pd.read_csv(f"{OUT}/das_location_jan13_onsets.csv")
rows = []
for cab in ("north", "south"):
    f = load_file(sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_{cab}_*"))[0])
    ch = f.channels()
    tt = f.times - T0
    w = (tt > -120) & (tt < 150)
    x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[:, w].astype(np.float64), axis=1)
    tw = tt[w]
    for r in P[P.cable == cab].itertuples():
        sel = np.argsort(np.abs(ch.opt_km.values - r.opt_km))[:5]
        e = (x[sel] ** 2).mean(0)
        m = (tw >= r.t - 25) & (tw <= r.t + 3)
        y = e[m]
        n = len(y)
        k = np.arange(5, n - 5)
        aic = np.array([kk * np.log(np.var(y[:kk]) + 1e-30) + (n - kk) * np.log(np.var(y[kk:]) + 1e-30) for kk in k])
        rows.append(dict(r._asdict(), t_aic=float(tw[m][k[np.argmin(aic)]])))
D = pd.DataFrame(rows).drop(columns="Index")
D = D[(D.t_aic - D.t).abs() <= 15].reset_index(drop=True)
print(D.groupby("stretch").agg(n=("t", "size"), t_half=("t", "median"), t_aic=("t_aic", "median")).round(2).to_string())

# ---- OBS picks (absolute)
inv = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
O = pd.read_csv("/data/wsd02/multispan_dvv/obs/dl_scan_picks_v2.csv")
O = O[(O.t > T0 - 30) & (O.t < T0 + 120)].copy()
O["tt"] = O.t - T0
O["lat"] = [crd[s][0] for s in O.sta]
O["lon"] = [crd[s][1] for s in O.sta]
print(O[["sta", "phase", "tt", "prob"]].round(2).to_string(index=False))

LATG = np.arange(44.70, 45.60, 0.01)
LONG = np.arange(-125.40, -123.85, 0.01)
ZS = [1, 5, 10, 15, 20, 25]
VS = np.arange(2.6, 4.21, 0.1)
KPS = (1.73, 1.8, 1.9)


def dist(lo, la, lat_):
    _, _, d = G.inv(np.tile(lo, len(LONG)), np.tile(la, len(LONG)), np.repeat(LONG, len(lo)), np.full(len(LONG) * len(lo), lat_))
    return d.reshape(len(LONG), len(lo)) / 1e3


def solve(use_das, use_obs, das_mode="free"):
    mis = np.full((len(LATG), len(LONG)), np.inf)
    best = (np.inf,)
    cab = D.cable.values
    isP = (O.phase == "P").values
    for ia, la in enumerate(LATG):
        dD, dO = dist(D.lon.values, D.lat.values, la), dist(O.lon.values, O.lat.values, la)
        for z in ZS:
            rD, rO = np.hypot(dD, z), np.hypot(dO, z)
            for v in VS:
                for kps in KPS:
                    tot = np.zeros(len(LONG))
                    if use_obs:
                        res_o = O.tt.values[None] - rO / np.where(isP, v * kps, v)[None]
                        t0 = np.median(res_o, axis=1, keepdims=True)
                        tot += np.mean(np.abs(res_o - t0), axis=1)
                    if use_das:
                        res_d = D.t_aic.values[None] - rD / v
                        off = np.zeros_like(res_d)
                        if das_mode == "free":
                            for c in ("north", "south"):
                                mm = cab == c
                                off[:, mm] = np.median(res_d[:, mm], axis=1, keepdims=True)
                        elif das_mode == "common":
                            off[:] = np.median(res_d, axis=1, keepdims=True)
                        else:  # absolute: DAS time - latency = UTC; origin shared with the OBS (or free if no OBS)
                            res_d = res_d - LAT_D512
                            if not use_obs:
                                t0 = np.median(res_d, axis=1, keepdims=True)
                            off[:] = t0
                        tot += np.mean(np.abs(res_d - off), axis=1)
                    k = np.argmin(tot)
                    mis[ia] = np.minimum(mis[ia], tot)
                    if tot[k] < best[0]:
                        out = dict(misfit=float(tot[k]), lat=float(la), lon=float(LONG[k]), depth_km=z, vs=float(v), vp_vs=kps)
                        if use_obs:
                            out["origin_minus_PNSN_s"] = float(t0[k, 0])
                        if use_das:
                            for c in ("north", "south"):
                                out[f"das_offset_{c}_s"] = float(off[k, cab == c][0])
                                if use_obs and das_mode != "absolute":
                                    out[f"das_latency_{c}_s"] = float(off[k, cab == c][0] - t0[k, 0])
                        best = (tot[k], out)
        if kps == KPS[-1] and ia % 20 == 0:
            print(".", end="", flush=True)
    return best[1], mis


res, maps = {}, {}
RUNS = (("DAS only, one offset per cable", True, False, "free"), ("DAS only, common latency (s67)", True, False, "common"),
        ("OBS only", False, True, "free"), ("joint, DAS latency free per cable", True, True, "free"),
        ("joint, DAS latency 5.8 s (s67)", True, True, "absolute"))
for name, ud, uo, mode in RUNS:
    b, mis = solve(ud, uo, mode)
    ia, io = np.where(mis <= b["misfit"] + DMIS)
    _, _, ds = G.inv(CAT["lon"], CAT["lat"], b["lon"], b["lat"])
    b.update(dist_to_PNSN_km=round(ds / 1e3, 1), region=dict(lat=(round(LATG[ia].min(), 2), round(LATG[ia].max(), 2)),
                                                            lon=(round(LONG[io].min(), 2), round(LONG[io].max(), 2))),
             region_area_km2=round(len(ia) * 1.112 * 1.112 * np.cos(np.radians(45.2)), 0))
    res[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()}
    maps[name] = mis
    print("\n", name, res[name], flush=True)
json.dump(res, open(f"{OUT}/das_location_jan13_joint.json", "w"), indent=1)

fig, axs = plt.subplots(1, len(RUNS), figsize=(7 * len(RUNS), 7))
for ax, name in zip(axs, res):
    r_ = res[name]
    pc = ax.pcolormesh(LONG, LATG, maps[name] - r_["misfit"], vmin=0, vmax=3, cmap="viridis_r", shading="auto")
    ax.contour(LONG, LATG, maps[name] - r_["misfit"], levels=[DMIS], colors="w", linewidths=1.3)
    for c in ("north", "south"):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.6", lw=1)
    if "DAS" in name or "joint" in name:
        sc = ax.scatter(D.lon, D.lat, c=D.t_aic, s=16, cmap="plasma", edgecolor="k", lw=0.3, zorder=4)
    oo = O.drop_duplicates("sta")
    ax.plot(oo.lon, oo.lat, "^", ms=9, color="w" if "OBS" in name or "joint" in name else "0.5", mec="k", label="Hydrate Ridge OBS")
    ax.plot(CAT["lon"], CAT["lat"], "*", ms=16, color="0.85", mec="k", label="PNSN (onshore only; comparison)")
    ax.plot(r_["lon"], r_["lat"], "*", ms=18, color="tab:red", mec="k", label=f"this solution ({r_['dist_to_PNSN_km']} km from PNSN)")
    extra = (f", origin {r_['origin_minus_PNSN_s']:+.1f} s" if "origin_minus_PNSN_s" in r_ else "") + \
            (f"\nDAS latency N {r_['das_latency_north_s']:.1f} s / S {r_['das_latency_south_s']:.1f} s" if "das_latency_north_s" in r_ else "")
    ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude",
           title=f"{name}: misfit {r_['misfit']:.2f} s, z {r_['depth_km']} km, Vs {r_['vs']:.1f}{extra}")
    ax.set_aspect(1 / np.cos(np.radians(45.2)))
    ax.legend(fontsize=7, loc="lower left")
axs[0].set_ylabel("latitude")
fig.colorbar(pc, ax=axs, shrink=0.7, label="misfit - minimum (s); white contour: +0.5 s")
fig.suptitle("Jan 13 2026 ML 2.0: multi-span DAS S onsets (relative, one offset per cable) + Hydrate Ridge OBS P/S (absolute)")
fig.savefig(f"{OUT}/figs/das_location_jan13_joint.png", dpi=95, bbox_inches="tight")
