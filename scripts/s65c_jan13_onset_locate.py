"""Item 8b (v3): locate the Jan 13 2026 ML 2.0 from S-envelope ONSETS on the fiber stretches that clearly record it.

Why: s65b (envelope image along both cables) shows the event clearly only on 4 stretches (north A1 25-35 km, north
A2 67-105 km, south A1 35-45 km, south A2 97-115 km; the rest of the fiber is noise-dominated), so the s65 envelope
back-projection over all bins with SNR >= 4.5x was dominated by noise bins and failed.
Bins: s65 1-km bins (2-3.6 Hz log envelope / pre-event median, 2-s smoothing) with event SNR >= SNR_MIN (3.5 = 33x).
Onset per bin: first time in [0, 120] s where the log envelope reaches half its maximum in that window (rising edge;
independent of any location). Location: grid search lat/lon 0.01 deg x depth x homogeneous V, L1 misfit; origin
time + latency solved as ONE offset per cable (median) - variant 'common' uses one offset for both cables.
Spatial coherence: only contiguous runs of >= 4 lit bins; onsets > 0.5 s and within 3 s of the running median of\n5 neighbours.\nUncertainty: grid nodes with L1 <= L1min + DL1 (0.5 s). Per-stretch subsets show what each span contributes.
Outputs: outputs_100m/das_location_jan13_onsets.csv, das_location_jan13_v3.json, figs/das_location_jan13_v3.png
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from msdvv.geometry import read_route
from msdvv.io import load_file

G = Geod(ellps="WGS84")
CAT = dict(t=pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp(), lat=45.1618, lon=-124.390)
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
SNR_MIN, DL1, RUN_MIN = 3.5, 0.5, 4
rows = []
for cab in ("north", "south"):
    f = load_file(sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_{cab}_*"))[0])
    ch = f.channels()
    g = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    lat = np.interp(ch.opt_km, g.opt_km, g.lat)
    lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    tt = f.times - CAT["t"]
    w = (tt > -120) & (tt < 150)
    x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[:, w].astype(np.float64), axis=1)
    tw = tt[w]
    a2 = np.abs(hilbert(x, axis=1)) ** 2
    for blk, gb in ch.groupby("block"):
        idx = gb.index.values
        for k in range(0, len(idx) - 4, 5):
            sel = idx[k:k + 5]
            if np.any(gb.pos_in_block.loc[sel] < 3) or np.any(gb.pos_in_block.loc[sel] > gb.block_len.iloc[0] - 4):
                continue
            env = uniform_filter1d(a2[sel].mean(0), int(round(2 * f.fs)))
            e = np.log(env / np.median(env[(tw > -90) & (tw < -5)]))
            m = (tw >= 0) & (tw <= 120)
            em = e[m].max()
            if em < SNR_MIN:
                continue
            j = np.argmax(e[m] >= 0.5 * em)
            j0 = max(j - 1, 0)  # sub-sample linear interpolation of the half-max crossing
            t_on = np.interp(0.5 * em, [e[m][j0], e[m][j]], [tw[m][j0], tw[m][j]]) if j > 0 else tw[m][j]
            rows.append(dict(cable=cab, block=blk, span=int(gb.span.iloc[0]) if "span" in gb else -1, opt_km=float(ch.opt_km[sel].mean()),
                             lat=float(lat[sel].mean()), lon=float(lon[sel].mean()), snr=float(em), t=float(t_on)))
P = pd.DataFrame(rows).sort_values(["cable", "opt_km"]).reset_index(drop=True)
# spatial coherence (data-driven, no location assumption): keep contiguous runs of >= RUN_MIN lit bins (gaps <= 1.6 km)
# and drop onsets at the window edge or > 3 s from the running median of 5 neighbouring bins in the run
P["run"] = (P.groupby("cable").opt_km.diff().fillna(99) > 1.6).cumsum()
P = P[P.groupby("run").t.transform("size") >= RUN_MIN]
P = P[P.t > 0.5]
P["t_rm"] = P.groupby("run").t.transform(lambda s: s.rolling(5, center=True, min_periods=3).median())
P = P[(P.t - P.t_rm).abs() <= 3].reset_index(drop=True)
P["stretch"] = P.cable + " " + P.opt_km.groupby(P.run).transform("min").round().astype(int).astype(str) + "-" + P.opt_km.groupby(P.run).transform("max").round().astype(int).astype(str) + " km"
P.to_csv(f"{OUT}/das_location_jan13_onsets.csv", index=False)
print(len(P), "bins;", P.groupby("cable").size().to_dict())
print(P.groupby("stretch").agg(n=("t", "size"), km0=("opt_km", "min"), km1=("opt_km", "max"), t_med=("t", "median"),
                                        t_min=("t", "min"), t_max=("t", "max"), lon0=("lon", "min"), lon1=("lon", "max")).round(2).to_string())

LATG = np.arange(44.80, 45.60, 0.01)
LONG = np.arange(-125.30, -123.85, 0.01)
ZS = [1, 5, 10, 15, 20]
VS = np.arange(2.4, 4.61, 0.2)


def locate(Q, per_cable=True):
    n = len(Q)
    cabs = Q.cable.values
    mis = np.full((len(LATG), len(LONG)), np.inf)
    best = (np.inf,)
    for ia, la in enumerate(LATG):
        _, _, d = G.inv(np.tile(Q.lon.values, len(LONG)), np.tile(Q.lat.values, len(LONG)), np.repeat(LONG, n), np.full(len(LONG) * n, la))
        d = d.reshape(len(LONG), n) / 1e3
        for z in ZS:
            r_ = np.hypot(d, z)
            for v in VS:
                r = Q.t.values[None] - r_ / v
                groups = ("north", "south") if per_cable else (None,)
                off = np.zeros_like(r)
                for c in groups:
                    mm = (cabs == c) if c else np.ones(n, bool)
                    if mm.any():
                        off[:, mm] = np.median(r[:, mm], axis=1, keepdims=True)
                L1 = np.mean(np.abs(r - off), axis=1)
                k = np.argmin(L1)
                mis[ia] = np.minimum(mis[ia], L1)
                if L1[k] < best[0]:
                    o = {c: float(np.median(r[k, cabs == c])) for c in ("north", "south") if (cabs == c).any()}
                    best = (float(L1[k]), float(la), float(LONG[k]), z, float(v), o)
    return best, mis


res, maps = {}, {}
runs = [("both cables, offset per cable", P, True), ("both cables, common offset", P, False),
        ("north only", P[P.cable == "north"], True), ("south only", P[P.cable == "south"], True)]
for name, Q, pc in runs:
    if len(Q) < 4:
        continue
    best, mis = locate(Q.reset_index(drop=True), pc)
    ia, io = np.where(mis <= best[0] + DL1)
    _, _, ds = G.inv(CAT["lon"], CAT["lat"], best[2], best[1])
    res[name] = dict(L1_s=round(best[0], 2), lat=round(best[1], 3), lon=round(best[2], 3), depth_km=best[3], v_kms=round(best[4], 2),
                     offsets_s={k: round(v, 2) for k, v in best[5].items()}, n_bins=len(Q), dist_to_PNSN_km=round(ds / 1e3, 1),
                     region=dict(lat=(round(LATG[ia].min(), 2), round(LATG[ia].max(), 2)), lon=(round(LONG[io].min(), 2), round(LONG[io].max(), 2))))
    maps[name] = mis
    print(name, res[name], flush=True)
json.dump(res, open(f"{OUT}/das_location_jan13_v3.json", "w"), indent=1)

names = list(maps)
fig, axs = plt.subplots(1, len(names), figsize=(6 * len(names), 6.5))
for ax, name in zip(axs, names):
    r_ = res[name]
    pc_ = ax.pcolormesh(LONG, LATG, maps[name] - r_["L1_s"], vmin=0, vmax=3, cmap="viridis_r", shading="auto")
    ax.contour(LONG, LATG, maps[name] - r_["L1_s"], levels=[DL1], colors="w", linewidths=1.2)
    for c in ("north", "south"):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.6", lw=1)
    Q = P if "both" in name else P[P.cable == name.split()[0]]
    sc = ax.scatter(Q.lon, Q.lat, c=Q.t, s=14, cmap="plasma", edgecolor="k", lw=0.3, zorder=4)
    ax.plot(CAT["lon"], CAT["lat"], "*", ms=16, color="0.85", mec="k", label="PNSN (onshore-only, gap 250 deg)")
    ax.plot(r_["lon"], r_["lat"], "*", ms=18, color="tab:red", mec="k", label="DAS onsets")
    ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude",
           title=f"{name}: {r_['n_bins']} bins\nL1 {r_['L1_s']} s, z {r_['depth_km']} km, V {r_['v_kms']} km/s", )
    ax.set_aspect(1 / np.cos(np.radians(45.2)))
    ax.legend(fontsize=7, loc="lower left")
axs[0].set_ylabel("latitude")
fig.colorbar(sc, ax=axs, shrink=0.6, label="S onset (s after PNSN origin, DAS time)", location="right")
fig.suptitle(f"Jan 13 2026 ML 2.0 from S-envelope onsets on the multi-span stretches that record it (SNR >= {SNR_MIN} log); "
             f"white: misfit <= min + {DL1} s; shading: misfit - min (s)")
fig.savefig(f"{OUT}/figs/das_location_jan13_v3.png", dpi=95, bbox_inches="tight")
