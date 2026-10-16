"""Item 8b: locate the Jan 13 2026 05:12:47 ML 2.0 earthquake with the DAS alone, using all spans of both cables,
by envelope back-projection (multi-span "beamforming for location").

Data: 8-Hz stream files around the event (raw_hf/eq20260113; 211.7 m channels, 7.45 Hz, D512), true per-sample
times. Channels averaged (energy) into ~1-km bins (5 channels) within each mask block, 3 channels kept clear of
block edges; bins with anomalous noise (> 3x median) skipped.
Envelope: 2-3.6 Hz, |Hilbert|^2 mean over the bin, smoothed 2 s, divided by the pre-event median ([-90, -5] s
relative to the catalogue origin), e = clip(log(env/noise), 0, None), sampled at 0.25 s.
Bins used: event SNR (max e in [0, 120] s) >= SNR_MIN - a data-driven cut that does not use the catalogue location.
Back-projection: for each trial hypocentre (lat/lon 0.01 deg, depth Z, homogeneous S velocity V) and origin time t0
(DAS time), B = mean over bins of e(t0 + d/V [+ dT_north for north bins]); maximum over t0. Runs: north-only,
south-only, joint (inter-cable offset dT in -4..4 s, absorbs any latency difference between the cables).
Uncertainty: grid region with B >= 0.9 x max.
Earlier versions (rejected, kept in docs/NEXT_STEPS.md): STA/LTA onset picks + L1 grid search. The 1 s / 20 s energy
STA/LTA saturates at ~21 on noise bursts as well as on the event; picks showed no moveout and the solutions sat on
the grid edges (and the first version searched around the catalogue-predicted S, i.e. was circular).
Outputs: outputs_100m/das_location_jan13.json, das_location_jan13_bins.csv, figs/das_location_jan13.png
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
CAT = dict(t=pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp(), lat=45.1618, lon=-124.390, z=1.0)
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
SNR_MIN = 1.5  # log units: event envelope >= e^1.5 = 4.5x pre-event median
DTG = 0.25
TG = np.arange(-100, 200, DTG)
rows, E = [], []
for cab in ("north", "south"):
    p = sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_{cab}_*"))[0]
    f = load_file(p)
    ch = f.channels()
    g = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    lat = np.interp(ch.opt_km, g.opt_km, g.lat)
    lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    tt = f.times - CAT["t"]
    w = (tt > -200) & (tt < 300)
    sos = butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos")
    x = sosfiltfilt(sos, f.data[:, w].astype(np.float64), axis=1)
    tw = tt[w]
    a2 = np.abs(hilbert(x, axis=1)) ** 2
    noise_ch = np.median(a2[:, (tw > -90) & (tw < -5)], axis=1)
    for blk, gb in ch.groupby("block"):
        idx = gb.index.values
        for k in range(0, len(idx) - 4, 5):
            sel = idx[k:k + 5]
            if np.any(gb.pos_in_block.loc[sel] < 3) or np.any(gb.pos_in_block.loc[sel] > gb.block_len.iloc[0] - 4):
                continue
            if np.median(noise_ch[sel]) > 3 * np.median(noise_ch):
                continue
            env = uniform_filter1d(a2[sel].mean(0), int(round(2 * f.fs)))
            nz = np.median(env[(tw > -90) & (tw < -5)])
            e = np.clip(np.log(np.interp(TG, tw, env) / nz), 0, None)
            la_, lo_ = float(lat[sel].mean()), float(lon[sel].mean())
            _, _, d = G.inv(lo_, la_, CAT["lon"], CAT["lat"])
            snr = float(e[(TG >= 0) & (TG <= 120)].max())
            rows.append(dict(cable=cab, block=blk, opt_km=float(ch.opt_km[sel].mean()), lat=la_, lon=lo_, snr=snr, d_cat_km=d / 1e3))
            E.append(e)
    print(cab, "bins:", sum(r["cable"] == cab for r in rows), flush=True)
P = pd.DataFrame(rows)
E = np.array(E)
P.to_csv(f"{OUT}/das_location_jan13_bins.csv", index=False)
keep = P.snr.values >= SNR_MIN
P, E = P[keep].reset_index(drop=True), E[keep]
print(f"{len(P)} bins with event SNR >= {SNR_MIN} ({P.groupby(['cable', 'block']).size().to_dict()})", flush=True)

LATG = np.arange(44.80, 45.60, 0.01)
LONG = np.arange(-125.05, -123.85, 0.01)
ZS = [1, 5, 10, 20]
VS = np.arange(2.6, 4.21, 0.2)
T0 = np.arange(-20, 40, DTG)


def backproject(sel, dTs=(0.0,)):
    """Max over (z, v, dT, t0) of the bin-mean envelope at t0 + travel time; returns best tuple and the B(lat, lon) map."""
    lo, la, ee, isn = P.lon.values[sel], P.lat.values[sel], E[sel], (P.cable.values[sel] == "north")
    n = int(sel.sum())
    Bmap = np.zeros((len(LATG), len(LONG)))
    best = (-1,)
    for ia, lat_ in enumerate(LATG):
        _, _, d = G.inv(np.tile(lo, len(LONG)), np.tile(la, len(LONG)), np.repeat(LONG, n), np.full(len(LONG) * n, lat_))
        d = d.reshape(len(LONG), n) / 1e3
        for z in ZS:
            r = np.hypot(d, z)
            for v in VS:
                for dT in dTs:
                    tt = r / v + dT * isn  # (nlon, n)
                    it = np.rint((T0[None, None, :] + tt[:, :, None] - TG[0]) / DTG).astype(int)  # (nlon, n, nt0)
                    it = np.clip(it, 0, len(TG) - 1)
                    B = ee[np.arange(n)[None, :, None], it].mean(1)  # (nlon, nt0)
                    j = np.unravel_index(np.argmax(B), B.shape)
                    Bmap[ia] = np.maximum(Bmap[ia], B.max(1))
                    if B[j] > best[0]:
                        best = (float(B[j]), float(lat_), float(LONG[j[0]]), z, float(v), float(dT), float(T0[j[1]]))
    return best, Bmap


res, maps = {}, {}
for name, sel, dTs in (("north", P.cable.values == "north", (0.0,)), ("south", P.cable.values == "south", (0.0,)),
                       ("joint", np.ones(len(P), bool), np.arange(-4, 4.01, 0.5))):
    if sel.sum() < 5:
        continue
    best, Bm = backproject(sel, dTs)
    _, _, ds = G.inv(CAT["lon"], CAT["lat"], best[2], best[1])
    ia, io = np.where(Bm >= 0.9 * best[0])
    res[name] = dict(B=round(best[0], 3), lat=round(best[1], 3), lon=round(best[2], 3), depth_km=best[3], vs_kms=round(best[4], 2),
                     dT_north_minus_south_s=best[5], t0_das_minus_cat_s=best[6], n_bins=int(sel.sum()), shift_from_catalog_km=round(ds / 1e3, 1),
                     region_0p9=dict(lat=(round(LATG[ia].min(), 2), round(LATG[ia].max(), 2)), lon=(round(LONG[io].min(), 2), round(LONG[io].max(), 2))))
    maps[name] = Bm
    print(name, res[name], flush=True)
json.dump(res, open(f"{OUT}/das_location_jan13.json", "w"), indent=1)

fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
for ax, name in zip(axs, ("north", "south", "joint")):
    if name not in maps:
        continue
    r_ = res[name]
    pc = ax.pcolormesh(LONG, LATG, maps[name] / r_["B"], vmin=0.6, vmax=1, cmap="magma", shading="auto")
    ax.contour(LONG, LATG, maps[name] / r_["B"], levels=[0.9], colors="w", linewidths=1)
    for c, col in (("north", "tab:cyan"), ("south", "tab:green")):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.7", lw=1)
        q = P[P.cable == c] if name in ("joint", c) else P.iloc[:0]
        ax.scatter(q.lon, q.lat, s=10 + 6 * q.snr, c=col, edgecolor="k", lw=0.3, zorder=4)
    ax.plot(CAT["lon"], CAT["lat"], "*", ms=18, color="0.8", mec="k", label="PNSN catalogue")
    ax.plot(r_["lon"], r_["lat"], "*", ms=18, color="tab:red", mec="k", label=f"DAS back-projection ({r_['shift_from_catalog_km']} km off)")
    ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude",
           title=f"{name}{' (both cables)' if name == 'joint' else ' cable only'}: {r_['n_bins']} bins, z {r_['depth_km']} km, V {r_['vs_kms']} km/s"
                 + (f", dT {r_['dT_north_minus_south_s']:+.1f} s" if name == "joint" else ""))
    ax.set_aspect(1 / np.cos(np.radians(45.2)))
    ax.legend(fontsize=7, loc="lower left")
axs[0].set_ylabel("latitude")
fig.colorbar(pc, ax=axs, shrink=0.8, label="stacked envelope / max (white: 0.9 contour)")
fig.suptitle("Jan 13 2026 ML 2.0 located with the DAS only: S-envelope back-projection over all 1-km bins with event SNR >= 4.5x (8-Hz stream)")
fig.savefig(f"{OUT}/figs/das_location_jan13.png", dpi=100, bbox_inches="tight")
