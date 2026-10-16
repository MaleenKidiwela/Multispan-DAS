"""Item 8h: LOCATE the Jan 16 2026 M6.0 Blanco earthquake by back-projection with all spans (multi-span beamforming
steered to points), compared with single spans, each cable, and both cables.

Data: continuous 100-m stream, both cables, true per-chunk times (regridded), 1-km bins (south C3 excluded), DAS -> UTC
with the latency of the file's setup (D256 25.2 s, D128 11.8 s; Jan 16 is D128). Onset function per bin: STA/LTA (2 s / 20 s) of the 0.4-0.9 Hz energy, minus its
pre-event median, normalised to its maximum in the event window (every bin equal weight; polarity-independent, so the
north-cable sign does not matter). 4 Hz.
Back-projection: grid lat 40.0-47.0 N, lon 131.0-123.5 W (0.05 deg), depth 10 km; first-P travel times from iasp91
(TauP table vs distance); B(node) = max over origin time t0 in [catalogue - 40 s, catalogue + 40 s] of the mean over
bins of cf_n(t0 + T_n(node)). The origin time is free, so neither the catalogue location nor its origin time is used.
ORIGIN=known: t0 restricted to catalogue origin +- 3 s (latency uncertainty), i.e. absolute DAS timing is used.
Resolution: region with B >= 0.9 x max (area) and the distance of the maximum from the catalogue epicentre (global
networks; the Blanco Transform is well covered by land and OBS stations, error ~5-15 km).
Outputs: outputs_100m/beamform/backproject_m6.json, figs/backproject_m6.png
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
import requests
from obspy.taup import TauPyModel
from pyproj import Geod
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from msdvv import load_config
from msdvv.geometry import read_route
from msdvv.interspan import make_bins
from msdvv.io import load_cfg

cfg = load_config(str(Path(__file__).resolve().parents[1] / "config_100m_v2.yaml"))
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
G = Geod(ellps="WGS84")
LAT, FSU = {256: 25.2, 128: 11.8}, 4.0  # latency per setup (Jan 16 is in the Jan 13-21 D128 period)
ev = pd.read_csv(__import__("io").StringIO(requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=60, params=dict(
    format="csv", starttime="2026-01-16T03:25:00", endtime="2026-01-16T03:27:00", minmagnitude=5.5)).text)).iloc[0]
T_EV = pd.Timestamp(ev.time).timestamp()
print(f"catalogue: {ev.time} M{ev.mag} {ev.latitude:.3f} {ev.longitude:.3f} z {ev.depth} km ({ev.id})", flush=True)

B = []
for cab in ("north", "south"):
    b, bid = make_bins(pd.read_csv(OUT / f"channel_quality_{cab}.csv"), 1.0, 3)
    b["cable"] = cab
    b["chans"] = [np.where(bid == k)[0] for k in b.bin]
    B.append(b[~((b.cable == "south") & (b.seg == "C3"))])
B = pd.concat(B, ignore_index=True)
inv = pd.read_csv(OUT / "inventory.csv")
tg = np.arange(T_EV - 300, T_EV + 260, 1 / FSU)
CF = np.zeros((len(B), len(tg)))
for cab in ("north", "south"):
    c = inv[(inv.cable == cab) & (inv.t0 <= tg[0]) & (inv.t1 >= tg[-1])].iloc[0]
    f = load_cfg(c.path, cfg)
    tt = f.times - LAT[int(c.D)]
    print(cab, 'setup D', int(c.D), 'latency', LAT[int(c.D)], flush=True)
    w = (tt >= tg[0] - 60) & (tt <= tg[-1] + 60)
    sos = butter(4, (0.4, 0.9), "bandpass", fs=f.fs, output="sos")
    n_s, n_l = int(2 * f.fs) | 1, int(20 * f.fs)
    for ix, b in B[B.cable == cab].iterrows():
        x = sosfiltfilt(sos, f.data[b.chans][:, w].astype(np.float64), axis=1).mean(0)
        a2 = np.abs(hilbert(x)) ** 2
        sta = uniform_filter1d(a2, n_s)
        lta = np.convolve(a2, np.ones(n_l) / n_l, "full")[: len(a2)]
        CF[ix] = np.interp(tg, tt[w], sta / (lta + 1e-12 * a2.max()))
pre = (tg >= T_EV - 280) & (tg <= T_EV - 60)
evw = (tg >= T_EV) & (tg <= T_EV + 150)
CF = CF - np.median(CF[:, pre], axis=1, keepdims=True)
CF = np.clip(CF / (CF[:, evw].max(1, keepdims=True) + 1e-12), 0, None)

# travel-time table (first P, iasp91, 10 km depth)
taup = TauPyModel("iasp91")
DG = np.arange(0, 8.01, 0.05)
TT = np.array([min(a.time for a in taup.get_travel_times(10, d, ["P", "p", "Pn", "Pg"])) for d in DG])
LATG = np.arange(40.0, 47.0001, 0.05)
LONG = np.arange(-131.0, -123.4999, 0.05)
import os
ORIGIN = os.environ.get("ORIGIN", "free")  # free: t0 in +-40 s (no catalogue time); known: t0 in +-3 s (catalogue origin
# from global networks, +- the ~3-s uncertainty of the 25.2-s latency) - removes the distance / origin-time trade-off
T0 = np.arange(-40, 40.01, 0.5) if ORIGIN == "free" else np.arange(-3, 3.01, 0.25)
TAG = "" if ORIGIN == "free" else "_origin_known"
CONF = {f"{c} {s} (single span)": ((B.cable == c) & (B.seg == s)).values for (c, s), g in B.groupby(["cable", "seg"]) if len(g) >= 10}
CONF.update({"north, all spans": (B.cable == "north").values, "south loop, all spans": (B.cable == "south").values, "both cables": np.ones(len(B), bool)})


def backproject(sel):
    lo, la, cf = B.lon.values[sel], B.lat.values[sel], CF[sel]
    n = sel.sum()
    M = np.zeros((len(LATG), len(LONG)))
    T0B = np.zeros_like(M)
    for ia, lat_ in enumerate(LATG):
        _, _, d = G.inv(np.tile(lo, len(LONG)), np.tile(la, len(LONG)), np.repeat(LONG, n), np.full(len(LONG) * n, lat_))
        T = np.interp(d.reshape(len(LONG), n) / 111195.0, DG, TT)  # (nlon, n)
        idx = np.rint((T_EV + T0[None, None, :] + T[:, :, None] - tg[0]) * FSU).astype(int)  # (nlon, n, nt0)
        S = cf[np.arange(n)[None, :, None], np.clip(idx, 0, len(tg) - 1)].mean(1)  # (nlon, nt0)
        M[ia], T0B[ia] = S.max(1), T0[S.argmax(1)]
    return M, T0B


res, maps = {}, {}
for name, sel in CONF.items():
    M, T0B = backproject(sel)
    ia, io = np.unravel_index(np.argmax(M), M.shape)
    _, _, dist = G.inv(ev.longitude, ev.latitude, LONG[io], LATG[ia])
    r90 = M >= 0.9 * M.max()
    area = float(r90.sum() * (0.05 * 111.195) ** 2 * np.cos(np.radians(43.5)))
    res[name] = dict(n_bins=int(sel.sum()), lat=round(float(LATG[ia]), 2), lon=round(float(LONG[io]), 2), origin_minus_cat_s=float(T0B[ia, io]),
                     dist_to_catalogue_km=round(dist / 1e3, 1), region90_km2=round(area), peak=round(float(M.max()), 3))
    maps[name] = M
    print(f"{name:28s} {res[name]}", flush=True)
json.dump(dict(event=dict(id=ev.id, time=ev.time, mag=float(ev.mag), lat=float(ev.latitude), lon=float(ev.longitude)), results=res),
          open(OUT / f"beamform/backproject_m6{TAG}.json", "w"), indent=1)

show = ["south A1 (single span)", "north A2 (single span)", "south loop, all spans", "north, all spans", "both cables"]
show = [s for s in show if s in maps]
fig, axs = plt.subplots(1, len(show), figsize=(5.2 * len(show), 6.6))
for ax, name in zip(axs, show):
    M = maps[name] / maps[name].max()
    pc = ax.pcolormesh(LONG, LATG, M, vmin=0.5, vmax=1, cmap="magma", shading="auto")
    ax.contour(LONG, LATG, M, levels=[0.9], colors="c", linewidths=1.2)
    for c in ("north", "south"):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.7", lw=0.8)
    sel = CONF[name]
    ax.plot(B.lon[sel], B.lat[sel], ".", color="lime", ms=2.5, label=f"bins used ({sel.sum()})")
    r_ = res[name]
    ax.plot(ev.longitude, ev.latitude, "*", ms=17, color="w", mec="k", label="catalogue (global networks)")
    ax.plot(r_["lon"], r_["lat"], "P", ms=13, color="tab:cyan", mec="k", label=f"back-projection max: {r_['dist_to_catalogue_km']:.0f} km off")
    ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude")
    ax.set_title(f"{name}\n90 % region {r_['region90_km2']:,} km$^2$", fontsize=10)
    ax.set_aspect(1 / np.cos(np.radians(43.5)))
    ax.legend(fontsize=7, loc="upper left")
axs[0].set_ylabel("latitude")
fig.colorbar(pc, ax=axs, shrink=0.7, label="back-projected P onset stack / max (cyan: 0.9)")
fig.suptitle(f"Jan 16 2026 M6.0 Blanco located by DAS back-projection (P onsets 0.4-0.9 Hz, 100-m stream, origin time {'free' if ORIGIN == 'free' else 'known +-3 s'}): single span vs multi-span")
fig.savefig(OUT / f"figs/backproject_m6{TAG}.png", dpi=100, bbox_inches="tight")
