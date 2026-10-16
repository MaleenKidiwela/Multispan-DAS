"""Map of catalogued earthquakes (USGS/PNSN, M>=1.5, Nov 5 2025 - Jan 28 2026) with the cables, kept spans,
faults and Hydrate Ridge, plus a 1-3.5 Hz record section of the 8-Hz stream around a chosen local event.

usage: python scripts/s14_event_map.py ["2026-01-13 05:12:47" 45.162 -124.390 "ML 2.0, 32 km W of Neskowin"]
Needs the 8-Hz-stream raw files covering the event in /data/wsd02/multispan_dvv/raw_hf/eq<YYYYMMDD>/.
"""
import glob
import io
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
from pyproj import Geod
from scipy.signal import butter, sosfiltfilt

from msdvv import ROOT
from msdvv.geometry import read_route, repeaters
from msdvv.io import load_file

T_EQ, LAT, LON, LABEL = ("2026-01-13 05:12:47", 45.162, -124.390, "ML 2.0, 32 km W of Neskowin") if len(sys.argv) < 5 else (
    sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
G = Geod(ellps="WGS84")
LF = Path("/data/wsd02/multispan_dvv/outputs_100m")
SPC = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}

r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=60, params=dict(
    format="csv", starttime="2025-11-05", endtime="2026-01-29", minmagnitude=1.5, latitude=45.0, longitude=-125.5, maxradiuskm=300))
ev = pd.read_csv(io.StringIO(r.text))
ev["time"] = pd.to_datetime(ev.time)
ev.to_csv(LF / "catalog_M1.5_300km.csv", index=False)
fc = json.load(open(ROOT / "resources" / "qfaults_oregon_margin_strikeslip.geojson"))
q = {c: pd.read_csv(LF / f"channel_quality_{c}.csv") for c in ("north", "south")}


def base(ax, xl, yl):
    for f in fc["features"]:
        c = np.array(f["geometry"]["coordinates"])
        ax.plot(c[:, 0], c[:, 1], color="brown", lw=1.2, alpha=0.8)
    for cab in ("north", "south"):
        rt = read_route(cab)
        ax.plot(rt.lon, rt.lat, "-", color="0.6", lw=1.5)
        for sp, gg in q[cab].groupby("span"):
            ax.scatter(gg.lon, gg.lat, s=3, color=SPC[sp], zorder=2)
        rp = repeaters(cab)
        ax.plot(rp.lon, rp.lat, "k^", ms=6, zorder=3)
    ax.plot(-125.148, 44.569, "D", color="gold", mec="k", ms=8, zorder=4)
    ax.annotate("S. Hydrate Ridge", (-125.148, 44.569), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for nm, (la, lo) in {"Pacific City": (45.202, -123.967), "Newport": (44.637, -124.053)}.items():
        ax.plot(lo, la, "ks", ms=4)
        ax.annotate(nm, (lo, la), xytext=(4, -3), textcoords="offset points", fontsize=8)
    ax.set(xlim=xl, ylim=yl, xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1 / np.cos(np.radians(44.8)))


fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1])
ax = fig.add_subplot(gs[:, 0])
base(ax, (-130.2, -123.6), (42.3, 46.3))
sc = ax.scatter(ev.longitude, ev.latitude, s=8 * np.exp(ev.mag), c=(ev.time - ev.time.min()).dt.days, cmap="plasma",
                edgecolor="k", lw=0.4, zorder=5)
for e in ev[ev.mag >= 4].itertuples():
    ax.annotate(f"M{e.mag:.1f} {e.time:%b %d}", (e.longitude, e.latitude), xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.add_patch(plt.Rectangle((-125.6, 44.4), 1.75, 0.95, fill=False, ec="k", ls="--"))
plt.colorbar(sc, ax=ax, label="days since 2025-11-05", shrink=0.6)
ax.set_title("Earthquakes M≥1.5 (USGS/PNSN), Nov 5 – Jan 28\ncable spans coloured, ▲ repeaters, brown = faults", fontsize=9)

ax2 = fig.add_subplot(gs[0, 1:])
base(ax2, (-125.6, -123.85), (44.4, 45.35))
near = ev[(ev.longitude > -125.6) & (ev.longitude < -123.85) & (ev.latitude > 44.4) & (ev.latitude < 45.35)]
ax2.scatter(near.longitude, near.latitude, s=40 * np.exp(near.mag - 1.5), c="red", edgecolor="k", zorder=6)
for e in near.itertuples():
    ax2.annotate(f"{e.magType.upper()}{e.mag:.1f} {e.time:%Y-%m-%d %H:%M} UTC, {e.depth:.0f} km", (e.longitude, e.latitude),
                 xytext=(6, 6), textcoords="offset points", fontsize=8, color="darkred")
ax2.set_title("Zoom (dashed box): local events near the cables", fontsize=9)

t_eq = pd.Timestamp(T_EQ, tz="UTC").timestamp()
for col, cab in zip((1, 2), ("south", "north")):
    ax3 = fig.add_subplot(gs[1, col])
    files = [p for p in glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq{pd.Timestamp(T_EQ):%Y%m%d}/nokia*") if f"_{cab}_" in p]
    if not files:
        ax3.set_title(f"{cab}: no 8-Hz file downloaded")
        continue
    f = load_file(files[0])
    i0 = int((t_eq - 60 - f.t0) * f.fs)
    f.data = f.data[:, max(i0, 0) : max(i0, 0) + int(240 * f.fs)]
    ch = f.channels()
    g = pd.read_csv(ROOT / "outputs" / f"geometry_{cab}.csv")  # 211.7-m geometry
    sel = ch[(ch.block == "A") & (ch.opt_km < 70)].ch.values
    x = sosfiltfilt(butter(4, [1.0, 3.5], "bandpass", fs=f.fs, output="sos"), f.data[sel], axis=1)
    x /= np.median(np.abs(x), axis=1, keepdims=True) * 8
    tt = np.arange(x.shape[1]) / f.fs - 60
    km = np.interp(ch.opt_km.values[sel], g.opt_km, g.route_km)
    ax3.pcolormesh(tt, km, x, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
    _, _, d = G.inv(g.lon.values, g.lat.values, np.full(len(g), LON), np.full(len(g), LAT))
    k = d.argmin()
    ax3.axhline(g.route_km[k], color="k", ls=":", lw=0.8)
    ax3.axvline(0, color="k", lw=0.6)
    near_ch = (km > g.route_km[k] - 8) & (km < g.route_km[k] + 8)
    rms = np.sqrt(np.mean(x[near_ch] ** 2, 0))
    print(f"{cab}: closest channel {d[k] / 1e3:.1f} km at route {g.route_km[k]:.1f} km; "
          f"max/pre-event RMS (1-3.5 Hz, ±8 km) = {rms.max() / np.median(rms[tt < -5]):.1f} at t = {tt[rms.argmax()]:.0f} s")
    ax3.set(xlabel="s after catalog origin (DAS times lag ~7-13 s)", ylabel="route km",
            title=f"{cab} span A, 8-Hz stream, 1-3.5 Hz (dotted: closest channel)")
fig.suptitle(f"{T_EQ} UTC {LABEL} relative to the OOI cables", fontsize=11)
fig.tight_layout()
out = LF / "figs" / f"earthquakes_map_{pd.Timestamp(T_EQ):%Y%m%d}.png"
fig.savefig(out, dpi=110)
print(out)
