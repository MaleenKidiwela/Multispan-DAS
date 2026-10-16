"""Plot of the genuine local earthquakes near the cables (hand-verified with the OOI Hydrate Ridge OBS):
map + OBS record sections with predicted P/S from the catalogue hypocentre (1-D: vp 6.0, vs 3.45 km/s)."""
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
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from pyproj import Geod

from msdvv.geometry import read_route, repeaters

G = Geod(ellps="WGS84")
C = Client("https://service.earthscope.org", timeout=120)
LF = Path("/data/wsd02/multispan_dvv/outputs_100m")
EVENTS = [  # hand-verified genuine local events
    dict(tag="Jan 13 2026 ML 2.0", t="2026-01-13T05:12:47.0", lat=45.162, lon=-124.390, z=0.0, col="tab:red"),
    dict(tag="Dec 13 2025 ML 1.6", t="2025-12-13T15:40:16.0", lat=44.623, lon=-124.356, z=24.6, col="tab:purple"),
]
MISLOC = dict(tag="Jan 26 ML 2.2 (catalogue)", lat=44.845, lon=-124.849, rlat=43.58, rlon=-126.82)
VP, VS = 6.0, 3.45
STA = ["HYS14", "HYS11", "HYS12", "HYS13", "HYSB1"]
inv = C.get_stations(network="OO", station=",".join(STA), starttime=UTCDateTime("2025-12-01"), endtime=UTCDateTime("2026-01-29"))
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
q = {c: pd.read_csv(LF / f"channel_quality_{c}.csv") for c in ("north", "south")}
cat = pd.read_csv(LF / "catalog_M1.5_300km.csv")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
ax = fig.add_subplot(gs[:, 0])
for c in ("north", "south"):
    rt = read_route(c)
    ax.plot(rt.lon, rt.lat, "-", color="0.65", lw=1.5)
    g = q[c][q[c].good]
    ax.scatter(g.lon, g.lat, s=2, color="tab:blue", zorder=2)
    rp = repeaters(c)
    ax.plot(rp.lon, rp.lat, "k^", ms=6)
ax.scatter(cat.longitude, cat.latitude, s=5 * np.exp(cat.mag), facecolor="none", edgecolor="0.6", lw=0.6, label="other catalogued events (mostly Blanco area, off map to SW)")
for s, (la, lo) in crd.items():
    ax.plot(lo, la, "v", color="k", ms=8, mec="w", zorder=5)
ax.annotate("HYS11-14", (-125.148, 44.569), xytext=(6, 4), textcoords="offset points", fontsize=8)
ax.annotate("HYSB1", (-125.405, 44.51), xytext=(6, -10), textcoords="offset points", fontsize=8)
for e in EVENTS:
    ax.plot(e["lon"], e["lat"], "*", color=e["col"], ms=20, mec="k", zorder=7, label=f"{e['tag']} (genuine, OBS-verified)")
ax.plot(MISLOC["lon"], MISLOC["lat"], "X", color="0.4", ms=10, label="Jan 26 ML 2.2 catalogue position (mislocated)")
ax.annotate("relocated ~211 km\nto 43.58N 126.82W", (MISLOC["lon"], MISLOC["lat"]), xytext=(-125.9, 44.2),
            arrowprops=dict(arrowstyle="<-", color="0.4"), fontsize=8, color="0.3")
for nm, (la, lo) in {"Pacific City": (45.202, -123.967), "Neskowin": (45.10, -123.98), "Newport": (44.637, -124.053)}.items():
    ax.plot(lo, la, "ks", ms=4)
    ax.annotate(nm, (lo, la), xytext=(4, -3), textcoords="offset points", fontsize=8)
ax.set(xlim=(-126.2, -123.7), ylim=(44.0, 45.5), xlabel="longitude", ylabel="latitude",
       title="Genuine local earthquakes near the cables, Nov 2025 - Jan 2026\n(blue dots = usable DAS channels, ▼ OOI seismometers, ▲ repeaters)")
ax.set_aspect(1 / np.cos(np.radians(44.8)))
ax.legend(fontsize=7, loc="lower left")

for k, e in enumerate(EVENTS):
    a2 = fig.add_subplot(gs[k, 1])
    t0 = UTCDateTime(e["t"])
    st = C.get_waveforms("OO", ",".join(STA), "*", "HH?,EH?", t0 - 20, t0 + 70)
    st.merge(fill_value=0)
    st.detrend("demean")
    st.taper(0.02)
    st.filter("bandpass", freqmin=3, freqmax=20, corners=4, zerophase=True)
    rows = []
    for sta in STA:
        _, _, d = G.inv(e["lon"], e["lat"], crd[sta][1], crd[sta][0])
        rows.append((sta, np.hypot(d / 1e3, e["z"])))
    rows.sort(key=lambda r: r[1])
    for j, (sta, hyp) in enumerate(rows):
        for comp, col, off in (("Z", "k", 0.0), ("H", "tab:orange", 0.0)):
            trs = [t for t in st.select(station=sta) if (t.stats.channel.endswith("Z")) == (comp == "Z")]
            if not trs:
                continue
            if comp == "Z":
                y, tt = trs[0].data, trs[0].times(reftime=t0)
            else:
                n = min(len(t.data) for t in trs)
                y, tt = np.sqrt(sum(t.data[:n].astype(float) ** 2 for t in trs)), trs[0].times(reftime=t0)[:n]
            y = y / (np.abs(y).max() + 1e-30)
            a2.plot(tt, hyp + (1.5 if comp == "Z" else -1.5) + 2.2 * y, color=col, lw=0.35)
        a2.text(-19, hyp, sta, fontsize=7, va="center")
    hh = np.linspace(min(r[1] for r in rows) - 5, max(r[1] for r in rows) + 5, 10)
    a2.plot(hh / VP, hh, "r--", lw=1, label=f"P pred (vp {VP})")
    a2.plot(hh / VS, hh, "b--", lw=1, label=f"S pred (vs {VS})")
    a2.set(xlim=(-20, 70), xlabel="s after catalogue origin", ylabel="hypocentral distance (km)",
           title=f"{e['tag']}: OOI Hydrate Ridge OBS, 3-20 Hz (black Z, orange horizontal envelope)")
    a2.legend(fontsize=7, loc="lower right")
fig.tight_layout()
out = LF / "figs" / "genuine_local_events.png"
fig.savefig(out, dpi=110)
print(out)
