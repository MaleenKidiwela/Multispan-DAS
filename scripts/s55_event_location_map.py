"""Location map of uw62216067 from the consensus deep-learning picks (s51/s52).

Misfit grids (oceanic 1-D model, depth 5 km, origin solved analytically) for (a) OBS P+S + land P (preferred, as s52)
and (b) OBS P+S only; contours at rms = 1.2x and 1.5x the minimum. Background: ComCat M>=3 2010-2026 (Blanco, Gorda,
Juan de Fuca boundaries). Also the PNSN catalogue location, the Hydrate Ridge seismometers, the OOI cables, and the
T-phase-only constraint (distance ring from HYS14 implied by the T-phase delay after the relocated origin).
Figure: figs/uw62216067_location_map.png
"""
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
from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from pyproj import Geod

from msdvv.geometry import read_route

OBS = Path("/data/wsd02/multispan_dvv/obs")
T0 = UTCDateTime("2026-01-26T09:09:21.46")
CAT = (44.8453, -124.8487)
G = Geod(ellps="WGS84")
inv = read_inventory(str(OBS / "OO_stations.xml"))
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
J = json.load(open(OBS / "uw62216067_relocation_ml.json"))
OP = pd.DataFrame([dict(sta=p["sta"], lat=crd[p["sta"]][0], lon=crd[p["sta"]][1], phase=p["phase"], t=UTCDateTime(p["t"]).timestamp, w=1.0) for p in J["picks"]])
LA = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/uw62216067_arrivals.csv")
fd = Client("https://service.earthscope.org", timeout=120)
lc = {}
for net, g in LA.groupby("net"):
    try:
        iv = fd.get_stations(network=net, station=",".join(g.sta.unique()), starttime=T0 - 86400, endtime=T0 + 86400, level="station")
        lc.update({(n.code, s.code): (s.latitude, s.longitude) for n in iv for s in n})
    except Exception:  # noqa: BLE001
        pass
LA = LA[[(r.net, r.sta) in lc for r in LA.itertuples()]]
LP = pd.DataFrame(dict(sta=LA.sta, lat=[lc[(r.net, r.sta)][0] for r in LA.itertuples()], lon=[lc[(r.net, r.sta)][1] for r in LA.itertuples()],
                       phase="P", t=[UTCDateTime(x).timestamp for x in LA.pick_time], w=0.3))
m = TauPyModel(str(Path(__file__).resolve().parents[1] / "resources/taup/ocean6.npz"))
DG = np.arange(0.02, 6.0, 0.03)
Z = 5
TAB = {ph: np.array([min((a.time for a in m.get_travel_times(Z, d, phase_list=n)), default=np.nan) for d in DG])
       for ph, n in (("P", ["p", "P", "Pn", "Pg"]), ("S", ["s", "S", "Sn", "Sg"]))}
LAT = np.arange(42.0, 46.6, 0.02)
LON = np.arange(-129.8, -123.6, 0.025)


def grid(D):
    out = np.full((len(LAT), len(LON)), np.nan)
    best = (np.inf,)
    for ia, la in enumerate(LAT):
        _, _, dd = G.inv(np.repeat(LON, len(D)), np.full(len(LON) * len(D), la), np.tile(D.lon.values, len(LON)), np.tile(D.lat.values, len(LON)))
        dd = dd.reshape(len(LON), len(D)) / 1e3 / 111.19
        pred = np.where(D.phase.values[None] == "P", np.interp(dd, DG, TAB["P"]), np.interp(dd, DG, TAB["S"]))
        r = D.t.values[None] - pred
        w = D.w.values[None]
        o = (w * r).sum(1) / w.sum()
        rms = np.sqrt((w * (r - o[:, None]) ** 2).sum(1) / w.sum())
        out[ia] = rms
        k = int(np.argmin(rms))
        if rms[k] < best[0]:
            best = (rms[k], la, LON[k], o[k])
    return out, best


Gall, Ball = grid(pd.concat([OP, LP], ignore_index=True))
Gobs, Bobs = grid(OP)
for nm, b in (("OBS+land", Ball), ("OBS only", Bobs)):
    _, _, dc = G.inv(CAT[1], CAT[0], b[2], b[1])
    print(f"{nm}: {b[1]:.2f}N {b[2]:.2f}W origin {UTCDateTime(b[3])} rms {b[0]:.2f} s, {dc / 1e3:.0f} km from catalogue")
r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(
    format="csv", starttime="2010-01-01", endtime="2026-02-01", minmagnitude=3.0, minlatitude=41.5, maxlatitude=47, minlongitude=-130.5, maxlongitude=-123.5))
BG = pd.read_csv(io.StringIO(r.text))
# T-phase ring: distance from HYS14 implied by T-phase peak delay after the preferred origin (peak, so an upper bound)
tdel = J["tphase_obs_s"]["HYS14"] - (Ball[3] - T0.timestamp)
ring_km = tdel * 1.485

fig, axs = plt.subplots(1, 2, figsize=(19, 9.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
for k, ax in enumerate(axs):
    xl, yl = ((-130.3, -123.6), (41.6, 46.8)) if k == 0 else ((-128.6, -125.6), (42.6, 44.8))
    sel = BG.longitude.between(*xl) & BG.latitude.between(*yl)
    ax.scatter(BG.longitude[sel], BG.latitude[sel], s=2 + 2.5 * (BG.mag[sel] - 3) ** 2, color="0.72", lw=0, zorder=1,
               label="ComCat M>=3, 2010-2026" if k == 0 else None)
    for c in ("north", "south"):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="tab:purple", lw=1.4, zorder=2, label="OOI cables" if (k == 0 and c == "north") else None)
    ax.contour(LON, LAT, Gall, levels=[Ball[0] * 1.2, Ball[0] * 1.5], colors=["tab:red", "tab:red"], linewidths=[1.6, 0.8], linestyles=["-", "--"], zorder=3)
    ax.contour(LON, LAT, Gobs, levels=[Bobs[0] * 1.2, Bobs[0] * 1.5], colors=["tab:blue", "tab:blue"], linewidths=[1.4, 0.7], linestyles=["-", "--"], zorder=3)
    th = np.linspace(0, 360, 361)
    rl = [G.fwd(crd["HYS14"][1], crd["HYS14"][0], a, ring_km * 1e3) for a in th]
    ax.plot([p[0] for p in rl], [p[1] for p in rl], ":", color="green", lw=1.5, zorder=3, label=f"T-phase ring from HYS14 ({ring_km:.0f} km; peak, upper bound)" if k == 0 else None)
    for s in ("HYSB1", "HYS11", "HYS12", "HYS13", "HYS14"):
        ax.plot(crd[s][1], crd[s][0], "v", color="w", mec="k", ms=9, zorder=5)
    ax.annotate("HYSB1 / HYS11-14", (crd["HYS14"][1], crd["HYS14"][0]), xytext=(8, 8), textcoords="offset points", fontsize=8, fontweight="bold")
    if k == 0:
        ax.plot(LP.lon, LP.lat, "^", color="0.3", ms=5, zorder=4, label="land stations with P picks (>225 km)")
        for la, lo in zip(LP.lat, LP.lon):
            ax.plot([Ball[2], lo], [Ball[1], la], color="0.6", lw=0.3, zorder=2)
    for s in ("HYSB1", "HYS14"):
        ax.plot([Ball[2], crd[s][1]], [Ball[1], crd[s][0]], color="tab:red", lw=0.6, zorder=2)
    ax.plot(CAT[1], CAT[0], "*", ms=20, color="0.55", mec="k", zorder=6, label="PNSN catalogue: ML 2.2 '62 km W of Depoe Bay'")
    ax.plot(Ball[2], Ball[1], "*", ms=22, color="tab:red", mec="k", zorder=7,
            label=f"relocated (OBS P+S + land P): {Ball[1]:.2f}N {abs(Ball[2]):.2f}W, 09:08:{UTCDateTime(Ball[3]).second + UTCDateTime(Ball[3]).microsecond / 1e6:04.1f} UTC")
    ax.plot(Bobs[2], Bobs[1], "*", ms=16, color="tab:blue", mec="k", zorder=7, label=f"OBS P+S only: {Bobs[1]:.2f}N {abs(Bobs[2]):.2f}W")
    if k == 0:
        ax.annotate("", xy=(Ball[2], Ball[1]), xytext=(CAT[1], CAT[0]), arrowprops=dict(arrowstyle="->", color="k", lw=1.2), zorder=6)
        _, _, dc = G.inv(CAT[1], CAT[0], Ball[2], Ball[1])
        ax.text((CAT[1] + Ball[2]) / 2 + 0.1, (CAT[0] + Ball[1]) / 2, f"{dc / 1e3:.0f} km", fontsize=9)
        for txt, (lo, la) in (("Blanco Transform", (-128.2, 43.85)), ("Gorda Ridge", (-127.4, 42.3)), ("Juan de Fuca Ridge", (-129.6, 45.8)), ("Oregon", (-123.9, 44.3))):
            ax.text(lo, la, txt, fontsize=9, style="italic", color="0.3")
    ax.set(xlim=xl, ylim=yl, xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1 / np.cos(np.radians(np.mean(yl))))
axs[0].legend(fontsize=7.5, loc="upper left")
axs[0].set_title("uw62216067, 2026-01-26: deep-learning picks relocation (oceanic 1-D model, depth 5 km)\n"
                 "contours: misfit 1.2x (solid) and 1.5x (dashed) minimum; red = OBS P+S + land P, blue = OBS only", fontsize=10)
axs[1].set_title("zoom: relocation vs Blanco Transform / Gorda Ridge seismicity", fontsize=10)
fig.tight_layout()
fig.savefig("/data/wsd02/multispan_dvv/outputs_100m/figs/uw62216067_location_map.png", dpi=105)
print(f"T-phase ring {ring_km:.0f} km; background events {len(BG)}")
