"""Item 8h+: locate the Jan 16 2026 M6.0 Blanco earthquake by joint back-projection of ALL DAS spans + seismometers,
with the origin time FREE (no catalogue information), to show what each data set contributes.

Elements and groups (each group weighted equally in the stack, so 188 fiber bins do not drown ~20 stations):
  north cable (89 bins), south cable (99 bins; C3 excluded) - 100-m stream, true times, latency of the file's setup
    (Jan 16: D128, 11.8 s);
  Hydrate Ridge OBS (OO HYSB1, HYS11-14), Axial Seamount OBS (OO AXAS1/2, AXBA1, AXCC1, AXEC1-3, AXID1), coastal and
    Coast Range land stations (UW) - vertical component, absolute UTC (IRIS FDSN; local copy for Hydrate Ridge).
DAS timing: the onset-function lag of the south bins within 10 km of the colocated seismometer HYSB1 is measured on
this event and applied to both cables (DAS_HYSB1=1; same latency on both cables, s67) - no catalogue information.
Travel times: oceanic 1-D model (thin crust, as s52) for offshore receivers, iasp91 for land stations.
Onset function for every element: STA/LTA (2 s / 20 s) of the 0.4-0.9 Hz energy, minus the pre-event median,
normalised to its event-window maximum (as s70). Back-projection: grid 40-47.5 N, 131.5-122.5 W (0.05 deg); B = mean over groups of the group-mean onset function at
t0 + T(node); t0 free in [catalogue - 40 s, + 40 s].
Runs: DAS only; seismometers only (offshore OBS, and all); DAS + offshore OBS; DAS + all seismometers.
Outputs: outputs_100m/beamform/backproject_m6_joint.json, figs/backproject_m6_joint.png
"""
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
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
LAT, FSU = {256: 25.2, 128: 11.8}, 4.0
ev = pd.read_csv(io.StringIO(requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=60, params=dict(
    format="csv", starttime="2026-01-16T03:25:00", endtime="2026-01-16T03:27:00", minmagnitude=5.5)).text)).iloc[0]
T_EV = pd.Timestamp(ev.time).timestamp()
tg = np.arange(T_EV - 300, T_EV + 260, 1 / FSU)
pre = (tg >= T_EV - 280) & (tg <= T_EV - 60)
evw = (tg >= T_EV) & (tg <= T_EV + 150)


def onset(x, t, fs):
    """STA/LTA onset function of a 0.4-0.9 Hz trace, on the common 4-Hz grid."""
    x = sosfiltfilt(butter(4, (0.4, 0.9), "bandpass", fs=fs, output="sos"), x)
    a2 = np.abs(hilbert(x)) ** 2
    n_s, n_l = int(2 * fs) | 1, int(20 * fs)
    sta = uniform_filter1d(a2, n_s)
    lta = np.convolve(a2, np.ones(n_l) / n_l, "full")[: len(a2)]
    return np.interp(tg, t, sta / (lta + 1e-12 * a2.max()))


EL = []  # elements: dict(group, name, lat, lon, cf)
# ---- DAS bins
inv = pd.read_csv(OUT / "inventory.csv")
for cab in ("north", "south"):
    b, bid = make_bins(pd.read_csv(OUT / f"channel_quality_{cab}.csv"), 1.0, 3)
    b = b[~((cab == "south") & (b.seg == "C3"))]
    c = inv[(inv.cable == cab) & (inv.t0 <= tg[0]) & (inv.t1 >= tg[-1])].iloc[0]
    f = load_cfg(c.path, cfg)
    tt = f.times - LAT[int(c.D)]
    w = (tt >= tg[0] - 60) & (tt <= tg[-1] + 60)
    for _, r in b.iterrows():
        x = f.data[np.where(bid == r.bin)[0]][:, w].astype(np.float64).mean(0)
        EL.append(dict(group=f"{cab} cable", name=f"{cab}{r.seg}", lat=r.lat, lon=r.lon, cf=onset(x, tt[w], f.fs)))
    print(cab, "latency", LAT[int(c.D)], flush=True)
# ---- seismometers
cl = Client("IRIS")
t0u, t1u = UTCDateTime(tg[0] - 120), UTCDateTime(tg[-1] + 120)
STA = {"Hydrate Ridge OBS": ("OO", ["HYSB1", "HYS11", "HYS12", "HYS13", "HYS14"]),
       "Axial OBS": ("OO", ["AXAS1", "AXAS2", "AXBA1", "AXCC1", "AXEC1", "AXEC2", "AXEC3", "AXID1"]),
       "land": ("UW", ["HEBO", "JEDS", "MPO", "UMPQ", "SEAS2", "WPO", "RADR", "NEMA"])}
for grp, (net, stas) in STA.items():
    for s in stas:
        try:
            st = cl.get_waveforms(net, s, "*", "HHZ,BHZ,EHZ", t0u, t1u, attach_response=False)
            ic = cl.get_stations(network=net, station=s, starttime=t0u, endtime=t1u)[0][0]
        except Exception as e:  # noqa: BLE001 - station without data is skipped and reported
            print("no data", s, str(e)[:60], flush=True)
            continue
        st.merge(fill_value=0)
        tr = sorted(st, key=lambda t: "HBE".index(t.stats.channel[0]) if t.stats.channel[0] in "HBE" else 9)[0]
        tr.detrend("linear")
        tt = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) / tr.stats.sampling_rate
        EL.append(dict(group=grp, name=s, lat=ic.latitude, lon=ic.longitude, cf=onset(tr.data.astype(np.float64), tt, tr.stats.sampling_rate)))
    print(grp, sum(e["group"] == grp for e in EL), "stations", flush=True)
E = pd.DataFrame([{k: v for k, v in e.items() if k != "cf"} for e in EL])
CF = np.array([e["cf"] for e in EL])
CF = CF - np.median(CF[:, pre], axis=1, keepdims=True)
CF = np.clip(CF / (CF[:, evw].max(1, keepdims=True) + 1e-12), 0, None)
# DAS timing check against the colocated seismometer HYSB1 (same P wave at the same place -> any lag is DAS timing):
# median lag of the onset function of south bins within 10 km of HYSB1; applied to both cables (same latency, s67).
iH = int(np.where(E.name.values == "HYSB1")[0][0])
wv = (tg >= T_EV - 40) & (tg <= T_EV + 200)
near = [i for i in range(len(E)) if E.group[i] == "south cable" and G.inv(E.lon[iH], E.lat[iH], E.lon[i], E.lat[i])[2] < 10e3]
ref = CF[iH, wv] - CF[iH, wv].mean()
lags = [(np.argmax(np.correlate(CF[i, wv] - CF[i, wv].mean(), ref, "full")) - (len(ref) - 1)) / FSU for i in near]
DAS_CORR = float(np.median(lags)) if os.environ.get("DAS_HYSB1", "1") == "1" else 0.0
print(f"DAS - HYSB1 onset lag: median {np.median(lags):.2f} s over {len(near)} bins (spread {np.percentile(lags, 16):.2f}-{np.percentile(lags, 84):.2f}); applied {DAS_CORR:.2f} s", flush=True)
isdas = E.group.str.contains("cable").values
sh = int(round(DAS_CORR * FSU))
if sh:
    CF[isdas] = np.concatenate([CF[isdas][:, sh:], np.zeros((isdas.sum(), sh))], axis=1)
np.savez(OUT / "beamform/backproject_m6_joint_cf.npz", CF=CF, tg=tg, T_EV=T_EV, **{k: E[k].values for k in ("group", "name", "lat", "lon")})
import os
if os.environ.get("CF_ONLY"):
    sys.exit(0)

# travel times: oceanic 1-D model (s52; thin crust) for offshore receivers (fiber, OBS), iasp91 for land stations
OCEAN = str(Path(__file__).resolve().parents[1] / "resources/taup/ocean6.npz")
MODEL_OFF = os.environ.get("MODEL_OFF", "oceanic")
DG = np.arange(0, 10.01, 0.05)
TTM = {}
for nm, mdl in (("iasp91", TauPyModel("iasp91")), ("oceanic", TauPyModel(OCEAN))):
    TTM[nm] = np.array([min(a.time for a in mdl.get_travel_times(10, d, ["P", "p", "Pn", "Pg"])) for d in DG])
E["model"] = np.where(E.group == "land", "iasp91", MODEL_OFF)
LATG = np.arange(40.0, 47.5001, 0.05)
LONG = np.arange(-131.5, -122.4999, 0.05)
T0 = np.arange(-40, 40.01, 0.5)
OBS_OFF = ["Hydrate Ridge OBS", "Axial OBS"]
RUNS = {"DAS only (all spans, both cables)": ["north cable", "south cable"],
        "offshore seismometers only": OBS_OFF,
        "all seismometers (offshore + land)": OBS_OFF + ["land"],
        "DAS + offshore seismometers": ["north cable", "south cable"] + OBS_OFF,
        "DAS + all seismometers": ["north cable", "south cable"] + OBS_OFF + ["land"]}


def backproject(groups):
    sel = E.group.isin(groups).values
    lo, la, cf, gr, md = E.lon.values[sel], E.lat.values[sel], CF[sel], E.group.values[sel], E.model.values[sel]
    gw = np.array([1.0 / (gr == g).sum() / len(groups) for g in gr])  # equal weight per group
    n = sel.sum()
    M = np.zeros((len(LATG), len(LONG)))
    T0B = np.zeros_like(M)
    for ia, lat_ in enumerate(LATG):
        _, _, d = G.inv(np.tile(lo, len(LONG)), np.tile(la, len(LONG)), np.repeat(LONG, n), np.full(len(LONG) * n, lat_))
        dd = d.reshape(len(LONG), n) / 111195.0
        T = np.where(md[None] == "iasp91", np.interp(dd, DG, TTM["iasp91"]), np.interp(dd, DG, TTM["oceanic"]))
        idx = np.clip(np.rint((T_EV + T0[None, None, :] + T[:, :, None] - tg[0]) * FSU).astype(int), 0, len(tg) - 1)
        S = np.einsum("lnt,n->lt", cf[np.arange(n)[None, :, None], idx], gw)
        M[ia], T0B[ia] = S.max(1), T0[S.argmax(1)]
    return M, T0B


res, maps = {}, {}
for name, groups in RUNS.items():
    M, T0B = backproject(groups)
    ia, io_ = np.unravel_index(np.argmax(M), M.shape)
    _, _, dist = G.inv(ev.longitude, ev.latitude, LONG[io_], LATG[ia])
    area = float((M >= 0.9 * M.max()).sum() * (0.05 * 111.195) ** 2 * np.cos(np.radians(43.5)))
    res[name] = dict(groups=groups, n_elements=int(E.group.isin(groups).sum()), lat=round(float(LATG[ia]), 2), lon=round(float(LONG[io_]), 2),
                     origin_minus_cat_s=float(T0B[ia, io_]), dist_to_catalogue_km=round(dist / 1e3, 1), region90_km2=round(area))
    maps[name] = M
    print(f"{name:38s} {res[name]}", flush=True)
json.dump(dict(event=dict(id=ev.id, lat=float(ev.latitude), lon=float(ev.longitude)), stations=E.groupby("group").name.apply(list).to_dict(),
               results=res), open(OUT / "beamform/backproject_m6_joint.json", "w"), indent=1)

fig, axs = plt.subplots(1, len(RUNS), figsize=(5.0 * len(RUNS), 6.8))
mk = {"north cable": (".", "lime", 2.5), "south cable": (".", "lime", 2.5), "Hydrate Ridge OBS": ("^", "c", 8), "Axial OBS": ("^", "deepskyblue", 8), "land": ("s", "w", 6)}
for ax, name in zip(axs, RUNS):
    M = maps[name] / maps[name].max()
    pc = ax.pcolormesh(LONG, LATG, M, vmin=0.5, vmax=1, cmap="magma", shading="auto")
    ax.contour(LONG, LATG, M, levels=[0.9], colors="c", linewidths=1.2)
    for c in ("north", "south"):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.7", lw=0.8)
    for g in RUNS[name]:
        q = E[E.group == g]
        m_, c_, s_ = mk[g]
        ax.plot(q.lon, q.lat, m_, color=c_, ms=s_, mec="k" if m_ != "." else c_, mew=0.5, label=f"{g} ({len(q)})")
    r_ = res[name]
    ax.plot(ev.longitude, ev.latitude, "*", ms=17, color="w", mec="k", label="catalogue")
    ax.plot(r_["lon"], r_["lat"], "P", ms=13, color="tab:cyan", mec="k", label=f"maximum: {r_['dist_to_catalogue_km']:.0f} km off")
    ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude")
    ax.set_title(f"{name}\norigin {r_['origin_minus_cat_s']:+.1f} s; 90 % region {r_['region90_km2']:,} km$^2$", fontsize=9)
    ax.set_aspect(1 / np.cos(np.radians(43.5)))
    ax.legend(fontsize=6, loc="lower left")
axs[0].set_ylabel("latitude")
fig.colorbar(pc, ax=axs, shrink=0.7, label="back-projected P onset stack / max (cyan: 0.9)")
fig.suptitle("Jan 16 2026 M6.0 Blanco: joint back-projection, origin time FREE (no catalogue information) - DAS spans and seismometers")
fig.savefig(OUT / "figs/backproject_m6_joint.png", dpi=100, bbox_inches="tight")
