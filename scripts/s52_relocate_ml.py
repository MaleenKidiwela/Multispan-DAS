"""Relocate uw62216067 with consensus deep-learning picks (s51) and check with T-phase arrivals.

Picks: per Hydrate Ridge seismometer, median of the model picks (PhaseNet/EQTransformer, 'obs' + land weights) with
probability >= 0.3 (P) / >= 0.2 (S) inside the expected windows (P -10..+10 s, S +10..+35 s after catalogue origin);
PNSN/CC/PB analyst P picks (phase-data product, uw62216067_arrivals.csv), down-weighted (0.3).
Grid search (lat, lon, depth 5/10/20 km), origin time solved analytically, for two 1-D models:
  iasp91 (continental, 35-km crust) and an oceanic model (3 km sediments+upper crust 4.5/2.6 km/s over 4 km lower crust
  6.8/3.8 km/s over mantle 8.0/4.5 km/s), built with TauP.
T-phase check: envelope maxima (2-8 Hz, 10-s smoothing) at the five OBS vs t0 + distance / 1.485 km/s.
Outputs: obs/uw62216067_relocation_ml.json, figs/uw62216067_relocation_ml.png
"""
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
from obspy import UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client
from obspy.signal.filter import envelope
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from pyproj import Geod

from msdvv.geometry import read_route

OBS = Path("/data/wsd02/multispan_dvv/obs")
FIG = Path("/data/wsd02/multispan_dvv/outputs_100m/figs")
T0 = UTCDateTime("2026-01-26T09:09:21.46")
CAT = (44.8462, -124.8798)  # catalogue epicentre (ComCat uw62216067)
G = Geod(ellps="WGS84")
HR = ["HYSB1", "HYS11", "HYS12", "HYS13", "HYS14"]
inv = read_inventory(str(OBS / "OO_stations.xml"))
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}

# ---------------- consensus picks
ML = pd.read_csv(OBS / "uw62216067_ml_picks.csv")
picks = []
for s in HR:
    for ph, (a, b, pmin) in (("P", (-10, 10, 0.3)), ("S", (10, 35, 0.2))):
        q = ML[(ML.station == s) & (ML.phase == ph) & ML.t_rel.between(a, b) & (ML.prob >= pmin)]
        if len(q) >= 2:
            picks.append(dict(sta=s, lat=crd[s][0], lon=crd[s][1], phase=ph, t=T0.timestamp + q.t_rel.median(),
                              spread=float(q.t_rel.max() - q.t_rel.min()), n_models=len(q), w=1.0))
OP = pd.DataFrame(picks)
print(OP.assign(t_rel=OP.t - T0.timestamp).round(2).to_string(index=False))
LA = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/uw62216067_arrivals.csv")
fd = Client("https://service.earthscope.org", timeout=120)
lc = {}
for net, g in LA.groupby("net"):
    try:
        iv = fd.get_stations(network=net, station=",".join(g.sta.unique()), starttime=T0 - 86400, endtime=T0 + 86400, level="station")
        lc.update({(n.code, s.code): (s.latitude, s.longitude) for n in iv for s in n})
    except Exception as e:  # noqa: BLE001
        print("station lookup failed", net, e)
LA = LA[[(r.net, r.sta) in lc for r in LA.itertuples()]]
LP = pd.DataFrame(dict(sta=LA.sta, lat=[lc[(r.net, r.sta)][0] for r in LA.itertuples()], lon=[lc[(r.net, r.sta)][1] for r in LA.itertuples()],
                       phase="P", t=[UTCDateTime(x).timestamp for x in LA.pick_time], w=0.3))
D = pd.concat([OP[["sta", "lat", "lon", "phase", "t", "w"]], LP], ignore_index=True)
print(f"{len(OP)} OBS picks (ML consensus), {len(LP)} land P picks")

# ---------------- velocity models
mdir = Path(__file__).resolve().parents[1] / "resources/taup"
mdir.mkdir(parents=True, exist_ok=True)
tv = mdir / "ocean6.tvel"
if not (mdir / "ocean6.npz").exists():
    base = [l.split() for l in open(os.path.join(os.path.dirname(__import__("obspy").__file__), "taup", "data", "iasp91.tvel")).read().splitlines()[2:]]
    deep = [r for r in base if float(r[0]) >= 60.0]
    lines = ["ocean6 - P", "ocean6 - S",
             "0.0 4.50 2.60 2.40", "3.0 4.50 2.60 2.40", "3.0 6.80 3.80 2.90", "7.0 6.80 3.80 2.90",
             "7.0 8.00 4.50 3.30", "60.0 8.05 4.48 3.35"] + [" ".join(r) for r in deep if float(r[0]) > 60.0]
    tv.write_text("\n".join(lines) + "\n")
    build_taup_model(str(tv), output_folder=str(mdir))
MODELS = {"iasp91": TauPyModel("iasp91"), "oceanic": TauPyModel(str(mdir / "ocean6.npz"))}
DG = np.arange(0.02, 6.0, 0.03)
DEPTHS = [5, 10, 20]


def table(m):
    return {(ph, z): np.array([min((a.time for a in m.get_travel_times(z, d, phase_list=n)), default=np.nan) for d in DG])
            for z in DEPTHS for ph, n in (("P", ["p", "P", "Pn", "Pg"]), ("S", ["s", "S", "Sn", "Sg"]))}


LAT = np.arange(42.0, 46.8, 0.025)
LON = np.arange(-129.5, -123.5, 0.03)
res = {}
for name, m in MODELS.items():
    TAB = table(m)
    best, grid = (np.inf,), np.full((len(DEPTHS), len(LAT), len(LON)), np.nan)
    for iz, z in enumerate(DEPTHS):
        for ia, la in enumerate(LAT):
            for io, lo in enumerate(LON):
                _, _, dd = G.inv(np.full(len(D), lo), np.full(len(D), la), D.lon.values, D.lat.values)
                deg = dd / 1e3 / 111.19
                pred = np.where(D.phase == "P", np.interp(deg, DG, TAB[("P", z)]), np.interp(deg, DG, TAB[("S", z)]))
                r_ = D.t.values - pred
                ok = np.isfinite(r_)
                w = D.w.values[ok]
                o = np.sum(w * r_[ok]) / w.sum()
                rms = np.sqrt(np.sum(w * (r_[ok] - o) ** 2) / w.sum())
                grid[iz, ia, io] = rms
                if rms < best[0]:
                    best = (rms, la, lo, z, o)
    iz = DEPTHS.index(best[3])
    g2 = grid[iz]
    ok_region = g2 <= best[0] * 1.2  # rough uncertainty: rms within 20 % of the minimum
    ia, io = np.where(ok_region)
    _, _, dcat = G.inv(CAT[1], CAT[0], best[2], best[1])
    res[name] = dict(lat=round(best[1], 3), lon=round(best[2], 3), depth_km=best[3], origin=str(UTCDateTime(best[4]))[:23], rms_s=round(best[0], 2),
                     dt0_vs_catalog_s=round(best[4] - T0.timestamp, 1), shift_from_catalog_km=round(dcat / 1e3), grid=g2,
                     region=(LAT[ia].min(), LAT[ia].max(), LON[io].min(), LON[io].max()))
    print(f"[{name}] {best[1]:.3f}N {best[2]:.3f}E z={best[3]} km origin {UTCDateTime(best[4])} (dt0 {best[4] - T0.timestamp:+.1f} s) rms {best[0]:.2f} s; "
          f"{dcat / 1e3:.0f} km from catalogue; rms<=1.2*min box lat {LAT[ia].min():.2f}-{LAT[ia].max():.2f} lon {LON[io].min():.2f}-{LON[io].max():.2f}", flush=True)

# ---------------- T-phase check
st = read(str(OBS / "uw62216067_OO.mseed"))
st.merge(fill_value=0)
tobs = {}
for s in HR:
    z = [tr for tr in st.select(station=s) if tr.stats.channel.endswith("Z")][0].copy()
    z.data = np.nan_to_num(z.data.astype(float))
    z.detrend("demean")
    z.filter("bandpass", freqmin=2, freqmax=8, zerophase=True)
    e = envelope(z.data)
    k = int(10 * z.stats.sampling_rate)
    es = np.convolve(e, np.ones(k) / k, "same")
    tt = z.times(reftime=T0)
    m = (tt > 60) & (tt < 200)
    tobs[s] = float(tt[m][np.argmax(es[m])])
tphase = {}
for name in ("catalog",) + tuple(MODELS):
    if name == "catalog":
        la, lo, t0r = CAT[0], CAT[1], 0.0
    else:
        la, lo, t0r = res[name]["lat"], res[name]["lon"], res[name]["dt0_vs_catalog_s"]
    pred = {s: t0r + G.inv(lo, la, crd[s][1], crd[s][0])[2] / 1e3 / 1.485 for s in HR}
    d = np.array([tobs[s] - pred[s] for s in HR])
    tphase[name] = dict(pred={s: round(pred[s], 1) for s in HR}, resid_mean=round(float(d.mean()), 1), resid_spread=round(float(d.max() - d.min()), 1))
    print(f"T-phase [{name}]: observed envelope peaks {', '.join(f'{s} {tobs[s]:.0f}' for s in HR)} s; predicted {', '.join(f'{s} {pred[s]:.0f}' for s in HR)} s; "
          f"mean residual {d.mean():+.0f} s, spread {d.max() - d.min():.0f} s")

json.dump(dict(picks=OP.assign(t=[str(UTCDateTime(x))[:23] for x in OP.t]).to_dict("records"),
               relocations={k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in res.items()},
               tphase_obs_s=tobs, tphase=tphase, note="t relative to catalogue origin 09:09:21.46 where given as t_rel; T phase 1.485 km/s"),
          open(OBS / "uw62216067_relocation_ml.json", "w"), indent=1, default=float)

# ---------------- figure
fig, axs = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
for c in ("north", "south"):
    rt = read_route(c)
    ax.plot(rt.lon, rt.lat, "-", color="0.75", lw=1.5)
cols = {"iasp91": "tab:red", "oceanic": "tab:blue"}
for name, r in res.items():
    ax.contour(LON, LAT, r["grid"], levels=[r["rms_s"] * 1.2, r["rms_s"] * 1.5], colors=[cols[name]], linewidths=[1.2, 0.6])
    ax.plot(r["lon"], r["lat"], "*", ms=18, color=cols[name], mec="k", label=f"{name}: {r['lat']:.2f}N {r['lon']:.2f}W, z {r['depth_km']} km, rms {r['rms_s']} s")
ax.plot(CAT[1], CAT[0], "*", ms=18, color="0.5", mec="k", label="catalogue (PNSN ML 2.2)")
for s in HR:
    ax.plot(crd[s][1], crd[s][0], "kv", ms=8)
ax.plot(LP.lon, LP.lat, "^", color="0.3", ms=5, label="land stations with P picks")
ax.set(xlim=(-129.5, -122), ylim=(42, 47.2), xlabel="longitude", ylabel="latitude",
       title="uw62216067 relocation with consensus deep-learning picks\n(contours: rms = 1.2x and 1.5x minimum at best depth)")
ax.set_aspect(1 / np.cos(np.radians(44.5)))
ax.legend(fontsize=8, loc="lower right")
ax = axs[1]
for k, s in enumerate(HR):
    tr = st.select(station=s)
    z = [t for t in tr if t.stats.channel.endswith("Z")][0].copy()
    z.data = np.nan_to_num(z.data.astype(float))
    z.detrend("demean")
    z.filter("bandpass", freqmin=2, freqmax=15, zerophase=True)
    y = z.data / (np.abs(z.data).max() + 1e-30)
    ax.plot(z.times(reftime=T0), k + 0.45 * y, "k", lw=0.3)
    for r in OP[OP.sta == s].itertuples():
        ax.plot([r.t - T0.timestamp] * 2, [k - 0.45, k + 0.45], color="r" if r.phase == "P" else "b", lw=1.2)
    ax.plot([tobs[s]] * 2, [k - 0.45, k + 0.45], color="g", lw=1.2)
    ax.text(-28, k, s, fontsize=8, va="center")
ax.set(xlim=(-30, 170), yticks=[], xlabel="s after catalogue origin 09:09:21.46 (Z, 2-15 Hz)",
       title="consensus picks: red P, blue S (ML); green = T-phase envelope peak")
fig.tight_layout()
fig.savefig(FIG / "uw62216067_relocation_ml.png", dpi=100)
print("done")
