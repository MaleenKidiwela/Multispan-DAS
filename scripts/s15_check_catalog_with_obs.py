"""Check offshore catalogue locations against the OOI seafloor seismometers at Hydrate Ridge.

For every catalogued event (M>=1.5, <=300 km of the cables, west of 124.2W, Nov 5 2025 - Jan 28 2026):
  * catalogue quality (gap, errors, nearest station) and PNSN/USGS P picks from the phase-data product
  * OO.HYS11-14 / OO.HYSB1 waveforms: STA/LTA + AIC P pick (vertical) and S pick (horizontals), SNR checks
  * observed vs predicted S-P at the catalogue location (iasp91)
  * grid-search relocation (lat, lon, depth in {5,10,20,30} km, iasp91) with OBS P+S (+ land P, down-weighted)
Outputs: /data/wsd02/multispan_dvv/obs/catalog_check.csv and figs/catalog_check_*.png
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn import Client
from obspy.signal.trigger import aic_simple, classic_sta_lta
from obspy.taup import TauPyModel
from pyproj import Geod

OUT = Path("/data/wsd02/multispan_dvv/obs")
FIG = Path("/data/wsd02/multispan_dvv/outputs_100m/figs")
G = Geod(ellps="WGS84")
FDSN = Client("https://service.earthscope.org", timeout=120)
STA = ["HYSB1", "HYS11", "HYS12", "HYS13", "HYS14"]
DEPTHS = [5, 10, 20, 30]

# ---------------- travel-time tables
m = TauPyModel("iasp91")
DG = np.arange(0.02, 8.0, 0.04)
TAB = {(ph, z): np.array([min((a.time for a in m.get_travel_times(z, d, phase_list=n)), default=np.nan) for d in DG])
       for z in DEPTHS for ph, n in (("P", ["p", "P", "Pn", "Pg"]), ("S", ["s", "S", "Sn", "Sg"]))}


def tt(ph, z, dist_km):
    return np.interp(np.asarray(dist_km) / 111.19, DG, TAB[(ph, z)])


# ---------------- catalogue
r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=60, params=dict(
    format="csv", starttime="2025-11-05", endtime="2026-01-29", minmagnitude=1.5, latitude=45.0, longitude=-125.5, maxradiuskm=300))
cat = pd.read_csv(io.StringIO(r.text))
cat = cat[cat.longitude < -124.2].reset_index(drop=True)
inv = FDSN.get_stations(network="OO", station=",".join(STA), starttime=UTCDateTime("2025-11-05"), endtime=UTCDateTime("2026-01-29"), level="channel")
coords = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
print(f"{len(cat)} offshore catalogued events", flush=True)


def land_picks(eid):
    try:
        d = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params=dict(eventid=eid, format="geojson"), timeout=60).json()
        prods = d["properties"]["products"]
        if "phase-data" not in prods:
            return pd.DataFrame()
        url = [c["url"] for n, c in prods["phase-data"][0]["contents"].items() if n.endswith("quakeml.xml")][0]
        e = read_events(io.BytesIO(requests.get(url, timeout=60).content))[0]
        o = e.preferred_origin() or e.origins[0]
        pk = {p.resource_id.id: p for p in e.picks}
        rows = []
        for a in o.arrivals:
            p = pk.get(a.pick_id.id)
            if p is None or a.distance is None or a.phase is None or not a.phase.upper().startswith(("P", "S")):
                continue
            slon, slat, _ = G.fwd(o.longitude, o.latitude, a.azimuth, a.distance * 111.19e3)
            rows.append(dict(sta=p.waveform_id.station_code, phase=a.phase[0].upper(), t=p.time.timestamp, lat=slat, lon=slon))
        return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001 - no phase data available
        return pd.DataFrame()


def obs_picks(t0):
    """Automatic P/S picks at the Hydrate Ridge OBS around catalogue origin t0 (UTCDateTime)."""
    out = {}
    try:
        st = FDSN.get_waveforms("OO", ",".join(STA), "*", "HH?,EH?", t0 - 90, t0 + 150)
    except Exception:  # noqa: BLE001
        return out, None
    st.merge(fill_value=0)
    st.detrend("demean")
    st.taper(0.02)
    st.filter("bandpass", freqmin=3, freqmax=20, corners=4, zerophase=True)
    for sta in STA:
        s = st.select(station=sta)
        z = [t for t in s if t.stats.channel.endswith("Z")]
        h = [t for t in s if not t.stats.channel.endswith("Z")]
        if not z:
            continue
        z = z[0]
        fs = z.stats.sampling_rate
        tz = z.times(reftime=t0)
        cft = classic_sta_lta(z.data, int(0.5 * fs), int(10 * fs))
        win = (tz > -60) & (tz < 100)
        if not win.any() or cft[win].max() < 4:
            continue
        itr = np.where(win & (cft >= 4))[0][0]
        w = (tz > tz[itr] - 4) & (tz < tz[itr] + 2)
        tp = tz[w][np.argmin(aic_simple(z.data[w]))]
        noise = np.std(z.data[(tz > tp - 6) & (tz < tp - 1)])
        snr_p = np.std(z.data[(tz > tp) & (tz < tp + 1.5)]) / max(noise, 1e-30)
        ts, snr_s = np.nan, np.nan
        if len(h) == 2 and len(h[0].data) == len(h[1].data):
            e = np.sqrt(h[0].data.astype(float) ** 2 + h[1].data.astype(float) ** 2)
            th = h[0].times(reftime=t0)
            w2 = (th > tp + 1.0) & (th < tp + 45)
            if w2.sum() > 50:
                ts = th[w2][np.argmin(aic_simple(e[w2]))]
                pre = np.std(e[(th > ts - 3) & (th < ts - 0.3)])
                snr_s = np.std(e[(th > ts) & (th < ts + 1.5)]) / max(pre, 1e-30)
        out[sta] = dict(tp=tp, ts=ts, snr_p=snr_p, snr_s=snr_s)
    return out, st


def relocate(t0, op, lp):
    rows = [(coords[s][0], coords[s][1], "P", t0.timestamp + v["tp"], 1.0) for s, v in op.items() if v["snr_p"] >= 3]
    rows += [(coords[s][0], coords[s][1], "S", t0.timestamp + v["ts"], 1.0) for s, v in op.items()
             if np.isfinite(v["ts"]) and v["snr_s"] >= 2.5 and v["ts"] - v["tp"] > 0.5]
    if len(lp):
        rows += [(r.lat, r.lon, r.phase, r.t, 0.3) for r in lp.itertuples()]
    D = pd.DataFrame(rows, columns=["lat", "lon", "phase", "t", "w"])
    if (D.w == 1).sum() < 4:
        return None
    best = (np.inf,)
    for z in DEPTHS:
        for la in np.arange(42.6, 47.0, 0.03):
            for lo in np.arange(-130.0, -123.5, 0.04):
                _, _, dd = G.inv(np.full(len(D), lo), np.full(len(D), la), D.lon.values, D.lat.values)
                pred = np.where(D.phase == "P", tt("P", z, dd / 1e3), tt("S", z, dd / 1e3))
                res = D.t.values - pred
                ok = np.isfinite(res)
                o = np.sum(D.w.values[ok] * res[ok]) / np.sum(D.w.values[ok])
                rms = np.sqrt(np.sum(D.w.values[ok] * (res[ok] - o) ** 2) / np.sum(D.w.values[ok]))
                if rms < best[0]:
                    best = (rms, la, lo, z, o)
    return dict(rms=best[0], lat=best[1], lon=best[2], depth=best[3], t0=best[4] - t0.timestamp, n_obs=int((D.w == 1).sum()), n_land=len(lp))


rows = []
for i, e in cat.iterrows():
    t0 = UTCDateTime(e.time)
    op, st = obs_picks(t0)
    lp = land_picks(e.id)
    row = dict(id=e.id, time=str(t0)[:19], mag=e.mag, magType=e.magType, net=e.net, lat=e.latitude, lon=e.longitude, depth=e.depth,
               gap=e.gap, horizErr=e.horizontalError, n_land=len(lp), n_obs_P=sum(v["snr_p"] >= 3 for v in op.values()),
               n_obs_S=sum(np.isfinite(v["ts"]) and v["snr_s"] >= 2.5 for v in op.values()))
    sp_obs, sp_cat = [], []
    for s, v in op.items():
        if v["snr_p"] >= 3 and np.isfinite(v["ts"]) and v["snr_s"] >= 2.5 and v["ts"] - v["tp"] > 0.5:
            _, _, d = G.inv(e.longitude, e.latitude, coords[s][1], coords[s][0])
            zz = min(DEPTHS, key=lambda q: abs(q - max(e.depth, 0)))
            sp_obs.append(v["ts"] - v["tp"])
            sp_cat.append(float(tt("S", zz, d / 1e3) - tt("P", zz, d / 1e3)))
    row["SP_obs_med"] = np.median(sp_obs) if sp_obs else np.nan
    row["SP_catalog_pred_med"] = np.median(sp_cat) if sp_cat else np.nan
    rl = relocate(t0, op, lp) if row["n_obs_P"] >= 3 else None
    if rl:
        _, _, dshift = G.inv(e.longitude, e.latitude, rl["lon"], rl["lat"])
        row.update(reloc_lat=rl["lat"], reloc_lon=rl["lon"], reloc_depth=rl["depth"], reloc_dt0=rl["t0"], reloc_rms=rl["rms"], shift_km=dshift / 1e3)
    rows.append(row)
    print(f"{row['time']} M{e.mag:.1f} {e.net} gap {e.gap}: OBS P {row['n_obs_P']} S {row['n_obs_S']}, land {row['n_land']}, "
          f"S-P obs {row['SP_obs_med']:.1f} vs catalog {row['SP_catalog_pred_med']:.1f}"
          + (f" -> reloc {rl['lat']:.2f},{rl['lon']:.2f} z{rl['depth']} shift {dshift / 1e3:.0f} km rms {rl['rms']:.2f}" if rl else ""), flush=True)
    if st is not None and row["n_obs_P"] >= 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        for k, s in enumerate(STA):
            z = [t for t in st.select(station=s) if t.stats.channel.endswith("Z")]
            if not z:
                continue
            tz = z[0].times(reftime=t0)
            y = z[0].data / (np.abs(z[0].data).max() + 1e-30)
            ax.plot(tz, k + 0.45 * y, "k", lw=0.3)
            if s in op:
                ax.plot([op[s]["tp"]] * 2, [k - 0.45, k + 0.45], "r")
                if np.isfinite(op[s]["ts"]):
                    ax.plot([op[s]["ts"]] * 2, [k - 0.45, k + 0.45], "b")
            ax.text(-88, k, s, fontsize=7, va="center")
        ax.set(xlim=(-90, 150), yticks=[], xlabel="s after catalogue origin",
               title=f"{row['time']} M{e.mag:.1f} ({e.place}) — Hydrate Ridge Z, 3-20 Hz; red P, blue S")
        fig.tight_layout()
        fig.savefig(FIG / f"catalog_check_{e.id}.png", dpi=90)
        plt.close(fig)

R = pd.DataFrame(rows)
R.to_csv(OUT / "catalog_check.csv", index=False)
print(R.round(2).to_string(index=False))
