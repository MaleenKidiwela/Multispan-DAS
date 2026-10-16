"""Item 3c: magnitude of uw62216067 (Jan 26 2026, catalogue ML 2.2 at the wrong place; relocated 43.34N 127.17W,
origin 09:08:56.6) by amplitude comparison with catalogued Blanco Transform events recorded by the same Hydrate Ridge
seafloor seismometers (HYSB1, HYS11-14; 20-Hz continuous data, Dec 23 - Jan 28).

Amplitude: peak of the 2-8 Hz horizontal (vector N/E) envelope in the S window [t0 + d/4.6 - 5, t0 + d/4.6 + 30] s
(Sn/So; the window starts before the S pick of uw62216067, which is d/4.6 ~ 24 s after its origin at 110 km... see
README), noise-checked (peak / pre-P RMS >= 5). Magnitude per station: M = log10 A + 1.11 log10 r + 0.00189 r + c_sta
(Hutton & Boore 1987 distance term), c_sta = median over the reference events of (M_cat - log10 A - ...).
Output: obs/uw62216067_magnitude.json, figs/uw62216067_magnitude.png
"""
import glob
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from obspy import Stream, UTCDateTime, read, read_inventory
from pyproj import Geod

G = Geod(ellps="WGS84")
OBS = "/data/wsd02/multispan_dvv/obs"
inv = read_inventory(f"{OBS}/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
STA = ["HYSB1", "HYS11", "HYS12", "HYS13", "HYS14"]
r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(
    format="csv", starttime="2025-12-23T20:00", endtime="2026-01-28T12:00", minmagnitude=2.5, minlatitude=42.0, maxlatitude=45.0,
    minlongitude=-129.5, maxlongitude=-125.8))
REF = pd.read_csv(io.StringIO(r.text))
REF = REF[REF.mag < 5.5].reset_index(drop=True)  # exclude the Jan 16 M6.0 (clipping, very long coda)
TARGET = dict(time="2026-01-26T09:08:56.6", latitude=43.34, longitude=-127.17, mag=np.nan, magType="?", id="uw62216067 (relocated)")


def amp(ev, sta):
    t0 = UTCDateTime(ev["time"])
    _, _, d = G.inv(ev["longitude"], ev["latitude"], crd[sta][1], crd[sta][0])
    rkm = d / 1e3
    tS = t0 + rkm / 4.6
    st = Stream()
    for day in sorted({(t0 - 120).strftime("%Y%m%d"), (tS + 60).strftime("%Y%m%d")}):
        for fn in glob.glob(f"{OBS}/continuous/{day}/OO.{sta}.*.mseed"):
            st += read(fn)
    st = st.select(component="[NE12]")
    if len(st) < 2:
        return rkm, np.nan, np.nan
    st.merge(fill_value=0)
    st.trim(t0 - 120, tS + 60)
    if len(st) < 2 or min(tr.stats.npts for tr in st) < 1000:
        return rkm, np.nan, np.nan
    st.detrend("demean")
    st.filter("bandpass", freqmin=2, freqmax=8, zerophase=True)
    n = min(tr.stats.npts for tr in st)
    h = np.sqrt(sum(tr.data[:n].astype(float) ** 2 for tr in st[:2]))
    tt = st[0].times(reftime=t0)[:n]
    sig = h[(tt >= rkm / 4.6 - 5) & (tt <= rkm / 4.6 + 30)]
    noi = h[(tt >= -110) & (tt <= -10)]
    if len(sig) == 0 or len(noi) == 0:
        return rkm, np.nan, np.nan
    return rkm, float(sig.max()), float(sig.max() / (np.sqrt(np.mean(noi**2)) + 1e-30))


rows = []
for _, e in pd.concat([REF, pd.DataFrame([TARGET])], ignore_index=True).iterrows():
    for s in STA:
        rkm, A, snr = amp(e, s)
        rows.append(dict(id=e["id"], time=str(e["time"])[:19], mag=e["mag"], magType=e["magType"], sta=s, r_km=rkm, A=A, snr=snr))
D = pd.DataFrame(rows)
D["dist_term"] = 1.11 * np.log10(D.r_km) + 0.00189 * D.r_km
ok = (D.snr >= 5) & np.isfinite(D.A)
D["logA_corr"] = np.log10(D.A) + D.dist_term
ref = D[ok & D.mag.notna()]
c = (ref.mag - ref.logA_corr).groupby(ref.sta).median()
cs = (ref.mag - ref.logA_corr).groupby(ref.sta).std()
D["M_est"] = D.logA_corr + D.sta.map(c)
loo = []  # leave-one-event-out check of the method on the references
for eid, g in ref.groupby("id"):
    rest = ref[ref.id != eid]
    cc = (rest.mag - rest.logA_corr).groupby(rest.sta).median()
    loo.append(dict(id=eid, mag=g.mag.iloc[0], M_loo=float(np.median(g.logA_corr + g.sta.map(cc)))))
L = pd.DataFrame(loo)
tg = D[(D.id == TARGET["id"]) & ok]
Mt = float(np.median(tg.M_est))
Ms = float(1.4826 * np.median(np.abs(tg.M_est - Mt)))
pd.set_option("display.width", 200)
print(D[ok].groupby(["id", "time", "mag", "magType"]).agg(n_sta=("sta", "size"), r_km=("r_km", "median"), M_est=("M_est", "median")).round(2).to_string())
print("station constants:", c.round(2).to_dict(), "spread", cs.round(2).to_dict())
print("leave-one-out: M_loo - M_cat median", round(float((L.M_loo - L.mag).median()), 2), "MAD", round(float(1.4826 * np.median(np.abs(L.M_loo - L.mag - np.median(L.M_loo - L.mag)))), 2))
print(f"uw62216067 (relocated): M = {Mt:.2f} +- {Ms:.2f} from {len(tg)} stations (catalogue ML 2.2)")
json.dump(dict(M=round(Mt, 2), M_mad=round(Ms, 2), n_sta=int(len(tg)), method="amplitude ratio vs Blanco reference events, 2-8 Hz S envelope, Hutton-Boore distance term",
               n_ref_events=int(ref.id.nunique()), loo_bias=round(float((L.M_loo - L.mag).median()), 2)), open(f"{OBS}/uw62216067_magnitude.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(L.mag, L.M_loo, s=40, label="reference events (leave-one-out)")
lim = (2.3, 4.5)
ax.plot(lim, lim, "k--", lw=0.8)
ax.axhline(Mt, color="tab:red", lw=1.5, label=f"uw62216067 relocated: M {Mt:.2f} +- {Ms:.2f}")
ax.axhspan(Mt - Ms, Mt + Ms, color="tab:red", alpha=0.15)
ax.axhline(2.2, color="0.5", ls=":", label="catalogue ML 2.2 (wrong location)")
ax.set(xlim=lim, ylim=lim, xlabel="catalogue magnitude (USGS/PNSN)", ylabel="estimated from Hydrate Ridge OBS amplitudes",
       title="Magnitude of uw62216067 from OBS S amplitudes (Blanco references)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("/data/wsd02/multispan_dvv/outputs_100m/figs/uw62216067_magnitude.png", dpi=110)
