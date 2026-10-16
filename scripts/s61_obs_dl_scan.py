"""Item 3b: deep-learning scan of the Hydrate Ridge seafloor seismometers (HYSB1, HYS11-14; 20-Hz continuous data
Dec 23 2025 - Jan 28 2026) for earthquakes, compared with the ComCat catalogue.

PhaseNet with OBS-trained weights ("obs", PickBlue) on each station-day (resampled 20 -> 100 Hz); picks P >= P_THR (0.5),
S >= S_THR (0.4); picks within 5 s of the day-file edges dropped (spurious). Association: an event = >= 3 stations with a
P pick within ASSOC_S (4 s; the array spans ~25 km); its S-P = median over
stations with an S pick 1-60 s after their P. Local = S-P <= 10 s (hypocentral distance <~ 80 km).
Catalogue match: any ComCat event (M >= 1, within 400 km) whose predicted P at HYS14 (iasp91-like 6.5 km/s for P,
crude) is within 15 s of the detection. Run in env pyfetch.
Outputs: obs/dl_scan_picks.csv, obs/dl_scan_events.csv, figs/obs_dl_scan.png
"""
import io

import numpy as np
import scipy.signal


def _np_detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
    x = np.moveaxis(np.asarray(data, dtype=np.float64), axis, -1)
    if type in ("constant", "c"):
        y = x - x.mean(-1, keepdims=True)
    else:
        n = x.shape[-1]
        t = np.arange(n) - (n - 1) / 2.0
        y = x - x.mean(-1, keepdims=True) - (x * t).sum(-1, keepdims=True) / (t**2).sum() * t
    return np.moveaxis(y, -1, axis)


scipy.signal.detrend = _np_detrend
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seisbench.models as sbm
import torch
from obspy import UTCDateTime, read, read_inventory
from pyproj import Geod

OBS = "/data/wsd02/multispan_dvv/obs"
G = Geod(ellps="WGS84")
STA = ["HYSB1", "HYS11", "HYS12", "HYS13", "HYS14"]
inv = read_inventory(f"{OBS}/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
import os
PTH, STH = float(os.environ.get("P_THR", "0.5")), float(os.environ.get("S_THR", "0.4"))
AWIN = float(os.environ.get("ASSOC_S", "4.0"))
TAG = os.environ.get("TAG", "")
model = sbm.PhaseNet.from_pretrained("obs")
model.to("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for day in pd.date_range("2025-12-23", "2026-01-28"):
    for s in STA:
        fs_ = glob.glob(f"{OBS}/continuous/{day:%Y%m%d}/OO.{s}.*.mseed")
        if not fs_:
            continue
        st = read(fs_[0])
        st.merge(fill_value=0)
        if len(st) < 3:
            continue
        for tr in st:
            tr.data = np.nan_to_num(tr.data.astype(np.float32))
            tr.stats.channel = tr.stats.channel[:2] + tr.stats.channel[-1]
        a, b = max(t.stats.starttime for t in st), min(t.stats.endtime for t in st)
        st.trim(a, b)
        st.resample(100.0)
        out = model.classify(st, P_threshold=PTH, S_threshold=STH)
        picks = out.picks if hasattr(out, "picks") else (out[0] if isinstance(out, tuple) else out)
        for p in picks:
            tp = float(p.peak_time.timestamp)
            if tp - a.timestamp < 5 or b.timestamp - tp < 5:  # day-file edges: spurious picks at 00:00 / 23:59:59
                continue
            rows.append(dict(sta=s, phase=p.phase, t=tp, prob=float(p.peak_value)))
    print(day.date(), len(rows), flush=True)
P = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
P.to_csv(f"{OBS}/dl_scan_picks{TAG}.csv", index=False)

# ---- association
pp = P[P.phase == "P"].reset_index(drop=True)
ev, used = [], np.zeros(len(pp), bool)
for i in range(len(pp)):
    if used[i]:
        continue
    w = pp[(pp.t >= pp.t[i]) & (pp.t <= pp.t[i] + AWIN)]
    w = w.sort_values("prob", ascending=False).drop_duplicates("sta")
    if len(w) >= 3:
        used[w.index] = True
        sp = []
        for r in w.itertuples():
            s_ = P[(P.phase == "S") & (P.sta == r.sta) & (P.t > r.t + 1) & (P.t < r.t + 60)]
            if len(s_):
                sp.append(s_.t.iloc[0] - r.t)
        first = w.loc[w.t.idxmin()]
        ev.append(dict(t_first=float(w.t.min()), first_sta=first.sta, n_sta=len(w), mean_prob=float(w.prob.mean()),
                       sp_s=float(np.median(sp)) if sp else np.nan, n_sp=len(sp), t_HYSB1_minus_HYS14=float(w.set_index("sta").t.get("HYSB1", np.nan) - w.set_index("sta").t.get("HYS14", np.nan))))
E = pd.DataFrame(ev)
r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(
    format="csv", starttime="2025-12-23", endtime="2026-01-29", minmagnitude=1.0, latitude=44.57, longitude=-125.15, maxradiuskm=400))
C = pd.read_csv(io.StringIO(r.text))
C["t"] = [UTCDateTime(x).timestamp for x in C.time]
_, _, dd = G.inv(C.longitude.values, C.latitude.values, np.full(len(C), crd["HYS14"][1]), np.full(len(C), crd["HYS14"][0]))
C["d_km"] = dd / 1e3
C["tP"] = C.t + np.hypot(C.d_km, C.depth.clip(lower=0)) / 6.5
m_id, m_mag, m_d = [], [], []
for r_ in E.itertuples():
    q = C[np.abs(C.tP - r_.t_first) <= 15]
    if len(q):
        k = (q.tP - r_.t_first).abs().idxmin()
        m_id.append(q.id[k]); m_mag.append(q.mag[k]); m_d.append(q.d_km[k])
    else:
        m_id.append(""); m_mag.append(np.nan); m_d.append(np.nan)
E["cat_id"], E["cat_mag"], E["cat_dist_km"] = m_id, m_mag, m_d
E["time"] = [str(UTCDateTime(x))[:22] for x in E.t_first]
E["local"] = E.sp_s <= 10
E.to_csv(f"{OBS}/dl_scan_events{TAG}.csv", index=False)
pd.set_option("display.width", 220)
print(f"{len(P)} picks; {len(E)} associated events (>= 3 stations); {int(E.local.sum())} local (S-P <= 10 s); "
      f"{int((E.cat_id != '').sum())} matched to ComCat; catalogue events within 100 km of HYS14: {int((C.d_km <= 100).sum())}")
print(E[E.local].sort_values("t_first")[["time", "n_sta", "mean_prob", "sp_s", "n_sp", "first_sta", "t_HYSB1_minus_HYS14", "cat_id", "cat_mag", "cat_dist_km"]].round(2).to_string(index=False))
fig, axs = plt.subplots(2, 1, figsize=(13, 8))
ax = axs[0]
for loc, col in ((True, "tab:red"), (False, "0.5")):
    q = E[E.local == loc]
    ax.scatter(pd.to_datetime(q.t_first, unit="s"), q.sp_s, s=8 + 20 * (q.cat_id != ""), c=col, label=f"{'local (S-P <= 10 s)' if loc else 'regional'}: {len(q)} (large = catalogued)")
ax.set(ylabel="S-P (s)", yscale="log", title="Deep-learning detections on the Hydrate Ridge seismometers (>= 3 stations)")
ax.legend(fontsize=8)
ax = axs[1]
q = E[E.sp_s.notna()]
ax.hist(q.sp_s, bins=np.arange(0, 60, 1), color="0.6", label="all")
ax.hist(q[q.cat_id != ""].sp_s, bins=np.arange(0, 60, 1), color="tab:blue", label="matched to ComCat")
ax.set(xlabel="S-P (s)", ylabel="events")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"/data/wsd02/multispan_dvv/outputs_100m/figs/obs_dl_scan{TAG}.png", dpi=105)
print("done")
