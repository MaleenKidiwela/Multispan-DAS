"""Item 8c (real data): single-span vs multi-span DAS beamforming of earthquake P waves, Dec 23 2025 - Jan 28 2026.

Events: ComCat M >= 4.0 within 1500 km of (45 N, 125 W) plus M >= 6.0 worldwide (30-95 deg), well located by land
networks -> true back-azimuth from each configuration's centroid (great circle).
Data: continuous 100-m stream, both cables, true per-chunk times regridded (dropped chunks = gaps); 1-km bins of good
channels (south C3 excluded as tonal); DAS stamp -> UTC with the setup latency (D256 25.2 s, D128 11.8 s; same on
both cables, s67). Band 0.4-0.9 Hz (above the microseism peak); envelope |hilbert| smoothed 2 s, divided by the
pre-event median, log, clipped at 0; upsampled to 4 Hz.
Beam (incoherent delay-and-sum of envelopes, polarity- and coupling-independent): plane wave from back-azimuth baz with
horizontal slowness s arrives at bin n at t0 - s (x_n sin baz + y_n cos baz); B(baz, s) = max over t0 in
[tP - 20, tP + 40] s of the mean over bins of e_n(t0 + delay_n). Noise reference: same beam in [tP - 400, tP - 200] s.
Configurations: every single span with >= 10 bins (conventional DAS reach), each cable's spans together, both cables.
FEATURE=stalta (default): beam of the per-bin STA/LTA (2 s / 20 s) normalised to its event-window maximum (every bin
equal weight; aligns onsets, not site-amplified energy); t0 searched in [tP - 15, tP + 15] s. (FEATURE=env, the log-
envelope beam in [tP - 20, tP + 40], failed on the M6.0 control: 43 deg off, slowness 0.04 - S/Lg energy dominated.)
Score: |baz error| of the beam maximum; event counted as recorded if the both-cable beam peak / noise-beam peak >= SNR_REC (1.5).
Outputs: outputs_100m/beamform/event_beam_compare.csv, figs/event_beam_compare.png
"""
import io
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
from msdvv.interspan import make_bins
from msdvv.io import load_cfg

cfg = load_config(str(Path(__file__).resolve().parents[1] / "config_100m_v2.yaml"))
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
G = Geod(ellps="WGS84")
TAUP = TauPyModel("iasp91")
LAT = {256: 25.2, 128: 11.8}
FS_UP = 4.0
import os
FEATURE = os.environ.get("FEATURE", "stalta")  # stalta: STA/LTA (2 s / 20 s) of the band energy, normalised per bin; env: log envelope
ONLY = os.environ.get("ONLY", "")  # substring of the event time to run a single event (validation)
SNR_REC = float(os.environ.get("SNR_REC", "1.5"))
WIN = [float(v) for v in os.environ.get("WIN", "-15,15").split(",")]  # search window for t0 relative to predicted first P
BAZ = np.arange(0, 360, 2.0)
SLO = np.arange(0.0, 0.2001, 0.005)

# ---- bins of both cables, common local frame
B = []
for cab in ("north", "south"):
    q = pd.read_csv(OUT / f"channel_quality_{cab}.csv")
    b, bid = make_bins(q, 1.0, 3)
    b["cable"] = cab
    b["chans"] = [np.where(bid == k)[0] for k in b.bin]
    if cab == "south":
        b = b[b.seg != "C3"]
    B.append(b)
B = pd.concat(B, ignore_index=True)
CONF = {f"{c} {s} (single span)": (B.cable == c) & (B.seg == s) for (c, s), g in B.groupby(["cable", "seg"]) if len(g) >= 10}
CONF.update({"north, all spans": B.cable == "north", "south loop, all spans": B.cable == "south", "both cables": np.ones(len(B), bool)})
print({k: int(v.sum()) for k, v in CONF.items()}, flush=True)

# ---- events
par = dict(format="csv", starttime="2025-12-23", endtime="2026-01-29", orderby="time-asc")
r1 = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(par, minmagnitude=4.0, latitude=45, longitude=-125, maxradiuskm=1500))
r2 = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(par, minmagnitude=6.0))
E = pd.concat([pd.read_csv(io.StringIO(r1.text)), pd.read_csv(io.StringIO(r2.text))]).drop_duplicates("id").reset_index(drop=True)
lat0, lon0 = B.lat.mean(), B.lon.mean()
_, baz0, dd = G.inv(E.longitude.values, E.latitude.values, np.full(len(E), lon0), np.full(len(E), lat0))
E["dist_deg"] = dd / 111195.0
E = E[(E.dist_deg <= 13.5) | ((E.dist_deg >= 30) & (E.dist_deg <= 95))].reset_index(drop=True)
E["t"] = pd.to_datetime(E.time).astype("int64") / 1e9
tP = []
for e in E.itertuples():
    arr = TAUP.get_travel_times(max(e.depth, 0), e.dist_deg, ["P", "p", "Pn", "Pg"])
    tP.append(e.t + min(a.time for a in arr) if arr else np.nan)
E["tP"] = tP
print(f"{len(E)} events", flush=True)

inv = pd.read_csv(OUT / "inventory.csv")


def env_series(path, cab, t_a, t_b, D):
    f = load_cfg(path, cfg)
    lat_ = LAT[int(D)]
    tt = f.times - lat_
    w = (tt >= t_a - 60) & (tt <= t_b + 60)
    if w.sum() < 100:
        return None
    sos = butter(4, (0.4, 0.9), "bandpass", fs=f.fs, output="sos")
    tw = tt[w]
    valid = f.valid[w] if f.valid is not None else np.ones(w.sum(), bool)
    tg = np.arange(t_a, t_b, 1 / FS_UP)
    rows = {}
    sub = B[B.cable == cab]
    for ix, b in sub.iterrows():
        x = sosfiltfilt(sos, f.data[b.chans][:, w].astype(np.float64), axis=1).mean(0)
        a2 = np.abs(hilbert(x)) ** 2
        if FEATURE == "stalta":
            sta = uniform_filter1d(a2, int(2 * f.fs) | 1)
            lta = np.convolve(a2, np.ones(int(20 * f.fs)) / int(20 * f.fs), "full")[: len(a2)]  # trailing 20-s mean
            e = sta / (lta + 1e-30 * a2.max())
        else:
            e = uniform_filter1d(a2, int(2 * f.fs) | 1)
        rows[ix] = np.interp(tg, tw, e)
    vg = np.interp(tg, tw, valid.astype(float)) > 0.99
    return tg, rows, vg


def beam(Ee, tg, sel_idx, x, y, t_lo, t_hi):
    m = (tg >= t_lo) & (tg <= t_hi)
    i0 = np.where(m)[0]
    best = (-np.inf, 0, 0)
    P = np.zeros((len(BAZ), len(SLO)))
    for ia, bz in enumerate(np.radians(BAZ)):
        proj = x * np.sin(bz) + y * np.cos(bz)
        dl = -SLO[:, None] * proj[None]  # (nslo, nbin) arrival delay
        idx = np.clip(i0[None, None, :] + np.rint(dl[:, :, None] * FS_UP).astype(int), 0, len(tg) - 1)
        S = Ee[np.arange(len(x))[None, :, None], idx].mean(1)  # (nslo, nt)
        P[ia] = S.max(1)
    j = np.unravel_index(np.argmax(P), P.shape)
    return P, BAZ[j[0]], SLO[j[1]]


rows = []
if ONLY:
    E = E[E.time.str.contains(ONLY)]
for e in E.itertuples():
    t_a, t_b = e.tP - 450, e.tP + 120
    series = {}
    for cab in ("north", "south"):
        c = inv[(inv.cable == cab) & (inv.t0 - 30 <= t_a) & (inv.t1 + 60 >= t_b)]
        if not len(c):
            continue
        s = env_series(c.path.iloc[0], cab, t_a, t_b, c.D.iloc[0])
        if s is not None:
            series[cab] = s
    if len(series) < 2:
        print(e.time[:19], "missing data", list(series), flush=True)
        continue
    tg = series["north"][0]
    vg = series["north"][2] & series["south"][2]
    if vg[(tg >= e.tP - 400) & (tg <= e.tP + 40)].mean() < 0.95:
        print(e.time[:19], "gap in window", flush=True)
        continue
    Eall = np.zeros((len(B), len(tg)))
    for cab, (_, rr, _) in series.items():
        for ix, v in rr.items():
            Eall[ix] = v
    pre = (tg >= e.tP - 400) & (tg <= e.tP - 200)
    if FEATURE == "stalta":  # each bin: CF minus its pre-event median, scaled by its max in the event window -> equal weight
        evw = (tg >= e.tP + WIN[0] - 40) & (tg <= e.tP + WIN[1] + 40)
        Eall = Eall - np.median(Eall[:, pre], axis=1, keepdims=True)
        Eall = np.clip(Eall / (Eall[:, evw].max(1, keepdims=True) + 1e-12), 0, None)
    else:
        Eall = np.clip(np.log(Eall / (np.median(Eall[:, pre], axis=1, keepdims=True) + 1e-30) + 1e-30), 0, None)
    for name, sel in CONF.items():
        si = np.where(np.asarray(sel))[0]
        la_c, lo_c = B.lat.values[si].mean(), B.lon.values[si].mean()
        az, _, d = G.inv(np.full(len(si), lo_c), np.full(len(si), la_c), B.lon.values[si], B.lat.values[si])
        x, y = d / 1e3 * np.sin(np.radians(az)), d / 1e3 * np.cos(np.radians(az))
        P, bz, sl = beam(Eall[si], tg, si, x, y, e.tP + WIN[0], e.tP + WIN[1])
        Pn, *_ = beam(Eall[si], tg, si, x, y, e.tP - 400, e.tP - 400 + WIN[1] - WIN[0])
        true = G.inv(lo_c, la_c, e.longitude, e.latitude)[0] % 360
        err = (bz - true + 180) % 360 - 180
        rows.append(dict(event=e.id, time=e.time[:19], mag=e.mag, dist_deg=round(e.dist_deg, 1), config=name, n_bins=len(si),
                         baz_true=round(true, 1), baz_est=bz, baz_err=err, slowness=sl, beam_peak=P.max(), noise_peak=Pn.max(),
                         beam_snr=P.max() / max(Pn.max(), 1e-9)))
    r_ = [r for r in rows if r["event"] == e.id]
    print(e.time[:19], f"M{e.mag}", f"{e.dist_deg:.0f} deg", " | ".join(f"{r['config'].split(' (')[0]}: {r['baz_err']:+.0f}" for r in r_), flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / f"beamform/event_beam_compare_{FEATURE}{'_' + ONLY[:10] if ONLY else ''}.csv", index=False)
ok = R[(R.config == "both cables") & (R.beam_snr >= SNR_REC)].event
S = R[R.event.isin(ok)].assign(abs_err=lambda d: d.baz_err.abs(), wrong=lambda d: d.baz_err.abs() > 20)
summ = S.groupby("config", sort=False).agg(n_events=("event", "nunique"), median_abs_err=("abs_err", "median"), frac_wrong_gt20=("wrong", "mean"))
pd.set_option("display.width", 200)
print(f"\n{len(ok)} events recorded (both-cable beam SNR >= {SNR_REC}) of {R.event.nunique()}")
print(summ.round(2).to_string())

fig, axs = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw=dict(width_ratios=[2, 1]))
ax = axs[0]
names = list(CONF)
for i, n in enumerate(names):
    q = S[S.config == n]
    ax.scatter(np.full(len(q), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(q)), q.abs_err, s=14,
               c=["tab:red" if "single" in n else "tab:blue"] * len(q))
    ax.plot([i - 0.3, i + 0.3], [q.abs_err.median()] * 2, "k-", lw=2)
ax.set_xticks(range(len(names)), [n.replace(" (single span)", "\n(single)") for n in names], rotation=45, ha="right", fontsize=8)
ax.set(ylabel="|back-azimuth error| (deg)", yscale="symlog", title=f"{len(ok)} recorded earthquakes: back-azimuth error per array configuration (black: median)")
ax = axs[1]
ax.barh(range(len(summ)), summ.frac_wrong_gt20.values * 100, color=["tab:red" if "single" in n else "tab:blue" for n in summ.index])
ax.set_yticks(range(len(summ)), summ.index, fontsize=8)
ax.set(xlabel="% of events with |error| > 20 deg", title="wrong direction")
fig.tight_layout()
fig.savefig(OUT / f"figs/event_beam_compare_{FEATURE}{'_' + ONLY[:10] if ONLY else ''}.png", dpi=100)
