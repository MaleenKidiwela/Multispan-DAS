"""Item 8c (real data, long period): single-span vs multi-span beamforming of earthquake Rayleigh waves.

Why: the 0.4-0.9 Hz P beam (s68) records only the Jan 16 M6.0 (teleseismic P is near-vertical, the horizontal fiber
barely senses it; small regional events are below the 1.86-Hz stream's band). Large earthquakes' long-period surface
waves are recorded well along the fiber, and at 0.04-0.07 Hz the Rayleigh wavelength (~55-100 km) exceeds any single
span (20-60 km): the case where the multi-span aperture (120-300 km) should matter most.
Events: ComCat M >= 6.0 worldwide (20-150 deg) and M >= 5.0 within 20 deg, Dec 23 2025 - Jan 28 2026.
Data: continuous 100-m stream, both cables, true times regridded; 1-km bins (south C3 excluded); 0.02-0.1 Hz.
Window: Rayleigh group arrivals VG 3.8 -> 3.0 km/s (after the Love waves, whose axial strain flips sign with cable
orientation; capped at 1200 s); sub-windows need >= 90 % valid samples (gaps zero-filled); noise: same length ending 60 s before the first P.
Beam: Bartlett, CSDM averaged over 300-s sub-windows (50 % overlap), spectra whitened per bin (unit amplitude),
diagonal removed, axial-strain Rayleigh weighting cos^2(baz - cable azimuth) (no polarity flips for Rayleigh), at
0.04/0.05/0.06/0.07 Hz, normalised beams averaged over frequency; slowness 0-0.5 s/km.
Score: |baz error| of the beam maximum vs the great-circle back-azimuth; event 'recorded' if the both-cable event beam
maximum / noise beam maximum >= SNR_REC (1.3).
Outputs: outputs_100m/beamform/surface_wave_beam.csv, figs/surface_wave_beam.png, figs/surface_wave_beam_examples.png
"""
import io
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
from obspy.taup import TauPyModel
from pyproj import Geod
from scipy.signal import butter, sosfiltfilt

from msdvv import load_config
from msdvv.interspan import make_bins
from msdvv.io import load_cfg

cfg = load_config(str(Path(__file__).resolve().parents[1] / "config_100m_v2.yaml"))
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
G = Geod(ellps="WGS84")
TAUP = TauPyModel("iasp91")
LAT = {256: 25.2, 128: 11.8}
FREQS = (0.04, 0.05, 0.06, 0.07)
BAZ = np.radians(np.arange(0, 360, 2.0))
SLO = np.arange(0.0, 0.5001, 0.01)
SUB_S = 300.0
SNR_REC = float(os.environ.get("SNR_REC", "1.3"))
ONLY = os.environ.get("ONLY", "")  # comma-separated substrings of event times (diagnostics)
NORTH_SIGN = float(os.environ.get("NORTH_SIGN", "1"))  # polarity test for the north cable
NORTH_SHIFT = float(os.environ.get("NORTH_SHIFT", "0"))  # s, added to north-cable times (inter-cable timing test)
VG = [float(v) for v in os.environ.get("VG", "3.8,3.0").split(",")]  # Rayleigh group-velocity window (km/s)
TAG = os.environ.get("TAG", "")

B = []
for cab in ("north", "south"):
    q = pd.read_csv(OUT / f"channel_quality_{cab}.csv")
    b, bid = make_bins(q, 1.0, 3)
    b["cable"] = cab
    b["chans"] = [np.where(bid == k)[0] for k in b.bin]
    if cab == "south":
        b = b[b.seg != "C3"]
    azs = np.full(len(b), np.nan)
    for _, g in b.groupby("seg"):
        idx = list(g.index)
        for i, ix in enumerate(idx):
            j0, j1 = idx[max(i - 2, 0)], idx[min(i + 2, len(idx) - 1)]
            azs[b.index.get_loc(ix)] = G.inv(b.lon[j0], b.lat[j0], b.lon[j1], b.lat[j1])[0]
    b["cab_az"] = np.radians(azs)
    B.append(b)
B = pd.concat(B, ignore_index=True)
CONF = {f"{c} {s} (single span)": (B.cable == c) & (B.seg == s) for (c, s), g in B.groupby(["cable", "seg"]) if len(g) >= 10}
CONF.update({"north, all spans": B.cable == "north", "south loop, all spans": B.cable == "south", "both cables": np.ones(len(B), bool)})

par = dict(format="csv", starttime="2025-12-23", endtime="2026-01-29", orderby="time-asc")
r1 = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(par, minmagnitude=6.0))
r2 = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(par, minmagnitude=5.0, latitude=45, longitude=-125, maxradius=20))
E = pd.concat([pd.read_csv(io.StringIO(r1.text)), pd.read_csv(io.StringIO(r2.text))]).drop_duplicates("id").reset_index(drop=True)
lat0, lon0 = B.lat.mean(), B.lon.mean()
_, _, dd = G.inv(E.longitude.values, E.latitude.values, np.full(len(E), lon0), np.full(len(E), lat0))
E["dist_km"] = dd / 1e3
E["dist_deg"] = E.dist_km / 111.195
E = E[((E.mag >= 6.0) & (E.dist_deg >= 20) & (E.dist_deg <= 150)) | ((E.mag >= 5.0) & (E.dist_deg < 20))].reset_index(drop=True)
E["t"] = pd.to_datetime(E.time).astype("int64") / 1e9
E["tP"] = [e.t + min(a.time for a in TAUP.get_travel_times(max(e.depth, 0), e.dist_deg, ["P", "Pn", "Pg", "p", "Pdiff"])) for e in E.itertuples()]
E["tR0"] = E.t + E.dist_km / VG[0]
E["tR1"] = np.minimum(E.t + E.dist_km / VG[1], E.tR0 + 1200)
if ONLY:
    E = E[E.time.str.contains("|".join(ONLY.split(",")))].reset_index(drop=True)
print(f"{len(E)} events", flush=True)
inv = pd.read_csv(OUT / "inventory.csv")


def series(cab, t_a, t_b):
    c = inv[(inv.cable == cab) & (inv.t0 - 30 <= t_a) & (inv.t1 + 60 >= t_b)]
    if not len(c):
        return None
    f = load_cfg(c.path.iloc[0], cfg)
    tt = f.times - LAT[int(c.D.iloc[0])] + (NORTH_SHIFT if cab == "north" else 0.0)
    w = (tt >= t_a - 300) & (tt <= t_b + 300)
    sos = butter(4, (0.02, 0.1), "bandpass", fs=f.fs, output="sos")
    sub = B[B.cable == cab]
    sg = NORTH_SIGN if cab == "north" else 1.0
    Z = {ix: sg * sosfiltfilt(sos, f.data[b.chans][:, w].astype(np.float64), axis=1).mean(0) for ix, b in sub.iterrows()}
    valid = f.valid[w] if f.valid is not None else np.ones(w.sum(), bool)
    return tt[w], Z, valid, f.fs


def beam(X, x, y, ca, fs):
    """X: (nsub, nbin, nt) windows -> normalised Bartlett beam (nbaz, nslo), averaged over FREQS."""
    n = X.shape[-1]
    F = np.fft.rfft(X * np.hanning(n), axis=-1)
    fr = np.fft.rfftfreq(n, 1 / fs)
    P = np.zeros((len(BAZ), len(SLO)))
    A = np.cos(BAZ[:, None, None] - ca[None, None]) ** 2  # (nbaz, 1, nbin)
    for f0 in FREQS:
        u = F[:, :, int(np.argmin(np.abs(fr - f0)))]
        u = u / (np.abs(u) + 1e-30)  # whitened
        W = A * np.exp(-2j * np.pi * f0 * (-SLO[None, :, None] * (x[None, None] * np.sin(BAZ)[:, None, None] + y[None, None] * np.cos(BAZ)[:, None, None])))
        num = np.mean(np.abs(np.einsum("abn,rn->abr", np.conj(W), u)) ** 2, -1) - (np.abs(W) ** 2).sum(-1)
        P += num / (np.abs(W).sum(-1) ** 2 - (np.abs(W) ** 2).sum(-1))
    return P / len(FREQS)


rows, examples = [], {}
for e in E.itertuples():
    L = e.tR1 - e.tR0
    n_a, n_b = e.tP - 60 - L, e.tP - 60
    S = {c: series(c, n_a, e.tR1) for c in ("north", "south")}
    if any(v is None for v in S.values()):
        print(e.time[:19], "no file covering the window", flush=True)
        continue
    fs = S["north"][3]
    tg = np.arange(n_a, e.tR1, 1 / fs)
    Z = np.zeros((len(B), len(tg)))
    vg = np.ones(len(tg), bool)
    for c, (tt, zz, valid, _) in S.items():
        for ix, z in zz.items():
            Z[ix] = np.interp(tg, tt, z)
        vg &= np.interp(tg, tt, valid.astype(float)) > 0.99  # gaps are zero-filled in the data; sub-windows need >= 90 % valid
    nsub = int(SUB_S * fs)

    def windows(t_a, t_b):
        st = [i for i in range(np.searchsorted(tg, t_a), np.searchsorted(tg, t_b) - nsub, nsub // 2) if vg[i:i + nsub].mean() >= 0.9]
        return np.array([Z[:, i:i + nsub] for i in st]) if st else None

    We, Wn = windows(e.tR0, e.tR1), windows(n_a, n_b)
    if We is None or Wn is None:
        print(e.time[:19], "gaps", flush=True)
        continue
    for name, sel in CONF.items():
        si = np.where(np.asarray(sel))[0]
        la_c, lo_c = B.lat.values[si].mean(), B.lon.values[si].mean()
        az, _, d = G.inv(np.full(len(si), lo_c), np.full(len(si), la_c), B.lon.values[si], B.lat.values[si])
        x, y = d / 1e3 * np.sin(np.radians(az)), d / 1e3 * np.cos(np.radians(az))
        Pe = beam(We[:, si], x, y, B.cab_az.values[si], fs)
        Pn = beam(Wn[:, si], x, y, B.cab_az.values[si], fs)
        ia, js = np.unravel_index(np.argmax(Pe), Pe.shape)
        true = G.inv(lo_c, la_c, e.longitude, e.latitude)[0] % 360
        est = np.degrees(BAZ[ia])
        rows.append(dict(event=e.id, time=e.time[:19], mag=e.mag, dist_deg=round(e.dist_deg, 1), config=name, n_bins=len(si), baz_true=round(true, 1),
                         baz_est=est, baz_err=(est - true + 180) % 360 - 180, slowness=SLO[js], beam_snr=Pe.max() / max(Pn.max(), 1e-9),
                         n_win=len(We)))
        if name in ("south A2 (single span)", "both cables"):
            examples[(e.id, name)] = (Pe, true)
    r_ = [r for r in rows if r["event"] == e.id]
    print(e.time[:19], f"M{e.mag} {e.dist_deg:.0f} deg SNR(both) {r_[-1]['beam_snr']:.2f}", " | ".join(f"{r['config'].split(' (')[0]}: {r['baz_err']:+.0f}" for r in r_), flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / f"beamform/surface_wave_beam{TAG}.csv", index=False)
ok = R[(R.config == "both cables") & (R.beam_snr >= SNR_REC)].event
Sx = R[R.event.isin(ok)].assign(abs_err=lambda d: d.baz_err.abs(), wrong=lambda d: d.baz_err.abs() > 20)
summ = Sx.groupby("config", sort=False).agg(n_events=("event", "nunique"), median_abs_err=("abs_err", "median"), frac_wrong_gt20=("wrong", "mean"))
pd.set_option("display.width", 200)
print(f"\n{len(ok)} events recorded (both-cable beam SNR >= {SNR_REC}) of {R.event.nunique()}")
print(summ.round(2).to_string())

fig, axs = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw=dict(width_ratios=[2, 1]))
names = list(CONF)
for i, n in enumerate(names):
    q = Sx[Sx.config == n]
    axs[0].scatter(np.full(len(q), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(q)), q.abs_err, s=16, c="tab:red" if "single" in n else "tab:blue")
    axs[0].plot([i - 0.3, i + 0.3], [q.abs_err.median()] * 2, "k-", lw=2)
axs[0].set_xticks(range(len(names)), [n.replace(" (single span)", "\n(single)") for n in names], rotation=45, ha="right", fontsize=8)
axs[0].set(ylabel="|back-azimuth error| (deg)", yscale="symlog",
           title=f"Rayleigh waves 0.04-0.07 Hz, {len(ok)} recorded earthquakes: error per configuration (black: median)")
axs[1].barh(range(len(summ)), summ.frac_wrong_gt20.values * 100, color=["tab:red" if "single" in n else "tab:blue" for n in summ.index])
axs[1].set_yticks(range(len(summ)), summ.index, fontsize=8)
axs[1].set(xlabel="% of events with |error| > 20 deg", title="wrong direction")
fig.tight_layout()
fig.savefig(OUT / f"figs/surface_wave_beam{TAG}.png", dpi=100)

ev = [k for k in ok][:4]
if ev:
    fig, axs = plt.subplots(2, len(ev), figsize=(4.5 * len(ev), 9), subplot_kw=dict(projection="polar"), squeeze=False)
    for j, eid in enumerate(ev):
        for i, name in enumerate(("south A2 (single span)", "both cables")):
            if (eid, name) not in examples:
                continue
            Pe, true = examples[(eid, name)]
            a = axs[i, j]
            a.pcolormesh(BAZ, SLO, (Pe / Pe.max()).T, cmap="viridis", shading="auto", vmin=0, vmax=1)
            a.plot([np.radians(true)] * 2, [0, SLO[-1]], "r--", lw=1.5)
            a.set_theta_zero_location("N")
            a.set_theta_direction(-1)
            a.set_yticklabels([])
            r0 = R[(R.event == eid) & (R.config == name)].iloc[0]
            a.set_title(f"{r0.time[:10]} M{r0.mag} {r0.dist_deg:.0f} deg\n{name}: error {r0.baz_err:+.0f} deg", fontsize=8)
    fig.suptitle("Rayleigh-wave beams (0.04-0.07 Hz): single span vs both cables; red dashed = true back-azimuth")
    fig.tight_layout()
    fig.savefig(OUT / f"figs/surface_wave_beam_examples{TAG}.png", dpi=100)
