"""Item 8: multi-span beamforming with the south-cable loop (one interrogator = one clock -> all spans exactly
synchronised; dropped chunks hit every channel at once).

Array: good 1-km bins of the south cable (same bins as the cross-span work; mean of the good channels per bin),
positions in a local x/y frame (km). Data: 100-m stream, RAM-normalised per bin, dropped chunks zero-filled on the true
time grid (regrid), windows with > 10 % gap or overlapping catalogue events rejected (except in EVENT mode).
Per window (WIN_S) and frequency f: bin spectra normalised to unit amplitude (whitened) -> cross-spectral matrix
(CSDM, diagonal removed); averaged over the windows of each hour. Bartlett beam power
    B(baz, s) = Re[ w^H C w ] / (N (N-1)),  w_n = exp(-2 pi i f s (x_n sin baz + y_n cos baz)) / ... (plane wave from baz)
on baz 0-358 deg (2 deg) and slowness 0-3.0 s/km (0.02). WEIGHT=rayleigh (default): bins weighted by the Rayleigh-wave
axial-strain sensitivity cos^2(baz - cable azimuth), beam normalised by the same weighting applied to a perfect plane
wave (1 = fully coherent); WEIGHT=none: unit weights. EVENT_LEN (s): length of the event window.
Modes: EVENT=<UTC> (single window around a known event, validation) | hourly (default) over the DATES range.
Outputs: outputs_100m/beamform/{beam_<tag>.npz, peaks_<tag>.csv}, figs/beamform_<tag>.png
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyproj import Geod
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from msdvv import load_config
from msdvv.interspan import make_bins
from msdvv.io import load_cfg

cfg = load_config()  # MSDVV_CONFIG=config_100m_v2.yaml (regrid_drops)
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m/beamform")
OUT.mkdir(parents=True, exist_ok=True)
G = Geod(ellps="WGS84")
FREQS = [float(x) for x in os.environ.get("FREQS", "0.12,0.16,0.20,0.25").split(",")]
WIN_S = float(os.environ.get("WIN_S", "300"))
EVENT = os.environ.get("EVENT", "")
TAG = os.environ.get("TAG", "test")
DATES = os.environ.get("DATES", "2025-12-23,2026-01-29").split(",")
BAZ = np.radians(np.arange(0, 360, 2.0))
SLO = np.arange(0.0, 3.001, 0.02)
q = pd.read_csv(Path(cfg["output_dir"]) / "channel_quality_south.csv")
bins, bid = make_bins(q, 1.0, 3)
bins = bins[~bins.seg.isin(["C3"])].reset_index(drop=True)
lat0, lon0 = bins.lat.mean(), bins.lon.mean()
az, dist = G.inv(np.full(len(bins), lon0), np.full(len(bins), lat0), bins.lon.values, bins.lat.values)[0], G.inv(np.full(len(bins), lon0), np.full(len(bins), lat0), bins.lon.values, bins.lat.values)[2]
X = dist / 1e3 * np.sin(np.radians(az))
Y = dist / 1e3 * np.cos(np.radians(az))
ev = pd.read_csv(Path(cfg["output_dir"]) / "events.csv", parse_dates=["win0", "win1"])
EVW = [(a.timestamp(), b.timestamp()) for a, b in zip(ev.win0, ev.win1)]
# cable azimuth per bin (from neighbouring bins of the same segment) -> Rayleigh-wave axial-strain sensitivity cos^2
cab_az = np.full(len(bins), np.nan)
for k in range(len(bins)):
    nb = [j for j in (k - 1, k + 1) if 0 <= j < len(bins) and bins.seg[j] == bins.seg[k]]
    if len(nb) == 2:
        cab_az[k] = G.inv(bins.lon[nb[0]], bins.lat[nb[0]], bins.lon[nb[1]], bins.lat[nb[1]])[0]
    elif len(nb) == 1:
        cab_az[k] = G.inv(bins.lon[k], bins.lat[k], bins.lon[nb[0]], bins.lat[nb[0]])[0]
cab_az = np.radians(np.nan_to_num(cab_az))
WEIGHT = os.environ.get("WEIGHT", "rayleigh")  # rayleigh: w_n ~ cos^2(baz - cable azimuth); none: unit weights
AMP = np.cos(BAZ[:, None] - cab_az[None, :]) ** 2 if WEIGHT == "rayleigh" else np.ones((len(BAZ), len(bins)))
AMP = AMP / np.sqrt((AMP**2).sum(1, keepdims=True))  # unit-norm weights per baz
# steering phases for every (baz, s, bin): plane wave arriving FROM baz travels toward baz+180
KX = -np.sin(BAZ)[:, None, None] * SLO[None, :, None]
KY = -np.cos(BAZ)[:, None, None] * SLO[None, :, None]
DT = KX * X[None, None, :] + KY * Y[None, None, :]  # (nbaz, nslo, nbin) delay (s) relative to array centre


def bin_series(path):
    f = load_cfg(path, cfg)
    tt = f.times if f.times is not None else f.t0 + np.arange(f.nt) / f.fs
    sos = butter(4, [0.08, 0.45], "bandpass", fs=f.fs, output="sos")
    Z = np.zeros((len(bins), f.nt), np.float32)
    for k, b in enumerate(bins.bin):
        ch = np.where(bid == b)[0]
        x = sosfiltfilt(sos, f.data[ch].astype(np.float64), axis=1).mean(0)
        a = uniform_filter1d(np.abs(x), max(3, int(20 * f.fs)) | 1) + 1e-30
        Z[k] = np.clip(x / a, -10, 10)
    valid = f.valid if f.valid is not None else np.ones(f.nt, bool)
    return tt, Z, valid, f.fs


def beam_from_windows(Z, valid, tt, fs, starts, n):
    """Average CSDM over the given windows -> beam power (nf, nbaz, nslo)."""
    nf = len(FREQS)
    C = np.zeros((nf, len(bins), len(bins)), np.complex128)
    used = 0
    fr = np.fft.rfftfreq(n, 1 / fs)
    fi = [int(np.argmin(np.abs(fr - f0))) for f0 in FREQS]
    taper = np.hanning(n)
    for s0 in starts:
        if valid[s0:s0 + n].mean() < 0.9:
            continue
        a, b = tt[s0], tt[s0 + n - 1]
        if not EVENT and any(a < e1 and b > e0 for e0, e1 in EVW):
            continue
        S = np.fft.rfft(Z[:, s0:s0 + n] * taper, axis=1)[:, fi]  # (nbin, nf)
        S = S / (np.abs(S) + 1e-30)
        for j in range(nf):
            C[j] += np.outer(S[:, j], np.conj(S[:, j]))
        used += 1
    if used == 0:
        return None, 0
    C /= used
    P = np.zeros((nf, len(BAZ), len(SLO)))
    for j, f0 in enumerate(FREQS):
        Cj = C[j] - np.diag(np.diag(C[j]))
        W = AMP[:, None, :] * np.exp(-2j * np.pi * f0 * DT)  # (nbaz, nslo, nbin), unit-norm amplitude weights
        num = np.real(np.einsum("abn,nm,abm->ab", np.conj(W), Cj, W))
        den = np.abs(W).sum(-1) ** 2 - (np.abs(W) ** 2).sum(-1)  # same weighting for a perfect plane wave (|C_nm| = 1): beam <= 1
        P[j] = num / den
    return P, used


inv = pd.read_csv(Path(cfg["output_dir"]) / "inventory.csv")
inv = inv[inv.cable == "south"].sort_values("t0").reset_index(drop=True)


def run_file(path):
    tt, Z, valid, fs = bin_series(path)
    n = int(WIN_S * fs)
    out = []
    if EVENT:
        t_ev = pd.Timestamp(EVENT, tz="UTC").timestamp()
        L = float(os.environ.get("EVENT_LEN", "1800"))
        for lab, (a, b) in (("event", (t_ev, t_ev + L)), ("pre-event noise", (t_ev - 3600, t_ev - 600))):
            i0, i1 = np.searchsorted(tt, a), np.searchsorted(tt, b)
            P, u = beam_from_windows(Z, valid, tt, fs, range(i0, max(i0 + 1, i1 - n + 1), max(1, n // 4)), n)
            out.append((lab, a, P, u))
        return out
    hours = np.floor(tt / 3600)
    for h in np.unique(hours):
        idx = np.where(hours == h)[0]
        if len(idx) < n:
            continue
        P, u = beam_from_windows(Z, valid, tt, fs, range(idx[0], idx[-1] - n + 1, n // 2), n)
        if P is not None:
            out.append(("hour", h * 3600, P, u))
    return out


if EVENT:
    t_ev = pd.Timestamp(EVENT, tz="UTC").timestamp()
    path = inv[(inv.t0 <= t_ev - 3600) & (inv.t1 >= t_ev + 1800)].path.values
    if not len(path):
        path = inv.iloc[(np.abs(inv.t0 - t_ev)).argsort()[:1]].path.values
    res = run_file(path[0])
else:
    a, b = (pd.Timestamp(x, tz="UTC").timestamp() for x in DATES)
    paths = inv[(inv.t1 > a) & (inv.t0 < b)].path.values
    res = [r for rr in Parallel(n_jobs=int(os.environ.get("NJOBS", "8")))(delayed(run_file)(p) for p in paths) for r in rr]

rows, Ps = [], []
for lab, t, P, u in res:
    if P is None:
        continue
    for j, f0 in enumerate(FREQS):
        k = np.unravel_index(np.argmax(P[j]), P[j].shape)
        rows.append(dict(label=lab, t=t, f=f0, baz=np.degrees(BAZ[k[0]]), slowness=SLO[k[1]], vel=1 / SLO[k[1]] if SLO[k[1]] > 0 else np.inf,
                         beam_max=P[j][k], beam_med=float(np.median(P[j])), n_win=u))
    Ps.append(P)
PK = pd.DataFrame(rows).sort_values(["t", "f"])
PK.to_csv(OUT / f"peaks_{TAG}.csv", index=False)
np.savez(OUT / f"beam_{TAG}.npz", P=np.array(Ps, np.float32), t=np.array([r[1] for r in res if r[2] is not None]),
         labels=np.array([r[0] for r in res if r[2] is not None]), baz=np.degrees(BAZ), slo=SLO, freqs=FREQS, x=X, y=Y)
pd.set_option("display.width", 200)
print(PK.round(3).to_string(index=False) if EVENT else PK.groupby("f")[["baz", "slowness", "beam_max"]].describe().round(2).to_string())

# figure: polar beam maps for event mode, or baz-time images for hourly mode
if EVENT:
    fig, axs = plt.subplots(len(res), len(FREQS), figsize=(4 * len(FREQS), 4 * len(res)), subplot_kw=dict(projection="polar"), squeeze=False)
    for i, (lab, t, P, u) in enumerate(res):
        for j, f0 in enumerate(FREQS):
            ax = axs[i, j]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.pcolormesh(BAZ, SLO, P[j].T, cmap="viridis", shading="auto")
            ax.set_ylim(0, 3)
            ax.set_title(f"{lab}, {f0:.2f} Hz ({u} windows)", fontsize=9)
    fig.suptitle(f"South-loop DAS beamforming around {EVENT} (radius = slowness 0-3 s/km; Bartlett, whitened bins)", fontsize=11)
else:
    fig, axs = plt.subplots(len(FREQS) + 1, 1, figsize=(14, 3 * (len(FREQS) + 1)), sharex=True)
    Pa = np.array(Ps)
    tt_ = pd.to_datetime([r[1] for r in res if r[2] is not None], unit="s", utc=True)
    o = np.argsort(tt_)
    for j, f0 in enumerate(FREQS):
        bp = Pa[o, j].max(axis=2)  # max over slowness -> (nt, nbaz)
        axs[j].pcolormesh(tt_[o], np.degrees(BAZ), bp.T, cmap="viridis", shading="auto")
        axs[j].set_ylabel(f"{f0:.2f} Hz\nback-azimuth (deg)", fontsize=8)
    for f0 in FREQS:
        g = PK[PK.f == f0]
        axs[-1].plot(pd.to_datetime(g.t, unit="s", utc=True), g.vel.clip(upper=5), ".", ms=2, label=f"{f0:.2f} Hz")
    axs[-1].set_ylabel("apparent velocity of\nbeam peak (km/s)", fontsize=8)
    axs[-1].legend(fontsize=7, ncol=4)
    fig.suptitle("South-loop DAS beamforming of ambient noise, hourly (max over slowness)", fontsize=11)
fig.tight_layout()
fig.savefig(f"/data/wsd02/multispan_dvv/outputs_100m/figs/beamform_{TAG}.png", dpi=100)
print("done")
