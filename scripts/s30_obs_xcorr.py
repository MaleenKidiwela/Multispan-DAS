"""OOI seafloor seismometer (OO.HYS11-14, OO.HYSB1) ambient-noise cross-correlation, vertical components.

Daily 20-Hz MiniSEED from d04_download_obs.py -> per window (600 s, 50 % overlap): detrend, taper, bandpass,
one-bit -> rFFT -> per band whitening -> cross-spectra for every station pair -> hourly substacks.
Windows with gaps, event windows (outputs_100m/events.csv) or outlier RMS are skipped.
Output: /data/wsd02/multispan_dvv/obs/ccf/obs_zz.h5 (t_sub, nwin, per band: spec (nsub, npair, nb), bins).
HYS14 clock: chronfix shows dt = 0 for Dec 2025 - Jan 2026 (healthy after the 2025-05-08 resync).
"""
import glob
import itertools
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.fft as sfft
from joblib import Parallel, delayed
from obspy import UTCDateTime, read
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

ROOT = Path("/data/wsd02/multispan_dvv/obs")
STA = ["HYS14", "HYSB1", "HYS11", "HYS12", "HYS13"]
PAIRS = list(itertools.combinations(range(len(STA)), 2))
FS, WIN, STEP, NFFT, SUB = 20.0, 600.0, 300.0, 32768, 3600
BANDS = {"b1": (0.12, 0.30), "b2": (0.30, 0.70), "b3": (1.0, 2.0)}
TAPER_HZ = {"b1": 0.03, "b2": 0.03, "b3": 0.1}
PREFILT = (0.08, 3.0)
EV = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/events.csv", parse_dates=["win0", "win1"])
EVW = [(a.timestamp(), b.timestamp()) for a, b in zip(EV.win0, EV.win1)]
f = sfft.rfftfreq(NFFT, 1 / FS)
BIN = {}
for b, (lo, hi) in BANDS.items():
    tw = TAPER_HZ[b]
    idx = np.where((f >= lo - tw) & (f <= hi + tw))[0]
    w = np.ones(len(idx))
    fb = f[idx]
    w[fb < lo] = 0.5 * (1 - np.cos(np.pi * (fb[fb < lo] - (lo - tw)) / tw))
    w[fb > hi] = 0.5 * (1 + np.cos(np.pi * (fb[fb > hi] - hi) / tw))
    BIN[b] = (idx, w.astype(np.float32))


def load_day(day):
    """Vertical traces of all stations for one UTC day on a common 20-Hz grid; NaN where missing."""
    t0 = UTCDateTime(day)
    n = int(86400 * FS)
    X = np.full((len(STA), n), np.nan, np.float32)
    for i, s in enumerate(STA):
        for fn in glob.glob(str(ROOT / "continuous" / f"{day:%Y%m%d}" / f"OO.{s}.*.mseed")):
            for tr in read(fn).select(component="Z"):
                if abs(tr.stats.sampling_rate - FS) > 1e-3:
                    continue
                i0 = int(round((tr.stats.starttime - t0) * FS))
                a, b = max(i0, 0), min(i0 + tr.stats.npts, n)
                if b > a:
                    X[i, a:b] = tr.data[a - i0 : b - i0]
    return t0.timestamp, X


def one_day(day):
    t0, X = load_day(day)
    sos = butter(4, PREFILT, "bandpass", fs=FS, output="sos")
    n, s = int(WIN * FS), int(STEP * FS)
    taper = tukey(n, 0.05).astype(np.float32)
    starts = np.arange(0, X.shape[1] - n + 1, s)
    centres = t0 + (starts + n / 2) / FS
    sid = np.floor(centres / SUB).astype(np.int64)
    usub = np.unique(sid)
    acc = {b: np.zeros((len(usub), len(PAIRS), len(BIN[b][0])), np.complex64) for b in BANDS}
    nwin = np.zeros((len(usub), len(PAIRS)), np.int32)
    rms_all = []
    segs = []
    for w0 in starts:
        seg = X[:, w0 : w0 + n].astype(np.float64)
        ok = ~np.isnan(seg).any(1)
        segs.append((w0, ok))
        r = np.full(len(STA), np.nan)
        if ok.any():
            r[ok] = np.std(seg[ok], axis=1)
        rms_all.append(r)
    rms_all = np.array(rms_all)
    med = np.nanmedian(rms_all, 0)
    mad = np.nanmedian(np.abs(rms_all - med), 0) * 1.4826
    for (w0, ok), r, c, si in zip(segs, rms_all, centres, sid):
        wt0, wt1 = t0 + w0 / FS, t0 + (w0 + n) / FS
        if any(wt0 < e1 and wt1 > e0 for e0, e1 in EVW):
            continue
        ok = ok & (r < med + 5 * mad)
        if ok.sum() < 2:
            continue
        seg = X[:, w0 : w0 + n].astype(np.float64)
        seg[~ok] = 0.0
        seg = sosfiltfilt(sos, seg, axis=1)
        seg = np.sign(seg - seg.mean(1, keepdims=True)) * taper
        F = sfft.rfft(seg, n=NFFT, axis=1)
        j = np.searchsorted(usub, si)
        for b, (idx, wt) in BIN.items():
            Fb = F[:, idx]
            Fb = Fb / (np.abs(Fb) + 1e-20) * wt
            for k, (a, bb) in enumerate(PAIRS):
                if ok[a] and ok[bb]:
                    acc[b][j, k] += (Fb[a] * np.conj(Fb[bb])).astype(np.complex64)
        for k, (a, bb) in enumerate(PAIRS):
            nwin[j, k] += int(ok[a] and ok[bb])
    return usub * SUB, nwin, acc


days = pd.date_range("2025-12-23", "2026-01-28", freq="D")
res = Parallel(n_jobs=16, verbose=5)(delayed(one_day)(d) for d in days)
t = np.concatenate([r[0] for r in res])
ut, inv = np.unique(t, return_inverse=True)
(ROOT / "ccf").mkdir(exist_ok=True)
with h5py.File(ROOT / "ccf" / "obs_zz.h5", "w") as h:
    h.attrs.update(dict(fs=FS, nfft=NFFT, maxlag_s=300.0, substack_s=SUB, win_s=WIN, stations=",".join(STA),
                        pairs=";".join(f"{STA[a]}-{STA[b]}" for a, b in PAIRS), clock_note="HYS14 chronfix dt=0 in this period"))
    h["t_sub"] = ut
    nw = np.zeros((len(ut), len(PAIRS)), np.int32)
    np.add.at(nw, inv, np.concatenate([r[1] for r in res]))
    h["nwin"] = nw
    for b in BANDS:
        S = np.zeros((len(ut), len(PAIRS), len(BIN[b][0])), np.complex64)
        np.add.at(S, inv, np.concatenate([r[2][b] for r in res]))
        g = h.create_group(b)
        g["spec"], g["bins"] = S, BIN[b][0]
print("hours", len(ut), "pairs", [f"{STA[a]}-{STA[b]}" for a, b in PAIRS])
print("median windows per hour per pair:", np.median(nw, 0))
