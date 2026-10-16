"""Item 2/8b: absolute latency of the 8-Hz stream (D512, 7.45 Hz) on Jan 13 2026, per cable, WITHOUT any earthquake or
catalogue: the 100-m stream (D256, 1.86 Hz; latency +25.2 s calibrated against the colocated seismometer HYSB1,
docs/DAS_clock_note.md) recorded the same fibers at the same time (north 01:43-08:03, south 01:49-05:33 UTC).
Both files regridded on true per-chunk times (dropped chunks = flagged gaps). For ~40 fiber positions per cable in good
blocks, the 8-Hz record is low-passed and decimated x4 to the 100-m sample rate, both are interpolated onto one grid of
time STAMPS, band-passed 0.1-0.5 Hz (microseism), and cross-correlated in 20-min windows (gaps excluded); lag tau with
x_8Hz(t) ~ x_100m(t - tau) means latency_D512 = latency_D256 + tau. Parabolic sub-sample peak; median over windows and
positions, scatter = MAD.
Output: outputs_100m/stream_latency_jan13.csv (per position) and printed summary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, sosfiltfilt

from msdvv.io import load_file

LAT_D256 = 25.2
PAIRS = {
    "north": ("/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_north_8Hz_20260112_191838_00000000000000000024",
              "/data/wsd02/multispan_dvv/raw/2026/01/12/north-100m/nokia_DAS_north-100m_8Hz_20260112_125800_00000000000000000012"),
    "south": ("/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_south_8Hz_20260112_215335_00000000000000000040",
              "/data/wsd02/multispan_dvv/raw/2026/01/12/south-100m/nokia_DAS_south-100m_8Hz_20260112_180718_00000000000000000020"),
}
WIN_S, MAXLAG = 1200.0, 40.0
rows = []
for cab, (p8, p1) in PAIRS.items():
    a = load_file(p8, regrid=True)
    b = load_file(p1, regrid=True)
    ca, cb = a.channels(), b.channels()
    fs = b.fs
    t0, t1 = max(a.times[0], b.times[0]), min(a.times[-1], b.times[-1])
    tg = np.arange(t0 + 60, t1 - 60, 1 / fs)
    sos = butter(4, (0.1, 0.5), "bandpass", fs=fs, output="sos")
    good = ca.block.notna().values & (ca.pos_in_block.values >= 5) & (ca.pos_in_block.values <= ca.block_len.values - 6)
    idx_a = np.where(good)[0][:: max(1, good.sum() // 40)]
    va = np.asarray(a.valid if a.valid is not None else np.ones(a.data.shape[1], bool))
    vb = np.asarray(b.valid if b.valid is not None else np.ones(b.data.shape[1], bool))
    ma = np.interp(tg, a.times, va.astype(float)) > 0.999
    mb = np.interp(tg, b.times, vb.astype(float)) > 0.999
    ok_t = ma & mb
    print(cab, f"overlap {(t1 - t0) / 3600:.2f} h, {ok_t.mean():.0%} of samples valid in both, {len(idx_a)} positions", flush=True)
    nw, nlag = int(WIN_S * fs), int(MAXLAG * fs)
    for i in idx_a:
        j = int(np.argmin(np.abs(cb.opt_km.values - ca.opt_km.values[i])))
        if abs(cb.opt_km.values[j] - ca.opt_km.values[i]) > 0.06:
            continue
        xa_full = decimate(a.data[i].astype(np.float64), 4, ftype="fir", zero_phase=True)
        ta = a.times[::4][: len(xa_full)]
        xa = sosfiltfilt(sos, np.interp(tg, ta, xa_full))
        xb = sosfiltfilt(sos, np.interp(tg, b.times, b.data[j].astype(np.float64)))
        lags = []
        for s0 in range(0, len(tg) - nw, nw):
            sl = slice(s0, s0 + nw)
            if ok_t[sl].mean() < 0.98:
                continue
            u, v = xa[sl] - xa[sl].mean(), xb[sl] - xb[sl].mean()
            c = np.correlate(v, u[nlag:-nlag], "valid")  # c[k] = sum_t v[t + k] u[t + nlag] -> peak at k = nlag - tau * fs
            k = int(np.argmax(c))
            y0, y1, y2 = c[max(k - 1, 0)], c[k], c[min(k + 1, len(c) - 1)]
            dk = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if 0 < k < len(c) - 1 else 0.0
            tau = (nlag - (k + dk)) / fs  # u(t) ~ v(t - tau)
            cc = y1 / np.sqrt(np.sum(u[nlag:-nlag] ** 2) * np.sum(v[k:k + len(u) - 2 * nlag] ** 2))
            lags.append((tau, cc))
        if not lags:
            continue
        L = np.array(lags)
        keep = L[:, 1] > 0.5
        if keep.sum() == 0:
            continue
        rows.append(dict(cable=cab, opt_km=float(ca.opt_km.values[i]), n_win=int(keep.sum()), tau_s=float(np.median(L[keep, 0])),
                         tau_mad=float(np.median(np.abs(L[keep, 0] - np.median(L[keep, 0])))), cc=float(np.median(L[keep, 1]))))
R = pd.DataFrame(rows)
R.to_csv("/data/wsd02/multispan_dvv/outputs_100m/stream_latency_jan13.csv", index=False)
pd.set_option("display.width", 200)
for cab, g in R.groupby("cable"):
    tau = np.median(g.tau_s)
    print(f"{cab}: {len(g)} positions, median CC {g.cc.median():.2f}, tau = {tau:+.3f} s (MAD {np.median(np.abs(g.tau_s - tau)):.3f}) "
          f"-> latency D512 = {LAT_D256 + tau:.2f} s (given D256 = {LAT_D256} s)")
print(R.round(3).to_string(index=False))
