"""Track the DAS clock error through a file with the colocated HYSB1 x SB3 correlation (template matching)."""
import sys, glob; sys.path.insert(0, "/home/seismic/Multispan_DV")
import os; os.environ.setdefault("MSDVV_CONFIG", "config_100m.yaml")
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from obspy import read
from msdvv import load_config
from msdvv.io import load_cfg
from msdvv.interspan import make_bins
cfg = load_config()
inv = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/inventory.csv"); inv = inv[inv.cable == "south"].sort_values("t0")
q = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/channel_quality_south.csv")
bins, bid = make_bins(q, 1.0, 3)
B3 = bins.index[bins.seg == "B3"][:8]  # bins within ~5 km of HYSB1
WIN, STEP, ML = 600.0, 300.0, 450.0

def obs(times, fs, comp):
    days = pd.date_range(pd.Timestamp(times[0] - 600, unit="s").floor("D"), pd.Timestamp(times[-1] + 600, unit="s").floor("D"))
    y = np.full(len(times), np.nan); sos = butter(6, 0.45 * fs, "lowpass", fs=20.0, output="sos")
    for d in days:
        for fn in glob.glob(f"/data/wsd02/multispan_dvv/obs/continuous/{d:%Y%m%d}/OO.HYSB1.*.mseed"):
            for tr in read(fn).select(component=comp):
                if tr.stats.npts < 400: continue
                tt = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) / 20.0
                m = (times >= tt[0]) & (times <= tt[-1]); y[m] = np.interp(times[m], tt, sosfiltfilt(sos, tr.data.astype(float)))
    return y

def window_ccfs(path, time_fn=None):
    """Per 10-min window: sum over B3 bins and N/E of OBS x DAS CCF (one-bit, whitened), lags +-ML. OBS sampled at the
    (optionally corrected) DAS sample times."""
    f = load_cfg(path, cfg)
    times = f.t0 + np.arange(f.nt) / f.fs if time_fn is None else time_fn(f)
    sos = butter(4, [0.1, 0.35], "bandpass", fs=f.fs, output="sos")
    X = np.stack([np.sign(sosfiltfilt(sos, f.data[bid == b], axis=1)).mean(0) for b in B3])
    O = []
    for c in "NE":
        y = obs(times, f.fs, c); ok = np.isfinite(y); O.append((np.sign(sosfiltfilt(sos, np.nan_to_num(y))), ok))
    n, st = int(WIN * f.fs), int(STEP * f.fs); N = 4096; L = int(ML * f.fs)
    out, tc = [], []
    for s0 in range(0, f.nt - n, st):
        acc = np.zeros(2 * L + 1)
        for y, ok in O:
            if not ok[s0:s0 + n].all(): continue
            Y = np.fft.rfft(y[s0:s0 + n], N); Y /= np.abs(Y) + 1e-9
            for x in X:
                Xf = np.fft.rfft(x[s0:s0 + n], N); Xf /= np.abs(Xf) + 1e-9
                c = np.fft.irfft(Y * np.conj(Xf), N)
                acc += np.abs(np.concatenate([c[-L:], c[:L + 1]]))  # envelope-like (polarity varies with bin/comp)
        out.append(acc); tc.append((s0 + n / 2) / f.fs / 3600)
    return np.array(out), np.array(tc), np.arange(-L, L + 1) / f.fs, f

def pick_file(when, label, time_fn=None):
    r = inv.iloc[(np.abs(inv.t0 - pd.Timestamp(when, tz="UTC").timestamp())).argmin()]
    W, tc, lags, f = window_ccfs(r.path, time_fn)
    ok = W.sum(1) > 0
    sm = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, "same"), 1, W)
    pk = lags[np.argmax(sm, 1)]; snr = sm.max(1) / np.median(sm, 1)
    print(f"{label} {pd.Timestamp(f.t0, unit='s'):%m-%d %H:%M} ({f.nt / f.fs / 3600:.2f} h):")
    print("   " + " ".join(f"{a:.2f}h:{b:+.0f}s({c:.1f})" for a, b, c, o in zip(tc, pk, snr, ok) if o))
    return tc[ok], pk[ok], snr[ok], r, f

if __name__ == "__main__":
    pick_file("2026-01-03 07:00", "GOOD")
    tc, pk, snr, r, f = pick_file("2026-01-08 00:00", "BAD ")
    g = snr > 2.5
    A = np.vstack([tc[g], np.ones(g.sum())]).T
    sl, ic = np.linalg.lstsq(A, pk[g], rcond=None)[0]
    res = pk[g] - (sl * tc[g] + ic)
    print(f"linear fit (SNR>2.5, {g.sum()} windows): lag = {ic:+.1f} s {sl:+.2f} s/h  -> rate error {sl / 3600 * 100:+.3f} %; residual std {res.std():.1f} s, max {np.abs(res).max():.1f} s")
    np.savez("/tmp/claude-39217/-home-seismic-Multispan-DV/bdcf973f-3351-4060-bd01-4915969ab0b9/scratchpad/clock_jan8.npz", tc=tc, pk=pk, snr=snr)
