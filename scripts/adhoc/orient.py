"""Horizontal orientation check of HYSB1 (and HYS14) with the colocated / nearest south-cable bins."""
import sys, glob, os; sys.path.insert(0, "/home/seismic/Multispan_DV")
os.environ.setdefault("MSDVV_CONFIG", "config_100m.yaml")
import numpy as np, pandas as pd, json
from scipy.signal import butter, sosfiltfilt, hilbert
from obspy import read, read_inventory
from pyproj import Geod
from joblib import Parallel, delayed
from msdvv import load_config
from msdvv.io import load_cfg
from msdvv.interspan import make_bins
cfg = load_config(); G = Geod(ellps="WGS84")
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
q = pd.read_csv(f"{OUT}/channel_quality_south.csv")
bins, bid = make_bins(q, 1.0, 3)
inv_o = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv_o for s in n}
lat = json.load(open(f"{OUT}/obsdas/latency_calibration.json"))["b1"]
qc = pd.read_csv(f"{OUT}/obsdas_combined/file_timing_qc_T.csv")
inv = pd.read_csv(f"{OUT}/inventory.csv"); inv = inv[inv.cable == "south"]
inv["file"] = [p.split("/")[-1] for p in inv.path]
inv = inv.merge(qc[["file", "status"]], on="file")
files = inv[(inv.status == "good") & (inv.t0 > pd.Timestamp("2025-12-26", tz="UTC").timestamp()) & (inv.t0 < pd.Timestamp("2026-01-06", tz="UTC").timestamp())].path.values[:20]
# cable azimuth per bin from neighbouring bins (same segment)
az = np.full(len(bins), np.nan)
for i in range(len(bins)):
    j = [x for x in (i - 1, i + 1) if 0 <= x < len(bins) and bins.seg[x] == bins.seg[i]]
    if len(j) == 2:
        az[i] = G.inv(bins.lon[j[0]], bins.lat[j[0]], bins.lon[j[1]], bins.lat[j[1]])[0] % 180
STA = {"HYSB1": 6.0, "HYS14": 26.0}
sel = {s: [i for i in range(len(bins)) if G.inv(crd[s][1], crd[s][0], bins.lon[i], bins.lat[i])[2] / 1e3 < r and np.isfinite(az[i])] for s, r in STA.items()}
WIN, NFFT, LAGMAX = 600.0, 2048, 120.0

def obs_ne(times, sta, fs):
    days = pd.date_range(pd.Timestamp(times[0] - 600, unit="s").floor("D"), pd.Timestamp(times[-1] + 600, unit="s").floor("D"))
    out = {}
    sos = butter(6, 0.45 * fs, "lowpass", fs=20.0, output="sos")
    for c in "NE":
        y = np.full(len(times), np.nan)
        for d in days:
            for fn in glob.glob(f"/data/wsd02/multispan_dvv/obs/continuous/{d:%Y%m%d}/OO.{sta}.*.mseed"):
                for tr in read(fn).select(component=c):
                    if tr.stats.npts < 400: continue
                    tt = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) / 20.0
                    m = (times >= tt[0]) & (times <= tt[-1]); y[m] = np.interp(times[m], tt, sosfiltfilt(sos, tr.data.astype(float)))
        out[c] = y
    return out

def one(path):
    f = load_cfg(path, cfg)
    times = f.times if f.times is not None else f.t0 + np.arange(f.nt) / f.fs
    sos = butter(4, [0.12, 0.3], "bandpass", fs=f.fs, output="sos")
    res = {}
    for s, idx in sel.items():
        X = np.stack([sosfiltfilt(sos, f.data[bid == bins.index[i]], axis=1).mean(0) for i in idx])
        ne = obs_ne(times, s, f.fs)
        N, E = (np.nan_to_num(sosfiltfilt(sos, np.nan_to_num(ne[c]))) for c in "NE")
        ok = np.isfinite(ne["N"]) & np.isfinite(ne["E"])
        n = int(WIN * f.fs); L = int(LAGMAX * f.fs)
        accN = np.zeros((len(idx), 2 * L + 1)); accE = np.zeros_like(accN)
        for s0 in range(0, f.nt - n, n // 2):
            if not ok[s0:s0 + n].all(): continue
            sc = np.sqrt(np.mean(N[s0:s0 + n] ** 2 + E[s0:s0 + n] ** 2)) + 1e-30
            FN, FE = np.fft.rfft(N[s0:s0 + n] / sc, NFFT), np.fft.rfft(E[s0:s0 + n] / sc, NFFT)
            FX = np.fft.rfft(X[:, s0:s0 + n], NFFT, axis=1); FX /= np.abs(FX) + 1e-20
            for acc, FO in ((accN, FN), (accE, FE)):
                c = np.fft.irfft(FO[None] * np.conj(FX), NFFT, axis=1)
                acc += np.concatenate([c[:, -L:], c[:, :L + 1]], axis=1)
        res[s] = (accN, accE)
    return res, f.fs

R = Parallel(n_jobs=10)(delayed(one)(p) for p in files)
fs = R[0][1]; L = int(LAGMAX * fs); lags = np.arange(-L, L + 1) / fs
TH = np.radians(np.arange(0, 180, 5))
for s, idx in sel.items():
    CN = sum(r[0][s][0] for r in R); CE = sum(r[0][s][1] for r in R)
    tau = lat["latency_s"]["D256_g256_native"] * lat["ramp_sign"]
    # colocated / nearest arrival window: around the latency-shifted zero lag (+ travel time for HYS14)
    best = []
    for j, i in enumerate(idx):
        dkm = G.inv(crd[s][1], crd[s][0], bins.lon[i], bins.lat[i])[2] / 1e3
        win = np.abs(lags - (-tau if False else tau)) < 20 + dkm / 0.5  # broad: colocated + slow waves
        amp = [np.abs(hilbert(np.cos(t) * CN[j] + np.sin(t) * CE[j]))[win].max() for t in TH]
        tb = np.degrees(TH[int(np.argmax(amp))])
        best.append(dict(sta=s, bin=int(bins.bin[i]), dist_km=round(dkm, 1), cable_az=round(az[i], 0), best_az=tb,
                         diff=((tb - az[i] + 90) % 180) - 90, contrast=round(max(amp) / min(amp), 2)))
    B = pd.DataFrame(best)
    print(B.to_string(index=False))
    good = B[B.contrast > 1.5]
    print(f"{s}: median (best - cable azimuth) = {good['diff'].median():+.0f} deg (n={len(good)} bins with contrast > 1.5); circular spread {good['diff'].std():.0f} deg")
