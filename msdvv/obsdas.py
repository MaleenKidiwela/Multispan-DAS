"""Seismometer-DAS correlation helpers (s32 output): latency-corrected hourly CCFs, along-fibre neighbour stacking,
24-h back-stacked moving stacks and per-pair stretching (shared by s36/s37)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .dvv import one_sided, stretch
from .xcorr import spectra_to_ccf

OUT = Path(os.environ.get("MSDVV_OBSDAS_OUT", "/data/wsd02/multispan_dvv/outputs_100m"))  # Nov: outputs_nov_obsdas


def obsdas_dir(comp: str = "Z", cable: str = "south", prep: str = "onebit") -> Path:
    return OUT / ("obsdas" + ("" if cable == "south" else f"_{cable}") + ("" if comp == "Z" else f"_{comp}") + ("" if prep == "onebit" else f"_{prep}"))


def load_hourly(comp: str, band: str = "b1", inject: float = 0.0, inject_at=None, prep: str = "onebit", maxlag: float | None = None):
    """Hourly CCFs (nh, npath, nlag) float32, latency-corrected per setup with the Z calibration, partial hours merged.
    inject: multiply lag time by (1 + inject) in every hourly CCF from `inject_at` on (synthetic dv/v = +inject)."""
    od = obsdas_dir(comp, prep=prep)
    lat = json.load(open(OUT / "obsdas" / "latency_calibration.json"))[band]
    T, SP = [], []
    for f in sorted(od.glob("*.h5")):
        with h5py.File(f, "r") as h:
            s = h[band]["spec"][:]
            if s.shape[0] == 0:
                continue
            bins, fs, nfft = h[band]["bins"][:], h.attrs["fs"], int(h.attrs["nfft"])
            ml = float(maxlag) if maxlag else float(h.attrs["maxlag_s"])
            fr = np.fft.rfftfreq(nfft, 1 / fs)[bins]
            tau = lat["latency_s"][h.attrs["setup"]] * lat["ramp_sign"]
            T.append(h["t_sub"][:])
            SP.append(s.reshape(s.shape[0], -1, s.shape[3]) * np.exp(-2j * np.pi * fr * tau).astype(np.complex64))
    t, inv = np.unique(np.concatenate(T), return_inverse=True)
    S = np.zeros((len(t),) + SP[0].shape[1:], np.complex64)
    np.add.at(S, inv, np.concatenate(SP))
    del SP
    H, lags = spectra_to_ccf(S, bins, nfft, fs, ml)
    H = H.astype(np.float32)
    if inject:
        late = t >= pd.Timestamp(inject_at, tz="UTC").timestamp()
        for p in range(H.shape[1]):
            H[late, p] = np.array([np.interp(lags * (1 + inject), lags, h) for h in H[late, p]])
    return t, H, lags, pd.read_csv(od / "paths.csv")


def shift_lag(x, sh, dt):
    """Shift CCF(s) along the last axis by sh seconds (positive: later), by Fourier phase."""
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], dt)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * sh), n=x.shape[-1], axis=-1).astype(np.float32)


def best_shift(a, b, dt, maxs=5.0):
    m = int(maxs / dt)
    return (np.argmax([np.dot(np.roll(a, k), b) for k in range(-m, m + 1)]) - m) * dt


def neighbour_stack(H, P, dt, nbr=2):
    """Each path + paths of the same OBS to bins within +-nbr on the same segment, statically aligned (<= 5 s) to the
    centre path's all-period reference."""
    ref = H.sum(0)
    Hn = np.zeros_like(H)
    for p in range(H.shape[1]):
        r = P.iloc[p]
        for q in P.index[(P.obs == r.obs) & (P.seg == r.seg) & (np.abs(P.bin - r.bin) <= nbr)].values:
            sh = 0.0 if q == p else best_shift(ref[q], ref[p], dt)
            Hn[:, p] += H[:, q] if sh == 0 else shift_lag(H[:, q], sh, dt)
    return Hn


def moving_windows(t, sh_h=24, min_h=20):
    """Back-stacked windows [e - sh_h h, e) on the hourly grid with >= min_h hours: (t_end, i0, i1)."""
    grid = np.arange(t[0] + sh_h * 3600, t[-1] + 3601, 3600.0)
    i0, i1 = np.searchsorted(t, grid - sh_h * 3600), np.searchsorted(t, grid)
    ok = (i1 - i0) >= min_h
    return grid[ok], i0[ok], i1[ok]


def moving(H, i0, i1, w=None):
    """Moving stacks (n_stack, npath, nlag); w: optional (nh,) hour weights (even/odd split)."""
    X = H if w is None else H * w[:, None, None]
    cs = np.concatenate([np.zeros((1,) + X.shape[1:], np.float32), np.cumsum(X, 0, dtype=np.float64).astype(np.float32)])
    return cs[i1] - cs[i0]


def early_window(dist_km, ml, vg=0.77):
    t1 = np.maximum(0.7 * dist_km / vg, 3.0)
    t2 = np.minimum(np.maximum(2.5 * dist_km / vg, t1 + 30.0), ml - 10.0)
    return t1, t2


def convergence(C, ref, m):
    """Median over stacks of CC(stack, reference without it) inside lag mask m. C (n_stack, nlag)."""
    a = C[:, m] - C[:, m].mean(1, keepdims=True)
    r = (ref[None] - C)[:, m]
    r = r - r.mean(1, keepdims=True)
    return float(np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30)))


def stretch_series(C, ref, lags, t1, t2, rng, cc_min=0.4):
    """Both lag sides stretched separately, combined by 1/err^2. Returns (dv, W) with W = sum of 1/err^2 (0 if rejected)."""
    dt = lags[1] - lags[0]
    dv_s, w_s = [], []
    for side in ("pos", "neg"):
        c = one_sided(C, lags, side)
        st = stretch(c, np.broadcast_to(one_sided(ref, lags, side), c.shape), dt, t1, t2, 0.01, 101, rng)
        ok = (st["cc"] >= cc_min) & ~st["at_bound"]
        w = np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0)
        dv_s.append(np.where(ok, st["dv"], 0) * w)
        w_s.append(w)
    W = w_s[0] + w_s[1]
    return np.where(W > 0, (dv_s[0] + dv_s[1]) / np.maximum(W, 1e-30), np.nan), W


def file_timing_qc(comp: str = "T", band: str = "b1", ref_period=tuple(os.environ.get("MSDVV_QC_REF", "2025-12-26,2026-01-06").split(",")), near_km: float = 4.0,
                   cc_good: float = 0.6, path: Path | None = None) -> pd.DataFrame:
    """Per DAS file: lag shift (+-150 s) and CC of the colocated HYSB1 x SB3 correlation (bins < near_km) against the
    reference of ref_period. status: good (CC at zero shift >= cc_good), jump (whole-file shift, CC >= cc_good at
    the shift; corrected by shift_s), bad (no consistent lag: in-file drift / missing samples)."""
    path = path or OUT / "obsdas_combined" / f"file_timing_qc_{comp}.csv"
    if path.exists():
        return pd.read_csv(path)
    od = obsdas_dir(comp)
    lat = json.load(open(OUT / "obsdas" / "latency_calibration.json"))[band]
    P = pd.read_csv(od / "paths.csv")
    near = P.index[(P.obs == "HYSB1") & (P.dist_km < near_km)].values
    rows, ccfs = [], []
    for f in sorted(od.glob("*.h5")):
        with h5py.File(f, "r") as h:
            s = h[band]["spec"][:]
            if s.shape[0] == 0:
                continue
            bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], int(h.attrs["nfft"]), float(h.attrs["maxlag_s"])
            fr = np.fft.rfftfreq(nfft, 1 / fs)[bins]
            tau = lat["latency_s"][h.attrs["setup"]] * lat["ramp_sign"]
            S = s.reshape(s.shape[0], -1, s.shape[3])[:, near].sum(0) * np.exp(-2j * np.pi * fr * tau)
            c, lags = spectra_to_ccf(S, bins, nfft, fs, ml)
            ccfs.append(c)
            rows.append(dict(file=f.name[:-3], t0=float(h["t_sub"][0]), nh=s.shape[0], setup=h.attrs["setup"]))
    R = pd.DataFrame(rows)
    a, b = (pd.Timestamp(x, tz="UTC").timestamp() for x in ref_period)
    ref = np.sum([ccfs[i] for i in np.where((R.t0 > a) & (R.t0 < b))[0]], 0)
    dt = lags[1] - lags[0]
    m = int(150 / dt)
    # QC unit: each file (January: multi-hour files), or blocks of consecutive files of QC_BLOCK_H hours (MSDVV_QC_BLOCK_H;
    # November: 17-min files are too short for a stable correlation; each file inherits its block's verdict)
    block_h = float(os.environ.get("MSDVV_QC_BLOCK_H", "0"))
    if block_h > 0:
        R["block"] = (np.floor(R.t0 / (block_h * 3600)).astype(np.int64)).astype(str) + "_" + R.setup.astype(str)
        units = [np.where(R.block.values == b)[0] for b in R.block.unique()]
    else:
        units = [[i] for i in range(len(R))]
    for idx in units:
        c = np.sum([ccfs[i] for i in idx], 0)
        xc = np.array([np.sum(np.roll(c, k, axis=-1) * ref) for k in range(-m, m + 1)]) / np.sqrt(np.sum(c**2) * np.sum(ref**2))
        k = int(np.argmax(xc))
        for i in idx:
            R.loc[i, "shift_s"], R.loc[i, "cc"], R.loc[i, "cc0"] = (k - m) * dt, float(xc[k]), float(xc[m])
    R["status"] = np.where(R.cc0 >= cc_good, "good", np.where((R.cc >= cc_good) & (R.shift_s.abs() > 3), "jump", "bad"))
    R.loc[R.status == "good", "shift_s"] = 0.0
    R.to_csv(path, index=False)
    return R


def load_hourly_qc(comp: str, qc: pd.DataFrame, band: str = "b1", inject: float = 0.0, inject_at=None, prep: str = "onebit",
                   sub_band=None, taper_hz: float = 0.02, maxlag: float | None = None):
    """Like load_hourly, but per DAS file: drop 'bad' files, shift 'jump' files by shift_s (CCF domain), then merge
    partial hours. sub_band (lo, hi) Hz: cosine-tapered band limitation of the stored band before the CCF.
    Returns t, H (nh, npath, nlag), lags, paths."""
    od = obsdas_dir(comp, prep=prep)
    lat = json.load(open(OUT / "obsdas" / "latency_calibration.json"))[band]
    st = qc.set_index("file")
    T, CC = [], []
    for f in sorted(od.glob("*.h5")):
        name = f.name[:-3]
        if name not in st.index or st.loc[name, "status"] == "bad":
            continue
        with h5py.File(f, "r") as h:
            s = h[band]["spec"][:]
            if s.shape[0] == 0:
                continue
            bins, fs, nfft = h[band]["bins"][:], h.attrs["fs"], int(h.attrs["nfft"])
            ml = float(maxlag) if maxlag else float(h.attrs["maxlag_s"])  # 600-s windows, nfft 2048: no wrap below ~500 s
            fr = np.fft.rfftfreq(nfft, 1 / fs)[bins]
            tau = lat["latency_s"][h.attrs["setup"]] * lat["ramp_sign"]
            wsub = np.ones(len(fr), np.float32)
            if sub_band is not None:
                lo, hi = sub_band
                wsub = np.clip(np.minimum((fr - (lo - taper_hz)) / taper_hz, ((hi + taper_hz) - fr) / taper_hz), 0, 1)
                wsub = (0.5 - 0.5 * np.cos(np.pi * wsub)).astype(np.float32)
            c, lags = spectra_to_ccf(s.reshape(s.shape[0], -1, s.shape[3]) * (np.exp(-2j * np.pi * fr * tau) * wsub).astype(np.complex64), bins, nfft, fs, ml)
            c = c.astype(np.float32)
            if st.loc[name, "status"] == "jump":
                c = shift_lag(c, float(st.loc[name, "shift_s"]), lags[1] - lags[0])
            T.append(h["t_sub"][:])
            CC.append(c)
    t, inv = np.unique(np.concatenate(T), return_inverse=True)
    H = np.zeros((len(t),) + CC[0].shape[1:], np.float32)
    np.add.at(H, inv, np.concatenate(CC))
    if inject:
        late = t >= pd.Timestamp(inject_at, tz="UTC").timestamp()
        for p in range(H.shape[1]):
            H[late, p] = np.array([np.interp(lags * (1 + inject), lags, h) for h in H[late, p]])
    return t, H, lags, pd.read_csv(od / "paths.csv")
