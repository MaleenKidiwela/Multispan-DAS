"""Denoising test for the seismometer-DAS correlations (b1, early window, 24-h back-stacked, step 1 h).

Methods (each applied identically to the full, even-hour and odd-hour stacks, so the even/odd floor is fair):
  base       linear 24-h stacks (as s33)
  select     selective stacking: only hours whose CCF correlates with the reference above the pair's median
  nbr        along-fibre stacking: each bin + its +-NBR neighbours on the same segment (same OBS), each neighbour
             statically aligned to the centre bin's reference (lag shift <= 5 s) before summing
  svd        SVD (rank K) + 2-D Wiener filter of the pair's 24-h CCF series (Moreau et al. 2017)
  nbr+svd    both
Score per method: convergence CC per pair (24-h stack vs reference without it), number of converging pairs
(>= 0.3) by distance class, group dv/v std and even/odd floor, Dec 23-Jan 8 vs Jan 9-28.
INJECT=<eps> multiplies lag time by (1+eps) in every hourly CCF after 2026-01-15 00:00 (synthetic dv/v = -eps
in the stretching convention used here); INJECT_AT (default 2026-01-01), step measured over 5 days before vs 1-6 days after: recovered step per method shows whether denoising biases dv/v.
env: OBSDAS_COMP (Z|R|T), INJECT.
"""
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import hilbert, wiener

from msdvv.dvv import one_sided, stretch
from msdvv.xcorr import spectra_to_ccf

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
COMP = os.environ.get("OBSDAS_COMP", "Z")
INJECT = float(os.environ.get("INJECT", "0"))
INJ_AT = pd.Timestamp(os.environ.get("INJECT_AT", "2026-01-01"), tz="UTC")
od = OUT / ("obsdas" + ("" if COMP == "Z" else f"_{COMP}"))
res_dir = OUT / "obsdas_denoise"
res_dir.mkdir(exist_ok=True)
TAG = COMP + (f"_inj{INJECT:g}" if INJECT else "")
P = pd.read_csv(od / "paths.csv")
band, RNG = "b1", (0.12, 0.30)
SH, MIN_H, CONV_MIN, CC_MIN, NBR, K = 24, 20, 0.3, 0.4, 2, 6
VG = 0.77
lat = json.load(open(OUT / "obsdas" / "latency_calibration.json"))[band]  # from the Z colocated paths

# ---- hourly spectra, latency-corrected per setup, merged per hour
T, SP, SET = [], [], []
for f in sorted(od.glob("*.h5")):
    with h5py.File(f, "r") as h:
        s = h[band]["spec"][:]
        if s.shape[0] == 0:
            continue
        bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], int(h.attrs["nfft"]), float(h.attrs["maxlag_s"])
        fr = np.fft.rfftfreq(nfft, 1 / fs)[bins]
        tau = lat["latency_s"][h.attrs["setup"]] * lat["ramp_sign"]
        T.append(h["t_sub"][:])
        SP.append(s.reshape(s.shape[0], -1, s.shape[3]) * np.exp(-2j * np.pi * fr * tau).astype(np.complex64))
t, inv = np.unique(np.concatenate(T), return_inverse=True)
S = np.zeros((len(t),) + SP[0].shape[1:], np.complex64)
np.add.at(S, inv, np.concatenate(SP))
del SP
H, lags = spectra_to_ccf(S, bins, nfft, fs, ml)  # hourly CCFs (nh, npath, nlag)
H = H.astype(np.float32)
dt = lags[1] - lags[0]
if INJECT:
    late = t >= INJ_AT.timestamp()
    for p in range(H.shape[1]):
        H[late, p] = np.array([np.interp(lags * (1 + INJECT), lags, h) for h in H[late, p]])
npath = H.shape[1]
d = P.dist_km.values
t1 = np.maximum(0.7 * d / VG, 3.0)
t2 = np.minimum(np.maximum(2.5 * d / VG, t1 + 30.0), ml - 10.0)
la = np.abs(lags)
WM = [(la >= a) & (la <= b) for a, b in zip(t1, t2)]
hour = np.round(t / 3600).astype(np.int64)
grid = np.arange(t[0] + SH * 3600, t[-1] + 3601, 3600.0)
i0, i1 = np.searchsorted(t, grid - SH * 3600), np.searchsorted(t, grid)
okw = (i1 - i0) >= MIN_H
tend, i0, i1 = grid[okw], i0[okw], i1[okw]
ref_all = H.sum(0)


def shift_lag(x, sh):
    """Shift CCF(s) along the last axis by sh seconds (positive: later), by Fourier phase."""
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], dt)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * sh), n=x.shape[-1], axis=-1).astype(np.float32)


def best_shift(a, b, maxs=5.0):
    m = int(maxs / dt)
    c = [np.dot(np.roll(a, k), b) for k in range(-m, m + 1)]
    return (np.argmax(c) - m) * dt


# ---- along-fibre neighbour stack (static alignment on the reference)
nbr_idx = []
for p in range(npath):
    r = P.iloc[p]
    q = P.index[(P.obs == r.obs) & (P.seg == r.seg) & (np.abs(P.bin - r.bin) <= NBR)].values
    nbr_idx.append(q)
Hn = np.zeros_like(H)
for p in range(npath):
    for q in nbr_idx[p]:
        sh = 0.0 if q == p else best_shift(ref_all[q], ref_all[p])
        Hn[:, p] += H[:, q] if sh == 0 else shift_lag(H[:, q], sh)


def moving(Hx, sel=None):
    """24-h back-stacked moving stacks (n_stack, npath, nlag); sel: (nh, npath) 0/1 hour weights."""
    X = Hx if sel is None else Hx * sel[..., None]
    cs = np.concatenate([np.zeros((1,) + X.shape[1:], np.float32), np.cumsum(X, 0, dtype=np.float64).astype(np.float32)])
    return cs[i1] - cs[i0]


def svd_wiener(C):
    """C (n_stack, nlag) -> rank-K reconstruction + 2-D Wiener filter."""
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    R = (U[:, :K] * s[:K]) @ Vt[:K]
    return wiener(R, (5, 5)).astype(np.float32)


def selection(Hx, ref):
    """1 for hours whose CCF (early window) correlates with the reference above the pair's median."""
    sel = np.zeros(Hx.shape[:2], np.float32)
    for p in range(Hx.shape[1]):
        a, r = Hx[:, p][:, WM[p]], ref[p][WM[p]]
        cc = (a @ r) / (np.linalg.norm(a, axis=1) * np.linalg.norm(r) + 1e-30)
        sel[:, p] = cc > np.nanmedian(cc)
    return sel


def evaluate(name, Hx, use_sel, use_svd):
    ref = Hx.sum(0)
    sel = selection(Hx, ref) if use_sel else None
    even = (hour % 2 == 0).astype(np.float32)[:, None] * (np.ones((1, npath), np.float32) if sel is None else sel)
    odd = (hour % 2 == 1).astype(np.float32)[:, None] * (np.ones((1, npath), np.float32) if sel is None else sel)
    C, Ce, Co = moving(Hx, sel), moving(Hx, even), moving(Hx, odd)
    ref = Hx.sum(0) if sel is None else (Hx * sel[..., None]).sum(0)

    def per_path(p):
        c, ce, co = C[:, p], Ce[:, p], Co[:, p]
        raw = c.copy()
        if use_svd:
            c, ce, co = svd_wiener(c), svd_wiener(ce), svd_wiener(co)
        m = WM[p]
        a = c[:, m] - c[:, m].mean(1, keepdims=True)
        r = (ref[p][None] - raw)[:, m]
        r = r - r.mean(1, keepdims=True)
        conv = float(np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30)))
        out = {"conv": conv}
        if conv < CONV_MIN:
            return p, out, None
        series = []
        for X in (c, ce, co):
            dv_s, w_s = [], []
            for side in ("pos", "neg"):
                cc_ = one_sided(X, lags, side)
                st = stretch(cc_, np.broadcast_to(one_sided(ref[p], lags, side), cc_.shape), dt, t1[p], t2[p], 0.01, 101, RNG)
                ok = (st["cc"] >= CC_MIN) & ~st["at_bound"]
                w = np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0)
                dv_s.append(np.where(ok, st["dv"], 0) * w)
                w_s.append(w)
            W = w_s[0] + w_s[1]
            series.append((np.where(W > 0, (dv_s[0] + dv_s[1]) / np.maximum(W, 1e-30), np.nan), np.where(W > 0, W, 0.0)))
        return p, out, series

    res = Parallel(n_jobs=24, prefer="threads")(delayed(per_path)(p) for p in range(npath))
    conv = np.array([r[1]["conv"] for r in res])
    rows = []
    for o in P.obs.unique():
        ps = [r for r in res if r[2] is not None and P.obs.iloc[r[0]] == o]
        if len(ps) < 3:
            rows.append(dict(method=name, obs=o, n_paths=len(ps)))
            continue

        def gm(k):
            num = np.nansum([np.nan_to_num(r[2][k][0]) * r[2][k][1] for r in ps], 0)
            den = np.sum([r[2][k][1] for r in ps], 0)
            npp = np.sum([r[2][k][1] > 0 for r in ps], 0)
            return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan), npp

        g, npp = gm(0)
        ge, _ = gm(1)
        go, _ = gm(2)
        dd = (ge - go)[np.isfinite(ge - go)]
        floor = 1.4826 * np.median(np.abs(dd - np.median(dd))) / 2
        early = tend < pd.Timestamp("2026-01-08 12:00", tz="UTC").timestamp()
        row = dict(method=name, obs=o, n_paths=len(ps), floor_pct=100 * floor)
        for lab, m in (("Dec23-Jan8", early), ("Jan9-28", ~early)):
            row[f"std_{lab}"] = 100 * np.nanstd(g[m])
            row[f"paths/stack_{lab}"] = float(np.median(npp[m]))
            row[f"hourly_jump_{lab}"] = 100 * np.nanmedian(np.abs(np.diff(g[m])))
        if INJECT:
            a0 = INJ_AT.timestamp()
            inj = (tend >= a0 + 86400) & (tend < a0 + 6 * 86400)  # stacks fully after the injection (5 days)
            pre = (tend < a0) & (tend >= a0 - 5 * 86400)
            row["recovered_step_pct"] = 100 * (np.nanmedian(g[inj]) - np.nanmedian(g[pre]))
        rows.append(row)
        pd.DataFrame(dict(t=tend, dv=g, n=npp)).assign(method=name, obs=o).to_csv(res_dir / f"series_{TAG}_{name}_{o}.csv", index=False)
    pc = P[["obs", "bin", "seg", "dist_km"]].assign(conv=conv, method=name)
    return pd.DataFrame(rows), pc


summ, convs = [], []
for name, Hx, use_sel, use_svd in (("base", H, False, False), ("select", H, True, False), ("svd", H, False, True),
                                   ("nbr", Hn, False, False), ("nbr+svd", Hn, False, True), ("nbr+select", Hn, True, False)):
    sm, pc = evaluate(name, Hx, use_sel, use_svd)
    summ.append(sm)
    convs.append(pc)
    print(f"[{TAG}] {name}: converging {int((pc.conv >= CONV_MIN).sum())} "
          f"(<30 km {int(((pc.conv >= CONV_MIN) & (pc.dist_km < 30)).sum())}, 30-60 {int(((pc.conv >= CONV_MIN) & (pc.dist_km.between(30, 60))).sum())}, "
          f">60 {int(((pc.conv >= CONV_MIN) & (pc.dist_km > 60)).sum())})", flush=True)
SM = pd.concat(summ, ignore_index=True)
PC = pd.concat(convs, ignore_index=True)
SM.to_csv(res_dir / f"summary_{TAG}.csv", index=False)
PC.to_csv(res_dir / f"convergence_{TAG}.csv", index=False)
pd.set_option("display.width", 250)
print(SM.round(3).to_string(index=False))
