"""Vectorised stretching dv/v (same convention as noisepy.monitoring.monitoring_methods.stretching).

Convention: the current trace is resampled as s(t) = cur(t / (1 + eps)); the eps maximising
CC(s, ref) over the window is dv/v (positive = velocity increase, arrivals earlier).
Error: Weaver et al. (2011), eq. as in noisepy but with the (t2^3 - t1^3) term bracketed correctly.
"""
from __future__ import annotations

import numpy as np


def one_sided(cc: np.ndarray, lags: np.ndarray, side: str) -> np.ndarray:
    """Positive-lag half (side='pos') or time-reversed negative half (side='neg'), t >= 0."""
    i0 = np.searchsorted(lags, 0.0)
    if side == "pos":
        return cc[..., i0:]
    return cc[..., i0::-1]


def weaver_error(cc, fmin, fmax, t1, t2):
    X = np.clip(cc, 1e-6, 1.0)
    T = 1.0 / (fmax - fmin)
    wc = np.pi * (fmin + fmax)
    return np.sqrt(1 - X**2) / (2 * X) * np.sqrt(6 * np.sqrt(np.pi / 2) * T / (wc**2 * (t2**3 - t1**3)))


def _pearson(S, R):
    S = S - S.mean(-1, keepdims=True)
    R = R - R.mean(-1, keepdims=True)
    return np.sum(S * R, -1) / (np.sqrt(np.sum(S**2, -1) * np.sum(R**2, -1)) + 1e-30)


def _stretched(cur, t_win, eps, dt):
    """cur (N, L); eps (nE,) or (N, nE) -> (N, nE, nw) of cur(t/(1+eps))."""
    u = t_win / (1 + eps[..., None]) / dt  # (nE, nw) or (N, nE, nw)
    i0 = np.floor(u).astype(np.int64)
    w = (u - i0).astype(cur.dtype)
    i0 = np.clip(i0, 0, cur.shape[-1] - 2)
    if u.ndim == 2:
        return cur[:, i0] * (1 - w) + cur[:, i0 + 1] * w
    a = np.take_along_axis(cur[:, None, :], i0, axis=-1)
    b = np.take_along_axis(cur[:, None, :], i0 + 1, axis=-1)
    return a * (1 - w) + b * w


def stretch(cur, ref, dt, t1, t2, dv_range=0.01, nbtrial=101, band=None, chunk=4096):
    """Vectorised stretching on one-sided traces (t = arange(L) * dt).

    cur, ref: (N, L). Returns dict of arrays (N,): dv (fraction), cc, cc0, err, at_bound.
    """
    cur = np.asarray(cur, np.float32)
    ref = np.asarray(ref, np.float32)
    L = cur.shape[-1]
    t = np.arange(L) * dt
    iw = np.where((t >= t1) & (t <= t2))[0]
    tw = t[iw].astype(np.float32)
    eps = np.linspace(-dv_range, dv_range, nbtrial).astype(np.float32)
    de = eps[1] - eps[0]
    out = {k: np.empty(len(cur), np.float32) for k in ("dv", "cc", "cc0")}
    out["at_bound"] = np.empty(len(cur), bool)
    for s in range(0, len(cur), chunk):
        c, r = cur[s : s + chunk], ref[s : s + chunk]
        R = r[:, iw][:, None, :]
        cof = _pearson(_stretched(c, tw, eps, dt), R)  # (n, nE)
        ib = np.argmax(cof, axis=1)
        out["at_bound"][s : s + chunk] = (ib == 0) | (ib == nbtrial - 1)
        ibc = np.clip(ib, 1, nbtrial - 2)
        # fine grid +-1 coarse step around the best (as noisepy's second pass)
        fine = eps[ibc][:, None] + np.linspace(-de, de, 41, dtype=np.float32)[None, :]
        cf = _pearson(_stretched(c, tw, fine, dt), R)
        jf = np.argmax(cf, axis=1)
        out["dv"][s : s + chunk] = fine[np.arange(len(c)), jf]
        out["cc"][s : s + chunk] = cf[np.arange(len(c)), jf]
        out["cc0"][s : s + chunk] = _pearson(c[:, iw], r[:, iw])
    if band is not None:
        out["err"] = weaver_error(out["cc"], band[0], band[1], t1, t2).astype(np.float32)
    return out


def wcc(cur, ref, dt, t1, t2, win_s, step_s, max_shift_s=1.0, cc_min=0.5):
    """Vectorised windowed cross-correlation dv/v (Snieder et al. 2012), independent of stretching.

    In each sub-window the delay of cur relative to ref is the (parabolically refined) peak of their
    cross-correlation; dv/v = -slope of delay vs centre time (least squares through the origin,
    sub-windows with peak CC < cc_min are dropped). cur, ref: (N, L) one-sided traces.
    Returns dict dv (N,), n_win (N,).
    """
    cur = np.asarray(cur, np.float64)
    ref = np.asarray(ref, np.float64)
    nw = int(round(win_s / dt))
    starts = np.arange(int(np.ceil(t1 / dt)), int(t2 / dt) - nw + 1, max(1, int(round(step_s / dt))))
    taper = np.hanning(nw)
    nfft = 1 << int(np.ceil(np.log2(2 * nw)))
    ms = int(np.ceil(max_shift_s / dt))
    if len(starts) < 2:  # window too short for a slope
        return dict(dv=np.full(len(cur), np.nan, np.float32), n_win=np.zeros(len(cur), np.int32))
    shifts, weights, centres = [], [], []
    for s in starts:
        c = cur[:, s : s + nw] * taper
        r = ref[:, s : s + nw] * taper
        c = c - c.mean(1, keepdims=True)
        r = r - r.mean(1, keepdims=True)
        x = np.fft.irfft(np.fft.rfft(c, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)
        x = np.concatenate([x[:, -ms:], x[:, : ms + 1]], 1) / (np.sqrt((c**2).sum(1) * (r**2).sum(1))[:, None] + 1e-30)
        i = np.clip(np.argmax(x, 1), 1, 2 * ms - 1)
        y0, y1, y2 = (x[np.arange(len(x)), i + d] for d in (-1, 0, 1))
        den = y0 - 2 * y1 + y2
        p = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        shifts.append((i - ms + p) * dt)  # positive = cur delayed
        weights.append((y1 >= cc_min).astype(float))
        centres.append((s + nw / 2) * dt)
    sh, w, tc = np.array(shifts), np.array(weights), np.array(centres)[:, None]
    n = w.sum(0)
    slope = np.where(n >= 2, (w * sh * tc).sum(0) / np.maximum((w * tc**2).sum(0), 1e-30), np.nan)
    return dict(dv=(-slope).astype(np.float32), n_win=n.astype(np.int32))


def weighted_mean(dv, err, axis):
    """Inverse-variance weighted mean over axis (NaN-aware). Returns mean, std error, n."""
    w = np.where(np.isfinite(dv) & np.isfinite(err) & (err > 0), 1.0 / np.maximum(err, 1e-12) ** 2, 0.0)
    sw = w.sum(axis)
    m = np.where(sw > 0, np.nansum(np.where(w > 0, dv, 0) * w, axis) / np.maximum(sw, 1e-30), np.nan)
    e = np.where(sw > 0, 1.0 / np.sqrt(np.maximum(sw, 1e-30)), np.nan)
    return m, e, (w > 0).sum(axis)
