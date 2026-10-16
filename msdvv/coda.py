"""Coda-window rule shared by the three dv/v products (DAS-station, within-span, across-span).

Coda only, never ballistic. Preferred rule for groups of pairs (branch_velocities + branch_coda_window): the end of the
coherent ballistic energy, v_end = slowest velocity at which the envelope stack along lag = d / v is still >= 30 % of
its peak contrast (per segment pair / cluster and lag side); coda = [d / v_end + 10 s, + 90 s]. Per-pair fallback
(coda_windows), on the all-period reference CCF of a pair, per lag side:
  1. strong ballistic arrivals = envelope peaks (envelope smoothed over SMOOTH_S) with amplitude >= REL x the largest
     peak, at |lag| in [d / u_fast, d / u_search] (|lag| <= 15 s for colocated pairs, d < 3 km);
  2. t_last = the latest of them (fast and slow arrivals both count);
  3. coda start c0 = max(t_last + AFTER_S, d / u_slow + PAD_S)  -- never earlier than the slowest plausible wave;
  4. coda end   c1 = min(c0 + LEN_S, maxlag - 10); the side is usable only if c1 - c0 >= MIN_LEN_S.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert

AFTER_S, PAD_S, LEN_S, MIN_LEN_S, REL, SMOOTH_S = 15.0, 10.0, 90.0, 40.0, 0.5, 5.0


def coda_windows(ref, lags, dist_km, u_slow=0.40, u_fast=2.0, u_search=0.33, after=AFTER_S, pad=PAD_S, length=LEN_S,
                 min_len=MIN_LEN_S, rel=REL, colocated_km=3.0):
    """ref (nlag,), lags (nlag,) symmetric. Returns {side: dict(t_last, n_strong, c0, c1, usable)} for side in pos/neg."""
    dt = lags[1] - lags[0]
    env = uniform_filter1d(np.abs(hilbert(ref)), max(1, int(round(SMOOTH_S / dt))))
    ml = float(np.abs(lags).max())
    out = {}
    for side, sg in (("pos", 1), ("neg", -1)):
        la = sg * lags
        m = la >= 0
        e, l = env[m][np.argsort(la[m])], np.sort(la[m])
        if dist_km < colocated_km:
            lo, hi = 0.0, 15.0
        else:
            lo, hi = dist_km / u_fast, min(dist_km / u_search, ml - 20)
        w = (l >= lo) & (l <= hi)
        if w.sum() < 3:
            out[side] = dict(t_last=np.nan, n_strong=0, c0=np.nan, c1=np.nan, usable=False)
            continue
        pk, _ = find_peaks(e[w])
        if len(pk) == 0:
            pk = np.array([int(np.argmax(e[w]))])
        amp = e[w][pk]
        strong = pk[amp >= rel * amp.max()]
        t_last = float(l[w][strong].max())
        c0 = max(t_last + after, (dist_km / u_slow + pad) if dist_km >= colocated_km else after)
        c1 = min(c0 + length, ml - 10.0)
        out[side] = dict(t_last=t_last, n_strong=int(len(strong)), c0=c0, c1=c1, usable=bool(c1 - c0 >= min_len))
    return out


def branch_velocities(refs, lags, dists, v_grid=None, rel=0.5, min_pairs=20, rel_end=0.3):
    """Coherent ballistic branches of a set of pairs: stack the (smoothed, normalised) reference envelopes along
    lag = d / v for each v and side; branches = local maxima >= rel of the way from the median to the maximum.
    Returns {side: sorted array of branch velocities, side + "_end": v_end} where v_end = the slowest velocity at which
    the moveout stack is still >= rel_end of the way from the median to the maximum (end of the ballistic energy,
    also for broad packets); NaN if fewer than min_pairs pairs."""
    v_grid = np.arange(0.30, 2.01, 0.02) if v_grid is None else v_grid
    if len(refs) < min_pairs:
        return {"pos": np.array([]), "neg": np.array([]), "pos_end": np.nan, "neg_end": np.nan}
    dt = lags[1] - lags[0]
    env = uniform_filter1d(np.abs(hilbert(refs, axis=-1)), max(1, int(round(SMOOTH_S / dt))), axis=-1)
    env = env / (env.max(1, keepdims=True) + 1e-30)
    out = {}
    for side, sg in (("pos", 1), ("neg", -1)):
        vs = np.array([np.mean([np.interp(sg * d / v, lags, e) for d, e in zip(dists, env)]) for v in v_grid])
        bg = np.median(vs)
        pk, _ = find_peaks(vs)
        pk = pk[vs[pk] >= bg + rel * (vs.max() - bg)] if vs.max() > bg else np.array([], int)
        out[side] = np.sort(v_grid[pk])
        sig = vs >= bg + rel_end * (vs.max() - bg) if vs.max() > bg else np.zeros(len(vs), bool)
        out[side + "_end"] = float(v_grid[sig].min()) if sig.any() else np.nan
    return out


def branch_coda_window(dist_km, v_slow, maxlag, after=10.0, length=LEN_S, min_len=MIN_LEN_S, v_fallback=0.40):
    """Coda window after the end of the coherent ballistic energy: [d / v_end + after, + length] (v_fallback if none)."""
    v = v_slow if (v_slow is not None and np.isfinite(v_slow)) else v_fallback
    c0 = dist_km / v + after
    c1 = min(c0 + length, maxlag - 10.0)
    return c0, c1, bool(c1 - c0 >= min_len)


def energy_end_velocity(refs, lags, dists, ratio=2.0, v_grid=None, bg_min_lag=320.0, min_pairs=20):
    """End of the ballistic (+ early) energy for a group of pairs, per side: the slowest velocity in v_grid at which
    the median over pairs of envelope(d / v) / late-coda level (median envelope at |lag| >= bg_min_lag on that side)
    is still >= ratio. Coda windows starting at d / v_end + 10 s contain no ballistic energy above ratio x the
    late-coda level. Returns {"pos": v, "neg": v} (NaN if too few pairs / never above ratio)."""
    v_grid = np.arange(0.30, 1.51, 0.02) if v_grid is None else v_grid
    if len(refs) < min_pairs:
        return {"pos": np.nan, "neg": np.nan}
    dt = lags[1] - lags[0]
    env = uniform_filter1d(np.abs(hilbert(refs, axis=-1)), max(1, int(round(SMOOTH_S / dt))), axis=-1)
    out = {}
    for side, sg in (("pos", 1), ("neg", -1)):
        la = sg * lags
        bg = np.median(env[:, la >= bg_min_lag], axis=1) + 1e-30
        e = env / bg[:, None]
        vs = np.array([np.median([np.interp(sg * d / v, lags, x) for d, x in zip(dists, e)]) for v in v_grid])
        sig = v_grid[vs >= ratio]
        out[side] = float(sig.min()) if len(sig) else np.nan
    return out


def ballistic_end(refs, lags, dists, v_grid=None, bg_min_lag=320.0, min_pairs=20, rel_branch=0.4, rel_decay=0.5):
    """Moveout-based end of the ballistic arrivals for a group of pairs, per lag side.

    1. velocity spectrum vs(v) = median over pairs of envelope(d / v) / late-coda level (|lag| >= bg_min_lag);
       branches = local maxima >= min + rel_branch x (max - min) over v in 0.3-1.5 km/s (avoids zero-lag energy);
    2. v_b = slowest branch; pair envelopes aligned on t' = |lag| - d / v_b and stacked (median);
    3. t_after = first t' > peak (within +-15 s of 0) where the stack - 1 drops below rel_decay x (peak - 1).
    Coda start per pair = d / v_b + t_after + 10 s (msdvv.coda.ballistic_coda_start).
    Returns {side: dict(v_b, branches, t_after)}; NaN entries if too few pairs or no branch."""
    v_grid = np.arange(0.30, 1.51, 0.02) if v_grid is None else v_grid
    nan = {"v_b": np.nan, "branches": np.array([]), "t_after": np.nan}
    if len(refs) < min_pairs:
        return {"pos": dict(nan), "neg": dict(nan)}
    dt = lags[1] - lags[0]
    env = uniform_filter1d(np.abs(hilbert(refs, axis=-1)), max(1, int(round(SMOOTH_S / dt))), axis=-1)
    out = {}
    tp = np.arange(-30.0, 150.0, dt)
    for side, sg in (("pos", 1), ("neg", -1)):
        la = sg * lags
        bg = np.median(env[:, la >= bg_min_lag], axis=1) + 1e-30
        e = env / bg[:, None]
        vs = np.array([np.median([np.interp(sg * d / v, lags, x) for d, x in zip(dists, e)]) for v in v_grid])
        pk, _ = find_peaks(vs)
        pk = pk[vs[pk] >= vs.min() + rel_branch * (vs.max() - vs.min())]
        if len(pk) == 0:
            out[side] = dict(nan)
            continue
        vb = float(v_grid[pk].min())
        al = np.median([np.interp(sg * (d / vb + tp), lags, x) for d, x in zip(dists, e)], axis=0)
        w = np.abs(tp) <= 15
        i_pk = np.where(w)[0][np.argmax(al[w])]
        thr = 1 + rel_decay * (al[i_pk] - 1)
        after = np.where((tp > tp[i_pk]) & (al < thr))[0]
        t_after = float(tp[after[0]]) if len(after) else float(tp[-1])
        out[side] = dict(v_b=vb, branches=np.sort(v_grid[pk]), t_after=t_after)
    return out


def ballistic_coda_start(dist_km, be, maxlag, pad=10.0, length=LEN_S, min_len=MIN_LEN_S, v_fallback=0.40):
    """Coda window [d / v_b + t_after + pad, + length] from ballistic_end output `be` (one side); fallback d/v_fallback + pad."""
    if be is not None and np.isfinite(be.get("v_b", np.nan)):
        c0 = dist_km / be["v_b"] + be["t_after"] + pad
    else:
        c0 = dist_km / v_fallback + pad
    c1 = min(c0 + length, maxlag - 10.0)
    return c0, c1, bool(c1 - c0 >= min_len)
