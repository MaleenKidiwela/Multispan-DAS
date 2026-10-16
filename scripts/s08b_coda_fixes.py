"""Step 3b: can coda convergence be improved within the snapshots?

Compares median coda CC (60-min stacks vs leave-one-run-out reference, 5-km bins) for:
  base      : per-offset coda (as s08)
  kstack    : coda summed over all offsets k (coda ~ offset-independent at 0.4-3 km)
  early     : early-coda window [1.0, 2.5] x the slow direct-wave time (min 5 s)
  svd3      : substacks denoised by rank-3 SVD over time (per bin, offset) before stacking
  kstack+svd: both
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.stack import aggregate, coda_mask, distance_bins, load_runs, pair_mask, window_cc

cfg = load_config()
out = Path(cfg["output_dir"])
T = int(cfg.get("convergence", {}).get("fix_T_subs", 6))  # pilot: 6 x 10 min
BIN_KM = 5.0


def svd_denoise(cur, rank=3):
    """cur (nsub, nk, nbin, nlag): low-rank approximation along the substack axis per (k, bin)."""
    out = np.zeros_like(cur)
    for kk in range(cur.shape[1]):
        for b in range(cur.shape[2]):
            A = cur[:, kk, b]
            if not np.any(A):
                continue
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            r = min(rank, len(s))
            out[:, kk, b] = (U[:, :r] * s[:r]) @ Vt[:r]
    return out


def job(cab, band):
    geo = pd.read_csv(out / f"channel_quality_{cab}.csv")
    rs = load_runs(out / "ccf", cab, band)
    r0 = rs[0]
    pm = pair_mask(r0.valid, r0.ks, geo.good.values)
    bins_ch = distance_bins(geo, BIN_KM)
    bins_ch[~geo.good.values] = -1
    nbin = int(bins_ch.max()) + 1
    offs = r0.ks * r0.dx_km
    total = sum(r.spec.sum(0) for r in rs)
    res = {m: [] for m in ("base", "kstack", "early", "svd3", "kstack+svd")}
    for r in rs:
        nseg = len(r.t_sub) // T
        if nseg == 0:
            continue
        ref, lags = r0.ccf(aggregate(total - r.spec.sum(0), pm, bins_ch, nbin), workers=2)
        sub, _ = r0.ccf(aggregate(r.spec[: nseg * T], pm, bins_ch, nbin), workers=2)  # (nsub, nk, nbin, nlag)
        den = svd_denoise(sub)
        cur = sub.reshape((nseg, T) + sub.shape[1:]).sum(1)
        curd = den.reshape((nseg, T) + den.shape[1:]).sum(1)
        cm = [coda_mask(lags, o, cfg["dvv"], band) for o in offs]
        tdir = np.maximum(offs / cfg["dvv"]["v_direct"], 5.0)
        em = [(np.abs(lags) >= td) & (np.abs(lags) <= 2.5 * td) for td in tdir]
        has = np.abs(ref).sum(-1) > 0
        def cc_k(c, rr, masks):
            v = np.stack([window_cc(c[:, kk], rr[kk][None], masks[kk]) for kk in range(len(offs))], 1)
            v[:, ~has] = np.nan
            return v
        res["base"].append(cc_k(cur, ref, cm))
        res["early"].append(cc_k(cur, ref, em))
        res["svd3"].append(cc_k(curd, ref, cm))
        # offset-stack: common coda window = latest coda start over k
        cm_all = coda_mask(lags, offs.max(), cfg["dvv"], band)
        norm = lambda x: x / (np.sqrt((x[..., cm_all] ** 2).sum(-1, keepdims=True)) + 1e-30)
        rk, ck, cdk = norm(ref).sum(0), norm(cur).sum(1), norm(curd).sum(1)
        hk = has.any(0)
        v = window_cc(ck, rk[None], cm_all)
        v[:, ~hk] = np.nan
        res["kstack"].append(v[:, None])
        v = window_cc(cdk, rk[None], cm_all)
        v[:, ~hk] = np.nan
        res["kstack+svd"].append(v[:, None])
    return [dict(cable=cab, band=band, method=m, T_min=T * r0.substack_s / 60, cc_median_60min=float(np.nanmedian(np.concatenate(v))))
            for m, v in res.items()]


rows = Parallel(n_jobs=4)(delayed(job)(c, b) for c in ("north", "south") for b in cfg["xcorr"]["bands"])
df = pd.DataFrame([r for rr in rows for r in rr])
df.to_csv(out / "convergence" / "coda_fixes.csv", index=False)
print(df.pivot_table(index=["cable", "band"], columns="method", values="cc_median_60min").round(3).to_string())
