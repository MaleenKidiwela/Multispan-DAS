"""Within-span (DAS-DAS, short offsets) coda dv/v with moveout-based ballistic windows (same rule as s43 / s40).

Units: offset k (0.2-3.2 km) x distance bin (LEVEL, default 5 km) of one span, per cable.
Ballistic: per span and lag side, msdvv.coda.ballistic_end over all (k, bin) of that span (velocities 0.10-1.5 km/s,
late-coda level at |lag| >= 110 s); coda window c0 = max(d / v_b + t_after + 10 s, d / U_SLOW + 20 s), length
CODA_LEN (60 s), lags to MAXLAG (150 s; the stored spectra of 240-s windows / nfft 1024 allow ~300 s).
A (k, bin, side) is used if its coda convergence CC (stack vs reference without it) >= 0.3; per bin: sides and offsets
combined by 1/err^2; per span: inverse-variance mean over bins; even/odd floor; INJECT bias check.
env: MSDVV_CONFIG (config_100m_v2.yaml one-bit | config_100m_v2ram.yaml), BAND (b1 | b2), LEVEL (5km), STACK_H (24 | 48),
     INJECT, INJECT_AT, VARIANT (output: <output_dir>/within_coda/<VARIANT>/).
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.coda import ballistic_coda_start, ballistic_end
from msdvv.dvv import stretch
from msdvv.stack import bin_table, distance_bins, load_aggregated

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
out = Path(cfg["output_dir"])
BAND = os.environ.get("BAND", "b1")
LEVEL = os.environ.get("LEVEL", "5km")
SH = int(os.environ.get("STACK_H", "24"))
MIN_H = int(round(SH * 5 / 6))
ML = float(os.environ.get("MAXLAG", "150"))
CODA1 = float(os.environ.get("CODA_LEN", "60"))
U_SLOW = float(os.environ.get("U_SLOW", "0.15"))
INJECT = float(os.environ.get("INJECT", "0"))
INJ_AT = os.environ.get("INJECT_AT", "2026-01-01")
VARIANT = os.environ.get("VARIANT", "ref")
CABLES = os.environ.get("CABLES", "north,south").split(",")
SUF = f"_inj{INJECT:g}" if INJECT else ""
od = out / "within_coda" / VARIANT
od.mkdir(parents=True, exist_ok=True)
RNG = tuple(cfg["xcorr"]["bands"][BAND])
CONV_MIN, CC_MIN = 0.3, 0.4
VGRID = np.arange(0.10, 1.51, 0.02)
summ, sers, wrows = [], [], []
# injection runs reuse the clean run's windows and selected units (same measurements, fair recovery estimate)
FORCE = None
_cf = od / f"units_{BAND}_{SH}h.csv"
if INJECT and _cf.exists():
    _c = pd.read_csv(_cf)
    FORCE = {(r.cable, int(r.k), int(r.bin), r.side): (r.c0, r.c1, np.isfinite(r.scc) if "scc" in _c else False) for r in _c.itertuples()}

for cab in CABLES:
    geo = pd.read_csv(out / f"channel_quality_{cab}.csv")
    bins_ch = distance_bins(geo, float(LEVEL[:-2]))
    bins_ch[~geo.good.values] = -1
    nbin = int(bins_ch.max()) + 1
    BT = bin_table(geo, bins_ch).set_index("bin")
    R = load_aggregated(out / "ccf", cab, BAND, geo.good.values, bins_ch, nbin)
    R.maxlag_s = ML
    H, lags = R.ccf(R.spec, workers=8)  # (nsub, nk, nbin, nlag)
    H = H.astype(np.float32)
    t = R.t_sub
    ks, dx = R.ks, R.dx_km
    del R
    dt = lags[1] - lags[0]
    if INJECT:
        late = t >= pd.Timestamp(INJ_AT, tz="UTC").timestamp()
        idx = np.where(late)[0]
        for i in idx:
            H[i] = \
                np.apply_along_axis(lambda x: np.interp(lags * (1 + INJECT), lags, x), -1, H[i])
    ref = H.sum(0)  # (nk, nbin, nlag)
    has = np.abs(ref).sum(-1) > 0
    # ---- ballistic end per span & side (over all offsets and bins of the span)
    span_of_bin = np.array([f"{BT.loc[b, 'block']}{BT.loc[b, 'span']}" if b in BT.index else "" for b in range(nbin)])
    BE = {}
    for sp in np.unique(span_of_bin[span_of_bin != ""]):
        sel = [(kk, b) for kk in range(len(ks)) for b in np.where(span_of_bin == sp)[0] if has[kk, b]]
        if len(sel) < 20:
            continue
        refs = np.stack([ref[kk, b] for kk, b in sel])
        dist = np.array([ks[kk] * dx for kk, _ in sel])
        BE[sp] = ballistic_end(refs, lags, dist, v_grid=VGRID, bg_min_lag=ML - 40)
    # ---- moving stacks
    grid = np.arange(t[0] + SH * 3600, t[-1] + 3601, 3600.0)
    i0, i1 = np.searchsorted(t, grid - SH * 3600), np.searchsorted(t, grid)
    okw = (i1 - i0) >= MIN_H
    tend, i0, i1 = grid[okw], i0[okw], i1[okw]
    hour = np.round(t / 3600).astype(np.int64)

    def moving(w):
        X = H if w is None else H * w[:, None, None, None]
        cs = np.concatenate([np.zeros((1,) + X.shape[1:], np.float32), np.cumsum(X, 0, dtype=np.float64).astype(np.float32)])
        return cs[i1] - cs[i0]

    C, Ce, Co = moving(None), moving((hour % 2 == 0).astype(np.float32)), moving((hour % 2 == 1).astype(np.float32))
    del H
    pos = lags >= 0

    def side(x, k):
        m = pos if k == 0 else ~pos
        o = np.argsort(np.abs(lags[m]))
        return x[..., m][..., o], np.abs(lags[m])[o]

    def unit(kk, b):
        sp = span_of_bin[b]
        if sp not in BE or not has[kk, b]:
            return []
        dk = ks[kk] * dx
        res = []
        for k, sd in enumerate(("pos", "neg")):
            c0g, _, _ = ballistic_coda_start(dk, BE[sp][sd], ML, length=CODA1, v_fallback=U_SLOW)
            c0 = max(c0g, dk / U_SLOW + 20.0)
            c1 = min(c0 + CODA1, ML - 5.0)
            fz = FORCE.get((cab, int(ks[kk]), int(b), "+-"[k])) if FORCE is not None else None
            if FORCE is not None:
                if fz is None:
                    continue
                c0, c1 = fz[0], fz[1]
            row = dict(cable=cab, k=int(ks[kk]), offset_km=dk, bin=b, span=sp, side="+-"[k], c0=c0, c1=c1)
            if c1 - c0 < 30:
                res.append(row)
                continue
            c, la = side(C[:, kk, b], k)
            r, _ = side(ref[kk, b], k)
            m = (la >= c0) & (la <= c1)
            a = c[:, m] - c[:, m].mean(1, keepdims=True)
            rr = (r[None] - c)[:, m]
            rr = rr - rr.mean(1, keepdims=True)
            row["conv"] = float(np.nanmedian((a * rr).sum(1) / np.sqrt((a**2).sum(1) * (rr**2).sum(1) + 1e-30)))
            if (row["conv"] >= CONV_MIN) if FORCE is None else bool(fz[2]):
                for nm, X in (("", C), ("e", Ce), ("o", Co)):
                    x, _ = side(X[:, kk, b], k)
                    st = stretch(x, np.broadcast_to(r, x.shape), dt, c0, c1, 0.01, 101, RNG)
                    ok = (st["cc"] >= CC_MIN) & ~st["at_bound"]
                    row["dv" + nm] = np.where(ok, st["dv"], np.nan)
                    row["w" + nm] = np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0)
                    if nm == "":
                        row["scc"], row["kept"] = float(np.nanmedian(st["cc"])), float(np.mean(ok))
            res.append(row)
        return res

    U = [r for rr in Parallel(n_jobs=24, prefer="threads")(delayed(unit)(kk, b) for kk in range(len(ks)) for b in range(nbin)) for r in rr]
    wrows += [{k: v for k, v in r.items() if not k.startswith(("dv", "w"))} for r in U]
    used = [r for r in U if "dv" in r]
    for sp in sorted(set(r["span"] for r in used)) + ["all"]:
        uu = [r for r in used if sp == "all" or r["span"] == sp]
        if len(uu) < 3:
            continue
        g = {}
        for nm in ("", "e", "o"):
            num = np.sum([np.nan_to_num(r["dv" + nm]) * r["w" + nm] for r in uu], 0)
            den = np.sum([r["w" + nm] for r in uu], 0)
            g[nm] = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        dd = (g["e"] - g["o"])[np.isfinite(g["e"] - g["o"])]
        floor = 1.4826 * np.median(np.abs(dd - np.median(dd))) / 2 if len(dd) > 10 else np.nan
        bb = sorted(set(r["bin"] for r in uu))
        summ.append(dict(cable=cab, span=sp, n_units=len(uu), n_bins=len(bb), route_km=f"{BT.loc[bb, 'route_km'].min():.0f}-{BT.loc[bb, 'route_km'].max():.0f}",
                         depth_m=float(BT.loc[bb, "depth"].median()), stretch_cc=float(np.median([r["scc"] for r in uu])),
                         conv=float(np.median([r["conv"] for r in uu])), kept=float(np.median([r["kept"] for r in uu])),
                         floor_pct=100 * floor, std_pct=100 * np.nanstd(g[""])))
        sers.append(pd.DataFrame(dict(cable=cab, span=sp, t=tend, dv=g[""], floor=floor)))
    bes = {sp: {sd: (round(BE[sp][sd]["v_b"], 2), round(BE[sp][sd]["t_after"], 0)) for sd in ("pos", "neg")} for sp in BE}
    print(f"[{cab}] ballistic end per span (v_b km/s, packet s): {bes}; used units {len(used)} of {len(U)}", flush=True)

SUM = pd.DataFrame(summ)
SUM.to_csv(od / f"summary_{BAND}_{SH}h{SUF}.csv", index=False)
pd.concat(sers, ignore_index=True).to_csv(od / f"series_{BAND}_{SH}h{SUF}.csv", index=False)
pd.DataFrame(wrows).to_csv(od / f"units_{BAND}_{SH}h{SUF}.csv", index=False)
pd.set_option("display.width", 220)
print(f"[{VARIANT}{SUF}] {cfg['output_dir'].split('/')[-1]} {BAND} {LEVEL} {SH} h")
print(SUM.round(3).to_string(index=False))
