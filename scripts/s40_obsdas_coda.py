"""Seismometer-DAS coda dv/v: ballistic arrival picked on every correlation function, coda sampled after it.

For each pair, component (Z/R/T) and lag side (+: DAS after OBS, -: OBS after DAS), on the timing-QC'd reference:
  ballistic pick t_b = envelope peak with |lag| in [d / 2.0, d / 0.35 + 10] s (|lag| <= 10 s for d < 3 km); picks with
                      SNR < 4 are filled from the cluster's good picks on that side (+ moveout at 0.6 km/s)
  coda window       = [c0, c0 + CODA_LEN - 15] s with c0 = max(t_b + 15, d / U_FLOOR + 20) (U_FLOOR 0.40 km/s: never before
                      the slowest wave seen in the data); lags to MAXLAG = 300 s
Cluster CCF: each side of each pair shifted so its ballistic pick lands on the cluster's median pick, normalised by the
pair's reference amplitude, summed. dv/v: stretching of the coda windows (both sides, combined by 1/err^2) of the
STACK_H-h back-stacked cluster CCFs (default 48 h, step 1 h, >= 5/6 of the window with data) against the cluster reference; Z/R/T combined
by 1/err^2; even/odd floor; INJECT=<eps> (INJECT_AT) for the bias check.
Test variants (env): OBSDAS_PREP=ram, SUB_BAND=lo,hi, HOUR_REJECT=k, STACK_H=48, CODA_LEN=60; VARIANT=<name> writes to
obsdas_coda/variants/<name>/ (compared by s41_obsdas_variants.py).
Figures: obsdas_coda_picks.png (gathers with picks + coda windows), obsdas_coda_dvv.png
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from msdvv.dvv import stretch
from msdvv.obsdas import OUT, file_timing_qc, load_hourly_qc, moving, moving_windows, shift_lag

warnings.filterwarnings("ignore", category=RuntimeWarning)
COMPS = ["Z", "R", "T"]
# ---- test variants (README "Raising the seismometer-DAS CC"): all default to the reference configuration
PREP = os.environ.get("OBSDAS_PREP", "ram")               # ram (default since the s41 tests) | onebit
SUB = os.environ.get("SUB_BAND", "")                      # e.g. "0.15,0.25": analysis sub-band (Hz)
SUB_BAND = tuple(float(v) for v in SUB.split(",")) if SUB else None
HOUR_REJECT = float(os.environ.get("HOUR_REJECT", "0"))  # k: drop hours whose cluster CCF norm > median + k * MAD (0 = off)
STACK_H = int(os.environ.get("STACK_H", "48"))            # stack length (h; production 48 since the s41 tests); min data = 5/6 of it
VARIANT = os.environ.get("VARIANT", "")                   # output sub-directory name (empty = reference run)
RNG = SUB_BAND or (0.12, 0.30)
ML = float(os.environ.get("MAXLAG", "300"))               # CCF lag range (s); coda may run past 200 s with the floor below
U_FLOOR = float(os.environ.get("U_FLOOR", "0.40"))        # coda never starts before d / U_FLOOR + 20 s (slowest wave seen, msdvv.coda)
CC_MIN, CODA0, UMAX, UMIN, SNR_MIN, U_FILL = 0.4, 15.0, 2.0, 0.35, 4.0, 0.6
CODA1 = float(os.environ.get("CODA_LEN", "90"))
INJECT = float(os.environ.get("INJECT", "0"))
INJ_AT = os.environ.get("INJECT_AT", "2026-01-01")
SUF = (f"_inj{INJECT:g}" if INJECT else "") + ("" if (CODA1 == 90 or os.environ.get("VARIANT")) else f"_coda{CODA1:g}")
od = OUT / "obsdas_coda" / (f"variants/{VARIANT}" if VARIANT else "")
od.mkdir(parents=True, exist_ok=True)
fd = OUT / "figs" / "obsdas"
CL = pd.read_csv(OUT / "obsdas_combined" / "pairs.csv")
CLUSTERS = ["red: HYS14 x ACF -> SB3", "red: HYS14 x ACF -> SC4 west", "red: HYS14 x ACF -> SC4 east",
            "blue: HYSB1 control -> SB3", "blue: HYSB1 control -> SC4", "orange: HYS14 x Daisy Bank"]
COL = dict(zip(CLUSTERS, ["#8b0000", "#e31a1c", "#f768a1", "#08306b", "#4292c6", "tab:orange"]))
qc = file_timing_qc("T")


def pick(ref, lags, d):
    """Ballistic pick per side: (|lag| of the envelope peak, SNR = peak / median envelope of that side)."""
    e = np.abs(hilbert(ref))
    out = {}
    for side, sg in (("pos", 1), ("neg", -1)):
        la = sg * lags
        lo, hi = (0.0, 10.0) if d < 3 else (d / UMAX, d / UMIN + 10)
        m = (la >= lo) & (la <= hi)
        k = np.argmax(e[m])
        out[side] = (float(la[m][k]), float(e[m][k] / (np.median(e[la >= 0]) + 1e-30)))
    return out


def moveout_picks(raw, dist):
    """raw {p: {side: (t, snr)}} -> {p: {side: t}}. Picks with SNR >= SNR_MIN are kept (the strongest arrival on that
    side, fast or slow). Weak picks are filled from the cluster's good picks: median good pick + (d - d_med) / U_FILL
    (if no good pick on that side: d / U_FILL)."""
    out, info = {p: {} for p in raw}, {}
    for side in ("pos", "neg"):
        good = [p for p in raw if raw[p][side][1] >= SNR_MIN and (dist[p] < 3 or raw[p][side][0] > 0)]
        if good:
            tm, dm = np.median([raw[p][side][0] for p in good]), np.median([dist[p] for p in good])
        for p in raw:
            if p in good:
                out[p][side] = raw[p][side][0]
            elif dist[p] < 3:
                out[p][side] = 0.0
            else:
                out[p][side] = max(0.0, tm + (dist[p] - dm) / U_FILL) if good else dist[p] / U_FILL
        info[side] = len(good)
    return out, info


def side_part(x, lags, side):
    """One side of CCF(s) as a function of |lag| (ascending)."""
    m = lags >= 0 if side == "pos" else lags <= 0
    o = np.argsort(np.abs(lags[m]))
    return x[..., m][..., o], np.abs(lags[m])[o]


rows_pk, res, store = [], {}, {}
for comp in COMPS:
    t, H, lags, P = load_hourly_qc(comp, qc, inject=INJECT, inject_at=INJ_AT, prep=PREP, sub_band=SUB_BAND, maxlag=ML)
    dt = lags[1] - lags[0]
    MIN_H = int(round(STACK_H * 5 / 6))
    tend, i0, i1 = moving_windows(t, STACK_H, MIN_H)
    hour = np.round(t / 3600).astype(np.int64)
    ref_all = H.sum(0)
    for g in CLUSTERS:
        idx = np.where(CL.cluster.values == g)[0]
        raw = {p: pick(ref_all[p], lags, CL.dist_km[p]) for p in idx}
        pk, U = moveout_picks(raw, CL.dist_km)
        for p in idx:
            rows_pk.append(dict(comp=comp, cluster=g, obs=CL.obs[p], bin=CL.bin[p], dist_km=CL.dist_km[p], pick_pos_s=pk[p]["pos"], pick_neg_s=pk[p]["neg"],
                                raw_pos_s=raw[p]["pos"][0], snr_pos=raw[p]["pos"][1], raw_neg_s=raw[p]["neg"][0], snr_neg=raw[p]["neg"][1], n_good_pos=U["pos"], n_good_neg=U["neg"]))
        med = {s: float(np.median([pk[p][s] for p in idx])) for s in ("pos", "neg")}
        sides = {}
        for s in ("pos", "neg"):
            acc = None
            for p in idx:
                x, la = side_part(H[:, p], lags, s)
                xs = shift_lag(x, med[s] - pk[p][s], dt)  # ballistic pick -> cluster median pick
                xs = xs / (np.abs(ref_all[p]).max() + 1e-30)
                acc = xs if acc is None else acc + xs
            sides[s] = (acc, la)
        store[(g, comp)] = dict(sides=sides, med=med, pk=pk, raw=raw, idx=idx)
        per = {}
        for s in ("pos", "neg"):
            Hs, la = sides[s]
            keep = np.ones(len(t), np.float32)
            if HOUR_REJECT > 0:  # amplitude outliers (bursts) only: no comparison with the reference
                nrm = np.linalg.norm(Hs, axis=1)
                mdn = np.median(nrm)
                keep = (nrm <= mdn + HOUR_REJECT * 1.4826 * np.median(np.abs(nrm - mdn))).astype(np.float32)
            ckeep = np.concatenate([[0.0], np.cumsum(keep)])
            enough = (ckeep[i1] - ckeep[i0]) >= MIN_H
            ref = (Hs * keep[:, None]).sum(0)
            dmed = float(np.median(CL.dist_km.values[idx]))
            c0 = max(med[s] + CODA0, dmed / U_FLOOR + 20.0)  # after the ballistic pick AND never before the slowest wave seen
            c1 = min(c0 + CODA1 - CODA0, la[-1] - 10)
            X = Hs[:, None]
            out = []
            for w in (keep, keep * (hour % 2 == 0), keep * (hour % 2 == 1)):
                C = moving(X, i0, i1, w.astype(np.float32))[:, 0]
                st = stretch(C, np.broadcast_to(ref, C.shape), dt, c0, c1, 0.01, 101, RNG)
                ok = (st["cc"] >= CC_MIN) & ~st["at_bound"] & enough
                out.append((np.where(ok, st["dv"], np.nan), np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0), st["cc"]))
                if w is keep:
                    m = (la >= c0) & (la <= c1)
                    a = C[:, m] - C[:, m].mean(1, keepdims=True)
                    r = (ref[None] - C)[:, m]
                    r = r - r.mean(1, keepdims=True)
                    cv = float(np.nanmedian(((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30))[enough]))
                    store[(g, comp)][f"C_{s}"] = C
                    store[(g, comp)][f"ref_{s}"] = ref
            per[s] = dict(out=out, conv=cv, win=(c0, c1), rejected=float(1 - keep.mean()))
        res[(g, comp)] = dict(t=tend, per=per)
    print(f"[{comp}] {len(t)} good hours, {len(tend)} {STACK_H}-h stacks", flush=True)
    del H
PK = pd.DataFrame(rows_pk)
PK.to_csv(od / f"ballistic_picks{SUF}.csv", index=False)

# ---- combine sides and components
rows, ser = [], []
for g in CLUSTERS:
    num, den = np.zeros((3, len(tend))), np.zeros((3, len(tend)))
    row = dict(cluster=g)
    for comp in COMPS:
        for s in ("pos", "neg"):
            pr = res[(g, comp)]["per"][s]
            for k in range(3):
                dv, w, _ = pr["out"][k]
                num[k] += np.nan_to_num(dv) * w
                den[k] += w
            row[f"conv_{comp}{'+' if s == 'pos' else '-'}"] = pr["conv"]
            cc_ = pr["out"][0][2]
            row[f"scc_{comp}{'+' if s == 'pos' else '-'}"] = float(np.nanmedian(cc_))
            row[f"pass_{comp}{'+' if s == 'pos' else '-'}"] = float(np.mean(cc_ >= CC_MIN))
            row[f"rej_{comp}{'+' if s == 'pos' else '-'}"] = pr["rejected"]
            row[f"win_{comp}{'+' if s == 'pos' else '-'}"] = f"{pr['win'][0]:.0f}-{pr['win'][1]:.0f}"
    dv = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    d = (dv[1] - dv[2])[np.isfinite(dv[1] - dv[2])]
    floor = 1.4826 * np.median(np.abs(d - np.median(d))) / 2
    row.update(floor_pct=100 * floor, std_pct=100 * np.nanstd(dv[0]), kept=float(np.mean(np.isfinite(dv[0]))))
    rows.append(row)
    ser.append(pd.DataFrame(dict(cluster=g, t=tend, dv=dv[0], W=den[0], floor=floor)))
SUM = pd.DataFrame(rows)
SER = pd.concat(ser, ignore_index=True)
SUM.to_csv(od / f"summary{SUF}.csv", index=False)
SER.to_csv(od / f"series_{STACK_H}h{SUF}.csv", index=False)
pd.set_option("display.width", 260)
print(SUM.round(3).to_string(index=False))
if INJECT:
    sys.exit(0)

# ---- figure 1: per cluster and component, reference CCF of every pair with its picks and coda windows
XL = 290.0
lm = np.abs(lags) <= XL
fig, axs = plt.subplots(len(CLUSTERS), 3, figsize=(18, 3.4 * len(CLUSTERS)), sharex=True, squeeze=False)
t_, H_, lags_, _ = None, None, None, None
for j, comp in enumerate(COMPS):
    t_, H_, lags_, _ = load_hourly_qc(comp, qc, prep=PREP, sub_band=SUB_BAND, maxlag=ML)
    ref_all = H_.sum(0)
    del H_
    for i, g in enumerate(CLUSTERS):
        ax = axs[i, j]
        S = store[(g, comp)]
        idx = sorted(S["idx"], key=lambda p: CL.dist_km[p])
        for k, p in enumerate(idx):
            y = ref_all[p] / (np.abs(ref_all[p][lm]).max() + 1e-30)
            ax.plot(lags[lm], k + 0.45 * y[lm], color=COL[g], lw=0.5)
            for s, sg in (("pos", 1), ("neg", -1)):
                tb = S["pk"][p][s]
                ax.plot(sg * tb, k, "kv" if S["raw"][p][s][1] >= SNR_MIN and abs(S["raw"][p][s][0] - tb) < 0.5 else "wv", mec="k", ms=4)
                c0p = max(tb + CODA0, CL.dist_km[p] / U_FLOOR + 20.0)
                ax.plot([sg * c0p, sg * min(c0p + CODA1 - CODA0, lags[-1] - 10)], [k - 0.3, k - 0.3], color="k", lw=1.6, alpha=0.6)
        ax.set_yticks([0, len(idx) - 1])
        ax.set_yticklabels([f"{CL.dist_km[idx[0]]:.1f} km", f"{CL.dist_km[idx[-1]]:.1f} km"], fontsize=7)
        ax.axvline(0, color="0.6", lw=0.5)
        r = SUM[SUM.cluster == g].iloc[0]
        ax.set_title(f"{comp}: coda conv. CC +side {r[f'conv_{comp}+']:.2f} / -side {r[f'conv_{comp}-']:.2f}", fontsize=8)
        if j == 0:
            ax.set_ylabel(g.replace(": ", ":\n"), fontsize=8, color=COL[g])
axs[-1, 1].set_xlabel("lag (s), latency-corrected (+ = DAS after seismometer)")
fig.suptitle(f"Ballistic picks (black = picked, SNR >= {SNR_MIN:g}; white = weak pick, filled from the cluster) and coda windows (bars: max(pick + 15 s, d/{U_FLOOR:g} km/s + 20 s), 75 s long) on every pair's reference CCF (good-timing files only)", fontsize=11)
fig.tight_layout()
fig.savefig((od if VARIANT else fd) / f"obsdas_coda_picks{SUF}.png", dpi=100)
plt.close(fig)

# ---- figure 2: cluster coda dv/v
full = np.arange(tend[0], tend[-1] + 1, 3600.0)
td = pd.to_datetime(full, unit="s", utc=True)
fig, axs = plt.subplots(len(CLUSTERS), 1, figsize=(14, 2.1 * len(CLUSTERS)), sharex=True)
for ax, g in zip(axs, CLUSTERS):
    s = SER[SER.cluster == g].set_index("t").reindex(full)
    r = SUM[SUM.cluster == g].iloc[0]
    ax.fill_between(td, 100 * (s.dv - r.floor_pct / 100), 100 * (s.dv + r.floor_pct / 100), color=COL[g], alpha=0.25)
    ax.plot(td, 100 * s.dv, color=COL[g], lw=1)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("dv/v %")
    cvs = [r[c] for c in r.index if c.startswith("conv_")]
    ax.set_title(f"{g}: coda only, floor {r.floor_pct:.3f}%, std {r.std_pct:.3f}%, kept {100 * r.kept:.0f}%, coda conv. CC {np.nanmin(cvs):.2f}-{np.nanmax(cvs):.2f}", fontsize=9, loc="left")
fig.suptitle(f"Seismometer-DAS CODA dv/v per cluster: {STACK_H}-h back-stacked, step 1 h; running-absolute-mean; Z+R+T, both sides\nballistic picked per pair & side (coda only); DAS samples timed from per-chunk proc_start; shaded = even/odd floor", fontsize=10)
fig.tight_layout()
fig.savefig((od if VARIANT else fd) / f"obsdas_coda_dvv{SUF}.png", dpi=110)
print("done")
