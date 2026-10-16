"""Cross-span (DAS-DAS, south cable) coda dv/v with per-pair ballistic picks (same recipe as the seismometer-DAS s40).

For every bin pair (<= 70 km, C3 excluded) and lag side, on the all-period reference CCF (lags to MAXLAG = 300 s):
  coda window (WINDOW_RULE=branch, default): c0 = max(d / v_b + t_after + 10 s, d / U_SLOW + 20 s), length CODA_LEN, where
  v_b = slowest coherent ballistic branch of the pair's segment pair and side and t_after = measured decay of that
  packet (msdvv.coda.ballistic_end); U_SLOW = 0.40 km/s (slowest wave seen in the data); lags to MAXLAG = 400 s;
  usable only if >= 40 s. WINDOW_RULE=peak: per-pair strong-peak rule (coda_windows).
A side is used if the convergence CC of its coda window (24-h stack vs reference without it) >= 0.3. dv/v per
STACK_H-h back-stacked stack (24 or 48 h, step 1 h, >= 5/6 of the window with data): stretching of each used side, sides combined by 1/err^2, then
inverse-variance mean over the pairs of each group (as s23); even/odd floor; INJECT=<eps> bias check.
env: MSDVV_INTERSPAN (interspan_v3 = one-bit | interspan_v4_ram), BAND (b1), SUB_BAND=lo,hi, CODA_LEN (90),
     INJECT, INJECT_AT, VARIANT (output: outputs_100m/interspan_coda/<VARIANT>/).
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import hilbert

from msdvv import interspan_cfg, load_config
from msdvv.coda import ballistic_coda_start, ballistic_end, coda_windows
from msdvv.dvv import stretch
from msdvv.xcorr import spectra_to_ccf

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
xs = interspan_cfg(cfg)
out = Path(cfg["output_dir"])
src = out / xs.get("out_subdir", "interspan")
BAND = os.environ.get("BAND", "b1")
SUB = os.environ.get("SUB_BAND", "")
SUB_BAND = tuple(float(v) for v in SUB.split(",")) if SUB else None
RNG = SUB_BAND or tuple(xs["bands"][BAND])
CODA1 = float(os.environ.get("CODA_LEN", "90"))
INJECT = float(os.environ.get("INJECT", "0"))
INJ_AT = os.environ.get("INJECT_AT", "2026-01-01")
VARIANT = os.environ.get("VARIANT", "ref")
od = out / "interspan_coda" / VARIANT
od.mkdir(parents=True, exist_ok=True)
SUF = f"_inj{INJECT:g}" if INJECT else ""
CONV_MIN, CC_MIN, MAX_KM = 0.3, 0.4, 70.0
SH = int(os.environ.get("STACK_H", "24"))         # 24 or 48 (both are produced); back-stacked, step 1 h
MIN_H = int(round(SH * 5 / 6))
ML = float(os.environ.get("MAXLAG", "400"))       # CCF lag range built from the stored spectra (600-s windows, nfft 2048: no wrap < 500 s)
U_SLOW = float(os.environ.get("U_SLOW", "0.40"))  # coda never starts before d / U_SLOW + 10 s
P = pd.read_csv(src / "pairs.csv")
EXCL = set(xs.get("exclude_segs", []))
keep_pairs = np.where((~P.seg_i.isin(EXCL) & ~P.seg_j.isin(EXCL) & (P.dist_km <= MAX_KM)).values)[0]
P = P.iloc[keep_pairs].reset_index(drop=True)
FEAT = [c for c in P if c.startswith("x_")]
GROUPS = {f[2:]: P[f].values for f in FEAT}
GROUPS["control (no feature)"] = ~P[FEAT].any(axis=1).values
GROUPS["all paths"] = np.ones(len(P), bool)

# ---- hourly CCFs of the selected pairs
T, SP = [], []
for f in sorted(src.glob("*.h5")):
    with h5py.File(f, "r") as h:
        T.append(h["t_sub"][:])
        SP.append(h[BAND]["spec"][:][:, keep_pairs])
        bins, fs, nfft = h[BAND]["bins"][:], h.attrs["fs"], int(h.attrs["nfft"])
ml = ML
t, inv = np.unique(np.concatenate(T), return_inverse=True)
S = np.zeros((len(t),) + SP[0].shape[1:], np.complex64)
np.add.at(S, inv, np.concatenate(SP))
del SP
if SUB_BAND:
    fr = np.fft.rfftfreq(nfft, 1 / fs)[bins]
    lo, hi = SUB_BAND
    w = np.clip(np.minimum((fr - (lo - 0.02)) / 0.02, ((hi + 0.02) - fr) / 0.02), 0, 1)
    S *= (0.5 - 0.5 * np.cos(np.pi * w)).astype(np.complex64)
H, lags = spectra_to_ccf(S, bins, nfft, fs, ml)
H = H.astype(np.float32)
del S
dt = lags[1] - lags[0]
if INJECT:
    late = t >= pd.Timestamp(INJ_AT, tz="UTC").timestamp()
    for p in range(H.shape[1]):
        H[late, p] = np.array([np.interp(lags * (1 + INJECT), lags, x) for x in H[late, p]])
ref = H.sum(0)
SUPER_KM = float(os.environ.get("SUPER_KM", "0"))  # > 0: stack pairs into super-pairs before stretching
if SUPER_KM > 0:
    # super-pair = same segment pair, same SUPER_KM distance bin, same feature-crossing flags; members normalised by
    # their reference amplitude and summed (no shift: the coda is compared at lapse time, members differ by < SUPER_KM)
    flags = P[FEAT].astype(int).astype(str).agg("".join, axis=1)
    spk = ["-".join(sorted((a_, b_))) for a_, b_ in zip(P.seg_i, P.seg_j)]
    key = pd.Series([f"{s_}|{int(dd // SUPER_KM)}|{f_}" for s_, dd, f_ in zip(spk, P.dist_km, flags)])
    nrm = np.abs(ref).max(1) + 1e-30
    groups_ = [np.where(key.values == k_)[0] for k_ in pd.unique(key)]
    H = np.stack([(H[:, g_] / nrm[g_][None, :, None]).sum(1) for g_ in groups_], axis=1)
    ref = H.sum(0)
    first = [g_[0] for g_ in groups_]
    P = P.iloc[first].reset_index(drop=True).assign(dist_km=[float(np.median(P.dist_km.values[g_])) for g_ in groups_],
                                                     n_members=[len(g_) for g_ in groups_])
    GROUPS = {f[2:]: P[f].values for f in FEAT}
    GROUPS["control (no feature)"] = ~P[FEAT].any(axis=1).values
    GROUPS["all paths"] = np.ones(len(P), bool)
    print(f"super-pairs ({SUPER_KM:g} km): {len(P)} from {sum(len(g_) for g_ in groups_)} pairs, members median {int(np.median(P.n_members))}", flush=True)
d = P.dist_km.values
npair = len(P)

# ---- coda windows: after the slowest coherent ballistic branch of the pair's segment pair and side
# (msdvv.coda.branch_velocities: envelope stack along lag = d / v over all pairs of that segment pair; WINDOW_RULE=branch)
# or the per-pair strong-peak rule (WINDOW_RULE=peak, msdvv.coda.coda_windows)
pos = lags >= 0
RULE = os.environ.get("WINDOW_RULE", "branch")
segpair = np.array(["-".join(sorted((a, b))) for a, b in zip(P.seg_i, P.seg_j)])
BR = {sp: ballistic_end(ref[segpair == sp], lags, d[segpair == sp]) for sp in np.unique(segpair)}
picks = np.full((npair, 2), np.nan)   # end of the ballistic packet (s), for the figures
snrs = np.zeros((npair, 2))           # number of coherent branches found
wins = np.full((npair, 2, 2), np.nan)
usable = np.zeros((npair, 2), bool)
for p in range(npair):
    if RULE == "peak":
        Wp = coda_windows(ref[p], lags, d[p], u_slow=U_SLOW, length=CODA1)
    for k, sd in enumerate(("pos", "neg")):
        if RULE == "peak":
            picks[p, k], snrs[p, k] = Wp[sd]["t_last"], Wp[sd]["n_strong"]
            wins[p, k], usable[p, k] = (Wp[sd]["c0"], Wp[sd]["c1"]), Wp[sd]["usable"]
            continue
        be = BR[segpair[p]][sd]
        c0g, _, _ = ballistic_coda_start(d[p], be, ml, length=CODA1, v_fallback=U_SLOW)
        c0 = max(c0g, d[p] / U_SLOW + 20.0)  # never before the slowest wave seen in the data (+ packet)
        c1 = min(c0 + CODA1, ml - 10.0)
        wins[p, k], usable[p, k] = (c0, c1), bool(c1 - c0 >= 40)
        picks[p, k] = c0 - 10.0
        snrs[p, k] = len(be["branches"])
# injection runs reuse the clean run's windows and selected (pair, side) units, so the recovered step is measured
# on exactly the same measurements (the injected data must not change the selection)
FORCE = None
if INJECT and (od / "pair_sides.csv").exists():
    clean = pd.read_csv(od / "pair_sides.csv")
    for r in clean.itertuples():
        k = "+-".index(r.side)
        wins[r.pair, k] = (r.c0, r.c1)
    FORCE = {(r.pair, "+-".index(r.side)) for r in clean.itertuples() if np.isfinite(r.scc)}
U = {0: float(np.nanmedian(d / picks[:, 0])), 1: float(np.nanmedian(d / picks[:, 1]))}
pd.DataFrame([dict(segment_pair=sp, n_pairs=int((segpair == sp).sum()), **{f"{k}_{sd}": (",".join(f"{v:.2f}" for v in BR[sp][sd][k]) if k == "branches" else BR[sp][sd][k])
                                                                         for sd in ("pos", "neg") for k in ("branches", "v_b", "t_after")}) for sp in BR]).to_csv(od / f"ballistic_{BAND}.csv", index=False)
# ---- SH-h back-stacked moving stacks (step 1 h): full, even, odd
grid = np.arange(t[0] + SH * 3600, t[-1] + 3601, 3600.0)
i0, i1 = np.searchsorted(t, grid - SH * 3600), np.searchsorted(t, grid)
okw = (i1 - i0) >= MIN_H
tend, i0, i1 = grid[okw], i0[okw], i1[okw]
hour = np.round(t / 3600).astype(np.int64)


def moving(w):
    X = H if w is None else H * w[:, None, None]
    cs = np.concatenate([np.zeros((1,) + X.shape[1:], np.float32), np.cumsum(X, 0, dtype=np.float64).astype(np.float32)])
    return cs[i1] - cs[i0]


C = moving(None)
Ce = moving((hour % 2 == 0).astype(np.float32))
Co = moving((hour % 2 == 1).astype(np.float32))
del H


def side_arrays(X, p, k):
    m = pos if k == 0 else ~pos
    o = np.argsort(np.abs(lags[m]))
    return X[..., m][..., o], np.abs(lags[m])[o]


def one(p):
    res = []
    for k in range(2):
        if not usable[p, k]:
            continue
        c, la = side_arrays(C[:, p], p, k)
        r, _ = side_arrays(ref[p], p, k)
        a0, a1 = wins[p, k]
        m = (la >= a0) & (la <= a1)
        a = c[:, m] - c[:, m].mean(1, keepdims=True)
        rr = (r[None] - c)[:, m]
        rr = rr - rr.mean(1, keepdims=True)
        conv = float(np.nanmedian((a * rr).sum(1) / np.sqrt((a**2).sum(1) * (rr**2).sum(1) + 1e-30)))
        if (conv < CONV_MIN) if FORCE is None else ((p, k) not in FORCE):
            res.append(dict(p=p, k=k, conv=conv))
            continue
        out_ = dict(p=p, k=k, conv=conv)
        for nm, X in (("", C), ("e", Ce), ("o", Co)):
            x, _ = side_arrays(X[:, p], p, k)
            st = stretch(x, np.broadcast_to(r, x.shape), dt, a0, a1, 0.01, 101, RNG)
            ok = (st["cc"] >= CC_MIN) & ~st["at_bound"]
            out_["dv" + nm] = np.where(ok, st["dv"], np.nan)
            out_["w" + nm] = np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0)
            if nm == "":
                out_["scc"] = float(np.nanmedian(st["cc"]))
                out_["kept"] = float(np.mean(ok))
        res.append(out_)
    return res


R = [r for rr in Parallel(n_jobs=24, prefer="threads")(delayed(one)(p) for p in range(npair)) for r in rr]
used = [r for r in R if "dv" in r]
tab = pd.DataFrame([dict(pair=r["p"], side="+-"[r["k"]], conv=r["conv"], scc=r.get("scc", np.nan), kept=r.get("kept", np.nan)) for r in R])
tab = tab.merge(P[["bin_i", "bin_j", "seg_i", "seg_j", "dist_km"]].reset_index().rename(columns={"index": "pair"}), on="pair")
tab["pick_s"] = [picks[p, "+-".index(s)] for p, s in zip(tab.pair, tab.side)]
tab["n_strong"] = [snrs[p, "+-".index(s)] for p, s in zip(tab.pair, tab.side)]
tab["c0"] = [wins[p, "+-".index(s), 0] for p, s in zip(tab.pair, tab.side)]
tab["c1"] = [wins[p, "+-".index(s), 1] for p, s in zip(tab.pair, tab.side)]
tab.to_csv(od / f"pair_sides{SUF}.csv", index=False)

# per pair: combine sides; per group: inverse-variance mean
ns = len(tend)
num = {nm: np.zeros((ns, npair)) for nm in ("", "e", "o")}
den = {nm: np.zeros((ns, npair)) for nm in ("", "e", "o")}
for r in used:
    for nm in ("", "e", "o"):
        num[nm][:, r["p"]] += np.nan_to_num(r["dv" + nm]) * r["w" + nm]
        den[nm][:, r["p"]] += r["w" + nm]
rows, ser = [], []
for g, gm in GROUPS.items():
    sel = gm & (den[""].sum(0) > 0)
    if sel.sum() < 3:
        continue
    gs = {nm: np.where(den[nm][:, sel].sum(1) > 0, num[nm][:, sel].sum(1) / np.maximum(den[nm][:, sel].sum(1), 1e-30), np.nan) for nm in num}
    dd = (gs["e"] - gs["o"])[np.isfinite(gs["e"] - gs["o"])]
    floor = 1.4826 * np.median(np.abs(dd - np.median(dd))) / 2
    tg = tab[tab.pair.isin(np.where(sel)[0]) & tab.scc.notna()]
    rows.append(dict(group=g, n_pairs=int(sel.sum()), n_sides=len(tg), stretch_cc=float(tg.scc.median()), conv=float(tg.conv.median()),
                     kept=float(tg.kept.median()), floor_pct=100 * floor, std_pct=100 * np.nanstd(gs[""]), pairs_per_stack=float(np.median((den[""][:, sel] > 0).sum(1)))))
    ser.append(pd.DataFrame(dict(group=g, t=tend, dv=gs[""], floor=floor)))
SUM = pd.DataFrame(rows)
SUM.to_csv(od / f"summary{SUF}.csv", index=False)
pd.concat(ser, ignore_index=True).to_csv(od / f"series_{SH}h{SUF}.csv", index=False)
pd.set_option("display.width", 220)
print(f"[{VARIANT}{SUF}] {src.name} {BAND} {RNG} {SH} h, window rule {RULE}: ballistic-end median velocity +{U[0]:.2f} / -{U[1]:.2f} km/s; "
      f"usable sides {int(usable.sum())} of {2 * npair}; converging sides {len(used)}")
print(SUM.round(3).to_string(index=False))

# ---- window check figure (reference run only): gathers with the latest strong ballistic arrival and the coda windows
if not INJECT:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conv_p = tab.groupby("pair").conv.max().reindex(range(npair)).values
    fig, axs = plt.subplots(2, len(GROUPS) - 1, figsize=(5.2 * (len(GROUPS) - 1), 12), squeeze=False,
                            gridspec_kw=dict(height_ratios=[1.3, 1]))
    for j, (g, gm) in enumerate([(g, m) for g, m in GROUPS.items() if g != "all paths"]):
        ax = axs[0, j]
        q = np.where(gm & (conv_p >= CONV_MIN))[0]
        if len(q) == 0:
            ax.axis('off'); axs[1, j].axis('off'); continue
        q = q[np.argsort(d[q])]
        G = ref[q] / (np.abs(ref[q]).max(1, keepdims=True) + 1e-30)
        ax.pcolormesh(lags, np.arange(len(q)), G, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
        for k, sg in ((0, 1), (1, -1)):
            ax.plot(sg * picks[q, k], np.arange(len(q)), "k.", ms=1.5, label="end of ballistic packet" if k == 0 else None)
            ax.plot(sg * wins[q, k, 0], np.arange(len(q)), "g.", ms=1.5, label="coda start" if k == 0 else None)
            ax.plot(sg * wins[q, k, 1], np.arange(len(q)), "m.", ms=1.0, label="coda end" if k == 0 else None)
        yt = np.linspace(0, len(q) - 1, 6).astype(int) if len(q) else []
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{d[q][i]:.0f} km" for i in yt], fontsize=7)
        ax.set(xlim=(-ML, ML), xlabel="lag (s)", title=f"{g}: {len(q)} converging pairs (sorted by distance)")
        if j == 0:
            ax.legend(fontsize=7, loc="lower left", markerscale=6)
        ax = axs[1, j]
        if len(q):
            p = q[np.argmax(conv_p[q])]
            for k_, (nm, X) in enumerate((("reference", ref[p]),)):
                ax.plot(lags, X / np.abs(X).max(), "k", lw=0.8)
            daily = np.where(np.isclose(tend % 86400, 0))[0]
            for i, jx in enumerate(daily):
                y = C[jx, p] / (np.abs(C[jx, p]).max() + 1e-30)
                ax.plot(lags, -1.5 - 0.35 * i + 0.3 * y, color="0.3", lw=0.4)
            for k, sg in ((0, 1), (1, -1)):
                if usable[p, k]:
                    ax.axvspan(*sorted((sg * wins[p, k, 0], sg * wins[p, k, 1])), color="tab:green", alpha=0.15)
                ax.axvline(sg * picks[p, k], color="tab:red", ls="--", lw=0.8)
            rp = P.iloc[p]
            ax.set(xlim=(-ML, ML), yticks=[], xlabel="lag (s)")
            ax.set_title(f"best pair: S{rp.seg_i} {rp.bin_i} - S{rp.seg_j} {rp.bin_j}, {rp.dist_km:.1f} km, conv {conv_p[p]:.2f}\n"
                         f"reference (top) + daily {SH}-h stacks; red = end of ballistic energy, green = coda", fontsize=8)
    fig.suptitle(f"Cross-span coda windows ({src.name}, {BAND} {RNG[0]}-{RNG[1]} Hz, rule {RULE}): coda starts 10 s after the end of the slowest coherent "
                 f"ballistic packet of each segment pair & side, and never before d/{U_SLOW:g} km/s + 20 s", fontsize=11)
    fig.tight_layout()
    fig.savefig(od / f"windows_check_{BAND}_{SH}h.png", dpi=100)
    print("window figure:", od / f"windows_check_{BAND}_{SH}h.png")
