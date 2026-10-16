"""Seismometer-DAS dv/v v2: timing-QC'd files, one stacked correlation function per path cluster, windows on the
cluster's own arrival (arrival side only), coda (primary) and ballistic (secondary), Z/R/T combined by 1/err^2.

1. DAS timing: per-file QC from the colocated HYSB1 x SB3 correlation (msdvv.obsdas.file_timing_qc): 'bad' files
   (in-file drift / missing samples) dropped, 'jump' files shifted back.
2. Cluster CCF: the clusters of s37 (pairs.csv 'cluster'); every pair's hourly CCF normalised by its reference
   amplitude and statically aligned (<= 10 s) on the reference of the cluster's median-arrival pair, then summed.
3. Arrival t_b = envelope peak of the cluster reference. Windows on the arrival side (both sides if |t_b| < 3 s):
   coda [|t_b| + 15, |t_b| + 90] s; ballistic [|t_b| - 12, |t_b| + 12] s (only if |t_b| >= 15 s).
4. 24-h back-stacked stacks moving 1 h (>= 20 h of good data), stretching vs the cluster reference, even/odd floor;
   components combined per stack by 1/err^2. Convergence CC (stack vs reference without it) per component/window.
INJECT=<eps> (INJECT_AT, default 2026-01-01) for the bias check (suffix _inj<eps>).
Output: outputs_100m/obsdas_v2/  figures figs/obsdas/obsdas_v2_*.png
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from msdvv.dvv import stretch
from msdvv.obsdas import OUT, best_shift, file_timing_qc, load_hourly_qc, moving, moving_windows, shift_lag

warnings.filterwarnings("ignore", category=RuntimeWarning)
COMPS = ["Z", "R", "T"]
RNG = (0.12, 0.30)
CC_MIN = 0.4
INJECT = float(os.environ.get("INJECT", "0"))
INJ_AT = os.environ.get("INJECT_AT", "2026-01-01")
SUF = f"_inj{INJECT:g}" if INJECT else ""
od = OUT / "obsdas_v2"
od.mkdir(exist_ok=True)
fd = OUT / "figs" / "obsdas"
CL = pd.read_csv(OUT / "obsdas_combined" / "pairs.csv")
CLUSTERS = ["red: HYS14 x ACF -> SB3", "red: HYS14 x ACF -> SC4 west", "red: HYS14 x ACF -> SC4 east",
            "blue: HYSB1 control -> SB3", "blue: HYSB1 control -> SC4", "orange: HYS14 x Daisy Bank"]
COL = dict(zip(CLUSTERS, ["#8b0000", "#e31a1c", "#f768a1", "#08306b", "#4292c6", "tab:orange"]))
qc = file_timing_qc("T")


def windows(lags, tb):
    """Lag masks (coda, ballistic) on the arrival side; ballistic None if the arrival is too close to zero lag."""
    if abs(tb) < 3:
        la = np.abs(lags)
        return (la >= 15) & (la <= 90), None
    s = np.sign(tb) * lags
    a = abs(tb)
    return (s >= a + 15) & (s <= a + 90), ((s >= a - 12) & (s <= a + 12)) if a >= 15 else None


def measure(C, ref, lags, m):
    """Stretch every stack in C (n, nlag) against ref inside mask m (one side or both: done as a full-lag window
    with the rest zeroed, stretching about zero lag). Returns dv, W (=1/err^2, 0 if rejected), cc."""
    dt = lags[1] - lags[0]
    pos = lags >= 0
    out_dv, out_w, out_cc = [], [], []
    for side, sel in (("pos", pos), ("neg", ~pos)):
        mm = m & sel
        if mm.sum() < 10:
            continue
        lg = np.abs(lags[sel])
        o = np.argsort(lg)
        c, r = C[:, sel][:, o], ref[sel][o]
        lg = lg[o]
        t1, t2 = lg[(mm[sel])[o]].min(), lg[(mm[sel])[o]].max()
        st = stretch(c, np.broadcast_to(r, c.shape), dt, t1, t2, 0.01, 101, RNG)
        ok = (st["cc"] >= CC_MIN) & ~st["at_bound"]
        w = np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0)
        out_dv.append(np.where(ok, st["dv"], 0) * w)
        out_w.append(w)
        out_cc.append(st["cc"])
    W = np.sum(out_w, 0)
    return np.where(W > 0, np.sum(out_dv, 0) / np.maximum(W, 1e-30), np.nan), W, np.nanmean(out_cc, 0)


def conv(C, ref, m):
    a = C[:, m] - C[:, m].mean(1, keepdims=True)
    r = (ref[None] - C)[:, m]
    r = r - r.mean(1, keepdims=True)
    return (a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30)


res = {}  # (cluster, comp, window) -> dict
ccf_store = {}
for comp in COMPS:
    t, H, lags, P = load_hourly_qc(comp, qc, inject=INJECT, inject_at=INJ_AT)
    dt = lags[1] - lags[0]
    tend, i0, i1 = moving_windows(t)
    hour = np.round(t / 3600).astype(np.int64)
    ref_all = H.sum(0)
    for g in CLUSTERS:
        idx = np.where(CL.cluster.values == g)[0]
        env = np.abs(hilbert(ref_all[idx], axis=-1))
        tb_p = np.array([lags[np.argmax(np.where(np.abs(lags) < max(15, 2.5 * CL.dist_km[p] / 0.5 + 10), e, 0))] for e, p in zip(env, idx)])
        pm = idx[np.argsort(np.abs(tb_p - np.median(tb_p)))[0]]
        Hc = np.zeros((len(t), len(lags)), np.float32)
        for p in idx:
            sh = best_shift(ref_all[p], ref_all[pm], dt, 10.0)
            Hc += shift_lag(H[:, p], sh, dt) / (np.abs(ref_all[p]).max() + 1e-30)
        ref = Hc.sum(0)
        e = np.abs(hilbert(ref))
        tb = float(lags[np.argmax(np.where(np.abs(lags) < np.abs(tb_p).max() + 10, e, 0))])
        mc, mb = windows(lags, tb)
        X = Hc[:, None]
        C = moving(X, i0, i1)[:, 0]
        Ce = moving(X, i0, i1, (hour % 2 == 0).astype(np.float32))[:, 0]
        Co = moving(X, i0, i1, (hour % 2 == 1).astype(np.float32))[:, 0]
        ccf_store[(g, comp)] = (ref, C, tb)
        for wname, m in (("coda", mc), ("ballistic", mb)):
            if m is None:
                continue
            dv, W, cc = measure(C, ref, lags, m)
            dve, We, _ = measure(Ce, ref, lags, m)
            dvo, Wo, _ = measure(Co, ref, lags, m)
            res[(g, comp, wname)] = dict(t=tend, dv=dv, W=W, cc=cc, dve=dve, We=We, dvo=dvo, Wo=Wo, conv=np.nanmedian(conv(C, ref, m)), tb=tb)
    print(f"[{comp}] done ({len(t)} good hours, {len(tend)} 24-h stacks)", flush=True)
    del H

# ---- combine components per cluster & window
rows, ser = [], []
for g in CLUSTERS:
    for wname in ("coda", "ballistic"):
        ks = [(g, c, wname) for c in COMPS if (g, c, wname) in res]
        if not ks:
            continue
        tt = res[ks[0]]["t"]

        def comb(key_dv, key_w):
            num = np.sum([np.nan_to_num(res[k][key_dv]) * res[k][key_w] for k in ks], 0)
            den = np.sum([res[k][key_w] for k in ks], 0)
            return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan), den

        dv, W = comb("dv", "W")
        dve, _ = comb("dve", "We")
        dvo, _ = comb("dvo", "Wo")
        d = (dve - dvo)[np.isfinite(dve - dvo)]
        floor = 1.4826 * np.median(np.abs(d - np.median(d))) / 2 if len(d) > 10 else np.nan
        row = dict(cluster=g, window=wname, arrival_s=res[ks[0]]["tb"], floor_pct=100 * floor, std_pct=100 * np.nanstd(dv),
                   kept=float(np.mean(np.isfinite(dv))), n_stacks=len(dv))
        for k in ks:
            row[f"conv_{k[1]}"] = res[k]["conv"]
            row[f"stretchCC_{k[1]}"] = float(np.nanmedian(res[k]["cc"]))
        rows.append(row)
        ser.append(pd.DataFrame(dict(cluster=g, window=wname, t=tt, dv=dv, W=W, floor=floor)))
SUM = pd.DataFrame(rows)
SER = pd.concat(ser, ignore_index=True)
SUM.to_csv(od / f"summary{SUF}.csv", index=False)
SER.to_csv(od / f"series_24h{SUF}.csv", index=False)
pd.set_option("display.width", 250)
print(SUM.round(3).to_string(index=False))
if INJECT:
    sys.exit(0)
with h5py.File(od / "cluster_ccfs_v2.h5", "w") as h:
    h["lags"], h["t_end"] = lags, tend
    for (g, comp), (ref, C, tb) in ccf_store.items():
        k = g.replace(":", "").replace(" ", "_").replace(">", "")
        h[f"{comp}/{k}/ref"], h[f"{comp}/{k}/stack_24h"] = ref, C.astype(np.float32)
        h[f"{comp}/{k}"].attrs["arrival_s"] = tb

# ---- figure: per cluster, T-component 24-h CCF image (+ windows) and coda/ballistic dv/v
XL = 130.0
lm = np.abs(lags) <= XL
fig, axs = plt.subplots(len(CLUSTERS), 2, figsize=(18, 3.3 * len(CLUSTERS)), gridspec_kw=dict(width_ratios=[1, 1.6]), squeeze=False)
full = np.arange(tend[0], tend[-1] + 1, 3600.0)
for i, g in enumerate(CLUSTERS):
    ref, C, tb = ccf_store[(g, "T")]
    ax = axs[i, 0]
    Cn = C / (np.abs(C[:, lm]).max(1, keepdims=True) + 1e-30)
    G = np.full((len(full), lm.sum()), np.nan)
    G[np.searchsorted(full, tend)] = Cn[:, lm]
    ax.pcolormesh(lags[lm], pd.to_datetime(full, unit="s", utc=True), G, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
    mc, mb = windows(lags, tb)
    for m, cl in ((mc, "k"), (mb, "g")):
        if m is not None:
            edges = lags[np.where(np.diff(m.astype(int)) != 0)[0]]
            for x in edges:
                ax.axvline(x, color=cl, ls="--", lw=0.9)
    r = SUM[(SUM.cluster == g) & (SUM.window == "coda")].iloc[0]
    ax.set_title(f"T-component 24-h CCFs; arrival {tb:+.0f} s; coda conv. CC {r.conv_T:.2f} (black), ballistic (green)", fontsize=8)
    ax.set_ylabel(g.replace(": ", ":\n"), fontsize=8, color=COL[g])
    ax.set_xlim(-XL, XL)
    ax = axs[i, 1]
    for wname, cl in (("coda", COL[g]), ("ballistic", "0.4")):
        s = SER[(SER.cluster == g) & (SER.window == wname)]
        if s.empty:
            continue
        s = s.set_index("t").reindex(full)
        td = pd.to_datetime(full, unit="s", utc=True)
        fl = SUM[(SUM.cluster == g) & (SUM.window == wname)].iloc[0]
        if wname == "coda":
            ax.fill_between(td, 100 * (s.dv - fl.floor_pct / 100), 100 * (s.dv + fl.floor_pct / 100), color=cl, alpha=0.25)
        ax.plot(td, 100 * s.dv, color=cl, lw=1.0 if wname == "coda" else 0.7,
                label=f"{wname}: floor {fl.floor_pct:.3f}%, std {fl.std_pct:.3f}%, kept {100 * fl.kept:.0f}%")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel("dv/v %")
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Seismometer-DAS v2: timing-QC'd files, one stacked CCF per cluster, arrival-side windows; 24-h back-stacked, step 1 h "
             "(gaps = files with bad DAS timing); dv/v = Z+R+T combined, shaded = even/odd floor", fontsize=10)
fig.tight_layout()
fig.savefig(fd / "obsdas_v2_clusters.png", dpi=105)
print("done")
