"""Stable across-feature dv/v time series from inter-span bin-pair correlations.

For each feature (paths crossing a fault trace / passing a point feature), for paths crossing no feature
(control) and for all paths: the inverse-variance mean dv/v of the converging pairs at each stack time, plus
an empirical noise floor from an even/odd-hour split of every stack (each half measured separately against
the same reference; floor = 1.4826 * MAD(even - odd) / 2, the noise of the full stack).

Run with MSDVV_INTERSPAN=interspan_v3 (bands b0, b1; C3 bins excluded).
"""
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
from joblib import Parallel, delayed
from scipy.signal import hilbert

from msdvv import interspan_cfg, load_config
from msdvv.dvv import one_sided, stretch
from msdvv.xcorr import spectra_to_ccf

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
xs = interspan_cfg(cfg)
out = Path(cfg["output_dir"])
od = out / xs.get("out_subdir", "interspan")
fd = out / "figs"
P = pd.read_csv(od / "pairs.csv")
FEAT = [c for c in P if c.startswith("x_")]
STACKS = [tuple(s) for s in xs.get("stacks_h", [[24, 6], [168, 24]])]
EXCL = set(xs.get("exclude_segs", []))
CONV_MIN, MAX_KM, CC_MIN = 0.3, 70.0, 0.4
GROUPS = {f[2:]: P[f].values for f in FEAT}
GROUPS["control (no feature)"] = ~P[FEAT].any(axis=1).values
GROUPS["all paths"] = np.ones(len(P), bool)
base_ok = (~P.seg_i.isin(EXCL) & ~P.seg_j.isin(EXCL) & (P.dist_km <= MAX_KM)).values


def load_band(band):
    t_all, specs, setups = [], [], []
    for f in sorted(od.glob("*.h5")):
        with h5py.File(f, "r") as h:
            t_all.append(h["t_sub"][:])
            specs.append(h[band]["spec"][:])
            setups += [h.attrs["setup"]] * len(h["t_sub"])
            bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], h.attrs["nfft"], h.attrs["maxlag_s"]
    t = np.concatenate(t_all)
    ut, inv = np.unique(t, return_inverse=True)
    S = np.zeros((len(ut),) + specs[0].shape[1:], np.complex64)
    np.add.at(S, inv, np.concatenate(specs))
    return ut, S, bins, fs, nfft, ml


def pair_dv(cur, ref, pairs, t1, t2, dt, rng):
    """Side-combined dv/v per (stack, pair) for the given pairs. cur (nseg, npair_all, nlag)."""
    def chunk(pp):
        rows = []
        for p in pp:
            dv_s, w_s = [], []
            for side in ("pos", "neg"):
                c = one_sided(cur[:, p], np.arange(cur.shape[-1]) - cur.shape[-1] // 2, side)
                r = np.broadcast_to(one_sided(ref[p], np.arange(cur.shape[-1]) - cur.shape[-1] // 2, side), c.shape)
                st = stretch(c, r, dt, t1[p], t2, 0.01, 101, rng)
                ok = (st["cc"] >= CC_MIN) & ~st["at_bound"]
                w = np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0)
                dv_s.append(np.where(ok, st["dv"], 0.0) * w)
                w_s.append(w)
            W = w_s[0] + w_s[1]
            rows.append(pd.DataFrame(dict(seg=np.arange(cur.shape[0]), pair=p, dv=np.where(W > 0, (dv_s[0] + dv_s[1]) / np.maximum(W, 1e-30), np.nan),
                                          err=np.where(W > 0, 1 / np.sqrt(np.maximum(W, 1e-30)), np.nan))))
        return pd.concat(rows) if rows else pd.DataFrame()
    parts = Parallel(n_jobs=16, prefer="threads")(delayed(chunk)(c) for c in np.array_split(pairs, 32) if len(c))
    return pd.concat(parts).dropna()


def group_mean(D, mask):
    g = D[mask[D.pair.values]]
    w = 1 / g.err**2
    m = (g.dv * w).groupby(g.seg).sum() / w.groupby(g.seg).sum()
    n = g.groupby("seg").size()
    return m, n


series, summary = [], []
for band in xs["bands"]:
    t, S, bins, fs, nfft, ml = load_band(band)
    ref, lags = spectra_to_ccf(S.sum(0), bins, nfft, fs, ml)
    dt = lags[1] - lags[0]
    env = np.abs(hilbert(ref, axis=-1))
    tpk = np.argmax(one_sided(env, lags, "pos") + one_sided(env, lags, "neg"), axis=-1) * dt
    vpk = P.dist_km.values / np.maximum(tpk, 1e-6)
    vg = np.median(vpk[(tpk > 2) & (vpk > 0.3) & (vpk < 5) & base_ok])
    t1 = np.maximum(1.3 * P.dist_km.values / vg, 10.0)
    t2 = ml - 10.0
    rng = xs["bands"][band]
    hour = np.round(t / xs["substack_s"]).astype(np.int64)
    Se, So = S * (hour % 2 == 0)[:, None, None], S * (hour % 2 == 1)[:, None, None]
    for SH, STEP in STACKS:
        sub_s = xs["substack_s"]
        starts = [i for i in range(0, len(t) - SH + 1, STEP) if t[i + SH - 1] - t[i] == (SH - 1) * sub_s]
        tc = np.array([t[i] + SH * sub_s for i in starts])  # back-stacked: label = end of window

        def stacks(X):
            cs = np.concatenate([np.zeros((1,) + X.shape[1:], X.dtype), np.cumsum(X, 0)])
            return spectra_to_ccf(np.stack([cs[i + SH] - cs[i] for i in starts]), bins, nfft, fs, ml)[0]

        cur = stacks(S)
        # convergence of each pair (coda CC of stack vs reference minus itself) -> select converging pairs
        la = np.abs(lags)
        conv = np.full(len(P), np.nan)
        for p in np.where(base_ok & (t1 < t2 - 30))[0]:
            m = (la >= t1[p]) & (la <= t2)
            a = cur[:, p][:, m] - cur[:, p][:, m].mean(1, keepdims=True)
            r = (ref[p][None] - cur[:, p])[:, m]
            r = r - r.mean(1, keepdims=True)
            conv[p] = np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30))
        sel = np.where(conv >= CONV_MIN)[0]
        D = pair_dv(cur, ref, sel, t1, t2, dt, rng)
        del cur
        De = pair_dv(stacks(Se), ref, sel, t1, t2, dt, rng)
        Do = pair_dv(stacks(So), ref, sel, t1, t2, dt, rng)
        for gname, mask in GROUPS.items():
            m, n = group_mean(D, mask)
            me, _ = group_mean(De, mask)
            mo, _ = group_mean(Do, mask)
            d = (me - mo).dropna()
            if len(m) < 3 or len(d) < 3:
                continue
            floor = 1.4826 * np.median(np.abs(d - d.median())) / 2
            series.append(pd.DataFrame(dict(band=band, stack_h=SH, group=gname, t=tc[m.index], dv=m.values, n_pairs=n.reindex(m.index).values,
                                            floor=floor)))
            summary.append(dict(band=band, stack_h=SH, group=gname, n_pairs_used=int(mask[sel].sum()), n_times=len(m),
                                std_dv_pct=100 * m.std(), floor_pct=100 * floor, std_over_floor=m.std() / floor if floor > 0 else np.nan,
                                even_odd_median_pct=100 * d.median()))
        print(f"[{band} {SH}h] converging pairs {len(sel)} (of {int(base_ok.sum())} candidates), stacks {len(starts)}", flush=True)
    del S, Se, So

SER = pd.concat(series, ignore_index=True)
SUM = pd.DataFrame(summary)
SER.to_csv(od / "feature_series.csv", index=False)
SUM.to_csv(od / "feature_series_summary.csv", index=False)
pd.set_option("display.width", 200)
print(SUM.round(4).to_string(index=False))

env = pd.read_csv(out / "dvv" / "environment_timeseries.csv", index_col=0)
for SH in sorted(SER.stack_h.unique()):
    gnames = list(GROUPS)
    fig, axs = plt.subplots(len(gnames) + 1, 1, figsize=(14, 2.2 * (len(gnames) + 1)), sharex=True)
    for band, col in zip(xs["bands"], ["C2", "C0"]):
        for ax, gname in zip(axs, gnames):
            g = SER[(SER.stack_h == SH) & (SER.band == band) & (SER.group == gname)].sort_values("t")
            if g.empty:
                continue
            td = pd.to_datetime(g.t, unit="s", utc=True)
            ax.fill_between(td, 100 * (g.dv - g.floor), 100 * (g.dv + g.floor), color=col, alpha=0.2)
            ax.plot(td, 100 * g.dv, ".-", color=col, ms=3, lw=0.9, label=f"{band} (±floor {100 * g.floor.iloc[0]:.3f}%, ~{int(g.n_pairs.median())} pairs)")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.set_ylabel(f"{gname}\ndv/v (%)", fontsize=8)
            ax.legend(fontsize=7, loc="upper left")
    axs[-1].plot(pd.to_datetime(env.index.values, unit="s", utc=True), env.Hs, "k-", lw=0.8)
    axs[-1].set_ylabel("Hs (m)", fontsize=8)
    fig.suptitle(f"South cable across-feature dv/v from inter-span paths, {SH}-h stacks (shaded: even/odd noise floor)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / f"interspan_feature_series_{SH}h.png", dpi=110)
    plt.close(fig)
print("done")
