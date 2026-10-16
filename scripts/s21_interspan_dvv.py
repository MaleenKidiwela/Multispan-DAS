"""Inter-span analysis: gathers, tonal-bin rejection, convergence, pair dv/v, feature-wise dv/v by regression.

Model per time step (weighted least squares over pairs):
    dv_p(t) = b(t) + sum_f x_pf * delta_f(t)
x_pf = 1 if pair p's path crosses feature f (fault trace, or within radius of a point feature).
b(t) absorbs the common (noise-source) signal; delta_f(t) is the extra dv/v of paths through f.
Formal errors are rescaled by the misfit but still too small (pairs share bins); use s22 (random-line null).

Run with MSDVV_INTERSPAN=<variant> to use a config variant (e.g. interspan_v2).
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
tag = xs.get("out_subdir", "interspan")
P = pd.read_csv(od / "pairs.csv")
FEAT = [c for c in P if c.startswith("x_")]
STACKS = [tuple(s) for s in xs.get("stacks_h", [[24, 6]])]
CC_MIN = 0.4
TONAL = xs.get("tonal_peakiness_factor")


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
    setup = pd.Series(setups).groupby(inv).first().values
    return ut, S, setup, bins, fs, nfft, ml


def stretch_chunk(cur_c, ref_c, t1_c, t2, dt, rng, pairs_c):
    rows = []
    for k, p in enumerate(pairs_c):
        for side in ("pos", "neg"):
            c = cur_c[:, k]
            st = stretch(c, np.broadcast_to(ref_c[k], c.shape), dt, t1_c[k], t2, 0.01, 101, rng)
            rows.append(pd.DataFrame(dict(seg=np.arange(len(c)), pair=p, side=side, dv=st["dv"], cc=st["cc"], err=st["err"],
                                          bound=st["at_bound"])))
    return pd.concat(rows) if rows else pd.DataFrame()


res_all, flagged_all = [], {}
for band in xs["bands"]:
    t, S, setup, bins, fs, nfft, ml = load_band(band)
    # --- tonal-bin rejection: a persistent tone stacks coherently and makes the long-term spectrum peaky
    A = np.abs(S.sum(0))
    peaky = A.max(1) / (np.median(A, 1) + 1e-30)
    bp = pd.concat([pd.DataFrame(dict(bin=P.bin_i, pk=peaky)), pd.DataFrame(dict(bin=P.bin_j, pk=peaky))]).groupby("bin").pk.median()
    flagged = bp.index[bp > (TONAL or np.inf) * bp.median()].tolist()
    flagged_all[band] = flagged
    tonal_pair = (P.bin_i.isin(flagged) | P.bin_j.isin(flagged)).values
    P[f"tonal_{band}"] = tonal_pair
    ref, lags = spectra_to_ccf(S.sum(0), bins, nfft, fs, ml)
    dt = lags[1] - lags[0]
    env = np.abs(hilbert(ref, axis=-1))
    sym = one_sided(env, lags, "pos") + one_sided(env, lags, "neg")
    tpk = np.argmax(sym, axis=-1) * dt
    vpk = P.dist_km.values / np.maximum(tpk, 1e-6)
    ok = (tpk > 2) & (vpk < 5) & (vpk > 0.3) & ~tonal_pair
    vg = np.median(vpk[ok])
    t1 = np.maximum(1.3 * P.dist_km.values / vg, 10.0)
    t2 = ml - 10.0
    usable = (t1 < t2 - 30) & ~tonal_pair
    print(f"[{band}] group velocity {vg:.2f} km/s ({ok.sum()} picks); tonal bins dropped {len(flagged)} "
          f"({tonal_pair.sum()} pairs); usable pairs {usable.sum()}", flush=True)
    o = np.argsort(P.dist_km.values)
    fig, ax = plt.subplots(figsize=(9, 6))
    Gm = ref[o] / (np.abs(ref[o]).max(1, keepdims=True) + 1e-30)
    Gm[tonal_pair[o]] = 0
    ax.pcolormesh(lags, P.dist_km.values[o], Gm, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
    for s in (+1, -1):
        ax.plot(s * np.linspace(0, P.dist_km.max(), 10) / vg, np.linspace(0, P.dist_km.max(), 10), "k--", lw=0.8)
    ax.set(xlabel="lag (s)", ylabel="inter-bin distance (km)", title=f"{tag}: reference CCFs {band} {xs['bands'][band]} Hz (tonal pairs blanked; {vg:.2f} km/s)")
    fig.tight_layout()
    fig.savefig(fd / f"{tag}_gather_{band}.png", dpi=110)
    plt.close(fig)
    sub_s = xs["substack_s"]
    rng = xs["bands"][band]
    lag_abs = np.abs(lags)
    for SH, STEP in STACKS:
        starts = [i for i in range(0, len(t) - SH + 1, STEP) if t[i + SH - 1] - t[i] == (SH - 1) * sub_s]
        if len(starts) < 3:
            continue
        tc = np.array([t[i] + SH * sub_s for i in starts])  # back-stacked: label = end of window
        cs = np.concatenate([np.zeros((1,) + S.shape[1:], S.dtype), np.cumsum(S, 0)])
        cur, _ = spectra_to_ccf(np.stack([cs[i + SH] - cs[i] for i in starts]), bins, nfft, fs, ml)
        ccv = np.full((len(starts), len(P)), np.nan)
        for p in np.where(usable)[0]:
            m = (lag_abs >= t1[p]) & (lag_abs <= t2)
            a = cur[:, p][:, m] - cur[:, p][:, m].mean(1, keepdims=True)
            r = (ref[p][None] - cur[:, p])[:, m]
            r = r - r.mean(1, keepdims=True)
            ccv[:, p] = (a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30)
        P[f"conv_cc_{band}_{SH}h"] = np.nanmedian(ccv, 0)
        up = np.where(usable)[0]
        chunks = np.array_split(up, 32)
        parts = Parallel(n_jobs=16, prefer="threads")(
            delayed(stretch_chunk)(cur[:, ch], ref[ch], t1[ch], t2, dt, rng, ch) for ch in chunks if len(ch))
        M = pd.concat(parts)
        M = M[(M.cc >= CC_MIN) & ~M.bound]
        w = 1 / M.err**2
        pdv = M.assign(w=w, wdv=w * M.dv).groupby(["seg", "pair"]).agg(wsum=("w", "sum"), wdv=("wdv", "sum")).reset_index()
        pdv["dv"] = pdv.wdv / pdv.wsum
        pdv["err"] = 1 / np.sqrt(pdv.wsum)
        pdv["t"] = tc[pdv.seg]
        pdv["setup"] = setup[np.array(starts)[pdv.seg]]
        pdv["band"], pdv["stack_h"] = band, SH
        res_all.append(pdv.merge(P[["dist_km", "seg_i", "seg_j"] + FEAT], left_on="pair", right_index=True))
        print(f"[{band} {SH}h] stacks {len(starts)}, pair-stack dv/v kept {len(pdv)}; median convergence CC "
              f"{np.nanmedian(P[f'conv_cc_{band}_{SH}h']):.2f}", flush=True)
        del cur
    del S

D = pd.concat(res_all, ignore_index=True)
D.to_csv(od / "pair_dvv.csv.gz", index=False)
P.to_csv(od / "pairs_with_stats.csv", index=False)
pd.Series({b: f for b, f in flagged_all.items()}).to_json(od / "tonal_bins.json")


def regress(g):
    X = np.c_[np.ones(len(g)), g[FEAT].values.astype(float)]
    W = 1 / np.maximum(g.err.values, 1e-5) ** 2
    if len(g) < 3 * X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return pd.Series({k: np.nan for k in ["b"] + FEAT + [f"se_{c}" for c in FEAT]})
    XtW = X.T * W
    cov = np.linalg.pinv(XtW @ X)
    beta = cov @ (XtW @ g.dv.values)
    r = g.dv.values - X @ beta
    s2 = np.sum(W * r**2) / max(len(g) - X.shape[1], 1)
    se = np.sqrt(np.diag(cov) * s2)
    return pd.Series(dict(b=beta[0], **{c: beta[i + 1] for i, c in enumerate(FEAT)}, **{f"se_{c}": se[i + 1] for i, c in enumerate(FEAT)}))


R = D.groupby(["band", "stack_h", "t"]).apply(regress).reset_index()
R.to_csv(od / "feature_dvv.csv", index=False)
env = pd.read_csv(out / "dvv" / "environment_timeseries.csv", index_col=0)
for SH in sorted(R.stack_h.unique()):
    RR = R[R.stack_h == SH]
    fig, axs = plt.subplots(len(FEAT) + 2, 1, figsize=(14, 2.3 * (len(FEAT) + 2)), sharex=True)
    for band, col in zip(xs["bands"], ["C2", "C0", "C3"]):
        g = RR[RR.band == band].sort_values("t")
        if g.empty:
            continue
        td = pd.to_datetime(g.t, unit="s", utc=True)
        axs[0].plot(td, 100 * g.b, color=col, label=band)
        for ax, c in zip(axs[1:], FEAT):
            ax.errorbar(td, 100 * g[c], yerr=100 * g[f"se_{c}"], fmt=".-", color=col, ms=3, lw=0.8, elinewidth=0.4, label=band)
            ax.axhline(0, color="0.7", lw=0.5)
            ax.set_ylabel(f"{c[2:]}\nextra dv/v (%)", fontsize=8)
    axs[0].set_ylabel("common term (%)", fontsize=8)
    axs[-1].plot(pd.to_datetime(env.index.values, unit="s", utc=True), env.Hs, "k-", lw=0.8)
    axs[-1].set_ylabel("Hs (m)", fontsize=8)
    for ax in axs:
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{tag}: south-cable inter-span dv/v, {SH}-h stacks — common term and extra dv/v of paths crossing each feature", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / f"{tag}_feature_dvv_{SH}h.png", dpi=110)
    plt.close(fig)
print("done")
