"""Cross-span (DAS-DAS, south cable) correlation functions of the v3 inter-span run used for the across-feature dv/v (s23).

Per band (b0 0.06-0.15 Hz, b1 0.12-0.30 Hz):
  Figure 1 (interspan_v3_gathers_<band>.png): reference CCFs (Dec 23 - Jan 28) of the converging pairs (24-h
    convergence CC >= 0.3, <= 70 km, C3 excluded; as s23) per path group, averaged in 2-km distance bins
    (each pair normalised), with the ballistic moveout d / vg and the coda start max(1.3 d / vg, 10 s) used by s23.
  Figure 2 (interspan_v3_ccf24h_<band>.png): 24-h back-stacked CCFs (shown every 3 h) against time for the 3 best-
    converging pairs of each group.
Run with MSDVV_CONFIG=config_100m.yaml MSDVV_INTERSPAN=interspan_v3.
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
from scipy.signal import hilbert

from msdvv import interspan_cfg, load_config
from msdvv.dvv import one_sided
from msdvv.xcorr import spectra_to_ccf

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
xs = interspan_cfg(cfg)
out = Path(cfg["output_dir"])
od = out / xs.get("out_subdir", "interspan")
fd = out / "figs"
P = pd.read_csv(od / "pairs.csv")
EXCL = set(xs.get("exclude_segs", []))
CONV_MIN, MAX_KM, SH, EVERY = 0.3, 70.0, 24, 3
FEAT = [c for c in P if c.startswith("x_")]
GROUPS = {"control (no feature)": ~P[FEAT].any(axis=1).values}
GROUPS.update({f[2:]: P[f].values for f in FEAT})
base_ok = (~P.seg_i.isin(EXCL) & ~P.seg_j.isin(EXCL) & (P.dist_km <= MAX_KM)).values

for band, rng in xs["bands"].items():
    T, SP = [], []
    for f in sorted(od.glob("*.h5")):
        with h5py.File(f, "r") as h:
            T.append(h["t_sub"][:])
            SP.append(h[band]["spec"][:])
            bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], h.attrs["nfft"], h.attrs["maxlag_s"]
    t, inv = np.unique(np.concatenate(T), return_inverse=True)
    S = np.zeros((len(t),) + SP[0].shape[1:], np.complex64)
    np.add.at(S, inv, np.concatenate(SP))
    del SP
    ref, lags = spectra_to_ccf(S.sum(0), bins, nfft, fs, ml)
    dt = lags[1] - lags[0]
    env = np.abs(hilbert(ref, axis=-1))
    tpk = np.argmax(one_sided(env, lags, "pos") + one_sided(env, lags, "neg"), axis=-1) * dt
    vpk = P.dist_km.values / np.maximum(tpk, 1e-6)
    vg = float(np.median(vpk[(tpk > 2) & (vpk > 0.3) & (vpk < 5) & base_ok]))
    t1 = np.maximum(1.3 * P.dist_km.values / vg, 10.0)
    t2 = ml - 10.0
    # 24-h back-stacked stacks (every EVERY h) for convergence and the time-lapse panels
    starts = [i for i in range(0, len(t) - SH + 1, EVERY) if t[i + SH - 1] - t[i] == (SH - 1) * 3600]
    tend = np.array([t[i] + SH * 3600 for i in starts])
    cs = np.concatenate([np.zeros((1,) + S.shape[1:], S.dtype), np.cumsum(S, 0)])
    use = np.where(base_ok & (t1 < t2 - 30))[0]
    cur = spectra_to_ccf(np.stack([cs[i + SH] - cs[i] for i in starts])[:, use], bins, nfft, fs, ml)[0]
    del cs, S
    la = np.abs(lags)
    conv = np.full(len(P), np.nan)
    for k, p in enumerate(use):
        m = (la >= t1[p]) & (la <= t2)
        a = cur[:, k][:, m] - cur[:, k][:, m].mean(1, keepdims=True)
        r = (ref[p][None] - cur[:, k])[:, m]
        r = r - r.mean(1, keepdims=True)
        conv[p] = np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30))
    sel = conv >= CONV_MIN
    print(f"[{band}] vg {vg:.2f} km/s; converging pairs {int(sel.sum())} of {int(base_ok.sum())} (<= {MAX_KM:.0f} km)", flush=True)

    # ---- figure 1: distance-binned gathers per group
    fig, axs = plt.subplots(1, len(GROUPS), figsize=(5 * len(GROUPS), 7), sharey=True, squeeze=False)
    edges = np.arange(4, MAX_KM + 2, 2.0)
    for ax, (g, gm) in zip(axs[0], GROUPS.items()):
        m = gm & sel
        Gm = np.full((len(edges) - 1, len(lags)), np.nan)
        cnt = np.zeros(len(edges) - 1, int)
        for b in range(len(edges) - 1):
            q = np.where(m & (P.dist_km.values >= edges[b]) & (P.dist_km.values < edges[b + 1]))[0]
            if len(q):
                rr = ref[q] / (np.abs(ref[q]).max(1, keepdims=True) + 1e-30)
                x = rr.mean(0)
                Gm[b] = x / (np.abs(x).max() + 1e-30)
                cnt[b] = len(q)
        ax.pcolormesh(lags, edges[:-1] + 1, Gm, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
        dd = np.linspace(4, MAX_KM, 50)
        for sg in (1, -1):
            ax.plot(sg * dd / vg, dd, "k--", lw=0.8)
            ax.plot(sg * np.maximum(1.3 * dd / vg, 10), dd, "g-", lw=1.0)
        ax.set(xlim=(-ml, ml), ylim=(4, MAX_KM), xlabel="lag (s)", title=f"{g}\n{int(m.sum())} converging pairs")
    axs[0, 0].set_ylabel("inter-bin distance (km)")
    fig.suptitle(f"Cross-span DAS-DAS reference correlation functions, south cable, {band} {rng[0]}-{rng[1]} Hz (Dec 23 - Jan 28; 2-km distance bins, "
                 f"pairs normalised); black dashed = d/vg (vg {vg:.2f} km/s), green = coda start used for dv/v", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / f"interspan_v3_gathers_{band}.png", dpi=110)
    plt.close(fig)

    # ---- figure 2: 24-h CCFs vs time, best 3 pairs per group
    full = np.arange(tend[0], tend[-1] + 1, 3600.0)  # hourly grid (gaps stay blank)
    td = pd.to_datetime(full, unit="s", utc=True)
    fig, axs = plt.subplots(len(GROUPS), 3, figsize=(17, 3.0 * len(GROUPS)), sharex=True, squeeze=False)
    for i, (g, gm) in enumerate(GROUPS.items()):
        cand = np.where(gm & sel)[0]
        best = cand[np.argsort(conv[cand])[::-1][:3]]
        for j in range(3):
            ax = axs[i, j]
            if j >= len(best):
                ax.axis("off")
                continue
            p = best[j]
            k = int(np.where(use == p)[0][0])
            C = cur[:, k] / (np.abs(cur[:, k]).max(1, keepdims=True) + 1e-30)
            # one row per 3-hourly stack, NaN rows only where data gaps exceed 6 h
            gap = np.r_[np.diff(tend) > 2 * EVERY * 3600, False]
            rows_t, rows_c = [], []
            for tt_, cc_, gp in zip(tend, C, gap):
                rows_t.append(tt_)
                rows_c.append(cc_)
                if gp:
                    rows_t.append(tt_ + EVERY * 3600)
                    rows_c.append(np.full(len(lags), np.nan))
            ax.pcolormesh(lags, pd.to_datetime(np.array(rows_t), unit="s", utc=True), np.array(rows_c), cmap="RdBu_r", vmin=-1, vmax=1,
                          shading="nearest", rasterized=True)
            for sg in (1, -1):
                ax.axvline(sg * P.dist_km[p] / vg, color="k", ls="--", lw=0.8)
                ax.axvline(sg * t1[p], color="g", lw=1.0)
            r = P.iloc[p]
            ax.set_title(f"{g}: S{r.seg_i} bin {r.bin_i} - S{r.seg_j} bin {r.bin_j}, {r.dist_km:.1f} km, conv. CC {conv[p]:.2f}", fontsize=8)
            ax.set_xlim(-ml, ml)
    for ax in axs[-1]:
        ax.set_xlabel("lag (s)")
    fig.suptitle(f"Cross-span DAS-DAS 24-h back-stacked CCFs (shown every {EVERY} h), {band} {rng[0]}-{rng[1]} Hz: 3 best-converging pairs per group; "
                 "black dashed = d/vg, green = coda start", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / f"interspan_v3_ccf24h_{band}.png", dpi=100)
    plt.close(fig)
    del cur
print("done (gathers)")

# ---- figure 3 (interspan_v3_bestpair_<band>.png): stability of the single best-converging pair per group:
# daily 24-h back-stacked CCFs (stack ending 00 UTC each day) as wiggles over the reference, and the CC (coda window)
# of every hourly-stepped 24-h stack with the reference (leave-out reference, as for convergence)
for band, rng in xs["bands"].items():
    T, SP = [], []
    for f in sorted(od.glob("*.h5")):
        with h5py.File(f, "r") as h:
            T.append(h["t_sub"][:])
            SP.append(h[band]["spec"][:])
            bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], h.attrs["nfft"], h.attrs["maxlag_s"]
    t, inv = np.unique(np.concatenate(T), return_inverse=True)
    S = np.zeros((len(t),) + SP[0].shape[1:], np.complex64)
    np.add.at(S, inv, np.concatenate(SP))
    del SP
    ref, lags = spectra_to_ccf(S.sum(0), bins, nfft, fs, ml)
    dt = lags[1] - lags[0]
    env = np.abs(hilbert(ref, axis=-1))
    tpk = np.argmax(one_sided(env, lags, "pos") + one_sided(env, lags, "neg"), axis=-1) * dt
    vpk = P.dist_km.values / np.maximum(tpk, 1e-6)
    vg = float(np.median(vpk[(tpk > 2) & (vpk > 0.3) & (vpk < 5) & base_ok]))
    t1 = np.maximum(1.3 * P.dist_km.values / vg, 10.0)
    t2 = ml - 10.0
    starts = [i for i in range(0, len(t) - SH + 1) if t[i + SH - 1] - t[i] == (SH - 1) * 3600]
    tend = np.array([t[i] + SH * 3600 for i in starts])
    # convergence on 3-hourly stacks to pick the best pair per group (same metric as figures 1-2)
    use = np.where(base_ok & (t1 < t2 - 30))[0]
    cs = np.concatenate([np.zeros((1,) + S.shape[1:], S.dtype), np.cumsum(S, 0)])
    st3 = starts[::EVERY]
    cur3 = spectra_to_ccf(np.stack([cs[i + SH] - cs[i] for i in st3])[:, use], bins, nfft, fs, ml)[0]
    la = np.abs(lags)
    conv = np.full(len(P), np.nan)
    for k, p in enumerate(use):
        m = (la >= t1[p]) & (la <= t2)
        a = cur3[:, k][:, m] - cur3[:, k][:, m].mean(1, keepdims=True)
        r = (ref[p][None] - cur3[:, k])[:, m]
        r = r - r.mean(1, keepdims=True)
        conv[p] = np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30))
    del cur3
    best = {g: int(np.nanargmax(np.where(gm & base_ok, conv, -np.inf))) for g, gm in GROUPS.items()}
    sel_p = list(best.values())
    cur = spectra_to_ccf(np.stack([cs[i + SH] - cs[i] for i in starts])[:, sel_p], bins, nfft, fs, ml)[0]  # hourly step
    del cs, S
    fig, axs = plt.subplots(len(best), 2, figsize=(17, 4.2 * len(best)), gridspec_kw=dict(width_ratios=[1.5, 1]), squeeze=False)
    for i, (g, p) in enumerate(best.items()):
        C = cur[:, i]
        m = (la >= t1[p]) & (la <= t2)
        a = C[:, m] - C[:, m].mean(1, keepdims=True)
        r = (ref[p][None] - C)[:, m]
        r = r - r.mean(1, keepdims=True)
        cc = (a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30)
        rp = P.iloc[p]
        ax = axs[i, 0]
        daily = np.where(np.isclose(tend % 86400, 0))[0]
        rn = ref[p] / np.abs(ref[p]).max()
        for k, j in enumerate(daily):
            y = C[j] / (np.abs(C[j]).max() + 1e-30)
            ax.plot(lags, k + 0.45 * rn, color="0.6", lw=0.6)
            ax.plot(lags, k + 0.45 * y, color="k", lw=0.6)
        ax.plot(lags, len(daily) + 0.8 + 0.9 * rn, color="tab:red", lw=1.2)
        for sg in (1, -1):
            ax.axvspan(*sorted((sg * t1[p], sg * t2)), color="tab:green", alpha=0.08)
            ax.axvline(sg * rp.dist_km / vg, color="tab:blue", ls="--", lw=0.8)
        yt = list(range(0, len(daily), 5))
        ax.set_yticks(yt + [len(daily) + 0.8])
        ax.set_yticklabels([pd.Timestamp(tend[daily[k]], unit="s").strftime("%b %d") for k in yt] + ["reference"], fontsize=7)
        ax.set(xlim=(-ml, ml), xlabel="lag (s)")
        ax.set_title(f"{g}: S{rp.seg_i} bin {rp.bin_i} - S{rp.seg_j} bin {rp.bin_j}, {rp.dist_km:.1f} km; daily 24-h stacks (black) over reference (grey); "
                     f"green = coda window, blue = d/vg", fontsize=8, loc="left")
        ax = axs[i, 1]
        ax.plot(pd.to_datetime(tend, unit="s", utc=True), cc, "k.", ms=2)
        ax.axhline(np.nanmedian(cc), color="tab:red", lw=1)
        ax.set(ylim=(-0.2, 1), ylabel="CC (coda window)")
        ax.set_title(f"24-h stacks (1-h step) vs leave-out reference: median CC {np.nanmedian(cc):.2f}, 10-90 % {np.nanpercentile(cc, 10):.2f}-{np.nanpercentile(cc, 90):.2f}", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(f"Cross-span DAS-DAS, {band} {rng[0]}-{rng[1]} Hz: best-converging pair of each path group", fontsize=11)
    fig.tight_layout()
    fig.savefig(fd / f"interspan_v3_bestpair_{band}.png", dpi=105)
    plt.close(fig)
    print(f"[{band}] best pairs:", {g: (f"{P.dist_km[p]:.1f} km", round(float(conv[p]), 2)) for g, p in best.items()}, flush=True)
print("done (best pairs)")
