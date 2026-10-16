"""Remove noise-source effects from the within-span dv/v (5-km bins, primary window), then test what remains.

Source proxies, measured from the correlations themselves (not the ocean forcing):
  asym_coda   = (E+ - E-)/(E+ + E-) of coda energy (directionality of the scattered field)
  asym_direct = same in the direct-wave window (directionality of the ballistic noise)
  logpow      = log10 in-band noise power (span mean)
Model per span & band: dv = c + b1*asym_coda + b2*asym_direct + b3*logpow (+ per-setup offsets) -> residual.
Then: residual vs wave height Hs (NDBC), shelf vs deep spans, and the Jan 8-9 storm step test on residuals
with an empirical null (same statistic at all other times).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv import load_config
from msdvv.stack import contiguous_segments, lag_windows

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
D = cfg["dvv"]
out = Path(cfg["output_dir"])
dd, fd = out / D["out_subdir"], out / "figs" / ("" if D["out_subdir"] == "dvv" else D["out_subdir"])
fd.mkdir(parents=True, exist_ok=True)
LEVEL, PW = D.get("analysis_level", D["levels"][-1]), D.get("primary_window", "coda")
sp = pd.read_csv(dd / "span_timeseries.csv.gz")
sp = sp[(sp.level == LEVEL) & (sp.window == PW)].copy()
env = pd.read_csv(dd / "environment_timeseries.csv", index_col=0)
S, STEP = D["stack_subs"], D["step_subs"]


def asymmetry(cab, band):
    """Per (seg, span) coda and direct-window lag-energy asymmetry from the stored 1-h bin CCF substacks."""
    sub = np.load(dd / "refs" / f"sub_{cab}_{band}_{LEVEL}.npy", mmap_mode="r")
    M = np.load(dd / "refs" / f"submeta_{cab}_{band}_{LEVEL}.npz")
    R = np.load(dd / "refs" / f"{cab}_{band}_{LEVEL}.npz")
    lags, ks, t_sub, dx = R["lags"], R["ks"], M["t_sub"], float(M["dx_km"])
    bins = pd.read_csv(dd / "refs" / f"bins_{cab}_{LEVEL}.csv")
    bins["seg_id"] = bins.block + bins.span.astype(str)
    starts = contiguous_segments(t_sub, S, float(M["substack_s"]), STEP)
    masks = {}
    for w in ("coda", "direct"):
        mp = np.zeros((len(ks), len(lags)), bool)
        mn = np.zeros_like(mp)
        for kk, k in enumerate(ks):
            t1, t2 = lag_windows(k * dx, D, band)[w]
            mp[kk] = (lags >= t1) & (lags <= t2)
            mn[kk] = (lags <= -t1) & (lags >= -t2)
        masks[w] = (mp, mn)
    cs = np.cumsum(np.concatenate([np.zeros((1,) + sub.shape[1:], np.float64), np.asarray(sub, np.float64) ** 2]), 0)
    rows = []
    for g, s0 in enumerate(starts):
        E = cs[s0 + S] - cs[s0]  # (nk, nbin, nlag) energy of the 6-h stack... (sum of hourly energies)
        for sid, b in bins.groupby("seg_id"):
            r = dict(seg=g, seg_id=sid, t=t_sub[s0] + S * float(M["substack_s"]) / 2)
            for w, (mp, mn) in masks.items():
                ep = sum(E[kk][b.bin.values][:, mp[kk]].sum() for kk in range(len(ks)))
                en = sum(E[kk][b.bin.values][:, mn[kk]].sum() for kk in range(len(ks)))
                r[f"asym_{w}"] = (ep - en) / (ep + en + 1e-30)
            rows.append(r)
    return pd.DataFrame(rows).assign(cable=cab, band=band)


A = pd.concat([asymmetry(c, b) for c in ("north", "south") for b in cfg["xcorr"]["bands"]], ignore_index=True)
X = sp.merge(A, on=["cable", "band", "seg_id", "t"], how="inner").dropna(subset=["dv", "logpow", "asym_coda", "asym_direct"])
hs = env.reindex(X.t.values, method="nearest", tolerance=1800)
X["Hs"] = hs.Hs.values
X["depth_class"] = np.where(X.depth < 700, "shelf", "deep")
neff_fac = S / STEP
rows, parts = [], []
for (cab, band, sid), g in X.groupby(["cable", "band", "seg_id"]):
    g = g.sort_values("t").copy()
    Z = [(g[c] - g[c].mean()) / (g[c].std() + 1e-12) for c in ("asym_coda", "asym_direct", "logpow")]
    Xm = np.c_[np.ones(len(g)), np.array(Z).T]
    if g.setup.nunique() > 1:  # per-setup offset (D128 period)
        Xm = np.c_[Xm, (g.setup != g.setup.mode().iloc[0]).astype(float)]
    coef, *_ = np.linalg.lstsq(Xm, g.dv.values, rcond=None)
    g["resid"] = g.dv.values - Xm @ coef + coef[0]
    r2 = 1 - np.var(g.dv.values - Xm @ coef) / np.var(g.dv.values)
    ok = g.Hs.notna()
    r_hs_raw = np.corrcoef(g.dv[ok], g.Hs[ok])[0, 1]
    r_hs_res = np.corrcoef(g.resid[ok], g.Hs[ok])[0, 1]
    rows.append(dict(cable=cab, band=band, span=sid, depth_m=round(g.depth.mean()), cls=g.depth_class.iloc[0], n=len(g),
                     R2_source=r2, std_raw_pct=100 * g.dv.std(), std_resid_pct=100 * g.resid.std(),
                     b_asym_coda=coef[1], b_asym_direct=coef[2], b_logpow=coef[3], r_Hs_raw=r_hs_raw, r_Hs_resid=r_hs_res,
                     z_r_Hs_resid=r_hs_res * np.sqrt(max(len(g) / neff_fac - 3, 1))))
    parts.append(g)
T = pd.DataFrame(rows)
Y = pd.concat(parts, ignore_index=True)
Y.to_csv(dd / "source_corrected_span_dvv.csv.gz", index=False)
T.to_csv(dd / "source_correction_summary.csv", index=False)
pd.set_option("display.width", 220)
print(T.round(3).to_string(index=False))


# storm step test on residuals (Jan 8 18:00 UTC, 36 h before vs 24 h during), empirical null over all times
def step(tt, v, tc, pre=36, post=24):
    a = v[(tt >= tc - pre * 3600) & (tt < tc)]
    b = v[(tt >= tc) & (tt < tc + post * 3600)]
    return b.mean() - a.mean() if len(a) > 10 and len(b) > 10 else np.nan


STORMS = cfg.get("storms", ["2026-01-08 18:00"])  # storm onsets (UTC); step = mean(after 24 h) - mean(before 36 h)
srows = []
for storm in STORMS:
    t_storm = pd.Timestamp(storm, tz="UTC").timestamp()
    for (cab, band, sid), g in Y.groupby(["cable", "band", "seg_id"]):
        tt, v = g.t.values, g.resid.values
        j = step(tt, v, t_storm)
        cands = np.arange(tt.min() + 36 * 3600, tt.max() - 24 * 3600, 6 * 3600)
        nul = np.array([step(tt, v, c) for c in cands if all(abs(c - pd.Timestamp(s_, tz="UTC").timestamp()) > 3 * 86400 for s_ in STORMS)])
        nul = nul[np.isfinite(nul)]
        srows.append(dict(storm=storm, cable=cab, band=band, span=sid, cls=g.depth_class.iloc[0], jump_resid_pct=100 * j,
                          null_std_pct=100 * nul.std() if len(nul) else np.nan, z=j / nul.std() if len(nul) > 5 else np.nan))
ST = pd.DataFrame(srows)
ST.to_csv(dd / "storm_test_resid.csv", index=False)
print(f"\n== storm steps on source-corrected residuals: {STORMS}")
print(ST.round(3).to_string(index=False))
print("\nmean z by depth class and band:")
print(ST.groupby(["storm", "band", "cls"]).z.agg(["mean", "count"]).round(2).to_string())

# figure: raw vs residual per span, b2 & b1, with Hs
for cab in ("north", "south"):
    g0 = Y[Y.cable == cab]
    spans = sorted(g0.seg_id.unique())
    fig, axs = plt.subplots(len(spans) + 1, 1, figsize=(14, 1.9 * (len(spans) + 1)), sharex=True)
    for ax, sid in zip(axs, spans):
        for band, col in zip(cfg["xcorr"]["bands"], ["C0", "C3", "C2"]):
            g = g0[(g0.seg_id == sid) & (g0.band == band)].sort_values("t")
            if g.empty:
                continue
            td = pd.to_datetime(g.t, unit="s", utc=True)
            ax.plot(td, 100 * g.dv, color=col, lw=0.6, alpha=0.35)
            ax.plot(td, 100 * g.resid, color=col, lw=1.0, label=f"{band} source-corrected")
        for st_ in STORMS:  # 36 h before / 24 h after each storm onset (the step-test windows)
            ax.axvspan(pd.Timestamp(st_, tz="UTC"), pd.Timestamp(st_, tz="UTC") + pd.Timedelta(hours=24), color="0.88", zorder=0)
        dep = g0[g0.seg_id == sid].depth.mean()
        ax.set_ylabel(f"{sid}\n{dep:.0f} m\ndv/v %", fontsize=7)
        ax.legend(fontsize=6, loc="upper left")
    axs[-1].plot(pd.to_datetime(env.index.values, unit="s", utc=True), env.Hs, "k-", lw=0.8)
    axs[-1].set_ylabel("Hs (m)", fontsize=8)
    fig.suptitle(f"{cab}: within-span dv/v ({PW}, {LEVEL} bins, {S}-h back-stacked) raw (faint) vs source-corrected (asymmetry + noise power); grey = 24 h after storm onsets", fontsize=9)
    fig.tight_layout()
    fig.savefig(fd / f"dvv_source_corrected_{cab}.png", dpi=110)
    plt.close(fig)
print("figs ok")
