"""Step 5: span averages, span-level noise floor, tide regression, run offsets and setup check, figures."""
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
from msdvv.dvv import weighted_mean

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
PLEV = cfg["dvv"].get("analysis_level", cfg["dvv"]["levels"][-1])  # bin size used for span-level products
out = Path(cfg["output_dir"])
dd, fd = out / cfg["dvv"]["out_subdir"], out / "figs" / ("" if cfg["dvv"]["out_subdir"] == "dvv" else cfg["dvv"]["out_subdir"])
fd.mkdir(parents=True, exist_ok=True)
MIN_BINS_KM = 3.0  # spans with less usable length are not reported

dv = pd.read_csv(dd / "dvv.csv.gz")
STACK_S = cfg["dvv"]["stack_subs"] * cfg["xcorr"]["substack_s"]
if "tlabel" not in dv:  # legacy output labelled at the window centre -> back-stacked (end) label
    dv["t"] = dv["t"] + STACK_S / 2
null = pd.read_csv(dd / "null.csv.gz")
bins = {(c, l): pd.read_csv(dd / "refs" / f"bins_{c}_{l}.csv") for c in ("north", "south") for l in cfg["dvv"]["levels"]}


def add_span(df):
    parts = []
    for (c, l), g in df.groupby(["cable", "level"]):
        b = bins[(c, l)][["bin", "block", "span", "opt_km", "depth"]]
        parts.append(g.merge(b, on="bin", how="left"))
    x = pd.concat(parts, ignore_index=True)
    x["seg_id"] = x.block + x.span.astype(str)
    return x


dv = add_span(dv[dv.side == "both"])
null = add_span(null)

# ---- span-averaged dv/v (inverse-variance over bins) per time
keys = ["cable", "band", "level", "window", "run", "seg", "t", "setup", "seg_id"]
HAS_POW = "logpow" in dv.columns
def span_avg(g):
    m, e, n = weighted_mean(g.dv.values, g.err.values, 0)
    lp = np.nanmean(g.logpow.values) if HAS_POW else np.nan
    return pd.Series(dict(dv=m, err=e, n_bins=n, depth=g.depth.mean(), opt_km=g.opt_km.mean(), logpow=lp))
sp = dv.groupby(keys).apply(span_avg).reset_index()
sp[["dv", "err", "n_bins", "depth", "opt_km", "logpow"]] = sp[["dv", "err", "n_bins", "depth", "opt_km", "logpow"]].astype(float)

# ---- span-level null test: even/odd halves averaged over bins the same way
def span_null(g):
    me, ee, _ = weighted_mean(g.dv_even.values, g.err_even.values, 0)
    mo, eo, n = weighted_mean(g.dv_odd.values, g.err_odd.values, 0)
    return pd.Series(dict(d=me - mo, err=np.sqrt((ee**2 + eo**2) / 2), n_bins=n))
spn = null.groupby(["cable", "band", "level", "window", "run", "seg", "seg_id"]).apply(span_null).reset_index()
spn[["d", "err", "n_bins"]] = spn[["d", "err", "n_bins"]].astype(float)
floor = spn.dropna(subset=["d"]).groupby(["cable", "band", "level", "window", "seg_id"]).agg(
    N=("d", "size"), n_bins=("n_bins", "median"),
    floor_span=("d", lambda d: 1.4826 * np.median(np.abs(d - d.median())) / np.sqrt(2)),
    formal_span=("err", "median")).reset_index()
binfloor = null.assign(d=null.dv_even - null.dv_odd).dropna(subset=["d"]).groupby(
    ["cable", "band", "level", "window", "seg_id"]).d.apply(lambda d: 1.4826 * np.median(np.abs(d - d.median())) / np.sqrt(2)).rename("floor_bin").reset_index()
floor = floor.merge(binfloor, on=["cable", "band", "level", "window", "seg_id"])
floor["sqrtN_gain_expected"] = np.sqrt(floor.n_bins)
floor["gain_observed"] = floor.floor_bin / floor.floor_span
floor["reliable"] = floor.N >= 8  # number of independent 60-min even/odd blocks
floor.to_csv(dd / "noise_floor.csv", index=False)

# ---- tide regression (within-run demeaned), using span-level noise floor as sigma
def tide_table():
    """NOAA CO-OPS predictions (6-min, MSL, GMT) covering the data, cached in resources/."""
    st = cfg["tides"]["noaa_station"]
    t0 = pd.Timestamp(dv.t.min(), unit="s") - pd.Timedelta(days=1)
    t1 = pd.Timestamp(dv.t.max(), unit="s") + pd.Timedelta(days=1)
    path = Path(cfg["resources_dir"]) / f"noaa_{st}_predictions_{t0:%Y%m%d}_{t1:%Y%m%d}.csv"
    legacy = Path(cfg["resources_dir"]) / "noaa_9435380_predictions_jan2026.csv"  # pilot window Jan 15-28
    if not path.exists() and legacy.exists() and t0 >= pd.Timestamp("2026-01-15") and t1 <= pd.Timestamp("2026-01-28"):
        path = legacy
    if not path.exists():
        import requests
        r = requests.get("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter", timeout=60, params=dict(
            product="predictions", datum="MSL", station=st, begin_date=f"{t0:%Y%m%d}", end_date=f"{t1:%Y%m%d}",
            time_zone="gmt", units="metric", interval="6", format="csv", application="msdvv_pilot"))
        r.raise_for_status()
        path.write_text(r.text)
    t = pd.read_csv(path)
    t.columns = ["time", "h"]
    return t


tide = tide_table()
tt = pd.to_datetime(tide.time, utc=True).astype("int64").values / 1e9
# tide averaged over each (trailing) stack window [t - stack, t]; a 6-h stack attenuates the 12.4-h tide by ~1/3
ct = np.concatenate([[0.0], np.cumsum(0.5 * (tide.h.values[1:] + tide.h.values[:-1]) * np.diff(tt))])
sp["tide"] = (np.interp(sp.t, tt, ct) - np.interp(sp.t - STACK_S, tt, ct)) / STACK_S
rows = []
for key, g in sp.dropna(subset=["dv"]).groupby(["cable", "band", "level", "window", "seg_id"]):
    g = g.copy()
    g["dv_d"] = g.dv - g.groupby("run").dv.transform("mean")
    g["tide_d"] = g.tide - g.groupby("run").tide.transform("mean")
    if len(g) < 10 or g.tide_d.std() == 0:
        continue
    beta = np.sum(g.dv_d * g.tide_d) / np.sum(g.tide_d**2)
    neff = len(g) / (cfg["dvv"]["stack_subs"] / cfg["dvv"]["step_subs"])  # moving stacks are oversampled
    # standard error from regression residuals (the span null floor is unreliable for short south runs)
    res = g.dv_d - beta * g.tide_d
    se = res.std() / np.sqrt(np.sum(g.tide_d**2)) * np.sqrt(len(g) / neff)
    row = dict(zip(["cable", "band", "level", "window", "seg_id"], key), n=len(g), beta_per_m=beta, se_per_m=se,
               z=beta / se, r=np.corrcoef(g.dv_d.values, g.tide_d.values)[0, 1],
               tide_range_m=g.tide.max() - g.tide.min())
    if HAS_POW and g.logpow.notna().sum() > 10:
        # joint regression dv = a*tide + b*log10(noise power): is the dv/v variability source-driven?
        g = g[g.logpow.notna()].copy()
        # demean per acquisition setup too: amplitude scales with the internal decimation D (D128 vs D256)
        g["pow_d"] = g.logpow - g.groupby(["run", "setup"]).logpow.transform("mean")
        X = np.c_[g.tide_d.values, g.pow_d.values]
        coef, *_ = np.linalg.lstsq(X, g.dv_d.values, rcond=None)
        resid = g.dv_d.values - X @ coef
        cov = np.linalg.inv(X.T @ X) * resid.var(ddof=2) * (cfg["dvv"]["stack_subs"] / cfg["dvv"]["step_subs"])
        se2 = np.sqrt(np.diag(cov))
        row.update(beta_tide_joint=coef[0], z_tide_joint=coef[0] / se2[0], beta_per_dex=coef[1], se_per_dex=se2[1],
                   z_pow=coef[1] / se2[1], r_pow=np.corrcoef(g.dv_d.values, g.pow_d.values)[0, 1],
                   var_explained_pow=np.corrcoef(g.dv_d.values, g.pow_d.values)[0, 1] ** 2)
    rows.append(row)
tides = pd.DataFrame(rows)
tides.to_csv(dd / "tide_regression.csv", index=False)

# ---- per-run mean dv/v (vs all-January reference) and setup change check
runm = sp.dropna(subset=["dv"]).groupby(["cable", "band", "level", "window", "seg_id", "run", "setup"]).agg(
    dv=("dv", "mean"), t=("t", "mean"), n=("dv", "size")).reset_index()
runm.to_csv(dd / "run_means.csv", index=False)
sp.to_csv(dd / "span_timeseries.csv.gz", index=False)

pd.set_option("display.width", 200)
PW = cfg["dvv"].get("primary_window", "early")
BANDS = list(cfg["xcorr"]["bands"])
BAND_COLORS = list(zip(BANDS, ["C0", "C3", "C2", "C1"]))
print(f"== span-level noise floor (one stack length), 5-km bins, all windows")
print(floor[(floor.level == PLEV)].pivot_table(index=["cable", "seg_id"], columns=["band", "window"], values="floor_span").round(4).to_string())
print(f"\n== gain from span averaging: observed vs sqrt(n_bins) (5km, {PW})")
print(floor[(floor.level == PLEV) & (floor.window == PW)][["cable", "band", "seg_id", "n_bins", "floor_bin", "floor_span", "gain_observed", "sqrtN_gain_expected"]].round(4).to_string())
print(f"\n== tide regression (5km, {PW}): dv/v per metre of tide")
print(tides[(tides.level == PLEV) & (tides.window == PW)].round(5).to_string())
print(f"\n== run means (5km, {PW}, b2)")
print(runm[(runm.level == PLEV) & (runm.window == PW) & (runm.band == "b2")].pivot_table(index=["cable", "seg_id"], columns=["run"], values="dv").round(4).to_string())

# ---- figures: span time series with tide (5km, primary window) per cable
for cab in ("north", "south"):
    s = sp[(sp.cable == cab) & (sp.level == PLEV) & (sp.window == PW)]
    segs = [sid for sid in sorted(s.seg_id.unique())
            if bins[(cab, PLEV)].pipe(lambda b: (b.block + b.span.astype(str) == sid).sum()) * float(PLEV[:-2]) >= MIN_BINS_KM]
    runs = sorted(s.run.unique())
    fig, axs = plt.subplots(len(segs), len(runs), figsize=(max(3.2 * len(runs), 14), 2.0 * len(segs) + 1), sharey="row", squeeze=False)
    for i, sid in enumerate(segs):
        for j, run in enumerate(runs):
            ax = axs[i, j]
            single = len(runs) == 1  # long merged series: date axis
            for band, col in BAND_COLORS:
                g = s[(s.seg_id == sid) & (s.run == run) & (s.band == band)].sort_values("t")
                if len(g) == 0:
                    continue
                th = pd.to_datetime(g.t, unit="s") if single else (g.t - g.t.min()) / 3600
                ax.errorbar(th, 100 * g.dv, yerr=100 * g.err, fmt=".-", color=col, ms=3, lw=0.8, elinewidth=0.4, label=band)
            g = s[(s.seg_id == sid) & (s.run == run)]
            if len(g):
                ax2 = ax.twinx()
                tg = np.linspace(g.t.min(), g.t.max(), 2000 if single else 100)
                ax2.plot(pd.to_datetime(tg, unit="s") if single else (tg - g.t.min()) / 3600,
                         np.interp(tg, tt, tide.h.values), "k-", lw=0.4 if single else 0.6, alpha=0.4)
                ax2.set_ylim(-2, 2)
                ax2.set_yticks([] if j < len(runs) - 1 else [-1, 0, 1])
                if j == len(runs) - 1:
                    ax2.set_ylabel("tide (m)", fontsize=7)
            ax.axhline(0, color="0.7", lw=0.5)
            if single:  # shade periods recorded with a different acquisition setup
                gs = s[(s.seg_id == sid) & (s.run == run)].sort_values("t")
                if gs.setup.nunique() > 1:
                    major = gs.setup.mode().iloc[0]
                    odd = gs[gs.setup != major]
                    ax.axvspan(pd.to_datetime(odd.t.min(), unit="s"), pd.to_datetime(odd.t.max(), unit="s"),
                               color="0.85", zorder=0, label=f"{odd.setup.iloc[0]} (decimated)" if i == 0 else None)
            if i == 0:
                t0 = pd.Timestamp(s[s.run == run].t.min(), unit="s", tz="UTC")
                ax.set_title(f"{run} {t0:%m-%d %H:%M}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"span {sid}\ndv/v (%)", fontsize=8)
            if i == len(segs) - 1:
                ax.set_xlabel("UTC date" if single else "hours", fontsize=8)
    axs[0, 0].legend(fontsize=7)
    stack_h = cfg["dvv"]["stack_subs"] * cfg["xcorr"]["substack_s"] / 3600
    fig.suptitle(f"{cab}: span-averaged dv/v ({stack_h:g}-h stacks, 5-km bins, {PW} window) with NOAA tide (grey)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / f"dvv_timeseries_{cab}.png", dpi=120)
    plt.close(fig)

# ---- precision map: bin-level null floor vs distance
fig, axs = plt.subplots(2, 1, figsize=(12, 7))
for ax, cab in zip(axs, ("north", "south")):
    n = null[(null.cable == cab) & (null.level == PLEV)].assign(d=lambda x: x.dv_even - x.dv_odd)
    for (band, win), g in n.groupby(["band", "window"]):
        f = g.dropna(subset=["d"]).groupby("bin").agg(opt_km=("opt_km", "first"),
                                                       fl=("d", lambda d: 1.4826 * np.median(np.abs(d - d.median())) / np.sqrt(2)))
        ax.plot(f.opt_km, 100 * f.fl, ".", ms=5, label=f"{band} {win}")
    ax.set(yscale="log", ylabel="null-test dv/v noise (%)", xlabel="optical distance (km)",
           title=f"{cab}: empirical 30-min dv/v precision per 5-km bin (even/odd null test)")
    ax.legend(fontsize=7, ncol=6)
fig.tight_layout()
fig.savefig(fd / "dvv_precision_map.png", dpi=120)

# ---- dv/v vs local noise power (source-change diagnostic), primary window, 5 km
if HAS_POW and sp.logpow.notna().any():
    for cab in ("north", "south"):
        s = sp[(sp.cable == cab) & (sp.level == PLEV) & (sp.window == PW)].dropna(subset=["dv", "logpow"])
        segs = sorted(s.seg_id.unique())
        fig, axs = plt.subplots(len(BANDS), len(segs), figsize=(2.8 * len(segs), 2.8 * len(BANDS)), squeeze=False)
        for i, band in enumerate(BANDS):
            for j, sid in enumerate(segs):
                g = s[(s.seg_id == sid) & (s.band == band)]
                ax = axs[i, j]
                if len(g) > 2:
                    sc = ax.scatter(g.logpow - g.logpow.mean(), 100 * (g.dv - g.dv.mean()), c=pd.to_datetime(g.t, unit="s").map(pd.Timestamp.toordinal), s=4, cmap="viridis")
                    rr = np.corrcoef(g.logpow, g.dv)[0, 1]
                    ax.set_title(f"{sid} {band}  r={rr:.2f}", fontsize=8)
                ax.set_xlabel("log10 noise power (demeaned)", fontsize=7)
                if j == 0:
                    ax.set_ylabel("dv/v (%)", fontsize=8)
        fig.suptitle(f"{cab}: dv/v ({PW} window) vs in-band noise power, colour = time", fontsize=10)
        fig.tight_layout()
        fig.savefig(fd / f"dvv_vs_power_{cab}.png", dpi=110)
        plt.close(fig)
print("figs ok")
