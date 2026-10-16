"""Step 6: compare dv/v and DAS noise power with ocean conditions (NDBC buoys).

Buoys: 46050 (Stonewall Bank, off Newport) and 46089 (Tillamook, offshore), files in resources/ndbc/
(historical annual file for 2025, monthly file for Jan 2026). Significant wave height WVHT and
dominant period DPD are averaged over each dv/v stack window and compared with the span noise
power and with the first principal component (common mode) of the span dv/v series.
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
PLEV = cfg["dvv"].get("analysis_level", cfg["dvv"]["levels"][-1])  # bin size used for span-level products
out = Path(cfg["output_dir"])
ND = Path(cfg["resources_dir"]) / "ndbc"
PW = cfg["dvv"].get("primary_window", "coda")
STACK_S = cfg["dvv"]["stack_subs"] * cfg["xcorr"]["substack_s"]  # back-stacked windows [t - STACK_S, t]


def read_ndbc(station):
    parts = []
    for f in sorted(ND.glob(f"{station}*.txt.gz")):
        d = pd.read_csv(f, sep=r"\s+", skiprows=[1], na_values=[99.0, 99.00, 999, 999.0, 9999.0])
        d = d.rename(columns={"#YY": "YY"})
        d["time"] = pd.to_datetime(dict(year=d.YY, month=d.MM, day=d.DD, hour=d.hh, minute=d.mm), utc=True)
        parts.append(d[["time", "WVHT", "DPD", "APD", "WSPD"]])
    d = pd.concat(parts).drop_duplicates("time").set_index("time").sort_index()
    return d


def window_mean(series: pd.Series, t_centres: np.ndarray) -> np.ndarray:
    s = series.dropna()
    ts = s.index.astype("int64").values / 1e9
    return np.array([s.values[(ts >= t - STACK_S) & (ts < t)].mean() if np.any((ts >= t - STACK_S) & (ts < t)) else np.nan
                     for t in t_centres])


DD = out / cfg["dvv"]["out_subdir"]
sp = pd.read_csv(DD / "span_timeseries.csv.gz")
s = sp[(sp.level == PLEV) & (sp.window == PW)].copy()
s["key"] = s.cable.str[0].str.upper() + s.seg_id + "_" + s.band
W = s.pivot_table(index="t", columns="key", values="dv")
P = s.pivot_table(index="t", columns="key", values="logpow")
t = W.index.values

buoys = {st: read_ndbc(st) for st in ("46050", "46089")}
env = pd.DataFrame(index=t)
for st, d in buoys.items():
    env[f"Hs_{st}"] = window_mean(d.WVHT, t)
    env[f"Tp_{st}"] = window_mean(d.DPD, t)
env["Hs"] = env[[c for c in env if c.startswith("Hs_")]].mean(1)
env["Tp"] = env[[c for c in env if c.startswith("Tp_")]].mean(1)

rows = []
for band in cfg["xcorr"]["bands"]:
    cols = [c for c in W if c.endswith(band)]
    X = W[cols].dropna()
    Y = X - X.mean()
    u, sv, vt = np.linalg.svd(Y.values, full_matrices=False)
    pc1 = pd.Series(u[:, 0] * sv[0] * np.sign(vt[0].sum()), index=X.index)  # common-mode dv/v, sign: mean loading > 0
    pmean = P[cols].mean(1)
    e = env.loc[X.index]
    for name, v in (("Hs", e.Hs), ("Tp", e.Tp), ("log10 noise power (span mean)", pmean.loc[X.index])):
        ok = v.notna() & pc1.notna()
        rows.append(dict(band=band, variable=name, r_with_dvv_PC1=np.corrcoef(pc1[ok], v[ok])[0, 1],
                         PC1_share=sv[0] ** 2 / np.sum(sv**2), n=int(ok.sum())))
    ok = e.Hs.notna()
    rows.append(dict(band=band, variable="Hs vs noise power", r_with_dvv_PC1=np.corrcoef(e.Hs[ok], pmean.loc[X.index][ok])[0, 1], n=int(ok.sum())))
    env[f"dvvPC1_{band}"] = pc1.reindex(env.index)
    env[f"logpow_{band}"] = pmean.reindex(env.index)
res = pd.DataFrame(rows)
res.to_csv(DD / "environment_correlation.csv", index=False)
env.to_csv(DD / "environment_timeseries.csv")
print(res.round(3).to_string())

# figure: Hs, Tp, noise power, dv/v common mode
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
td = pd.to_datetime(env.index, unit="s", utc=True)
for st in buoys:
    axs[0].plot(td, env[f"Hs_{st}"], lw=0.9, label=f"NDBC {st}")
    axs[1].plot(td, env[f"Tp_{st}"], lw=0.9, label=f"NDBC {st}")
axs[0].set_ylabel("Hs (m)")
axs[1].set_ylabel("dominant period (s)")
for band, c in zip(cfg["xcorr"]["bands"], ["C0", "C3", "C2", "C1"]):
    axs[2].plot(td, env[f"logpow_{band}"], color=c, lw=0.9, label=band)
    axs[3].plot(td, 100 * env[f"dvvPC1_{band}"] / np.sqrt(len([k for k in W if k.endswith(band)])), color=c, lw=0.9, label=band)
axs[2].set_ylabel("log10 DAS noise power\n(span mean)")
axs[3].set_ylabel("dv/v common mode\n(PC1, % per span)")
setup = s.groupby("t").setup.first().reindex(env.index)
odd = setup[setup.str.contains("D128", na=False)]
for ax in axs:
    ax.legend(fontsize=7, loc="upper left")
    if len(odd):
        ax.axvspan(pd.to_datetime(odd.index.min(), unit="s", utc=True), pd.to_datetime(odd.index.max(), unit="s", utc=True), color="0.9", zorder=0)
fig.suptitle(f"Ocean conditions vs DAS noise power and common-mode dv/v ({PW} window, 6-h stacks); grey = D128 setup", fontsize=10)
fig.tight_layout()
fig.savefig(out / "figs" / ("" if DD.name == "dvv" else DD.name) / "environment.png", dpi=120)
print("fig ok")
