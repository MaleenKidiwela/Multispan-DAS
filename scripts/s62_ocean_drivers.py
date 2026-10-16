"""Item 4: what drives the larger within-span dv/v variability after Jan 13?

Candidate drivers, all as trailing means over the same window as each dv/v stack (24 h or 48 h, back-stacked):
  * OOI Hydrate Ridge seafloor oceanography (EarthScope FDSN, network OO, 1 Hz -> hourly): HYS14 (summit, 777 m)
    and HYSB1 (slope base, 2907 m): LDO bottom pressure (de-tided: least-squares fit of 8 constituents M2 S2 N2 K2
    K1 O1 P1 Q1 + linear trend), LK1 pressure-sensor temperature, LKO current-meter temperature, LOE/LON/LOZ bottom
    current (speed = sqrt(E^2+N^2));
  * waves: Hs, Tp (NDBC 46050), in-band DAS noise power (environment_timeseries.csv);
  * DAS acquisition health: dropped chunks per hour (both cables, clock_drops_per_file.csv) and the D128 setup flag.
Targets: within-span coda dv/v (RAM, 0.3-0.7 Hz) north-all, south-all and per span; cross-span all paths (RAM b1).
Statistics: Pearson r on the hourly series, and r on 6-h-decimated series with an AR(1)-corrected effective N (95 %
significance); split Dec 23 - Jan 13 vs Jan 13 - 28.
Outputs: outputs_100m/ocean_drivers/{ocean_hourly.csv, correlations.csv}, figs/ocean_drivers.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

B = Path("/data/wsd02/multispan_dvv")
OD = B / "outputs_100m/ocean_drivers"
OD.mkdir(parents=True, exist_ok=True)
CH = ["LDO", "LK1", "LKO", "LOE", "LON", "LOZ"]
T_START, T_END = UTCDateTime("2025-12-21"), UTCDateTime("2026-01-29")
hourly_f = OD / "ocean_hourly.csv"
if not hourly_f.exists():
    C = Client("https://service.earthscope.org", timeout=300)
    cols = {}
    for sta in ("HYS14", "HYSB1"):
        for ch in CH:
            vals = []
            t = T_START
            while t < T_END:
                try:
                    st = C.get_waveforms("OO", sta, "*", ch, t, t + 86400)
                    st.merge(fill_value=None)
                    for tr in st.split():
                        tt = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) / tr.stats.sampling_rate
                        s = pd.Series(tr.data.astype(float), index=pd.to_datetime(tt, unit="s", utc=True))
                        vals.append(s.resample("1h").mean())
                except Exception as e:  # noqa: BLE001
                    print(sta, ch, t.date, "failed", type(e).__name__)
                t += 86400
            if vals:
                s = pd.concat(vals).groupby(level=0).mean()
                cols[f"{sta}_{ch}"] = s
            print(sta, ch, "hours", len(vals) and len(s), flush=True)
    H = pd.DataFrame(cols)
    H.to_csv(hourly_f)
H = pd.read_csv(hourly_f, index_col=0, parse_dates=True)
H.index = pd.to_datetime(H.index, utc=True)

# ---- de-tide bottom pressure (and remove the tide from currents: speed computed from de-tided E/N)
th = (H.index.view("int64") / 1e9 - H.index.view("int64")[0] / 1e9) / 3600.0
PER = {"M2": 12.4206012, "S2": 12.0, "N2": 12.65834751, "K2": 11.96723606, "K1": 23.93447213, "O1": 25.81933871, "P1": 24.06588766, "Q1": 26.868350}


def detide(y):
    ok = np.isfinite(y)
    A = np.column_stack([np.ones(len(th)), th] + [f(2 * np.pi * th / p) for p in PER.values() for f in (np.cos, np.sin)])
    c = np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
    tide = A[:, 2:] @ c[2:]
    return y - tide - c[1] * th - c[0], tide


for sta in ("HYS14", "HYSB1"):
    for ch in ("LDO", "LOE", "LON"):
        k = f"{sta}_{ch}"
        if k in H:
            H[k + "_res"], H[k + "_tide"] = detide(H[k].values)
    if f"{sta}_LOE_res" in H:
        H[f"{sta}_speed_res"] = np.hypot(H[f"{sta}_LOE_res"], H[f"{sta}_LON_res"])
        H[f"{sta}_speed"] = np.hypot(H[f"{sta}_LOE"], H[f"{sta}_LON"])
env = pd.read_csv(B / "outputs_100m/dvv_24h/environment_timeseries.csv", index_col=0)
env.index = pd.to_datetime(env.index, unit="s", utc=True)
H = H.join(env[["Hs", "Tp", "logpow_b1", "logpow_b2"]].resample("1h").mean(), how="outer")
R = pd.read_csv(B / "outputs_100m/clock_drops_per_file.csv")
R = R[R.stream == "100-m"]
drop = pd.Series(0.0, index=H.index)
for r in R.itertuples():
    a, b = pd.Timestamp(r.t0, unit="s", tz="UTC"), pd.Timestamp(r.t0 + r.duration_h * 3600, unit="s", tz="UTC")
    drop[(drop.index >= a) & (drop.index < b)] += r.drops / r.duration_h / 2.0  # per hour, mean of the two cables
H["drops_per_h"] = drop
H["D128"] = ((H.index >= pd.Timestamp("2026-01-13 18:18", tz="UTC")) & (H.index < pd.Timestamp("2026-01-21 18:22", tz="UTC"))).astype(float)
H.to_csv(OD / "ocean_hourly_derived.csv")

DRIVERS = {"HYS14 bottom pressure (de-tided)": "HYS14_LDO_res", "HYSB1 bottom pressure (de-tided)": "HYSB1_LDO_res",
           "HYS14 temperature (LKO)": "HYS14_LKO", "HYSB1 temperature (LKO)": "HYSB1_LKO", "HYS14 temperature (LK1)": "HYS14_LK1",
           "HYS14 current speed (de-tided)": "HYS14_speed_res", "HYSB1 current speed (de-tided)": "HYSB1_speed_res",
           "Hs": "Hs", "Tp": "Tp", "DAS noise power 0.3-0.7 Hz": "logpow_b2", "DAS dropped chunks / h": "drops_per_h", "D128 setup": "D128"}


def trailing(x, sh):
    return x.rolling(f"{sh}h", min_periods=int(sh * 0.6)).mean()


def sig_r(a, b, step=6):
    """r of 6-h decimated series and its 95 % threshold with AR(1) effective N."""
    d = pd.concat([a, b], axis=1).dropna().iloc[::step]
    if len(d) < 15:
        return np.nan, np.nan
    r = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
    r1 = [np.corrcoef(d.iloc[:-1, i], d.iloc[1:, i])[0, 1] for i in (0, 1)]
    neff = len(d) * (1 - r1[0] * r1[1]) / (1 + r1[0] * r1[1])
    return r, 1.96 / np.sqrt(max(neff - 2, 1))


rows = []
targets = {}
for sh in (24, 48):
    W = pd.read_csv(B / f"outputs_100m_v2ram/within_coda/ram_b2_{sh}h/series_b2_{sh}h.csv")
    for (cab, sp), g in W.groupby(["cable", "span"]):
        targets[(sh, f"within {cab} {sp}")] = pd.Series(g.dv.values * 100, index=pd.to_datetime(g.t, unit="s", utc=True))
    X = pd.read_csv(B / f"outputs_100m_v2/interspan_coda/ram_b1_{sh}h/series_{sh}h.csv")
    g = X[X.group == "all paths"]
    targets[(sh, "cross-span all paths")] = pd.Series(g.dv.values * 100, index=pd.to_datetime(g.t, unit="s", utc=True))
for (sh, tn), y in targets.items():
    for dn, col in DRIVERS.items():
        if col not in H:
            continue
        x = trailing(H[col], sh).reindex(y.index)
        for per, (a, b) in (("all", ("2025-12-23", "2026-01-29")), ("Dec23-Jan13", ("2025-12-23", "2026-01-13 18:00")), ("Jan13-28", ("2026-01-13 18:00", "2026-01-29"))):
            m = (y.index >= pd.Timestamp(a, tz="UTC")) & (y.index < pd.Timestamp(b, tz="UTC"))
            r, thr = sig_r(y[m], x[m])
            rows.append(dict(stack_h=sh, target=tn, driver=dn, period=per, r=r, r95=thr, significant=bool(abs(r) > thr) if np.isfinite(r) else False))
Cr = pd.DataFrame(rows)
Cr.to_csv(OD / "correlations.csv", index=False)
pd.set_option("display.width", 250)
show = Cr[(Cr.stack_h == 48) & Cr.target.isin(["within north all", "within south all", "cross-span all paths"])]
print(show.pivot_table(index="driver", columns=["target", "period"], values="r").round(2).to_string())
print("significant (|r| > AR1-corrected 95 %), 48 h:")
print(show[show.significant].sort_values("r", key=np.abs, ascending=False).round(2).to_string(index=False))

# ---- figure
fig, axs = plt.subplots(7, 1, figsize=(14, 15), sharex=True)
for ax, (sh, tn) in zip(axs[:2], ((48, "within north all"), (48, "within south all"))):
    ax.plot(targets[(sh, tn)], "C3", lw=1)
    ax.set_ylabel(f"{tn}\ndv/v % (48 h)", fontsize=8)
axs[2].plot(targets[(48, "cross-span all paths")], "C0", lw=1)
axs[2].set_ylabel("cross-span all\ndv/v % (48 h)", fontsize=8)
for c, col in (("HYS14_LDO_res", "k"), ("HYSB1_LDO_res", "0.5")):
    if c in H:
        axs[3].plot(trailing(H[c], 48), color=col, lw=1, label=c)
axs[3].set_ylabel("de-tided bottom\npressure (48 h)", fontsize=8)
axs[3].legend(fontsize=7)
for c, col in (("HYS14_LKO", "tab:red"), ("HYSB1_LKO", "tab:orange")):
    if c in H:
        axs[4].plot(trailing(H[c], 48), color=col, lw=1, label=c)
axs[4].set_ylabel("bottom temperature\n(48 h)", fontsize=8)
axs[4].legend(fontsize=7)
for c, col in (("HYS14_speed_res", "tab:green"), ("HYSB1_speed_res", "tab:olive")):
    if c in H:
        axs[5].plot(trailing(H[c], 48), color=col, lw=1, label=c)
axs[5].set_ylabel("de-tided current\nspeed (48 h)", fontsize=8)
axs[5].legend(fontsize=7)
axs[6].plot(trailing(H.Hs, 48), "k", lw=1, label="Hs (m)")
a2 = axs[6].twinx()
a2.plot(trailing(H.drops_per_h, 48), "tab:purple", lw=1, label="DAS drops / h")
axs[6].set_ylabel("Hs (m)", fontsize=8)
a2.set_ylabel("DAS drops / h", fontsize=8, color="tab:purple")
for ax in axs:
    ax.axvspan(pd.Timestamp("2026-01-13 18:18", tz="UTC"), pd.Timestamp("2026-01-21 18:22", tz="UTC"), color="tab:blue", alpha=0.05)
fig.suptitle("Within-span and cross-span dv/v vs Hydrate Ridge seafloor oceanography, waves and DAS acquisition health (all 48-h trailing; blue = D128 setup)", fontsize=10)
fig.tight_layout()
fig.savefig(B / "outputs_100m/figs/ocean_drivers.png", dpi=100)
print("done")
