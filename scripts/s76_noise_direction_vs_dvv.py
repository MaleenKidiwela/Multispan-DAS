"""Item 8b follow-up: does the ambient-noise field (multi-span beamforming of the south loop, s64 hourly) explain the
within-span dv/v variations, in particular the larger variability after Jan 13?

Noise-field drivers per hour and frequency (0.16, 0.20, 0.25 Hz; peaks_hourly.csv): back-azimuth of the beam maximum as
its east (sin) and north (cos) components, beam maximum (coherence / directionality of the noise field), apparent
velocity of the peak. Trailing means over the dv/v stack length (24 / 48 h, as the back-stacked dv/v).
Targets: within-span coda dv/v (RAM; b1 0.12-0.3 Hz and b2 0.3-0.7 Hz), cable averages ('all') and south spans.
Significance: r of 6-h decimated series against the 95 % threshold with an AR(1) effective N (as s62).
Also: variability (std) of each driver before and after Jan 13 18:00.
Outputs: outputs_100m/beamform/noise_vs_dvv_correlations.csv, figs/noise_direction_vs_dvv.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

B = Path("/data/wsd02/multispan_dvv")
P = pd.read_csv(B / "outputs_100m/beamform/peaks_hourly.csv")
P = P[P.label == "hour"] if "label" in P else P
P["time"] = pd.to_datetime(P.t, unit="s", utc=True)
H = {}
for f0 in (0.16, 0.20, 0.25):
    q = P[np.isclose(P.f, f0)].set_index("time").sort_index()
    q = q[~q.index.duplicated()]
    H[f"east comp. of noise direction {f0:g} Hz"] = np.sin(np.radians(q.baz))
    H[f"north comp. of noise direction {f0:g} Hz"] = np.cos(np.radians(q.baz))
    H[f"beam max (noise coherence) {f0:g} Hz"] = q.beam_max
    H[f"apparent velocity {f0:g} Hz"] = q.vel.clip(upper=5)
H = pd.DataFrame(H).resample("1h").mean()
SPLIT = pd.Timestamp("2026-01-13 18:00", tz="UTC")


def trailing(x, sh):
    return x.rolling(f"{sh}h", min_periods=int(sh * 0.6)).mean()


def sig_r(a, b, step=6):
    d = pd.concat([a, b], axis=1).dropna().iloc[::step]
    if len(d) < 15:
        return np.nan, np.nan
    r = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
    r1 = [np.corrcoef(d.iloc[:-1, i], d.iloc[1:, i])[0, 1] for i in (0, 1)]
    neff = len(d) * (1 - r1[0] * r1[1]) / (1 + r1[0] * r1[1])
    return r, 1.96 / np.sqrt(max(neff - 2, 1))


targets = {}
for band in ("b1", "b2"):
    for sh in (24, 48):
        W = pd.read_csv(B / f"outputs_100m_v2ram/within_coda/ram_{band}_{sh}h/series_{band}_{sh}h.csv")
        for (cab, sp), g in W.groupby(["cable", "span"]):
            if sp == "all" or cab == "south":
                targets[(band, sh, f"within {cab} {sp}")] = pd.Series(g.dv.values * 100, index=pd.to_datetime(g.t, unit="s", utc=True))
rows = []
for (band, sh, tn), y in targets.items():
    for dn in H.columns:
        x = trailing(H[dn], sh).reindex(y.index)
        for per, (a, b) in (("all", ("2025-12-23", "2026-01-29")), ("Dec23-Jan13", ("2025-12-23", SPLIT)), ("Jan13-28", (SPLIT, "2026-01-29"))):
            a, b = pd.Timestamp(a, tz="UTC") if isinstance(a, str) else a, pd.Timestamp(b, tz="UTC") if isinstance(b, str) else b
            m = (y.index >= a) & (y.index < b)
            r, thr = sig_r(y[m], x[m])
            rows.append(dict(band=band, stack_h=sh, target=tn, driver=dn, period=per, r=r, r95=thr, significant=bool(abs(r) > thr) if np.isfinite(r) else False))
C = pd.DataFrame(rows)
C.to_csv(B / "outputs_100m/beamform/noise_vs_dvv_correlations.csv", index=False)
pd.set_option("display.width", 220)
S = C[(C.stack_h == 48) & C.target.isin(["within south all", "within north all"])]
print("significant (48 h, cable averages):")
print(S[S.significant].sort_values("r", key=np.abs, ascending=False).round(2).to_string(index=False) if S.significant.any() else "  none")
print("\nstrongest |r| (48 h, cable averages), per period:")
print(S.loc[S.groupby(["band", "target", "period"]).r.apply(lambda s: s.abs().idxmax())].round(2).to_string(index=False))
print("\nsignificant (any target incl. single south spans):", int(C.significant.sum()), "of", int(C.r.notna().sum()), "tests")
V = pd.DataFrame({"before Jan 13": H[H.index < SPLIT].std(), "after Jan 13": H[H.index >= SPLIT].std()})
V["ratio"] = V["after Jan 13"] / V["before Jan 13"]
print("\nnoise-driver variability (std of hourly values):")
print(V.round(3).to_string())

fig, axs = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
for ax, (band, tn) in zip(axs[:2], (("b2", "within south all"), ("b2", "within north all"))):
    y = targets[(band, 48, tn)]
    ax.plot(y.index, y, "k-", lw=1)
    ax.axvline(SPLIT, color="0.5", ls=":")
    ax.set(ylabel="dv/v (%)", title=f"{tn}, {band} (0.3-0.7 Hz), 48-h back-stacked")
ax = axs[2]
for f0, col in ((0.16, "tab:blue"), (0.20, "tab:orange")):
    ax.plot(trailing(H[f"east comp. of noise direction {f0:g} Hz"], 48), color=col, label=f"{f0:g} Hz: sin(back-azimuth), 48-h mean (-1 = from the west)")
ax.axvline(SPLIT, color="0.5", ls=":")
ax.set(ylabel="east component", title="Noise direction (south-loop beam maximum)")
ax.legend(fontsize=8)
ax = axs[3]
for f0, col in ((0.16, "tab:blue"), (0.20, "tab:orange"), (0.25, "tab:green")):
    ax.plot(trailing(H[f"beam max (noise coherence) {f0:g} Hz"], 48), color=col, label=f"{f0:g} Hz beam maximum, 48-h mean")
ax.axvline(SPLIT, color="0.5", ls=":")
ax.set(ylabel="beam max", title="Directionality / coherence of the noise field")
ax.legend(fontsize=8)
fig.suptitle("Within-span dv/v vs the ambient-noise field from multi-span beamforming (south loop)")
fig.tight_layout()
fig.savefig(B / "outputs_100m/figs/noise_direction_vs_dvv.png", dpi=100)
