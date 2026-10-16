"""Item 7: November - December 2025 (Nov 6 - Dec 17) vs Dec 23 - Jan 28, same final settings (RAM, coda only, back-stacked).

Products: within-span (b1, b2; 24 / 48 h; cable averages), cross-span south (interspan_v4_ram, all paths and feature
groups) and north (interspan_v6_north), b1, 24 / 48 h. Metrics as in s46: stretching CC, convergence, kept share,
even/odd noise floor, std, and recovery of the injected 0.5 % step (Jan: injected after Jan 1; Nov: after Nov 27;
median of injected minus clean over 5 days after the stack has filled vs 5 days before).
Variability: std of the cable-average within-span dv/v per period (Nov 6 - Dec 17; Dec 23 - Jan 13; Jan 13 - 28) -
is the larger variability after Jan 13 also seen in November?
Outputs: outputs_nov_ram/nov_vs_jan.csv, figs/nov_vs_jan_dvv.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

B = Path("/data/wsd02/multispan_dvv")
PER = {"Nov": dict(within=B / "outputs_nov_ram/within_coda", cross=B / "outputs_nov_ram/interspan_coda", inj="2025-11-27"),
       "Jan": dict(within=B / "outputs_100m_v2ram/within_coda", cross=B / "outputs_100m_v2/interspan_coda", inj="2026-01-01")}


def recovery(clean, inj, key, sh, at):
    a0 = pd.Timestamp(at, tz="UTC").timestamp()
    out = {}
    for u in clean[key].unique():
        d = (inj[inj[key] == u].set_index("t").dv - clean[clean[key] == u].set_index("t").dv).dropna()
        post = d[(d.index >= a0 + sh * 3600) & (d.index < a0 + sh * 3600 + 5 * 86400)].median()
        pre = d[(d.index < a0) & (d.index >= a0 - 5 * 86400)].median()
        out[u] = 100 * (post - pre) / 0.005
    return out


rows, series = [], {}
for per, P in PER.items():
    for band in ("b1", "b2"):
        for sh in (24, 48):
            d = P["within"] / f"ram_{band}_{sh}h"
            S = pd.read_csv(d / f"summary_{band}_{sh}h.csv")
            cl, ij = pd.read_csv(d / f"series_{band}_{sh}h.csv"), pd.read_csv(d / f"series_{band}_{sh}h_inj0.005.csv")
            cl["u"], ij["u"] = cl.cable + " " + cl.span, ij.cable + " " + ij.span
            rec = recovery(cl, ij, "u", sh, P["inj"])
            for _, r in S[S.span == "all"].iterrows():
                u = f"{r.cable} all"
                rows.append(dict(period=per, product=f"within-span {r.cable}", band=band, stack_h=sh, cc=r.stretch_cc, conv=r.conv, kept=r.kept,
                                 floor_pct=r.floor_pct, std_pct=r.std_pct, recovered_pct=rec.get(u, np.nan)))
                g = cl[cl.u == u]
                series[(per, f"within-span {r.cable}", band, sh)] = pd.Series(g.dv.values * 100, index=pd.to_datetime(g.t, unit="s", utc=True))
    for sub, name in (("ram", "cross-span south"), ("north_ram" if per == "Jan" else "north", "cross-span north")):
        for sh in (24, 48):
            d = P["cross"] / f"{sub}_b1_{sh}h"
            if not (d / "summary.csv").exists():
                continue
            S = pd.read_csv(d / "summary.csv")
            cl, ij = pd.read_csv(d / f"series_{sh}h.csv"), pd.read_csv(d / f"series_{sh}h_inj0.005.csv")
            rec = recovery(cl, ij, "group", sh, P["inj"])
            for _, r in S.iterrows():
                rows.append(dict(period=per, product=f"{name}: {r.group}", band="b1", stack_h=sh, cc=r.stretch_cc, conv=r.conv, kept=r.kept,
                                 floor_pct=r.floor_pct, std_pct=r.std_pct, recovered_pct=rec.get(r.group, np.nan)))
            g = cl[cl.group == "all paths"]
            series[(per, name, "b1", sh)] = pd.Series(g.dv.values * 100, index=pd.to_datetime(g.t, unit="s", utc=True))
D = pd.DataFrame(rows)
D.to_csv(B / "outputs_nov_ram/nov_vs_jan.csv", index=False)
pd.set_option("display.width", 220)
main = D[D["product"].isin(["within-span north", "within-span south", "cross-span south: all paths", "cross-span north: all paths",
                            "cross-span south: Alvin Canyon fault", "cross-span south: Daisy Bank fault", "cross-span south: control (no feature)"])]
print(main.pivot_table(index=["product", "band", "stack_h"], columns="period", values=["cc", "floor_pct", "std_pct", "recovered_pct"]).round(3).to_string())

# variability per period (within-span cable averages, 48 h)
V = []
split = pd.Timestamp("2026-01-13 18:00", tz="UTC")
for (per, prod, band, sh), s in series.items():
    if not prod.startswith("within") or sh != 48:
        continue
    if per == "Nov":
        V.append(dict(product=prod, band=band, window="Nov 6 - Dec 17", std_pct=s.std(), n_days=(s.index[-1] - s.index[0]).days))
    else:
        V.append(dict(product=prod, band=band, window="Dec 23 - Jan 13", std_pct=s[s.index < split].std(), n_days=21))
        V.append(dict(product=prod, band=band, window="Jan 13 - 28", std_pct=s[s.index >= split].std(), n_days=15))
V = pd.DataFrame(V)
V.to_csv(B / "outputs_nov_ram/nov_vs_jan_variability.csv", index=False)
print("\nwithin-span cable-average std (48 h):")
print(V.pivot_table(index=["product", "band"], columns="window", values="std_pct").round(4).to_string())

fig, axs = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
for ax, (prod, band) in zip(axs, (("within-span north", "b2"), ("within-span south", "b2"), ("cross-span south", "b1"), ("cross-span north", "b1"))):
    for per, col in (("Nov", "tab:blue"), ("Jan", "k")):
        for sh, lw in ((24, 0.7), (48, 1.6)):
            s = series.get((per, prod, band, sh))
            if s is not None:
                s = s.sort_index()
                gap = s.index.to_series().diff() > pd.Timedelta("3h")  # break the line at outages / data gaps
                s = pd.concat([s, pd.Series(np.nan, index=s.index[gap.values] - pd.Timedelta("1min"))]).sort_index()
                ax.plot(s.index, s, color=col, lw=lw, alpha=0.6 if sh == 24 else 1, label=f"{per} {sh} h" if prod == "within-span north" else None)
    ax.axvline(split, color="0.6", ls=":")
    ax.set(ylabel="dv/v (%)", title=f"{prod} ({band}), coda only, back-stacked")
axs[0].legend(fontsize=8, ncol=4)
fig.suptitle("November - December 2025 (blue) and Dec 23 - Jan 28 (black): same settings; gaps = outages (Nov 12-18, Dec 8-9, power cut Dec 17-23)")
fig.tight_layout()
fig.savefig(B / "outputs_100m/figs/nov_vs_jan_dvv.png", dpi=100)
