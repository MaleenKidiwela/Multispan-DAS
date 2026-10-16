"""Item 7, third product: November (Nov 6 - Dec 17) vs Dec 23 - Jan 28 seismometer-DAS coda dv/v (same pipeline: s32 Z/R/T
RAM, s33 latency per setup, timing QC, s37 clusters, s40 coda; 48 h). Nov specifics: latency 11.27 s (Nov 6-26,
D64 g256) / 15.57 s (Nov 26 - Dec 17, D64 g128); timing QC on 24-h blocks (17-min files); injection at Dec 12.
Output: figs/nov_vs_jan_obsdas.png, outputs_nov_obsdas/nov_vs_jan_obsdas.csv
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

B = Path("/data/wsd02/multispan_dvv")
D = {"Jan": (B / "outputs_100m/obsdas_coda", "2026-01-01"), "Nov": (B / "outputs_nov_obsdas/obsdas_coda/variants/final_ram_48h", "2025-12-12")}
rows, ser = [], {}
for per, (d, at) in D.items():
    S = pd.read_csv(d / "summary.csv")
    c, i = pd.read_csv(d / "series_48h.csv"), pd.read_csv(d / "series_48h_inj0.005.csv")
    a0 = pd.Timestamp(at, tz="UTC").timestamp()
    for _, r in S.iterrows():
        cc_ = c[c.cluster == r.cluster].set_index("t").dv
        dd = (i[i.cluster == r.cluster].set_index("t").dv - cc_).dropna()
        post = dd[(dd.index >= a0 + 48 * 3600) & (dd.index < a0 + 48 * 3600 + 5 * 86400)].median()
        pre = dd[(dd.index < a0) & (dd.index >= a0 - 5 * 86400)].median()
        rows.append(dict(period=per, cluster=r.cluster, stretch_cc=S.loc[_, [k for k in S if k.startswith("scc_")]].mean(), floor_pct=r.floor_pct,
                         std_pct=r.std_pct, recovered_pct=100 * (post - pre) / 0.005))
        ser[(per, r.cluster)] = pd.Series(cc_.values * 100, index=pd.to_datetime(cc_.index, unit="s", utc=True)).sort_index()
R = pd.DataFrame(rows)
R.to_csv(B / "outputs_nov_obsdas/nov_vs_jan_obsdas.csv", index=False)
pd.set_option("display.width", 200)
print(R.pivot_table(index="cluster", columns="period", values=["stretch_cc", "floor_pct", "recovered_pct"]).round(3).to_string())
print(R.groupby("period")[["stretch_cc", "floor_pct", "recovered_pct"]].median().round(3))

cl = list(dict.fromkeys(R.cluster))
fig, axs = plt.subplots(len(cl), 1, figsize=(15, 2.4 * len(cl)), sharex=True)
for ax, k in zip(axs, cl):
    for per, col in (("Nov", "tab:blue"), ("Jan", "k")):
        s = ser.get((per, k))
        if s is None:
            continue
        gap = s.index.to_series().diff() > pd.Timedelta("3h")
        s = pd.concat([s, pd.Series(np.nan, index=s.index[gap.values] - pd.Timedelta("1min"))]).sort_index()
        f = R[(R.period == per) & (R.cluster == k)].iloc[0]
        ax.plot(s.index, s, color=col, lw=1.2, label=f"{per}: CC {f.stretch_cc:.2f}, floor {f.floor_pct:.3f} %")
    ax.set_ylabel("dv/v (%)")
    ax.set_title(k, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Seismometer-DAS coda dv/v per path cluster, 48-h back-stacked: November (blue) vs Dec 23 - Jan 28 (black)")
fig.tight_layout()
fig.savefig(B / "outputs_100m/figs/nov_vs_jan_obsdas.png", dpi=95)
