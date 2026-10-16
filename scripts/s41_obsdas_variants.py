"""Compare the seismometer-DAS coda dv/v test variants (s40 with VARIANT=<name>; README "Raising the seismometer-DAS CC").

Per variant and cluster: median stretching CC and convergence CC over components/sides, share of 24-h stacks kept
(stretching CC >= 0.4), even/odd floor, std, and recovery of the injected 0.5 % step (relative to the same variant
without injection, 5 days before vs days 1-6 after 2026-01-01). Writes variants_comparison.csv / .md and a figure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv.obsdas import OUT


def md(df):
    """Minimal markdown table (no tabulate dependency)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)

VD = OUT / "obsdas_coda" / "variants"
DESC = {
    "ref": "reference: one-bit, 0.12-0.30 Hz, 24 h, coda pick+15..+90 s",
    "ram": "running-absolute-mean (20 s) instead of one-bit",
    "reject3": "drop hours with CCF norm > median + 3 MAD",
    "band15_25": "analysis band 0.15-0.25 Hz",
    "coda60": "coda pick+15..+60 s",
    "stack48": "48-h stacks (check only: breaks the 24-h standard)",
    "ram_stack48": "running-absolute-mean + 48-h stacks",
    "combo": "running-absolute-mean + 0.15-0.25 Hz (the two changes that helped)",
}
a0 = pd.Timestamp("2026-01-01", tz="UTC").timestamp()
rows = []
for v in sorted(p.name for p in VD.iterdir() if (p / "summary.csv").exists()):
    S = pd.read_csv(VD / v / "summary.csv")
    sf = sorted(f for f in (VD / v).glob("series_*h.csv"))[0]
    B = pd.read_csv(sf)
    xf = sf.with_name(sf.stem + "_inj0.005.csv")
    X = pd.read_csv(xf) if xf.exists() else None
    for _, r in S.iterrows():
        scc = [r[c] for c in S if c.startswith("scc_")]
        cnv = [r[c] for c in S if c.startswith("conv_")]
        pas = [r[c] for c in S if c.startswith("pass_")]
        rej = [r[c] for c in S if c.startswith("rej_")]
        rec = np.nan
        if X is not None:
            df = (X[X.cluster == r.cluster].set_index("t").dv - B[B.cluster == r.cluster].set_index("t").dv).dropna()
            post = df[(df.index >= a0 + 86400) & (df.index < a0 + 6 * 86400)].median()
            pre = df[(df.index < a0) & (df.index >= a0 - 5 * 86400)].median()
            rec = 100 * (post - pre) / 0.005
        rows.append(dict(variant=v, cluster=r.cluster.split(": ")[1], stretch_cc=np.nanmedian(scc), conv=np.nanmedian(cnv), kept=np.nanmedian(pas),
                         rejected_h=np.nanmedian(rej) if rej else 0.0, floor_pct=r.floor_pct, std_pct=r.std_pct, recovered_pct=rec))
D = pd.DataFrame(rows)
D.to_csv(VD / "variants_comparison.csv", index=False)
G = D.groupby("variant")[["stretch_cc", "conv", "kept", "rejected_h", "floor_pct", "std_pct", "recovered_pct"]].median()
G["description"] = [DESC.get(v, "") for v in G.index]
order = [v for v in DESC if v in G.index] + [v for v in G.index if v not in DESC]
G = G.loc[order]
pd.set_option("display.width", 220)
print(G.round(3).to_string())
print()
print(D.pivot(index="cluster", columns="variant", values="stretch_cc")[order].round(2).to_string())
with open(VD / "variants_comparison.md", "w") as fh:
    fh.write("Median over the 6 clusters (per cluster: median over Z/R/T x both lag sides)\n\n")
    fh.write(md(G.round(3)) + "\n\nStretching CC per cluster\n\n")
    fh.write(md(D.pivot(index="cluster", columns="variant", values="stretch_cc")[order].round(2)) + "\n\nFloor (%) per cluster\n\n")
    fh.write(md(D.pivot(index="cluster", columns="variant", values="floor_pct")[order].round(3)) + "\n\nInjection recovery (%) per cluster\n\n")
    fh.write(md(D.pivot(index="cluster", columns="variant", values="recovered_pct")[order].round(0)) + "\n")
fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, (col, lab) in zip(axs, (("stretch_cc", "stretching CC"), ("conv", "convergence CC"), ("floor_pct", "even/odd floor (%)"), ("recovered_pct", "injection recovered (%)"))):
    P = D.pivot(index="variant", columns="cluster", values=col).loc[order]
    for c in P.columns:
        ax.plot(range(len(order)), P[c], "o-", lw=0.8, ms=4, label=c)
    ax.plot(range(len(order)), P.median(1), "k-", lw=2.5, label="median")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_title(lab, fontsize=10)
    if col == "recovered_pct":
        ax.axhline(100, color="0.5", ls="--", lw=0.8)
axs[0].legend(fontsize=6)
fig.suptitle("Seismometer-DAS coda dv/v: test variants (24-h back-stacked unless stated; production = ram_stack48)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "figs" / "obsdas" / "obsdas_variants.png", dpi=110)
print("done")
