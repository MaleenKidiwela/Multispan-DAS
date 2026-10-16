"""Final comparison of the three coda-only dv/v products and their variants (docs/PLAN_final_dvv.md).

For every product / variant / unit (DAS-station cluster, cross-span group, within-span cable x span): stretching CC,
convergence CC, share kept, even/odd floor, std, and recovery of the injected 0.5 % step (injected minus clean series;
median over the 5 days before 2026-01-01 vs the 5 days after the stack has fully passed the step).
Writes outputs_100m/final_comparison.csv and .md (included in the README).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

B = Path("/data/wsd02/multispan_dvv")
A0 = pd.Timestamp("2026-01-01", tz="UTC").timestamp()


def recovery(clean, inj, key, sh):
    rows = {}
    for u in clean[key].unique():
        c = clean[clean[key] == u].set_index("t").dv
        x = inj[inj[key] == u].set_index("t").dv
        d = (x - c).dropna()
        post = d[(d.index >= A0 + sh * 3600) & (d.index < A0 + sh * 3600 + 5 * 86400)].median()
        pre = d[(d.index < A0) & (d.index >= A0 - 5 * 86400)].median()
        rows[u] = 100 * (post - pre) / 0.005
    return rows


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


rows = []
# ---- DAS-station (s40): production 48 h in obsdas_coda/, 24 h in variants/final_ram_24h
for sh, d in ((48, B / "outputs_100m/obsdas_coda"), (24, B / "outputs_100m/obsdas_coda/variants/final_ram_24h")):
    S = pd.read_csv(d / "summary.csv")
    rec = recovery(pd.read_csv(d / f"series_{sh}h.csv"), pd.read_csv(d / f"series_{sh}h_inj0.005.csv"), "cluster", sh)
    for _, r in S.iterrows():
        rows.append(dict(product="DAS-station", variant=f"ram {sh}h", unit=r.cluster.split(": ")[1], n=np.nan,
                         stretch_cc=np.nanmedian([r[c] for c in S if c.startswith("scc_")]),
                         conv=np.nanmedian([r[c] for c in S if c.startswith("conv_")]),
                         kept=np.nanmedian([r[c] for c in S if c.startswith("pass_")]), floor_pct=r.floor_pct, std_pct=r.std_pct,
                         recovered_pct=rec.get(r.cluster, np.nan)))
# ---- cross-span (s43)
for v in ("onebit_b1_24h", "onebit_b1_48h", "ram_b1_24h", "ram_b1_48h"):
    d = B / "outputs_100m_v2/interspan_coda" / v
    sh = int([x for x in v.split("_") if x.endswith("h")][0][:-1])
    S = pd.read_csv(d / "summary.csv")
    rec = recovery(pd.read_csv(d / f"series_{sh}h.csv"), pd.read_csv(d / f"series_{sh}h_inj0.005.csv"), "group", sh)
    for _, r in S.iterrows():
        rows.append(dict(product="cross-span", variant=v.replace("_", " "), unit=r.group, n=r.n_sides, stretch_cc=r.stretch_cc, conv=r.conv,
                         kept=r.kept, floor_pct=r.floor_pct, std_pct=r.std_pct, recovered_pct=rec.get(r.group, np.nan)))
# ---- within-span (s45)
for root, norm in ((B / "outputs_100m_v2/within_coda", "onebit"), (B / "outputs_100m_v2ram/within_coda", "ram")):
    for band in ("b1", "b2"):
        for sh in (24, 48):
            d = root / f"{norm}_{band}_{sh}h"
            f = d / f"summary_{band}_{sh}h.csv"
            if not f.exists():
                continue
            S = pd.read_csv(f)
            cl, ij = pd.read_csv(d / f"series_{band}_{sh}h.csv"), pd.read_csv(d / f"series_{band}_{sh}h_inj0.005.csv")
            cl["u"], ij["u"] = cl.cable + " " + cl.span, ij.cable + " " + ij.span
            rec = recovery(cl, ij, "u", sh)
            for _, r in S.iterrows():
                u = f"{r.cable} {r.span}"
                rows.append(dict(product="within-span", variant=f"{norm} {band} {sh}h", unit=u, n=r.n_units, stretch_cc=r.stretch_cc, conv=r.conv,
                                 kept=r.kept, floor_pct=r.floor_pct, std_pct=r.std_pct, recovered_pct=rec.get(u, np.nan)))
D = pd.DataFrame(rows)
D.to_csv(B / "outputs_100m/final_comparison.csv", index=False)
G = D.groupby(["product", "variant"])[["stretch_cc", "conv", "kept", "floor_pct", "std_pct", "recovered_pct"]].median().reset_index()
pd.set_option("display.width", 220)
print(G.round(3).to_string(index=False))
with open(B / "outputs_100m/final_comparison.md", "w") as fh:
    fh.write("Medians over units (DAS-station clusters / cross-span groups / within-span cable x span)\n\n" + md(G.round(3)) + "\n\nPer unit\n\n" + md(D.round(3)) + "\n")
