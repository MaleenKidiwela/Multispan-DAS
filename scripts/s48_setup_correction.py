"""Acquisition-setup step correction for all dv/v series (plan 0.3).

The 100-m stream ran with internal decimation D128 (3.73 Hz, decimated x2 in software) from 2026-01-13 18:18 to
2026-01-21 18:22 and with D256 otherwise. Every product shows a step of opposite sign at the two transitions (the D128
period reads ~0.1-0.3 % 'slow'): an acquisition (filter phase) effect, not the medium.
Per unit (cluster / group / span):
  1. stacks whose window [t - SH, t) mixes the two setups are dropped (NaN);
  2. step at each transition = median of the first 12 pure stacks after - median of the last 12 pure stacks before;
  3. D128 offset = mean(step_13, -step_21) (both transitions jointly; consistency = |step_13 + step_21|);
  4. the offset is subtracted from the pure-D128 stacks.
DAS-station: the D128 period is excluded instead (too few pure D128 stacks after the DAS timing QC).
The even/odd floor is unaffected (both halves carry the same step). Writes <series>_setupcorr.csv next to each series
and outputs_100m/setup_steps.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

B = Path("/data/wsd02/multispan_dvv")
T1 = pd.Timestamp("2026-01-13 18:00", tz="UTC").timestamp()  # first hour with D128 data (18:18)
T2 = pd.Timestamp("2026-01-21 19:00", tz="UTC").timestamp()  # first hour with only D256 data again (D128 ends 18:22)
N = 12


def correct(path, key, sh, product, variant, exclude=False):
    """exclude=True: the D128 (and mixed) stacks are set to NaN instead of offset-corrected (used for DAS-station, where
    many D128 files fail the DAS timing QC and too few pure D128 stacks constrain the offset)."""
    S = pd.read_csv(path)
    if key not in S:
        S[key] = S.cable + " " + S.span
    out, rows = [], []
    for u, g in S.groupby(key, sort=False):
        g = g.sort_values("t").copy()
        t0 = g.t - sh * 3600
        pure256 = (g.t <= T1) | (t0 >= T2)
        pure128 = (t0 >= T1) & (g.t <= T2 - 3600)
        g.loc[~(pure256 | pure128), "dv"] = np.nan
        a1 = g[pure256 & (g.t <= T1)].dv.dropna().tail(N).median()
        b1 = g[pure128].dv.dropna().head(N).median()
        a2 = g[pure128].dv.dropna().tail(N).median()
        b2 = g[pure256 & (t0 >= T2)].dv.dropna().head(N).median()
        s13, s21 = b1 - a1, b2 - a2
        off = np.nanmean([s13, -s21])
        if exclude:
            g.loc[pure128, "dv"] = np.nan
        elif np.isfinite(off):
            g.loc[pure128, "dv"] = g.loc[pure128, "dv"] - off
        rows.append(dict(product=product, variant=variant, unit=u, step_jan13_pct=100 * s13, step_jan21_pct=100 * s21,
                         offset_pct=100 * off, consistency_pct=100 * (s13 + s21),
                         std_before_pct=100 * S[S[key] == u].dv.std(), std_after_pct=100 * g.dv.std()))
        out.append(g)
    pd.concat(out).to_csv(path.with_name(path.stem + "_setupcorr.csv"), index=False)
    return rows


R = []
for sh, d in ((48, B / "outputs_100m/obsdas_coda"), (24, B / "outputs_100m/obsdas_coda/variants/final_ram_24h")):
    for suf in ("", "_inj0.005"):
        R += [r for r in correct(d / f"series_{sh}h{suf}.csv", "cluster", sh, "DAS-station", f"ram {sh}h{suf}", exclude=True) if not suf]
for v in sorted(p.name for p in (B / "outputs_100m_v2/interspan_coda").iterdir() if p.is_dir()):
    sh = int([x for x in v.split("_") if x.endswith("h")][0][:-1])
    for suf in ("", "_inj0.005"):
        f = B / "outputs_100m_v2/interspan_coda" / v / f"series_{sh}h{suf}.csv"
        if f.exists():
            rr = correct(f, "group", sh, "cross-span", v)
            R += rr if not suf else []
for root in (B / "outputs_100m_v2/within_coda", B / "outputs_100m_v2ram/within_coda"):
    for p in sorted(x for x in root.iterdir() if x.is_dir() and x.name != "figs"):
        band, sh = p.name.split("_")[1], int(p.name.split("_")[2][:-1])
        for suf in ("", "_inj0.005"):
            f = p / f"series_{band}_{sh}h{suf}.csv"
            if f.exists():
                rr = correct(f, "u", sh, "within-span", p.name)
                R += rr if not suf else []
D = pd.DataFrame(R)
D.to_csv(B / "outputs_100m/setup_steps.csv", index=False)
pd.set_option("display.width", 220)
print(D.groupby(["product", "variant"])[["step_jan13_pct", "step_jan21_pct", "offset_pct", "consistency_pct", "std_before_pct", "std_after_pct"]].median().round(3).to_string())
