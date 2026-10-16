"""Item 8c figure: Jan 16 2026 M6.0 (Blanco, 2.5 deg) P-onset beam directions per array configuration, on a map.
Input: beamform/event_beam_compare_stalta_2026-01-16.csv (s68, FEATURE=stalta). Arrows from each configuration's
centroid: estimated back-azimuth (colour = single span red / multi-span blue) and true back-azimuth (grey dashed).
Output: figs/beam_m6_single_vs_multi.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv.geometry import read_route
from msdvv.interspan import make_bins

OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
R = pd.read_csv(OUT / "beamform/event_beam_compare_stalta_2026-01-16.csv")
B = []
for cab in ("north", "south"):
    b, _ = make_bins(pd.read_csv(OUT / f"channel_quality_{cab}.csv"), 1.0, 3)
    b["cable"] = cab
    B.append(b[~((b.cable == "south") & (b.seg == "C3"))])
B = pd.concat(B, ignore_index=True)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw=dict(width_ratios=[1.4, 1]))
for c in ("north", "south"):
    rt = read_route(c)
    ax.plot(rt.lon, rt.lat, "-", color="0.8", lw=1)
cols = plt.cm.tab10.colors
L = 0.55  # arrow length (deg lat)
k = 0
for r in R.itertuples():
    if "single" in r.config:
        cab, seg = r.config.split()[:2]
        sel = (B.cable == cab) & (B.seg == seg)
        col = cols[k % 10]
        k += 1
        ax.plot(B.lon[sel], B.lat[sel], ".", color=col, ms=4)
        lw, ls_ = 1.5, "-"
    else:
        sel = (B.cable == r.config.split(",")[0]) if "," in r.config else np.ones(len(B), bool)
        col, lw = "tab:blue", 3
    la, lo = B.lat[sel].mean(), B.lon[sel].mean()
    sx = 1 / np.cos(np.radians(la))
    for bz, st, cc, w_ in ((r.baz_true, "--", "0.4", 1), (r.baz_est, "-", col, lw)):
        ax.annotate("", xy=(lo + L * sx * np.sin(np.radians(bz)), la + L * np.cos(np.radians(bz))), xytext=(lo, la),
                    arrowprops=dict(arrowstyle="->", color=cc, lw=w_, ls=st))
    ax.text(lo, la + 0.04, f"{r.config.replace(' (single span)', '')}: {r.baz_err:+.0f}°", fontsize=7, color=col if "single" in r.config else "tab:blue",
            ha="center")
ax.set(xlabel="longitude", ylabel="latitude", title="Jan 16 2026 M6.0 Blanco (2.5°): P-onset beam direction per span and per multi-span array\n"
                                                   "solid = estimated back-azimuth, grey dashed = true; arrows from each array's centroid")
ax.set_aspect(1 / np.cos(np.radians(45)))
q = R.assign(a=R.baz_err.abs()).sort_values("a")
ax2.barh(range(len(q)), q.a, color=["tab:red" if "single" in n else "tab:blue" for n in q.config])
ax2.set_yticks(range(len(q)), [f"{n} ({b} bins)" for n, b in zip(q.config, q.n_bins)], fontsize=8)
ax2.axvline(10, color="0.5", ls=":")
ax2.set(xlabel="|back-azimuth error| (deg)", title="blue = multi-span, red = single span")
fig.tight_layout()
fig.savefig(OUT / "figs/beam_m6_single_vs_multi.png", dpi=100)
