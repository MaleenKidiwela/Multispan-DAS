"""Item 8l: can the spans tell which DIRECTION a small earthquake comes from? Ground-truth test on the Nov 23 2025 M3.2
Amity earthquake (PNSN, 75 stations, gap 22 deg), plus the same measurement for the Jan 13 2026 ML 2.0.

Input: per-bin S-envelope delays within each coherent stretch (s74: amity_bin_times.csv; s65e:
das_location_jan13_channel_times.csv). A free time offset per stretch, so only the wavefront direction within each
stretch is used (no latency, no origin time).
Plane-wave fit: delay = c_k - s (x sin(baz) + y cos(baz)) for bins of stretch k; grid baz 0-358 (1 deg) x horizontal
slowness 0.05-0.6 s/km; L1 misfit (each stretch weighted equally). Uncertainty: baz range with misfit <= min + 10 %.
Single stretch: the same fit with only that stretch -> best baz and its mirror image about the stretch axis.
Truth (Amity): great-circle back-azimuth from each configuration's centroid to the PNSN epicentre.
Outputs: outputs_100m/small_event_direction.csv, figs/small_event_direction.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod

G = Geod(ellps="WGS84")
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
BAZ = np.arange(0, 360, 1.0)
SLO = np.arange(0.05, 0.601, 0.01)
AMITY = (45.0753, -123.1575)


def fit(Q):
    la0, lo0 = Q.lat.mean(), Q.lon.mean()
    az, _, d = G.inv(np.full(len(Q), lo0), np.full(len(Q), la0), Q.lon.values, Q.lat.values)
    x, y = d / 1e3 * np.sin(np.radians(az)), d / 1e3 * np.cos(np.radians(az))
    grp = [(Q.stretch == k).values for k in Q.stretch.unique()]
    M = np.zeros((len(BAZ), len(SLO)))
    for i, b in enumerate(np.radians(BAZ)):
        pred = -SLO[:, None] * (x * np.sin(b) + y * np.cos(b))[None]  # (nslo, nbin)
        r = Q.t.values[None] - pred
        for gm in grp:
            r[:, gm] -= np.median(r[:, gm], axis=1, keepdims=True)
        M[i] = np.mean([np.mean(np.abs(r[:, gm]), axis=1) for gm in grp], axis=0)
    ib, js = np.unravel_index(np.argmin(M), M.shape)
    prof = M.min(1)
    ok = BAZ[prof <= M.min() * 1.10]
    return dict(baz=BAZ[ib], slow=SLO[js], misfit=M.min(), baz_ok=ok, prof=prof / M.min(), lat0=la0, lon0=lo0)


rows, profs = [], {}
for ev, f_ in (("Amity M3.2 (Nov 23)", OUT / "amity_bin_times.csv"), ("Jan 13 ML 2.0", OUT / "das_location_jan13_channel_times.csv")):
    D = pd.read_csv(f_)
    if "t" not in D:
        D["t"] = D["d"]
    sets = {"all coherent stretches": D}
    sets.update({f"{k} (single)": D[D.stretch == k] for k in D.stretch.unique()})
    for name, Q in sets.items():
        if len(Q) < 3:
            continue
        r = fit(Q.reset_index(drop=True))
        true = G.inv(r["lon0"], r["lat0"], AMITY[1], AMITY[0])[0] % 360 if ev.startswith("Amity") else np.nan
        err = (r["baz"] - true + 180) % 360 - 180 if np.isfinite(true) else np.nan
        # width of the acceptable back-azimuth set (circular)
        ok = np.sort(r["baz_ok"])
        width = 360 - np.max(np.diff(np.r_[ok, ok[0] + 360])) if len(ok) > 1 else 1.0
        rows.append(dict(event=ev, config=name, n_stretches=Q.stretch.nunique(), n_bins=len(Q), baz_est=r["baz"], baz_true=round(true, 1) if np.isfinite(true) else np.nan,
                         error=err, slowness=r["slow"], width_deg_10pct=width, misfit=round(r["misfit"], 3)))
        profs[(ev, name)] = (r["prof"], true)
R = pd.DataFrame(rows)
R.to_csv(OUT / "small_event_direction.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(2).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
for ax, ev in zip(axs, ("Amity M3.2 (Nov 23)", "Jan 13 ML 2.0")):
    for (e_, name), (prof, true) in profs.items():
        if e_ != ev:
            continue
        lw, col = (2.5, "k") if name.startswith("all") else (1, None)
        ax.plot(BAZ, prof, lw=lw, color=col, label=name)
        if np.isfinite(true) and name.startswith("all"):
            ax.axvline(true, color="r", ls="--", label=f"true back-azimuth {true:.0f}° (PNSN)")
    ax.set(xlabel="back-azimuth (deg)", ylabel="misfit / minimum", ylim=(0.95, 2.0), title=f"{ev}: plane-wave direction from the S wavefront in each stretch")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figs/small_event_direction.png", dpi=100)
