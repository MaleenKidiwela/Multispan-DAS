"""Step 7: seafloor surface-wave (Scholte) group velocity along each cable from the reference CCFs.

For every 2-km bin and band, the envelope peak of the all-period reference CCF is picked on each
lag side for offsets k*dx (0.4-3.2 km); the group velocity is the least-squares slope of offset vs
pick time through the origin (picks outside 0.05-3 km/s are rejected). Scholte-wave group velocity
is a proxy for near-seafloor shear velocity; b1 (0.12-0.3 Hz) samples deeper than b2 (0.3-0.7 Hz).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from msdvv import load_config
from msdvv.dvv import one_sided

cfg = load_config()
out = Path(cfg["output_dir"])
rd = out / "dvv" / "refs"
PROF = cfg.get("profile", {})
LEVEL = PROF.get("level", "2km")
K_MIN_KM = PROF.get("k_min_km", 0.4)  # shortest offset used for the move-out fit
rows = []
for cab in ("north", "south"):
    bins = pd.read_csv(rd / f"bins_{cab}_{LEVEL}.csv")
    for band in cfg["xcorr"]["bands"]:
        R = np.load(rd / f"{cab}_{band}_{LEVEL}.npz")
        M = np.load(rd / f"submeta_{cab}_{band}_{LEVEL}.npz")
        ref, lags, ks, dx = R["ref"], R["lags"], R["ks"], float(M["dx_km"])
        dt = lags[1] - lags[0]
        offs = ks * dx
        use = offs >= K_MIN_KM
        for b in range(ref.shape[1]):
            for side in ("pos", "neg"):
                env = np.abs(hilbert(one_sided(ref[:, b], lags, side), axis=-1))
                if not np.any(env):
                    continue
                tpk = np.argmax(env, axis=-1) * dt
                ok = use & (tpk > 0) & np.isfinite(tpk)
                ok &= (offs / np.maximum(tpk, 1e-6) > 0.05) & (offs / np.maximum(tpk, 1e-6) < 3.0)
                if ok.sum() < 5:
                    continue
                slow = np.sum(tpk[ok] * offs[ok]) / np.sum(offs[ok] ** 2)  # s/km
                resid = tpk[ok] - slow * offs[ok]
                rows.append(dict(cable=cab, band=band, bin=b, side=side, v_kms=1 / slow, n=int(ok.sum()),
                                 rms_s=float(np.sqrt(np.mean(resid**2)))))
v = pd.DataFrame(rows)
# average the two propagation directions where both are well determined
vb = v[v.rms_s < 1.0].groupby(["cable", "band", "bin"]).agg(v_kms=("v_kms", "median"), n_sides=("side", "size"),
                                                           rms_s=("rms_s", "mean")).reset_index()
prof = []
for cab in ("north", "south"):
    b = pd.read_csv(rd / f"bins_{cab}_{LEVEL}.csv")
    prof.append(vb[vb.cable == cab].merge(b, on="bin"))
prof = pd.concat(prof, ignore_index=True)
prof.to_csv(out / "dvv" / "scholte_velocity_profile.csv", index=False)

summ = prof.assign(seg_id=prof.block + prof.span.astype(str)).groupby(["cable", "seg_id", "band"]).agg(
    depth_m=("depth", "median"), v_med=("v_kms", "median"), v_p10=("v_kms", lambda x: x.quantile(0.1)),
    v_p90=("v_kms", lambda x: x.quantile(0.9)), n_bins=("v_kms", "size")).round(3)
print(summ.unstack("band").to_string())
c = prof.groupby("band")[["v_kms", "depth"]].corr().iloc[0::2, -1]
print("\ncorrelation of velocity with water depth:", c.round(2).to_dict())

fig, axs = plt.subplots(2, 1, figsize=(13, 7.5))
for ax, cab in zip(axs, ("north", "south")):
    p = prof[prof.cable == cab]
    for band, col in (("b1", "C0"), ("b2", "C3")):
        q = p[p.band == band].sort_values("opt_km")
        for blk, g in q.groupby("block"):
            ax.plot(g.route_km, g.v_kms, "o-", color=col, ms=3, lw=0.8, label=f"{band} {cfg['xcorr']['bands'][band]} Hz" if blk == "A" else None)
    ax.set(ylabel="Scholte group velocity (km/s)", xlabel="distance along route (km)", title=f"{cab} cable")
    ax2 = ax.twinx()
    ax2.plot(p.route_km, p.depth, ".", color="0.6", ms=2)
    ax2.invert_yaxis()
    ax2.set_ylabel("water depth (m)", color="0.5")
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Seafloor surface-wave velocity from 5-week reference correlations (2-km bins); grey = water depth", fontsize=10)
fig.tight_layout()
fig.savefig(out / "figs" / "scholte_velocity_profile.png", dpi=120)
print("fig ok")
