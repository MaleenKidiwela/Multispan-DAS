"""Step 1b: per-channel physical/non-physical classification from inter-channel coherence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

import numpy as np
import pandas as pd

from msdvv import load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN tails at block ends
cfg = load_config()
out = Path(cfg["output_dir"])
q = cfg["quality"]
fig, axs = plt.subplots(2, 1, figsize=(13, 7))
for ax, cab in zip(axs, ("north", "south")):
    S = np.load(out / "noise" / f"spectra_{cab}.npz")
    geo = pd.read_csv(out / f"geometry_{cab}.csv")
    f, ks = S["f"], list(S["ks"])
    m = (f >= q["band"][0]) & (f <= q["band"][1])
    with np.errstate(all="ignore"):
        per_file = np.nanmean(S["coh_raw"][:, :, ks.index(q["k"])][:, :, m], axis=-1)  # (nfile, nch)
    geo["coh"] = np.nanmedian(per_file, axis=0)
    geo["coh_min_file"] = np.nanmin(per_file, axis=0)
    for b, band in cfg["xcorr"]["bands"].items():
        mb = (f >= band[0]) & (f <= band[1])
        with np.errstate(all="ignore"):
            geo[f"coh_{b}"] = np.nanmedian(np.nanmean(S["coh_raw"][:, :, ks.index(q["k"])][:, :, mb], axis=-1), axis=0)
            geo[f"logpsd_{b}"] = np.log10(np.median(S["psd_raw"][:, :, mb].mean(-1), axis=0) + 1e-40)
    geo["good"] = geo.usable_geom & (geo.coh >= q["coh_min"])
    geo.to_csv(out / f"channel_quality_{cab}.csv", index=False)
    for b, g in geo.groupby("block"):
        print(f"{cab} {b}: {g.good.sum():4d}/{len(g)} good channels ({g.good.mean():.0%})")
    for b, g in geo.groupby("block"):
        ax.plot(g.opt_km, g.coh, "-", color="0.6", lw=0.7)
        ax.plot(g.opt_km[g.good], g.coh[g.good], "g.", ms=3)
        ax.plot(g.opt_km[~g.good], g.coh[~g.good], "r.", ms=3)
    for p in pd.read_csv(out / f"repeater_picks_{cab}.csv").opt_km:
        ax.axvline(p, color="k", ls="--", lw=0.8)
    ax.axhline(q["coh_min"], color="b", lw=0.8)
    ax.set(title=f"{cab}: coherence k={q['k']} ({q['band'][0]}-{q['band'][1]} Hz); green = good, dashed = repeater",
           xlabel="optical distance (km)", ylabel="coherence", ylim=(0, 1))
fig.tight_layout()
fig.savefig(out / "figs" / "channel_quality.png", dpi=130)
