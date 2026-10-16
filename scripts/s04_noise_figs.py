"""Step 1 figures: PSD vs distance, coherence vs offset, f-k per block."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv import load_config

cfg = load_config()
out = Path(cfg["output_dir"])
nd, fd = out / "noise", out / "figs"
BANDS = cfg["xcorr"]["bands"]

for cab in ("north", "south"):
    S = np.load(nd / f"spectra_{cab}.npz")
    geo = pd.read_csv(out / f"geometry_{cab}.csv")
    f, ks = S["f"], S["ks"]
    DXKM = float(np.median(np.diff(geo.opt_km[geo.block == geo.block.iloc[0]])))
    fm = (f > 0.03) & (f < 3.7)
    # --- PSD vs distance (median over files), raw and common-mode removed
    fig, axs = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, tag in zip(axs, ("raw", "cm")):
        p = np.log10(np.median(S[f"psd_{tag}"], axis=0)[:, fm] + 1e-40)
        p[~geo.usable_geom.values] = np.nan
        im = ax.pcolormesh(f[fm], np.arange(len(geo)), p, shading="auto", cmap="magma", vmin=np.nanpercentile(p, 2), vmax=np.nanpercentile(p, 99.5))
        ax.set_xscale("log")
        for b, g in geo.groupby("block"):
            ax.axhline(g.ch.min(), color="c", lw=1)
            ax.text(0.035, g.ch.min() + 5, f"{b}: {g.opt_km.min():.0f}-{g.opt_km.max():.0f} km opt, {g.depth.min():.0f}-{g.depth.max():.0f} m", color="c", fontsize=8, va="top")
        for c in pd.read_csv(out / f"repeater_picks_{cab}.csv").ch:
            ax.axhline(c, color="w", ls="--", lw=0.8)
        ax.set(title=f"{cab} PSD log10[(1/s)^2/Hz] ({tag})", xlabel="Hz")
        plt.colorbar(im, ax=ax)
    axs[0].set_ylabel("channel (concatenated array)")
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(fd / f"psd_{cab}.png", dpi=120)
    plt.close(fig)

    # --- coherence vs frequency for each offset, per block
    blocks = sorted(geo.block.unique())
    fig, axs = plt.subplots(2, len(blocks), figsize=(4 * len(blocks), 7), sharex=True, sharey=True)
    for j, b in enumerate(blocks):
        idx = geo.index[(geo.block == b) & geo.usable_geom].values
        for i, tag in enumerate(("raw", "cm")):
            C = np.nanmedian(S[f"coh_{tag}"][:, idx], axis=(0, 1))  # (nk, nf)
            ax = axs[i, j]
            for kk, k in enumerate(ks):
                ax.plot(f[fm], C[kk, fm], label=f"k={k} ({k * DXKM:.1f} km)")
            ax.set(xscale="log", title=f"{cab} {b} ({tag})", ylim=(0, 1))
    axs[0, 0].legend(fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("Hz")
    for ax in axs[:, 0]:
        ax.set_ylabel("median coherence")
    fig.tight_layout()
    fig.savefig(fd / f"coherence_{cab}.png", dpi=120)
    plt.close(fig)

    # --- f-k per block
    F = np.load(nd / f"fk_{cab}.npz")
    fig, axs = plt.subplots(2, len(blocks), figsize=(4 * len(blocks), 7))
    for j, b in enumerate(blocks):
        for i, tag in enumerate(("raw", "cm")):
            P, k, ff = F[f"{tag}_{b}"], F[f"k_{b}"], F["f"]
            m = (ff > 0.03) & (ff < 3.7)
            ax = axs[i, j]
            ax.pcolormesh(k, ff[m], np.log10(P[:, m].T + 1e-40), shading="auto", cmap="viridis",
                          vmin=np.percentile(np.log10(P[:, m] + 1e-40), 5), vmax=np.percentile(np.log10(P[:, m] + 1e-40), 99.8))
            for v in (0.3, 1.5, 3.0):
                kk = np.linspace(-k.max(), k.max(), 50)
                ax.plot(kk, np.abs(kk) * v, "w--", lw=0.6)
            ax.set(ylim=(0.03, 3.7), title=f"{cab} {b} ({tag})", xlabel="k (1/km)", ylabel="Hz")
    fig.tight_layout()
    fig.savefig(fd / f"fk_{cab}.png", dpi=120)
    plt.close(fig)
print("figs written")
