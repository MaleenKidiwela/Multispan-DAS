"""Step 3a: reference CCF gathers (offset x lag) per block and band — physical move-out check."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv import load_config
from msdvv.stack import normalize, pair_mask, run_paths, total_spectra

cfg = load_config()
out = Path(cfg["output_dir"])
runs = sys.argv[1:] or None
for cab in ("north", "south"):
    geo = pd.read_csv(out / f"channel_quality_{cab}.csv")
    blocks = sorted(geo.block.unique())
    bands = list(cfg["xcorr"]["bands"])
    if runs:  # restrict to named runs: stream only those files
        import msdvv.stack as _st
        _rp = _st.run_paths
        _st.run_paths = lambda d, c: [p for p in _rp(d, c) if p.stem in runs]
    if not run_paths(out / "ccf", cab):
        continue
    fig, axs = plt.subplots(len(bands), len(blocks), figsize=(3.6 * len(blocks), 4.2 * len(bands)), squeeze=False)
    for i, band in enumerate(bands):
        spec, r0 = total_spectra(out / "ccf", cab, band)  # (nk, nch, nb) stack of everything
        cc, lags = r0.ccf(spec)  # (nk, nch, nlag)
        pm = pair_mask(r0.valid, r0.ks, geo.good.values)
        for j, b in enumerate(blocks):
            chs = geo.ch[geo.block == b].values
            G = []
            for kk in range(len(r0.ks)):
                sel = chs[pm[kk, chs]]
                G.append(normalize(cc[kk, sel]).mean(0) if len(sel) else np.zeros(len(lags)))
            G = np.array(G)
            G /= np.abs(G).max(1, keepdims=True) + 1e-30
            off = r0.ks * r0.dx_km
            ax = axs[i, j]
            ax.pcolormesh(lags, off, G, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest")
            for v in (0.3, 1.0, 3.0):
                ax.plot(off / v, off, "k:", lw=0.7)
                ax.plot(-off / v, off, "k:", lw=0.7)
            ax.set(xlim=(-40, 40), title=f"{cab} {b} {band} {cfg['xcorr']['bands'][band]} Hz", xlabel="lag (s)")
            ax.text(0.02, 0.97, f"{int(geo.good[geo.block == b].sum())} good ch", transform=ax.transAxes, va="top", fontsize=8)
        axs[i, 0].set_ylabel("offset (km)")
    fig.tight_layout()
    tag = "_".join(runs) if runs else "all"
    fig.savefig(out / "figs" / f"gathers_{cab}_{tag}.png", dpi=110)
    plt.close(fig)
print("ok")
