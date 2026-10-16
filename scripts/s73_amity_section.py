"""Item 8j: is the Nov 23 2025 M3.2 Amity (Oregon) earthquake (PNSN uw62214776: 45.0753 N 123.1575 W, 15.1 km, 75
stations, gap 22 deg - onshore, well located: ground truth) recorded along the spans? 8-Hz stream (7.45 Hz), both cables,
config_nov (virtual mask for the unmasked Nov 6-26 h5 files; times = proc_start + i/fs: these hand-downsampled files
have no per-chunk metadata, so drops cannot be corrected; files are 17 min long).
Envelope image: 2-3.6 Hz and 1-2 Hz, log(envelope / pre-event median), along optical distance, vs time after the PNSN
origin (DAS stamps, latency of this setup not yet known).
Output: figs/amity_m32_section.png
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from msdvv import load_config
from msdvv.io import load_file

cfg = load_config(str(Path(__file__).resolve().parents[1] / "config_nov.yaml"))
T0 = pd.Timestamp("2025-11-23 15:58:26", tz="UTC").timestamp()
EV = (45.0753, -123.1575)
G = Geod(ellps="WGS84")
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
fig, axs = plt.subplots(2, 2, figsize=(18, 11))
for r, cab in enumerate(("north", "south")):
    p = sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20251123/*_{cab}_*"))[0]  # the earlier file (filename times are one file off)
    f = load_file(p, virtual_mask=cfg["virtual_mask"], dx_default=cfg["dx_default"])  # full 7.45 Hz (load_cfg would decimate)
    ch = f.channels()
    tt = (f.times if f.times is not None else f.t0 + np.arange(f.nt) / f.fs) - T0
    print(cab, "fs", round(f.fs, 3), "dx", round(f.dx, 1), "nch", f.data.shape[0], "t range", round(tt[0]), round(tt[-1]), flush=True)
    g = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    lat = np.interp(ch.opt_km, g.opt_km, g.lat); lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    dkm = G.inv(np.full(len(lon), EV[1]), np.full(len(lat), EV[0]), lon, lat)[2] / 1e3
    w = (tt > -120) & (tt < 200)
    ok = ch.block.notna().values
    for c_, band in enumerate(((1.0, 2.0), (2.0, 3.6))):
        x = sosfiltfilt(butter(4, band, "bandpass", fs=f.fs, output="sos"), f.data[ok][:, w].astype(np.float64), axis=1)
        env = uniform_filter1d(np.abs(hilbert(x, axis=1)) ** 2, int(2 * f.fs), axis=1)
        e = np.log(env / np.median(env[:, (tt[w] > -110) & (tt[w] < -5)], axis=1, keepdims=True))
        print(cab, band, "e finite", float(np.isfinite(e).mean()), "pct 50/99", np.nanpercentile(e[np.isfinite(e)], [50, 99]).round(2), flush=True)
        e = np.nan_to_num(e, nan=0.0, neginf=0.0, posinf=3.0)
        ax = axs[r, c_]
        pc = ax.pcolormesh(tt[w], ch.opt_km.values[ok], uniform_filter1d(e, 5, axis=0), vmin=0, vmax=3, cmap="magma", shading="auto")
        for v, col in ((6.0, "c"), (3.5, "lime")):
            ax.plot(np.hypot(dkm[ok], 15.1) / v, ch.opt_km.values[ok], ".", color=col, ms=0.5)
        ax.set(xlabel="s after PNSN origin (DAS stamps, latency not removed)", ylabel="optical km")
        ax.set_title(f"{cab}, {band[0]}-{band[1]} Hz; cyan P 6 km/s, green S 3.5 km/s (no latency)", fontsize=9)
fig.colorbar(pc, ax=axs, shrink=0.6)
fig.suptitle("Nov 23 2025 M3.2 Amity, Oregon (onshore, well located) on the 8-Hz stream")
fig.savefig(f"{OUT}/figs/amity_m32_section.png", dpi=90, bbox_inches="tight")
