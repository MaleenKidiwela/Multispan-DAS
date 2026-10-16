"""Item 8b diagnostic: S-envelopes (2-3.6 Hz, log SNR as in s65) of all 1-km bins of both cables around the Jan 13 2026
ML 2.0, ordered along each cable (longitude) -> where and when the event is seen, without any location assumption.
Output: figs/das_location_jan13_envelopes.png
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
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from msdvv.io import load_file

T0 = pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp()
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
fig, axs = plt.subplots(2, 2, figsize=(18, 11), gridspec_kw=dict(width_ratios=[3, 1]))
for r, cab in enumerate(("north", "south")):
    f = load_file(sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_{cab}_*"))[0])
    ch = f.channels()
    g = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    tt = f.times - T0
    w = (tt > -120) & (tt < 150)
    x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[:, w].astype(np.float64), axis=1)
    tw = tt[w]
    a2 = np.abs(hilbert(x, axis=1)) ** 2
    env = uniform_filter1d(a2, int(round(2 * f.fs)), axis=1)
    e = np.log(env / np.median(env[:, (tw > -90) & (tw < -5)], axis=1, keepdims=True))
    ok = ch.block.notna().values & (ch.pos_in_block.values >= 3) & (ch.pos_in_block.values <= ch.block_len.values - 4)
    ax = axs[r, 0]
    pc = ax.pcolormesh(tw, ch.opt_km.values[ok], uniform_filter1d(e[ok], 5, axis=0), vmin=0, vmax=3, cmap="magma", shading="auto")
    ax.set(xlabel="s after PNSN origin (DAS time)", ylabel="optical distance (km)", title=f"{cab} cable: log(S-envelope / pre-event median), 2-3.6 Hz")
    fig.colorbar(pc, ax=ax)
    ax = axs[r, 1]
    snr = uniform_filter1d(e[ok][:, (tw > 0) & (tw < 120)].max(1), 5)
    ax.plot(snr, ch.opt_km.values[ok], ".", ms=2)
    ax2 = ax.twiny()
    ax2.plot(lon[ok], ch.opt_km.values[ok], "-", color="0.6", lw=1)
    ax.set(xlabel="event SNR (log)", title="SNR (blue) and channel longitude (grey, top axis)")
    ax2.axvline(-124.39, color="r", ls=":")
fig.tight_layout()
fig.savefig(f"{OUT}/figs/das_location_jan13_envelopes.png", dpi=95)
