"""Item 8b: does the Jan 13 2026 ML 2.0 appear on the curved south spans (B3, C4, D4) and north B3/C4 once the
channels of each span are stacked? Per 10-km stretch: mean over channels of the 2-3.6 Hz energy envelope (2-s
smoothing) / its pre-event median, no alignment (incoherent stack: noise std drops ~ sqrt(N)); event SNR = peak in
[0, 90] s relative to the pre-event std of the stacked trace.
Output: figs/das_location_jan13_span_stacks.png, outputs_100m/das_location_jan13_span_stacks.csv
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
rows, tr = [], []
for cab in ("north", "south"):
    f = load_file(sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_{cab}_*"))[0])
    ch = f.channels()
    g = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    lat = np.interp(ch.opt_km, g.opt_km, g.lat)
    lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    tt = f.times - T0
    w = (tt > -120) & (tt < 150)
    x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[:, w].astype(np.float64), axis=1)
    tw = tt[w]
    env = uniform_filter1d(np.abs(hilbert(x, axis=1)) ** 2, int(round(2 * f.fs)), axis=1)
    env /= np.median(env[:, (tw > -90) & (tw < -5)], axis=1, keepdims=True)
    ok = ch.block.notna().values & (ch.pos_in_block.values >= 3) & (ch.pos_in_block.values <= ch.block_len.values - 4)
    pre = (tw > -90) & (tw < -5)
    for blk, gb in ch[ok].groupby("block"):
        km = gb.opt_km.values
        for a in np.arange(np.floor(km.min() / 10) * 10, km.max(), 10):
            sel = gb.index.values[(km >= a) & (km < a + 10)]
            if len(sel) < 10:
                continue
            # robust incoherent stack: median over channels limits single noisy channels
            s = np.median(env[sel], axis=0)
            snr = (s[(tw >= 0) & (tw <= 90)].max() - np.median(s[pre])) / np.std(s[pre])
            tpk = tw[(tw >= 0) & (tw <= 90)][np.argmax(s[(tw >= 0) & (tw <= 90)])]
            rows.append(dict(cable=cab, block=blk, km0=a, n=len(sel), lat=float(lat[sel].mean()), lon=float(lon[sel].mean()), snr=float(snr), t_peak=float(tpk)))
            tr.append((f"{cab} {blk} {a:.0f}-{a + 10:.0f}", tw, s, snr))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/das_location_jan13_span_stacks.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(2).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(16, 12), sharex=True)
for ax, cab in zip(axs, ("north", "south")):
    k = 0
    for name, tw, s, snr in tr:
        if not name.startswith(cab):
            continue
        y = (s - np.median(s)) / (np.percentile(s, 99.5) - np.median(s) + 1e-9)
        ax.plot(tw, k + 0.9 * y, color="tab:red" if snr > 8 else "k", lw=0.8)
        ax.text(152, k, f"{name.split(' ', 1)[1]} km  SNR {snr:.0f}", fontsize=7, va="center")
        k += 1
    ax.set(xlabel="s after PNSN origin (DAS time; 8-Hz latency 5.8 s)", title=f"{cab}: median envelope per 10-km stretch (red: SNR > 8)", yticks=[])
fig.tight_layout()
fig.savefig(f"{OUT}/figs/das_location_jan13_span_stacks.png", dpi=95)
