"""Item 8b: is the Jan 13 2026 ML 2.0 recorded on the southern loop of the south cable (spans B3, C4, D4; 60-90 km
south of the event) once ALL its bins are stacked along predicted S moveout (delay-and-sum = back-projection beam)?
Bins: s65 1-km bins of blocks B, C, D (2-3.6 Hz energy envelope / pre-event median, 2-s smoothing).
Delays: S travel time from trial sources (PNSN epicentre; s65f joint solution; a grid of points) with Vs = 3.5 km/s,
z = 10 km; DAS time -> UTC with the 8-Hz latency 5.8 s (s67). Stack = mean over bins of the aligned log envelopes,
time axis = origin time relative to PNSN. Noise level: std of the stack in the pre-event part.
Also the unaligned stack and a 'null' stack with delays for a source mirrored 180 deg (wrong moveout).
Output: figs/das_location_jan13_southloop_stack.png
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

from msdvv.io import load_file

G = Geod(ellps="WGS84")
T0 = pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp()
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
LAT8, VS, Z = 5.8, 3.5, 10.0
f = load_file(sorted(glob.glob("/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_south_*"))[0])
ch = f.channels()
g = pd.read_csv(f"{OUT}/geometry_south.csv")
lat = np.interp(ch.opt_km, g.opt_km, g.lat)
lon = np.interp(ch.opt_km, g.opt_km, g.lon)
tt = f.times - LAT8 - T0
w = (tt > -150) & (tt < 200)
x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[:, w].astype(np.float64), axis=1)
tw = tt[w]
env = uniform_filter1d(np.abs(hilbert(x, axis=1)) ** 2, int(round(2 * f.fs)), axis=1)
e = np.log(env / np.median(env[:, (tw > -120) & (tw < -10)], axis=1, keepdims=True))
ok = ch.block.isin(["B", "C", "D"]).values & (ch.pos_in_block.values >= 3) & (ch.pos_in_block.values <= ch.block_len.values - 4)
B, LA, LO = [], [], []
for blk, gb in ch[ok].groupby("block"):
    idx = gb.index.values
    for k in range(0, len(idx) - 4, 5):
        sel = idx[k:k + 5]
        B.append(np.log(np.mean(np.exp(e[sel]), axis=0)))
        LA.append(lat[sel].mean()); LO.append(lon[sel].mean())
B, LA, LO = np.array(B), np.array(LA), np.array(LO)
print(len(B), "southern-loop bins")
tg = np.arange(-60, 120, 0.25)


def stack(src_lat, src_lon):
    _, _, d = G.inv(LO, LA, np.full(len(LO), src_lon), np.full(len(LA), src_lat))
    ts = np.hypot(d / 1e3, Z) / VS
    return np.mean([np.interp(tg + t, tw, b) for b, t in zip(B, ts)], axis=0), ts


cases = {"PNSN epicentre": (45.162, -124.39), "joint DAS+OBS (s65f)": (45.06, -124.53), "mirror (wrong side, null)": (43.9, -124.9)}
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
for name, (la_, lo_) in cases.items():
    s, ts = stack(la_, lo_)
    pre = tg < -10
    z = (s - np.median(s[pre])) / np.std(s[pre])
    axs[0].plot(tg, z, label=f"{name}: S times {ts.min():.0f}-{ts.max():.0f} s; peak {z[(tg > -10) & (tg < 30)].max():.1f} sigma "
                            f"at {tg[(tg > -10) & (tg < 30)][np.argmax(z[(tg > -10) & (tg < 30)])]:+.1f} s")
    print(name, f"S {ts.min():.1f}-{ts.max():.1f} s; max z in origin window {z[(tg > -10) & (tg < 30)].max():.1f}")
u = B.mean(0)
uz = (u - np.median(u[tw < -10])) / np.std(u[tw < -10])
axs[1].plot(tw, uz, "k", lw=0.8)
axs[0].axvline(0, color="0.5", ls=":")
axs[0].set(xlabel="origin time relative to PNSN (s, after removing predicted S travel time)", ylabel="stack (sigma above pre-event)",
           title=f"South-cable southern loop (B3, C4, D4; {len(B)} bins) delay-and-sum of S envelopes")
axs[0].legend(fontsize=8)
axs[1].set(xlabel="UTC time after PNSN origin (s), no alignment", ylabel="sigma", title="unaligned mean of the same bins")
fig.tight_layout()
fig.savefig(f"{OUT}/figs/das_location_jan13_southloop_stack.png", dpi=100)
