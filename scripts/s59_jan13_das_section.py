"""Item 3a: record section of the Jan 13 2026 05:12:47 ML 2.0 local earthquake (uw, 45.162N 124.390W, ~0.6 km from
the south cable A1 span) on both cables, 8-Hz stream (211.7 m, 7.45 Hz, D512), with TRUE per-sample times (the
Jan 12-13 files lost 31-46 chunks: nominal times are wrong by up to 164-243 s here).

Channels within 40 km (hypocentral), 2-3.6 Hz, envelope normalised per channel, sorted by hypocentral distance;
time axis = true DAS time - origin - LAT (DAS latency estimated from the S-aligned stack peak, s17: 13-14 s);
predicted P (6.0 km/s) and S (3.45 km/s) curves. Figure: figs/das_jan13_record_section.png
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
E = dict(t=pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp(), lat=45.1618, lon=-124.390, z=1.0)
LAT = 13.5
VP, VS = 6.0, 3.45
fig, axs = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, cab in zip(axs, ("south", "north")):
    p = sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_{cab}_*"))[0]
    f = load_file(p)
    ch = f.channels()
    g = pd.read_csv(f"/data/wsd02/multispan_dvv/outputs_100m/geometry_{cab}.csv")
    lat = np.interp(ch.opt_km, g.opt_km, g.lat)
    lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    _, _, d = G.inv(lon, lat, np.full(len(ch), E["lon"]), np.full(len(ch), E["lat"]))
    hyp = np.hypot(d / 1e3, E["z"])
    tt = f.times - E["t"] - LAT
    w = (tt > -20) & (tt < 40)
    sel = np.where((hyp <= 40) & (ch.pos_in_block.values > 3))[0]
    x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[sel][:, w], axis=1)
    env = uniform_filter1d(np.abs(hilbert(x, axis=1)), int(0.5 * f.fs), axis=1)
    env /= np.median(env, axis=1, keepdims=True)
    o = np.argsort(hyp[sel])
    ax.pcolormesh(tt[w], hyp[sel][o], np.log10(env[o]), cmap="magma", vmin=0, vmax=1.2, shading="nearest", rasterized=True)
    hh = np.linspace(hyp[sel].min(), hyp[sel].max(), 50)
    ax.plot(hh / VP, hh, "c--", lw=1.2, label=f"P, {VP} km/s")
    ax.plot(hh / VS, hh, "w--", lw=1.2, label=f"S, {VS} km/s")
    ax.set(xlabel=f"true DAS time - origin - {LAT:g} s latency (s)", ylabel="hypocentral distance (km)",
           title=f"{cab} cable: {len(sel)} channels (211.7 m) within 40 km, 2-3.6 Hz envelope (log10 / median)")
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Jan 13 2026 05:12:47 UTC ML 2.0 (32 km W of Neskowin) on the OOI DAS, true per-sample times", fontsize=11)
fig.tight_layout()
fig.savefig("/data/wsd02/multispan_dvv/outputs_100m/figs/das_jan13_record_section.png", dpi=105)
print("done")
