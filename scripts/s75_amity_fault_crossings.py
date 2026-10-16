"""Item 1 / 8k: do the Nov 23 2025 M3.2 Amity earthquake's waves show the Alvin Canyon and Daisy Bank faults where the
south cable crosses them (route km 274.5 and 295.6; s57)?

Data: south cable, 8-Hz stream (7.45 Hz, 105.9 m), FULL fiber (Nov 6-26 h5 files are unmasked; no virtual mask
applied here), proc_start + i/fs times. Section: optical channels from route km 240 to 330 (shelf off Newport).
Per channel (5-channel running mean to suppress single bad channels):
  - physical-channel check: coherence with neighbours in the pre-event noise (as item 1) - channels below 0.15 flagged;
  - S-wave envelope (2-3.6 Hz and 1-2 Hz): onset time (20 % of the channel's maximum above noise), peak amplitude
    (log10, relative to the section median), duration (time above 50 % of the maximum);
  - local apparent slowness along the cable: slope of the onset times over a 2-km sliding window.
A fault zone would show as a step in onset time, amplification and longer shaking, a slowness change, or arrivals
starting at the crossing and moving away (record section).
Outputs: figs/amity_fault_crossings.png, outputs_100m/amity_fault_crossings.csv
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from msdvv import load_config
from msdvv.io import load_file

cfg = load_config(str(Path(__file__).resolve().parents[1] / "config_nov.yaml"))
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
T0 = pd.Timestamp("2025-11-23 15:58:26", tz="UTC").timestamp()
CROSS = {"Alvin Canyon fault": 274.5, "Daisy Bank fault": 295.6}
p = sorted(glob.glob("/data/wsd02/multispan_dvv/raw_hf/eq20251123/*_south_*"))[0]
f = load_file(p, dx_default=cfg["dx_default"])  # full fiber: no virtual mask
nch = f.data.shape[0]
opt_km = np.arange(nch) * f.dx / 1e3
g = pd.read_csv(f"{OUT}/geometry_south.csv")
route_km = np.interp(opt_km, g.opt_km, g.route_km, left=np.nan, right=np.nan)
tt = (f.times if f.times is not None else f.t0 + np.arange(f.nt) / f.fs) - T0
sel = np.where((route_km >= 240) & (route_km <= 330))[0]
print(f"{nch} channels on the full fiber; section route km 240-330: {len(sel)} channels (optical {opt_km[sel[0]]:.1f}-{opt_km[sel[-1]]:.1f} km)", flush=True)
w = (tt > -150) & (tt < 220)
tw = tt[w]
X = f.data[sel][:, w].astype(np.float64)
pre = (tw > -140) & (tw < -10)
# physical-channel check: zero-lag coherence with the neighbour in 0.5-2 Hz pre-event noise
xn = sosfiltfilt(butter(4, (0.5, 2.0), "bandpass", fs=f.fs, output="sos"), X[:, pre], axis=1)
coh = np.r_[[np.corrcoef(xn[i], xn[i + 1])[0, 1] for i in range(len(sel) - 1)], np.nan]
rows = {"route_km": route_km[sel], "opt_km": opt_km[sel], "coh": coh}
fig, axs = plt.subplots(2, 3, figsize=(20, 11), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
for r, band in enumerate(((2.0, 3.6), (1.0, 2.0))):
    x = sosfiltfilt(butter(4, band, "bandpass", fs=f.fs, output="sos"), X, axis=1)
    x = uniform_filter1d(x, 5, axis=0)  # 5-channel running mean (~0.5 km)
    env = uniform_filter1d(np.abs(hilbert(x, axis=1)) ** 2, int(f.fs), axis=1)
    nz = np.median(env[:, pre], axis=1)
    ev = (tw >= 0) & (tw <= 150)
    mx = env[:, ev].max(1)
    on = np.array([tw[ev][np.argmax(env[i, ev] >= nz[i] + 0.2 * (mx[i] - nz[i]))] for i in range(len(sel))])
    dur = np.array([(env[i, ev] >= nz[i] + 0.5 * (mx[i] - nz[i])).sum() / f.fs for i in range(len(sel))])
    amp = np.log10(mx / np.median(mx))
    snr = np.log10(mx / nz)
    # local apparent slowness from onsets in 2-km windows
    slo = np.full(len(sel), np.nan)
    rk = route_km[sel]
    for i in range(len(sel)):
        m = np.abs(rk - rk[i]) <= 1.0
        if m.sum() >= 8:
            slo[i] = np.polyfit(rk[m], on[m], 1)[0]
    tag = f"{band[0]:g}-{band[1]:g}Hz"
    rows.update({f"onset_{tag}": on, f"amp_{tag}": amp, f"dur_{tag}": dur, f"snr_{tag}": snr, f"slow_{tag}": slo})
    ax = axs[r, 0]
    img = np.log10(env / nz[:, None])
    ax.pcolormesh(tw, rk, img, vmin=0, vmax=2.5, cmap="magma", shading="auto")
    for k_, v in CROSS.items():
        ax.axhline(v, color="c", ls="--", lw=1)
        ax.text(tw[-1], v, " " + k_, color="c", fontsize=8, va="center")
    ax.plot(on, rk, "w.", ms=1)
    ax.set(xlim=(-20, 150), xlabel="s after PNSN origin (DAS stamps)", ylabel="route km (south cable)",
           title=f"{tag}: log10 envelope / pre-event noise, 5-channel mean; white dots = onset (20 %)")
    ax = axs[r, 1]
    ax.plot(amp, rk, "k-", lw=0.8, label="log10 peak amplitude (rel. median)")
    ax.plot(snr / 2, rk, "-", color="tab:orange", lw=0.8, label="log10 SNR / 2")
    ax.plot(coh, rk, "-", color="tab:green", lw=0.8, label="noise coherence (physical >= 0.15)")
    for v in CROSS.values():
        ax.axhline(v, color="c", ls="--", lw=1)
    ax.set(xlabel="value", title="amplitude, SNR, channel coherence")
    ax.legend(fontsize=7)
    ax = axs[r, 2]
    ax.plot(on - np.nanmedian(on), rk, "k-", lw=0.8, label="onset - median (s)")
    ax.plot(slo * 10, rk, "-", color="tab:purple", lw=0.8, label="local slowness x10 (s/km)")
    ax.plot(dur / 10, rk, "-", color="tab:blue", lw=0.8, label="duration / 10 (s)")
    for v in CROSS.values():
        ax.axhline(v, color="c", ls="--", lw=1)
    ax.set(xlabel="value", title="onset, local slowness, duration")
    ax.legend(fontsize=7)
fig.suptitle("Nov 23 2025 M3.2 Amity on the south cable across the Alvin Canyon and Daisy Bank fault crossings (8-Hz stream, full fiber)")
fig.tight_layout()
fig.savefig(f"{OUT}/figs/amity_fault_crossings.png", dpi=95)
pd.DataFrame(rows).to_csv(f"{OUT}/amity_fault_crossings.csv", index=False)
R = pd.DataFrame(rows)
for k_, v in CROSS.items():
    near = (R.route_km - v).abs() <= 1.5
    ctrl = ((R.route_km - v).abs() > 4) & ((R.route_km - v).abs() <= 10)
    print(f"{k_} (route km {v}): channels +-1.5 km {near.sum()}, coherence {R.coh[near].median():.2f} (control {R.coh[ctrl].median():.2f}); "
          f"2-3.6 Hz amp {R['amp_2-3.6Hz'][near].median():+.2f} vs {R['amp_2-3.6Hz'][ctrl].median():+.2f}; "
          f"SNR {R['snr_2-3.6Hz'][near].median():.2f} vs {R['snr_2-3.6Hz'][ctrl].median():.2f}; "
          f"duration {R['dur_2-3.6Hz'][near].median():.1f} vs {R['dur_2-3.6Hz'][ctrl].median():.1f} s", flush=True)
