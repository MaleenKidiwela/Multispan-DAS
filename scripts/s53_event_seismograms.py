"""Seismograms of uw62216067 at the Hydrate Ridge seafloor seismometers with the final picks.

Left: Z/N/E (2-15 Hz) from -15 to +45 s after the catalogue origin, with the consensus deep-learning P (red) and S
(blue) picks and the rejected alternative S at ~+13.5 s (orange dashed). Right: vertical envelope from -20 to +170 s
with P, S and the T-phase peak (green) - the T-phase is only consistent with S at ~+24 s (README).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read
from obspy.signal.filter import envelope

T0 = UTCDateTime("2026-01-26T09:09:21.46")
J = json.load(open("/data/wsd02/multispan_dvv/obs/uw62216067_relocation_ml.json"))
PK = {(p["sta"], p["phase"]): UTCDateTime(p["t"]).timestamp - T0.timestamp for p in J["picks"]}
ALT = {"HYS11": 14.04, "HYS12": 14.35, "HYS13": 14.06, "HYS14": 13.88, "HYSB1": 5.42}
TP = J["tphase_obs_s"]
STA = ["HYSB1", "HYS14", "HYS11", "HYS12", "HYS13"]
st = read("/data/wsd02/multispan_dvv/obs/uw62216067_OO.mseed")
st.merge(fill_value=0)
for tr in st:
    tr.data = np.nan_to_num(tr.data.astype(float))
st.detrend("demean")
st.filter("bandpass", freqmin=2, freqmax=15, zerophase=True)
fig, axs = plt.subplots(len(STA), 2, figsize=(17, 2.6 * len(STA)), gridspec_kw=dict(width_ratios=[1.35, 1]), squeeze=False)
for i, s in enumerate(STA):
    tr = {t.stats.channel[-1]: t for t in st.select(station=s)}
    ax = axs[i, 0]
    tt = tr["Z"].times(reftime=T0)
    m = (tt >= -15) & (tt <= 45)
    sc = max(np.abs(tr[c].data[m]).max() for c in "ZNE") + 1e-30
    for k, (c, col) in enumerate((("Z", "k"), ("N", "C1"), ("E", "C2"))):
        ax.plot(tt[m], 2 - k + 0.9 * tr[c].data[m] / sc, color=col, lw=0.5)
        ax.text(-14.5, 2 - k + 0.25, f"{tr[c].stats.channel}", fontsize=7, color=col)
    ax.axvline(PK[(s, "P")], color="red", lw=1.5, label=f"P (deep-learning consensus)")
    ax.axvline(PK[(s, "S")], color="blue", lw=1.5, label="S (deep-learning consensus)")
    ax.axvline(ALT[s], color="darkorange", ls="--", lw=1.2, label="alternative 'S' (rejected)")
    ax.set(xlim=(-15, 45), yticks=[])
    ax.set_ylabel(s, rotation=0, labelpad=28, fontsize=11, fontweight="bold")
    ax.set_title(f"{s}: P {PK[(s, 'P')]:+.2f} s, S {PK[(s, 'S')]:+.2f} s, S-P {PK[(s, 'S')] - PK[(s, 'P')]:.1f} s", fontsize=9, loc="left")
    if i == 0:
        ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax = axs[i, 1]
    m2 = (tt >= -20) & (tt <= 170)
    env = envelope(tr["Z"].data)
    k_ = int(2 * tr["Z"].stats.sampling_rate)
    es = np.convolve(env, np.ones(k_) / k_, "same")
    ax.fill_between(tt[m2], 0, es[m2] / es[m2].max(), color="0.6")
    ax.axvline(PK[(s, "P")], color="red", lw=1.2)
    ax.axvline(PK[(s, "S")], color="blue", lw=1.2)
    ax.axvline(TP[s], color="green", lw=1.5, label="T-phase peak")
    ax.set(xlim=(-20, 170), ylim=(0, 1.05), yticks=[])
    ax.set_title(f"Z envelope (2-s smoothing): T-phase peak {TP[s]:+.0f} s", fontsize=9, loc="left")
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")
for ax in axs[-1]:
    ax.set_xlabel(f"s after catalogue origin {str(T0)[:22]} UTC (2-15 Hz)")
fig.suptitle("uw62216067 (catalogue ML 2.2 '62 km W of Depoe Bay') at the Hydrate Ridge seafloor seismometers: final picks\n"
             "relocated 43.35N 127.19W, origin 09:08:56.6 UTC (oceanic 1-D model); S at ~+24 s fits P/S and T-phase, the +13.5-s arrival does not", fontsize=11)
fig.tight_layout()
fig.savefig("/data/wsd02/multispan_dvv/outputs_100m/figs/uw62216067_seismograms_picks.png", dpi=105)
print("done")
