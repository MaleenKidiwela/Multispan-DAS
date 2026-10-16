"""Seismograms of uw62216067 at the Hydrate Ridge seafloor seismometers with the deep-learning phase probabilities.

Per station: Z/N/E (2-15 Hz) on top, and below them the P (red) and S (blue) probability traces of PhaseNet and
EQTransformer with OBS-trained ('obs', PickBlue) and land-trained (PhaseNet 'instance', EQTransformer 'original')
weights; vertical lines = final consensus picks. Left column -15..+45 s, right column -20..+170 s (incl. T-phase).
Run in env pyfetch (SeisBench). Figure: figs/uw62216067_ml_prob_seismograms.png
"""
import json

import numpy as np
import scipy.signal


def _np_detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
    """Plain-numpy detrend: the pyfetch env's MKL lstsq fails inside scipy.signal.detrend."""
    x = np.moveaxis(np.asarray(data, dtype=np.float64), axis, -1)
    if type in ("constant", "c"):
        y = x - x.mean(-1, keepdims=True)
    else:
        n = x.shape[-1]
        t = np.arange(n) - (n - 1) / 2.0
        y = x - x.mean(-1, keepdims=True) - (x * t).sum(-1, keepdims=True) / (t**2).sum() * t
    return np.moveaxis(y, -1, axis)


scipy.signal.detrend = _np_detrend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seisbench.models as sbm
import torch
from obspy import UTCDateTime, read

T0 = UTCDateTime("2026-01-26T09:09:21.46")
STA = ["HYSB1", "HYS14", "HYS11", "HYS12", "HYS13"]
J = json.load(open("/data/wsd02/multispan_dvv/obs/uw62216067_relocation_ml.json"))
PK = {(p["sta"], p["phase"]): UTCDateTime(p["t"]).timestamp - T0.timestamp for p in J["picks"]}
TP = J["tphase_obs_s"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"PhaseNet obs": (sbm.PhaseNet.from_pretrained("obs"), "-"), "EQTransformer obs": (sbm.EQTransformer.from_pretrained("obs"), "--"),
          "PhaseNet instance": (sbm.PhaseNet.from_pretrained("instance"), ":"), "EQTransformer original": (sbm.EQTransformer.from_pretrained("original"), "-.")}
for m, _ in MODELS.values():
    m.to(dev)
st0 = read("/data/wsd02/multispan_dvv/obs/uw62216067_OO.mseed")
st0.merge(fill_value=0)
for tr in st0:
    tr.data = np.nan_to_num(tr.data.astype(np.float64))

fig = plt.figure(figsize=(18, 4.6 * len(STA)))
gs = fig.add_gridspec(3 * len(STA), 2, height_ratios=[1.5, 1, 0.35] * len(STA), width_ratios=[1.3, 1], hspace=0.12, wspace=0.08)
for i, s in enumerate(STA):
    st = st0.select(station=s).copy()
    st.detrend("demean")
    st.resample(100.0)
    a, b = max(t.stats.starttime for t in st), min(t.stats.endtime for t in st)
    st.trim(a, b)
    for t in st:
        t.stats.starttime = a
        t.stats.channel = t.stats.channel[:2] + t.stats.channel[-1]
        t.data = t.data.astype(np.float32)
    prob = {name: m.annotate(st) for name, (m, _) in MODELS.items()}
    wf = st.copy()
    wf.filter("bandpass", freqmin=2, freqmax=15, zerophase=True)
    tr = {t.stats.channel[-1]: t for t in wf}
    tt = tr["Z"].times(reftime=T0)
    for j, (x0, x1) in enumerate(((-15, 45), (-20, 170))):
        axw = fig.add_subplot(gs[3 * i, j])
        axp = fig.add_subplot(gs[3 * i + 1, j], sharex=axw)
        m_ = (tt >= x0) & (tt <= x1)
        sc = max(np.abs(tr[c].data[m_]).max() for c in "ZNE") + 1e-30
        for k, (c, col) in enumerate((("Z", "k"), ("N", "C1"), ("E", "C2"))):
            axw.plot(tt[m_], 2 - k + 0.9 * tr[c].data[m_] / sc, color=col, lw=0.45)
            if j == 0:
                axw.text(x0 + 0.4, 2 - k + 0.3, tr[c].stats.channel, fontsize=7, color=col)
        for ax in (axw, axp):
            ax.axvline(PK[(s, "P")], color="red", lw=1.2, alpha=0.7)
            ax.axvline(PK[(s, "S")], color="blue", lw=1.2, alpha=0.7)
            if j == 1:
                ax.axvline(TP[s], color="green", lw=1.2, alpha=0.7)
        for name, (_, ls) in MODELS.items():
            for p in prob[name]:
                ph = p.stats.channel.split("_")[-1]
                if ph not in ("P", "S"):
                    continue
                pt = p.times(reftime=T0)
                mm = (pt >= x0) & (pt <= x1)
                axp.plot(pt[mm], p.data[mm], color="red" if ph == "P" else "blue", ls=ls, lw=1.0,
                         label=f"{name} {ph}" if (i == 0 and j == 0) else None)
        axw.set(yticks=[], xlim=(x0, x1))
        axp.set(ylim=(0, 1.02), xlim=(x0, x1))
        axp.set_yticks([0, 0.5, 1])
        axp.tick_params(labelsize=7)
        plt.setp(axw.get_xticklabels(), visible=False)
        if j == 0:
            axw.set_ylabel(s, rotation=0, labelpad=30, fontsize=11, fontweight="bold")
            axp.set_ylabel("probability", fontsize=8)
            axw.set_title(f"{s}: P {PK[(s, 'P')]:+.2f} s, S {PK[(s, 'S')]:+.2f} s (S-P {PK[(s, 'S')] - PK[(s, 'P')]:.1f} s)", fontsize=9, loc="left")
        else:
            axw.set_title(f"full record: T-phase peak {TP[s]:+.0f} s (green) - no model picks it as P or S", fontsize=9, loc="left")
            axp.set_yticklabels([])
        if i == len(STA) - 1:
            axp.set_xlabel(f"s after catalogue origin {str(T0)[:22]} UTC (waveforms 2-15 Hz)")
        if i == 0 and j == 0:
            axp.legend(fontsize=7, ncol=4, loc="upper right")
fig.suptitle("uw62216067 at the Hydrate Ridge seafloor seismometers: seismograms and deep-learning phase probabilities\n"
             "red = P, blue = S; solid PhaseNet-obs, dashed EQTransformer-obs (OBS-trained, PickBlue), dotted PhaseNet-instance, dash-dot EQTransformer-original; "
             "vertical lines = final picks", fontsize=11, y=0.905)
fig.savefig("/data/wsd02/multispan_dvv/outputs_100m/figs/uw62216067_ml_prob_seismograms.png", dpi=100, bbox_inches="tight")
print("done")
