"""Deep-learning phase picks for uw62216067 (2026-01-26 09:09:21, catalogue ML 2.2 "62 km W of Depoe Bay").

SeisBench (env pyfetch): PhaseNet and EQTransformer with the OBS-trained weights ("obs" = PickBlue, Bornstein et al.)
plus land-trained weights (PhaseNet "instance", EQTransformer "original") as a cross-check, on the 200-Hz records of
the Hydrate Ridge (HYSB1, HYS11-14) and Axial Seamount (AXBA1, AXCC1, AXEC2, AXAS1, AXID1) seafloor seismometers.
Outputs: /data/wsd02/multispan_dvv/obs/uw62216067_ml_picks.csv, figs/uw62216067_ml_picks.png
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal


def _np_detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
    """Linear (or constant) detrend in plain numpy: the pyfetch env's MKL lstsq fails inside scipy.signal.detrend."""
    x = np.moveaxis(np.asarray(data, dtype=np.float64), axis, -1)
    if type in ("constant", "c"):
        y = x - x.mean(-1, keepdims=True)
    else:
        n = x.shape[-1]
        t = np.arange(n) - (n - 1) / 2.0
        slope = (x * t).sum(-1, keepdims=True) / (t**2).sum()
        y = x - x.mean(-1, keepdims=True) - slope * t
    return np.moveaxis(y, -1, axis)


scipy.signal.detrend = _np_detrend
import seisbench.models as sbm
import torch
from obspy import UTCDateTime, read

OBS = "/data/wsd02/multispan_dvv/obs"
FIG = "/data/wsd02/multispan_dvv/outputs_100m/figs"
T0 = UTCDateTime("2026-01-26T09:09:21.46")
STA = ["HYSB1", "HYS11", "HYS12", "HYS13", "HYS14", "AXBA1", "AXCC1", "AXEC2", "AXAS1", "AXID1"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"PN-obs": sbm.PhaseNet.from_pretrained("obs"), "EQT-obs": sbm.EQTransformer.from_pretrained("obs"),
          "PN-instance": sbm.PhaseNet.from_pretrained("instance"), "EQT-original": sbm.EQTransformer.from_pretrained("original")}
for m in MODELS.values():
    m.to(dev)

st0 = read(f"{OBS}/uw62216067_OO.mseed")
st0.merge(fill_value=0)
for tr in st0:  # some records carry NaN/inf samples (gaps): the pickers need finite data
    tr.data = np.nan_to_num(tr.data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
print({tr.id: int(np.sum(tr.data == 0)) for tr in st0 if np.sum(tr.data == 0) > 100})
rows, ann = [], {}
for s in STA:
    st = st0.select(station=s).copy()
    if len(st) < 3:
        continue
    st.detrend("demean")
    st.resample(100.0)
    t_a, t_b = max(tr.stats.starttime for tr in st), min(tr.stats.endtime for tr in st)
    st.trim(t_a, t_b)  # identical spans: SeisBench pads mismatched components with NaN
    for tr in st:
        tr.data = np.nan_to_num(tr.data.astype(np.float32))
        tr.stats.starttime = t_a
    for c in st:  # SeisBench expects Z/N/E component codes
        c.stats.channel = c.stats.channel[:2] + c.stats.channel[-1]
    for name, m in MODELS.items():
        out = m.classify(st, P_threshold=0.2, S_threshold=0.2)
        if hasattr(out, "picks"):  # SeisBench >= 0.5: ClassifyOutput
            picks = out.picks
        elif isinstance(out, tuple):  # 0.4 EQTransformer: (picks, detections)
            picks = out[0]
        else:  # 0.4 PhaseNet: PickList
            picks = out
        for p in picks:
            rows.append(dict(station=s, model=name, phase=p.phase, time=str(p.peak_time), t_rel=float(p.peak_time - T0), prob=float(p.peak_value)))
        ann[(s, name)] = m.annotate(st)
P = pd.DataFrame(rows).sort_values(["station", "phase", "t_rel"])
P.to_csv(f"{OBS}/uw62216067_ml_picks.csv", index=False)
win = P[(P.t_rel > -90) & (P.t_rel < 200)]
pd.set_option("display.width", 200)
print(win.round(2).to_string(index=False))

# ---- figure: waveforms (2-15 Hz) + P/S probabilities per model
fig, axs = plt.subplots(len(STA), 1, figsize=(15, 2.3 * len(STA)), sharex=True)
for ax, s in zip(axs, STA):
    st = st0.select(station=s).copy()
    if len(st) < 3:
        ax.set_ylabel(s)
        continue
    st.detrend("demean")
    st.filter("bandpass", freqmin=2, freqmax=15, zerophase=True)
    for k, comp in enumerate("ZNE"):
        tr = [t for t in st if t.stats.channel.endswith(comp)][0]
        y = tr.data / (np.abs(tr.data).max() + 1e-30)
        ax.plot(tr.times(reftime=T0), 2.2 + k * 0.9 + 0.4 * y, color=("C0", "C1", "C2")[k], lw=0.35)
    for name, ls in (("PN-obs", "-"), ("EQT-obs", "--"), ("PN-instance", ":"), ("EQT-original", "-.")):
        a = ann.get((s, name))
        if a is None:
            continue
        for tr in a:
            ph = tr.stats.channel.split("_")[-1]
            if ph not in ("P", "S"):
                continue
            ax.plot(tr.times(reftime=T0), 2.0 * tr.data, color="red" if ph == "P" else "blue", ls=ls, lw=0.9,
                    label=f"{name} {ph}" if s == STA[0] else None)
    ax.set_ylabel(s, rotation=0, labelpad=25)
    ax.set_yticks([])
    ax.set_xlim(-60, 180)
axs[0].legend(fontsize=7, ncol=8, loc="upper right")
axs[-1].set_xlabel(f"s after catalogue origin {str(T0)[:22]} (waveforms 2-15 Hz, Z/N/E; red = P probability, blue = S probability)")
fig.suptitle("uw62216067: deep-learning phase probabilities (PhaseNet / EQTransformer, OBS-trained 'obs' and land-trained weights)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/uw62216067_ml_picks.png", dpi=95)
print("done")
