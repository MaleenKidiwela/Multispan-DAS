"""Item 1 (docs/NEXT_STEPS.md): are there physical DAS channels where the south cable crosses the mapped faults?

Crossings (Qfaults traces x cable route): Alvin Canyon fault at route km 274.5, Daisy Bank fault at route km 295.6
(both on the shelf, ~270 m). In the 100-m stream every channel within 5 km of both crossings fails the coherence test.
Here the same metric (magnitude-squared coherence between channel i and the channel 0.42 km further, mean over
0.12-0.6 Hz, 120-s segments; physical if >= 0.15) is computed for every recording configuration available on disk:
  Nov 5 / Nov 20  downsampled h5, 105.9 m, gauge 256, unmasked full fibre
  Nov 27 / Dec 1  raw, 52.9 m, gauge 128, unmasked full fibre
  Dec 5 / Dec 15  raw, 52.9 m, gauge 128, masked
  Dec 25-29 (fine-spacing 8-Hz stream sample) 13.2 m, gauge 13, masked
  Jan (100-m stream, reference: outputs_100m/channel_quality_south.csv)
Route km from optical km by the south-cable linear map (fit to the 100-m stream geometry).
Outputs: outputs_100m/fault_crossings.csv, figs/fault_crossings_coherence.png
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
from scipy.signal import butter, sosfiltfilt

from msdvv.io import load_file, read_meta

B = Path("/data/wsd02/multispan_dvv")
CROSS = {"Alvin Canyon fault": 274.5, "Daisy Bank fault": 295.6}
REF = (255.0, 267.0)  # route km of the good C4 bins used by the final within-span product
WIN_KM = 20.0
OFF_KM = 0.42
geo = pd.read_csv(B / "outputs_100m/geometry_south.csv")
a_, b_ = np.polyfit(geo.opt_km, geo.route_km, 1)  # route = a*opt + b
opt_of = lambda r: (r - b_) / a_  # noqa: E731
print(f"route = {a_:.4f} * opt {b_:+.2f}")
need = [(opt_of(REF[0]) - 2, opt_of(REF[1]) + 2)] + [(opt_of(r) - WIN_KM, opt_of(r) + WIN_KM) for r in CROSS.values()]

files = sorted(glob.glob(str(B / "raw_nov/sample/*south*"))) + sorted(p_ for p_ in glob.glob(str(B / "raw_hf/sample/**/nokia_DAS_south*"), recursive=True) if Path(p_).is_file())[::6]


def label(f):
    return f"{pd.Timestamp(f.t0, unit='s'):%b %d} {f.dx:.1f} m g{int(f.attrs['gauge'])} {'masked' if 'masked' in f.attrs and len(np.atleast_2d(f.attrs['masked'])) and f.nch * f.dx / 1e3 < 600 else 'full fibre'}"


rows = []
for p in files:
    m = read_meta(p, 105.85775068689657)
    tsl = slice(0, int(min(m.nt, 1800 * m.fs)))  # first 30 min
    f = load_file(p, tslice=tsl, dx_default=105.85775068689657)
    nch = f.data.shape[0]
    if "masked" in f.attrs and int(sum((np.asarray(f.attrs["masked"])[:, 1] - np.asarray(f.attrs["masked"])[:, 0]) / f.dx)) == nch:
        ct = f.channels()
        opt = ct.opt_km.values
        blk = ct.block.values
    else:
        opt = np.arange(nch) * f.dx / 1e3
        blk = np.zeros(nch, int)
    sel = np.zeros(nch, bool)
    for lo, hi in need:
        sel |= (opt >= lo) & (opt <= hi)
    idx = np.where(sel)[0]
    if len(idx) < 10:
        print("no channels near crossings:", Path(p).name)
        continue
    k = max(1, int(round(OFF_KM * 1e3 / f.dx)))
    x = f.data[idx[0]: idx[-1] + k + 1].astype(np.float64)
    o = opt[idx[0]: idx[-1] + k + 1]
    bl = blk[idx[0]: idx[-1] + k + 1]
    sos = butter(4, [0.1, min(0.8, 0.45 * f.fs)], "bandpass", fs=f.fs, output="sos")
    x = sosfiltfilt(sos, x, axis=1)
    n = int(120 * f.fs)
    segs = [x[:, s: s + n] for s in range(0, x.shape[1] - n + 1, n)]
    Xs = np.stack([np.fft.rfft(sg * np.hanning(n), axis=1) for sg in segs], axis=1)  # (nch, nseg, nf)
    fr = np.fft.rfftfreq(n, 1 / f.fs)
    band = (fr >= 0.12) & (fr <= min(0.6, 0.45 * f.fs))
    Xs = Xs[:, :, band]
    pxx = np.mean(np.abs(Xs) ** 2, axis=1)
    sxy = np.mean(Xs[:-k] * np.conj(Xs[k:]), axis=1)
    coh = np.mean(np.abs(sxy) ** 2 / (pxx[:-k] * pxx[k:] + 1e-40), axis=1)
    same = bl[:-k] == bl[k:]
    coh[~same] = np.nan
    lab = label(f)
    for oo, c in zip(o[:-k], coh):
        rows.append(dict(file=Path(p).name, config=lab, dx_m=f.dx, opt_km=oo, route_km=a_ * oo + b_, coh=c))
    print(f"{lab}: {len(coh)} channels analysed, k={k}", flush=True)
D = pd.DataFrame(rows)
D.to_csv(B / "outputs_100m/fault_crossings.csv", index=False)

# reference: 100-m stream (January) channel quality
Q = pd.read_csv(B / "outputs_100m/channel_quality_south.csv")
summary = []
for cfg_, g in list(D.groupby("config", sort=False)) + [("Jan 100-m stream 105.9 m g256 masked (s05)", Q.rename(columns={"coh": "coh"}))]:
    r = dict(config=cfg_)
    for nm, rk in list(CROSS.items()) + [("reference C4 255-267", None)]:
        w = g[(g.route_km - rk).abs() <= 3] if rk is not None else g[g.route_km.between(*REF)]
        r[f"{nm}: n"] = int(w.coh.notna().sum())
        r[f"{nm}: median coh"] = round(float(w.coh.median()), 3) if w.coh.notna().any() else np.nan
        r[f"{nm}: frac >= 0.15"] = round(float((w.coh >= 0.15).mean()), 2) if w.coh.notna().any() else np.nan
    summary.append(r)
S = pd.DataFrame(summary)
S.to_csv(B / "outputs_100m/fault_crossings_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.to_string(index=False))

cfgs = list(D.config.unique())
fig, axs = plt.subplots(len(CROSS), 1, figsize=(14, 4.5 * len(CROSS)), squeeze=False)
for ax, (nm, rk) in zip(axs[:, 0], CROSS.items()):
    for i, c in enumerate(cfgs):
        g = D[(D.config == c) & ((D.route_km - rk).abs() <= WIN_KM)].sort_values("route_km")
        sm = g.coh.rolling(max(1, int(round(500 / g.dx_m.iloc[0]))) if len(g) else 1, center=True, min_periods=1).median()
        ax.plot(g.route_km, sm, lw=1.0, label=c, color=plt.get_cmap("tab10")(i % 10))
    q = Q[(Q.route_km - rk).abs() <= WIN_KM]
    ax.plot(q.route_km, q.coh, "k.", ms=3, label="Jan 100-m stream (s05)")
    ax.axvline(rk, color="saddlebrown", lw=2, label=f"{nm} crossing")
    ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
    ax.set(xlim=(rk - WIN_KM, rk + WIN_KM), ylim=(0, 1), xlabel="south cable route km", ylabel="coherence (0.42 km, 0.12-0.6 Hz)",
           title=f"{nm}: channel coherence around the crossing (dashed = physical-channel threshold 0.15; gaps = masked/outside record)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(B / "outputs_100m/figs/fault_crossings_coherence.png", dpi=100)
print("done")
