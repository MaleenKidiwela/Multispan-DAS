"""Item 8j: multi-span DAS location of a GROUND-TRUTH local earthquake: Nov 23 2025 M3.2 Amity, Oregon (PNSN uw62214776,
45.0753 N 123.1575 W, 15.1 km depth; 75 stations, azimuthal gap 22 deg - onshore and well located).

Data: 8-Hz stream (7.45 Hz, 105.9 m channels), both cables, config_nov virtual mask; times = proc_start + i/fs (these
hand-downsampled Nov 6-26 files carry no per-chunk metadata). Band 2-3.6 Hz, S envelopes (2-s smoothing, log, divided by
the pre-event median).
1) Lit stretches, data-driven: 5-km segments whose median-envelope stack has event SNR >= SNR_SEG (log 1.5 above
   pre-event, i.e. 4.5x); contiguous segments merged; stretches >= 5 km kept.
2) Within each stretch: envelope cross-correlation against the stretch stack (3 iterations) -> relative delay per
   1-km bin (10 channels); bins with CC < 0.5 dropped. Stretch onset = 20 % of the aligned stack maximum.
   Per-bin time = stretch onset + (delay - mean delay).
3) Location: grid lat 44.0-46.4 N, lon 125.4-122.4 W (0.02 deg) x depth {5..25 km} x Vs 2.8-4.0 km/s, L1 misfit,
   MODE=wavefront (default): one free time offset per stretch, i.e. only the wavefront shape inside each stretch is used
   (robust to clock offsets between files and to which phase sets a stretch's onset) - multi-span triangulation;
   MODE=common: one time offset for all stretches (origin + latency; both cables share the latency, s67).
   Only stretches with median envelope CC >= 0.8 are used. Each stretch weighted equally. Uncertainty: misfit <= min + 0.3 s.
4) Comparison: the same with each single stretch alone, the stretches of each cable alone, and all stretches.
Outputs: outputs_100m/amity_location.json, amity_stretches.csv, figs/amity_location.png
"""
import glob
import json
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
from msdvv.geometry import read_route
from msdvv.io import load_file

cfg = load_config(str(Path(__file__).resolve().parents[1] / "config_nov.yaml"))
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
G = Geod(ellps="WGS84")
T0 = pd.Timestamp("2025-11-23 15:58:26", tz="UTC").timestamp()
EV = dict(lat=45.0753, lon=-123.1575, z=15.1)
SNR_SEG, MAXSH, DMIS, CC_STR = 1.5, 10.0, 0.3, 0.8
import os
MODE = os.environ.get("MODE", "wavefront")  # wavefront: one free offset per stretch (shape only); common: one offset for all
BINS = []  # rows: cable, stretch, lat, lon, t (DAS stamp - T0)
STR = []
TR = {}
for cab in ("north", "south"):
    p = sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20251123/*_{cab}_*"))[0]  # earlier file covers the event
    f = load_file(p, virtual_mask=cfg["virtual_mask"], dx_default=cfg["dx_default"])
    ch = f.channels()
    g = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    lat = np.interp(ch.opt_km, g.opt_km, g.lat)
    lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    tt = (f.times if f.times is not None else f.t0 + np.arange(f.nt) / f.fs) - T0
    w = (tt > -150) & (tt < 250)
    tw = tt[w]
    x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[:, w].astype(np.float64), axis=1)
    env = uniform_filter1d(np.abs(hilbert(x, axis=1)) ** 2, int(2 * f.fs), axis=1)
    pre = (tw > -140) & (tw < -10)
    e = np.log(env / (np.median(env[:, pre], axis=1, keepdims=True) + 1e-30) + 1e-30)
    ok = ch.block.notna().values & (ch.pos_in_block.values >= 3) & (ch.pos_in_block.values <= ch.block_len.values - 4)
    evm = (tw >= 0) & (tw <= 150)
    # 1) lit 5-km segments
    seg = []
    for blk, gb in ch[ok].groupby("block"):
        km = gb.opt_km.values
        for a in np.arange(np.floor(km.min() / 5) * 5, km.max(), 5):
            sel = gb.index.values[(km >= a) & (km < a + 5)]
            if len(sel) < 20:
                continue
            s = np.median(np.clip(e[sel], 0, None), axis=0)
            snr = s[evm].max() - np.median(s[pre])
            seg.append((blk, a, snr))
    lit = [(b, a) for b, a, s in seg if s >= SNR_SEG]
    # merge contiguous segments of the same block
    runs = []
    for b, a in sorted(lit):
        if runs and runs[-1][0] == b and abs(runs[-1][2] - a) < 1e-6:
            runs[-1][2] = a + 5
        else:
            runs.append([b, a, a + 5])
    for b, a0, a1 in runs:
        if a1 - a0 < 5:
            continue
        sel = np.where(ok & (ch.block.values == b) & (ch.opt_km.values >= a0) & (ch.opt_km.values < a1))[0]
        # 1-km bins of 10 channels
        nb = len(sel) // 10
        if nb < 3:
            continue
        E = np.array([np.log(np.mean(np.exp(e[sel[k * 10:(k + 1) * 10]]), axis=0)) for k in range(nb)])
        E = np.clip(E, 0, None)[:, (tw >= -10) & (tw <= 150)]
        tm = tw[(tw >= -10) & (tw <= 150)]
        dt = tm[1] - tm[0]
        ns = int(MAXSH / dt)
        sh = np.zeros(nb)
        for _ in range(3):
            ref = np.mean([np.roll(E[i], -int(round(sh[i] / dt))) for i in range(nb)], axis=0)
            ref = ref - ref.mean()
            for i in range(nb):
                u = E[i] - E[i].mean()
                c = np.array([np.dot(np.roll(u, -k), ref) for k in range(-ns, ns + 1)])
                k = int(np.argmax(c))
                dk = 0.5 * (c[k - 1] - c[k + 1]) / (c[k - 1] - 2 * c[k] + c[k + 1]) if 0 < k < len(c) - 1 else 0.0
                sh[i] = (k - ns + dk) * dt
        cc = np.array([np.corrcoef(np.roll(E[i], -int(round(sh[i] / dt))), ref)[0, 1] for i in range(nb)])
        good = cc >= 0.5
        if good.sum() < 3:
            continue
        st = np.mean([np.roll(np.exp(E[i]), -int(round((sh[i] - sh[good].mean()) / dt))) for i in np.where(good)[0]], axis=0)
        bg = np.median(st[tm < 0])
        t_on = float(tm[np.argmax(st >= bg + 0.2 * (st.max() - bg))])
        name = f"{cab} {b} {a0:.0f}-{a1:.0f} km"
        blat = np.array([lat[sel[k * 10:(k + 1) * 10]].mean() for k in range(nb)])
        blon = np.array([lon[sel[k * 10:(k + 1) * 10]].mean() for k in range(nb)])
        for i in np.where(good)[0]:
            BINS.append(dict(cable=cab, stretch=name, lat=blat[i], lon=blon[i], t=t_on + sh[i] - sh[good].mean(), cc=cc[i]))
        STR.append(dict(stretch=name, cable=cab, block=b, km0=a0, km1=a1, n_bins=int(good.sum()), cc_med=float(np.median(cc)), t_onset=t_on,
                        lat=float(blat.mean()), lon=float(blon.mean()), dist_to_event_km=float(G.inv(EV["lon"], EV["lat"], blon.mean(), blat.mean())[2] / 1e3)))
        TR[name] = (tm, E, sh, good)
    print(cab, "lit stretches:", [s["stretch"] for s in STR if s["cable"] == cab], flush=True)
D = pd.DataFrame(BINS)
S = pd.DataFrame(STR)
S["used"] = S.cc_med >= CC_STR  # coherent stretches only (near-shore stretches: CC ~0.6, one 'onset' before the origin)
D = D[D.stretch.isin(S.stretch[S.used])].reset_index(drop=True)
D.to_csv(f"{OUT}/amity_bin_times.csv", index=False)
if os.environ.get("EXTRACT_ONLY"):
    sys.exit(0)
S.to_csv(f"{OUT}/amity_stretches.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(2).to_string(index=False))

LATG = np.arange(44.0, 46.4001, 0.02)
LONG = np.arange(-125.4, -122.3999, 0.02)
ZS = [5, 10, 15, 20, 25]
VS = np.arange(2.8, 4.01, 0.1)


def locate(Q):
    """Grid search with one common time offset; each stretch weighted equally."""
    grp = [Q.stretch.values == k for k in Q.stretch.unique()]
    mis = np.full((len(LATG), len(LONG)), np.inf)
    best = (np.inf,)
    for ia, la in enumerate(LATG):
        _, _, d = G.inv(np.tile(Q.lon.values, len(LONG)), np.tile(Q.lat.values, len(LONG)), np.repeat(LONG, len(Q)), np.full(len(LONG) * len(Q), la))
        d = d.reshape(len(LONG), len(Q)) / 1e3
        for z in ZS:
            r = np.hypot(d, z)
            for v in VS:
                res = Q.t.values[None] - r / v
                if MODE == "wavefront":  # free offset per stretch: only the wavefront shape inside each stretch is used
                    for gm in grp:
                        res[:, gm] -= np.median(res[:, gm], axis=1, keepdims=True)
                else:
                    res = res - np.median(res, axis=1, keepdims=True)
                L1 = np.mean([np.mean(np.abs(res[:, gm]), axis=1) for gm in grp], axis=0)
                k = int(np.argmin(L1))
                mis[ia] = np.minimum(mis[ia], L1)
                if L1[k] < best[0]:
                    off = float(np.median(Q.t.values - r[k] / v))
                    best = (float(L1[k]), float(la), float(LONG[k]), z, float(v), off)
    return best, mis


RUNS = {"all stretches, both cables": np.ones(len(D), bool)}
RUNS.update({f"{c} cable stretches": (D.cable == c).values for c in ("north", "south")})
RUNS.update({f"{k} (single stretch)": (D.stretch == k).values for k in S.stretch[S.used]})
res, maps = {}, {}
for name, sel in RUNS.items():
    if sel.sum() < 4:
        continue
    b, mis = locate(D[sel].reset_index(drop=True))
    ia, io_ = np.where(mis <= b[0] + DMIS)
    _, _, dist = G.inv(EV["lon"], EV["lat"], b[2], b[1])
    area = float(len(ia) * (0.02 * 111.195) ** 2 * np.cos(np.radians(45)))
    res[name] = dict(n_stretches=int(D[sel].stretch.nunique()), n_bins=int(sel.sum()), misfit_s=round(b[0], 2), lat=round(b[1], 3), lon=round(b[2], 3),
                     depth_km=b[3], vs=round(b[4], 2), offset_s=round(b[5], 2), dist_to_PNSN_km=round(dist / 1e3, 1), region_km2=round(area))
    maps[name] = mis
    print(f"{name:42s} {res[name]}", flush=True)
json.dump(dict(event=EV, mode=MODE, stretches=STR, results=res), open(f"{OUT}/amity_location_{MODE}.json", "w"), indent=1, default=float)

show = ["all stretches, both cables", "north cable stretches", "south cable stretches"] + [k for k in res if "single" in k][:3]
show = [s for s in show if s in maps]
fig, axs = plt.subplots(2, 3, figsize=(19, 12))
for ax, name in zip(axs.flat, show):
    r_ = res[name]
    M = maps[name] - r_["misfit_s"]
    pc = ax.pcolormesh(LONG, LATG, M, vmin=0, vmax=3, cmap="viridis_r", shading="auto")
    ax.contour(LONG, LATG, M, levels=[DMIS], colors="w", linewidths=1.3)
    for c in ("north", "south"):
        rt = read_route(c)
        ax.plot(rt.lon, rt.lat, "-", color="0.6", lw=0.8)
    q = D[RUNS[name]]
    ax.scatter(q.lon, q.lat, c=q.t, s=10, cmap="plasma", zorder=4)
    ax.plot(EV["lon"], EV["lat"], "*", ms=18, color="w", mec="k", label="PNSN (75 stations, gap 22°)")
    ax.plot(r_["lon"], r_["lat"], "P", ms=13, color="tab:red", mec="k", label=f"DAS: {r_['dist_to_PNSN_km']:.0f} km off")
    ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude", ylabel="latitude")
    ax.set_title(f"{name}\n{r_['n_stretches']} stretch(es), {r_['n_bins']} bins; z {r_['depth_km']} km, Vs {r_['vs']}; region {r_['region_km2']:,} km²", fontsize=9)
    ax.set_aspect(1 / np.cos(np.radians(45.2)))
    ax.legend(fontsize=7, loc="lower left")
fig.colorbar(pc, ax=axs, shrink=0.6, label="misfit - minimum (s); white: +0.3 s")
fig.suptitle("Nov 23 2025 M3.2 Amity (ground truth, onshore): DAS-only location from S wavefronts on the coherent stretches (" + ("wavefront shape only, free offset per stretch" if MODE == "wavefront" else "one common time offset") + ")")
fig.savefig(f"{OUT}/figs/amity_location_{MODE}.png", dpi=95, bbox_inches="tight")
