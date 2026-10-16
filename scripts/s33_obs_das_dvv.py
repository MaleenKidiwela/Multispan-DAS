"""Seismometer-DAS dv/v (s32 output): DAS latency calibration per acquisition setup, gathers, convergence,
24-h back-stacked dv/v per path group (+ even/odd floor), map of the paths used.

Latency: the DAS time stamps are offset from UTC by a setup-dependent latency, which shifts the OBS-DAS correlations.
Per setup, the absolute latency = envelope peak of the stacked colocated HYSB1 paths (< 3 km, travel time < 1 s); it is
removed with a phase ramp per row before partial hours are merged; the residual shift of each setup vs the corrected
reference is reported as a check.
"""
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.collections import LineCollection
from obspy import read_inventory
from scipy.signal import hilbert

from msdvv import ROOT, load_config
from msdvv.dvv import one_sided, stretch
from msdvv.geometry import read_route, repeaters
from msdvv.xcorr import spectra_to_ccf

warnings.filterwarnings("ignore", category=RuntimeWarning)
cfg = load_config()
out = Path(cfg["output_dir"])
od = out / "obsdas"
fd = out / "figs" / "obsdas"
fd.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(od / "paths.csv")
OBS = ["HYSB1", "HYS14"]
FEAT = [c for c in P if c.startswith("x_")]
SH, STEP, CONV_MIN, CC_MIN, MIN_H = 24, 1, 0.3, 0.4, 20
RNG = {"b1": (0.12, 0.30), "b2": (0.30, 0.70)}
# early (default): [max(0.7 d/vg, 3 s), max(2.5 d/vg, t1 + 30 s)] -- the only window that converges in 24 h;
# coda: [max(1.3 d/vg, 10 s), maxlag - 10] (converges only at ~weekly stacks)
WINDOW = os.environ.get("OBSDAS_WINDOW", "early")
SETUP_CHANGES = []  # filled from the data (first/last hour of each non-majority setup run)
SUF = "" if WINDOW == "early" else f"_{WINDOW}"


def load_band(band):
    T, SP, SET = [], [], []
    for f in sorted(od.glob("*.h5")):
        with h5py.File(f, "r") as h:
            s = h[band]["spec"][:]  # (nsub, nobs, nbin, nb)
            if s.shape[0] == 0:  # no OBS data during this DAS file
                continue
            T.append(h["t_sub"][:])
            SP.append(s.reshape(s.shape[0], s.shape[1] * s.shape[2], s.shape[3]))  # -> (nsub, npath, nb), obs-major like paths.csv
            SET += [h.attrs["setup"]] * len(h["t_sub"])
            bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], h.attrs["nfft"], h.attrs["maxlag_s"]
    return np.concatenate(T), np.concatenate(SP), np.array(SET), bins, fs, nfft, ml


def merge_hours(t, S, setup):
    """Sum rows with the same hour (partial hours split across consecutive DAS files)."""
    ut, inv = np.unique(t, return_inverse=True)
    M = np.zeros((len(ut),) + S.shape[1:], np.complex64)
    np.add.at(M, inv, S)
    return ut, M, pd.Series(setup).groupby(inv).first().values


def ramp(freqs, tau):
    """Phase ramp that moves a CCF feature at lag +tau to lag 0 (sign verified numerically in the loop)."""
    return np.exp(-2j * np.pi * freqs * tau).astype(np.complex64)


def xcorr_shift(a, b, dt, maxs=20.0):
    """Lag (s) by which a must be shifted to best match b (both (..., nlag))."""
    m = int(maxs / dt)
    best = [np.sum(np.roll(a, k, axis=-1) * b, axis=-1) for k in range(-m, m + 1)]
    return (np.argmax(np.array(best), axis=0) - m) * dt


summary, series = [], []
lat_report = {}
for band in RNG:
    t, S, setup, bins, fs, nfft, ml = load_band(band)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)[bins]
    lags = spectra_to_ccf(S[:1].sum(0), bins, nfft, fs, ml)[1]
    dt = lags[1] - lags[0]
    # --- absolute DAS latency per setup: envelope peak of the stacked colocated HYSB1 paths (< 3 km, travel time < 1 s)
    near = P.index[(P.obs == "HYSB1") & (P.dist_km < 3)].values
    lat, sign = {}, None
    major = pd.Series(setup).value_counts().index[0]
    cc_maj, _ = spectra_to_ccf(S[setup == major].sum(0), bins, nfft, fs, ml)
    e = np.abs(hilbert(cc_maj[near], axis=-1))
    e = (e / e.max(1, keepdims=True)).mean(0)
    lat[major] = float(lags[np.argmax(np.where(np.abs(lags) < 60, e, 0))])
    snr0 = np.abs(hilbert(cc_maj, axis=-1)).max(1) / (np.median(np.abs(hilbert(cc_maj, axis=-1))[:, np.abs(lags) > 150], 1) + 1e-30)
    hi0 = np.argsort(snr0)[::-1][: max(10, len(P) // 5)]
    for su in np.unique(setup):
        if su != major:  # shift that best aligns this setup's stack with the majority stack (high-SNR paths)
            cc_su, _ = spectra_to_ccf(S[setup == su].sum(0), bins, nfft, fs, ml)
            lat[su] = lat[major] - float(np.median(xcorr_shift(cc_su[hi0], cc_maj[hi0], dt, 30.0)))
    for sg in (1, -1):  # verify the ramp sign: after correction the colocated peak must sit at ~0 lag
        c2, _ = spectra_to_ccf((S[setup == major] * ramp(freqs, sg * lat[major])).sum(0)[near], bins, nfft, fs, ml)
        e2 = np.abs(hilbert(c2, axis=-1))
        e2 = (e2 / e2.max(1, keepdims=True)).mean(0)
        if abs(lags[np.argmax(np.where(np.abs(lags) < 60, e2, 0))]) <= 1.0:
            sign = sg
            break
    if sign is None:
        raise RuntimeError("latency ramp did not move the colocated peak to zero lag")
    for su in lat:
        S[setup == su] *= ramp(freqs, sign * lat[su])
    # relative check after correction: residual shift of each setup stack vs the corrected reference (high-SNR paths)
    t, S, setup = merge_hours(t, S, setup)
    if not SETUP_CHANGES:
        ch = np.where(setup[1:] != setup[:-1])[0]
        SETUP_CHANGES.extend(pd.to_datetime(t[ch + 1], unit="s").strftime("%Y-%m-%d %H:%M"))
    ref, lags = spectra_to_ccf(S.sum(0), bins, nfft, fs, ml)
    env = np.abs(hilbert(ref, axis=-1))
    snr = env.max(1) / (np.median(env[:, np.abs(lags) > 150], 1) + 1e-30)
    hi = np.argsort(snr)[::-1][: max(10, len(P) // 5)]
    check = {su: float(np.median(xcorr_shift(spectra_to_ccf(S[setup == su].sum(0), bins, nfft, fs, ml)[0][hi], ref[hi], dt)))
             for su in np.unique(setup)}
    tau_abs = 0.0
    lat_report[band] = dict(latency_s=lat, ramp_sign=sign, residual_shift_vs_ref_s=check, n_colocated_paths=int(len(near)),
                            note="positive latency = DAS time stamps late relative to the OBS (UTC)")
    print(f"[{band}] DAS latency per setup (s): {lat}; residual shift after correction: {check}", flush=True)
    # --- gather
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, o in zip(axs, OBS):
        idx = P.index[P.obs == o].values
        oo = idx[np.argsort(P.dist_km.values[idx])]
        Gm = ref[oo] / (np.abs(ref[oo]).max(1, keepdims=True) + 1e-30)
        ax.pcolormesh(lags, P.dist_km.values[oo], Gm, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
        ax.axvline(tau_abs, color="k", ls="--", lw=0.8)
        ax.set(xlabel="lag (s), latency-corrected (+ = DAS after OBS)", ylabel="distance OBS-bin (km)", title=f"{o} x south-cable bins, {band} {RNG[band]} Hz")
    fig.tight_layout()
    fig.savefig(fd / f"obsdas_gather_{band}{SUF}.png", dpi=110)
    plt.close(fig)
    # --- coda windows, convergence, dv/v (24 h, step 1 h, back-stacked) + even/odd floor
    pk = np.argmax(one_sided(env, lags, "pos") + one_sided(env, lags, "neg"), axis=-1) * dt
    vpk = P.dist_km.values / np.maximum(pk, 1e-6)
    vg = np.nanmedian(vpk[(pk > 2) & (vpk > 0.3) & (vpk < 5)])
    if WINDOW == "early":
        t1 = np.maximum(0.7 * P.dist_km.values / vg, 3.0)
        t2 = np.minimum(np.maximum(2.5 * P.dist_km.values / vg, t1 + 30.0), ml - 10.0)
    else:
        t1 = np.maximum(1.3 * P.dist_km.values / vg, 10.0)
        t2 = np.full(len(P), ml - 10.0)
    usable = t1 < t2 - 20
    sub = 3600.0
    # back-stacked windows [e - 24 h, e) on the hourly grid; accepted with >= MIN_H hours of data (OBS/DAS gaps)
    grid = np.arange(t[0] + SH * sub, t[-1] + sub + 1, STEP * sub)
    i0, i1 = np.searchsorted(t, grid - SH * sub), np.searchsorted(t, grid)
    okw = (i1 - i0) >= MIN_H
    tend, i0, i1 = grid[okw], i0[okw], i1[okw]
    hour = np.round(t / sub).astype(np.int64)

    def stacks(X):
        cs = np.concatenate([np.zeros((1,) + X.shape[1:], X.dtype), np.cumsum(X, 0)])
        return spectra_to_ccf(cs[i1] - cs[i0], bins, nfft, fs, ml)[0]

    cur = stacks(S)
    la = np.abs(lags)
    conv = np.full(len(P), np.nan)
    for p in np.where(usable)[0]:
        m = (la >= t1[p]) & (la <= t2[p])
        a = cur[:, p][:, m] - cur[:, p][:, m].mean(1, keepdims=True)
        r = (ref[p][None] - cur[:, p])[:, m]
        r = r - r.mean(1, keepdims=True)
        conv[p] = np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30))
    P[f"conv_{band}"] = conv
    P[f"win_{band}"] = [f"{a:.0f}-{b:.0f}" for a, b in zip(t1, t2)]
    sel = np.where(conv >= CONV_MIN)[0]
    print(f"[{band}] convergence CC quantiles (10/50/90%): {np.round(np.nanpercentile(conv, [10, 50, 90]), 2)}", flush=True)

    def pair_dv(C, pp):
        out_ = []
        for p in pp:
            dv_s, w_s = [], []
            for side in ("pos", "neg"):
                c = one_sided(C[:, p], lags, side)
                st = stretch(c, np.broadcast_to(one_sided(ref[p], lags, side), c.shape), dt, t1[p], t2[p], 0.01, 101, RNG[band])
                ok = (st["cc"] >= CC_MIN) & ~st["at_bound"]
                w = np.where(ok, 1 / np.maximum(st["err"], 1e-5) ** 2, 0.0)
                dv_s.append(np.where(ok, st["dv"], 0) * w)
                w_s.append(w)
            W = w_s[0] + w_s[1]
            out_.append(pd.DataFrame(dict(seg=np.arange(C.shape[0]), path=p, dv=np.where(W > 0, (dv_s[0] + dv_s[1]) / np.maximum(W, 1e-30), np.nan),
                                          err=np.where(W > 0, 1 / np.sqrt(np.maximum(W, 1e-30)), np.nan))))
        return pd.concat(out_).dropna() if out_ else pd.DataFrame(columns=["seg", "path", "dv", "err"])

    def par(C):
        parts = Parallel(n_jobs=16, prefer="threads")(delayed(pair_dv)(C, c) for c in np.array_split(sel, 16) if len(c))
        return pd.concat(parts) if parts else pd.DataFrame(columns=["seg", "path", "dv", "err"])

    D = par(cur)
    De = par(stacks(S * (hour % 2 == 0)[:, None, None]))
    Do = par(stacks(S * (hour % 2 == 1)[:, None, None]))
    groups = {f[2:]: P[f].values for f in FEAT}
    groups["control (no feature)"] = ~P[FEAT].any(axis=1).values
    for o in OBS:
        groups[f"all {o} paths"] = (P.obs == o).values

    def gmean(DD, mask):
        g = DD[mask[DD.path.values.astype(int)]]
        w = 1 / g.err**2
        return (g.dv * w).groupby(g.seg).sum() / w.groupby(g.seg).sum(), g.groupby("seg").size()

    for gname, mask in groups.items():
        if mask[sel].sum() < 3:
            continue
        m_, n_ = gmean(D, mask)
        me, _ = gmean(De, mask)
        mo, _ = gmean(Do, mask)
        d = (me - mo).dropna()
        floor = 1.4826 * np.median(np.abs(d - d.median())) / 2 if len(d) > 5 else np.nan
        series.append(pd.DataFrame(dict(band=band, group=gname, t=tend[m_.index], dv=m_.values, n=n_.reindex(m_.index).values, floor=floor)))
        summary.append(dict(band=band, group=gname, n_paths=int(mask[sel].sum()), std_pct=100 * m_.std(), floor_pct=100 * floor,
                            std_over_floor=m_.std() / floor if floor else np.nan))
    print(f"[{band}] group velocity {vg:.2f} km/s, converging paths {len(sel)} of {usable.sum()} usable", flush=True)
    del S, cur

SER = pd.concat(series, ignore_index=True) if series else pd.DataFrame()
SUM = pd.DataFrame(summary)
SER.to_csv(od / f"obsdas_series_24h{SUF}.csv", index=False)
SUM.to_csv(od / f"obsdas_summary{SUF}.csv", index=False)
P.to_csv(od / f"paths_with_stats{SUF}.csv", index=False)
json.dump(lat_report, open(od / f"latency_calibration{SUF}.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(SUM.round(4).to_string(index=False))

# --- figures: series and map of paths used (b1)
if len(SER):
    gn = list(SER.group.unique())
    fig, axs = plt.subplots(len(gn), 1, figsize=(14, 2.0 * len(gn)), sharex=True, squeeze=False)
    for ax, g in zip(axs[:, 0], gn):
        for band, col in (("b1", "C0"), ("b2", "C3")):
            s = SER[(SER.group == g) & (SER.band == band)].sort_values("t")
            if s.empty:
                continue
            s = s.set_index("t").reindex(np.arange(s.t.min(), s.t.max() + 1, 3600.0)).reset_index()  # NaN in gaps: no lines across them
            s["floor"] = s.floor.ffill()
            td = pd.to_datetime(s.t, unit="s", utc=True)
            ax.fill_between(td, 100 * (s.dv - s.floor), 100 * (s.dv + s.floor), color=col, alpha=0.2)
            ax.plot(td, 100 * s.dv, color=col, lw=0.8, label=f"{band} (floor {100 * s.floor.iloc[0]:.3f}%, ~{int(s.n.median())} paths)")
        for x in SETUP_CHANGES:
            ax.axvline(pd.Timestamp(x, tz="UTC"), color="k", ls=":", lw=0.8)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_ylabel(f"{g}\ndv/v %", fontsize=7)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Seismometer-DAS dv/v (HYSB1/HYS14 x south-cable bins), {WINDOW} window, 24-h back-stacked, step 1 h; shaded = even/odd floor", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / f"obsdas_series_24h{SUF}.png", dpi=110)
    plt.close(fig)
inv_obs = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv_obs for s in n}
fig, ax = plt.subplots(figsize=(12, 9))
rt = read_route("south")
ax.plot(rt.lon, rt.lat, "-", color="0.75", lw=2)
U = P[P["conv_b1"] >= CONV_MIN] if "conv_b1" in P else P.iloc[:0]
cols = {"HYSB1": "tab:blue", "HYS14": "tab:red"}
for o in OBS:
    g = U[U.obs == o]
    ax.add_collection(LineCollection([[(crd[o][1], crd[o][0]), (r.bin_lon, r.bin_lat)] for r in g.itertuples()], colors=[cols[o]], linewidths=0.8,
                                     alpha=0.6, label=f"{o} paths used ({len(g)})"))
    ax.plot(crd[o][1], crd[o][0], "v", color=cols[o], mec="k", ms=12, zorder=5)
    ax.annotate(o, (crd[o][1], crd[o][0]), xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold")
B = pd.read_csv(od / "bins.csv")
ax.scatter(B.lon, B.lat, s=12, color="k", zorder=4, label="DAS bins (good channels)")
fc = json.load(open(ROOT / cfg["interspan"]["features_geojson"]))
for f in fc["features"]:
    c = np.array(f["geometry"]["coordinates"])
    ax.plot(c[:, 0], c[:, 1], color="saddlebrown", lw=2)
th = np.linspace(0, 2 * np.pi, 100)
ax.plot(-125.148 + 5 / (111.2 * np.cos(np.radians(44.569))) * np.cos(th), 44.569 + 5 / 111.2 * np.sin(th), "--", color="goldenrod")
rp = repeaters("south")
ax.plot(rp.lon, rp.lat, "k^", ms=8)
ax.set(xlim=(-125.8, -123.85), ylim=(44.25, 45.3), xlabel="longitude", ylabel="latitude",
       title=f"Seismometer-DAS paths used (b1, {WINDOW} window, 24-h stacks, convergence CC >= {CONV_MIN})")
ax.set_aspect(1 / np.cos(np.radians(44.8)))
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(fd / f"obsdas_paths_map_b1{SUF}.png", dpi=120)
print("done")
