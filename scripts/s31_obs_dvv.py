"""dv/v from the OOI seafloor seismometer correlations (s30): 24-h stacks moving 1 h, back-stacked.

Per pair & band: reference = all-period stack; direct time t_pk = envelope peak of the symmetric reference;
coda window [max(2 t_pk, min_coda), +coda_len] (both sides, stretched separately, combined by 1/err^2).
Noise floor: even/odd-hour split of each stack. Clock check: daily lag of the (direct) envelope peak and of the
cross-correlation between each daily stack and the reference (a clock error shifts it; dv/v stretches it).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read_inventory
from pyproj import Geod
from scipy.signal import hilbert

from msdvv.dvv import one_sided, stretch
from msdvv.xcorr import spectra_to_ccf

ROOT = Path("/data/wsd02/multispan_dvv/obs")
FIG = Path("/data/wsd02/multispan_dvv/outputs_100m/figs/obs")
FIG.mkdir(parents=True, exist_ok=True)
S, STEP = 24, 1
MIN_CODA = {"b1": 15.0, "b2": 10.0, "b3": 5.0}
CODA_LEN = {"b1": 120.0, "b2": 90.0, "b3": 40.0}
RNG = {"b1": (0.12, 0.30), "b2": (0.30, 0.70), "b3": (1.0, 2.0)}
CC_MIN = 0.5
import os
WINDOW = os.environ.get("OBS_WINDOW", "coda")  # coda (late) or early: [max(t_pk, min), max(2.5 t_pk, min + 20 s)]
MIN_EARLY = {"b1": 5.0, "b2": 3.0, "b3": 2.0}
SUFFIX = "" if WINDOW == "coda" else f"_{WINDOW}"
G = Geod(ellps="WGS84")
inv = read_inventory(str(ROOT / "continuous" / "OO_response.xml"))
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
env_df = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/dvv_24h/environment_timeseries.csv", index_col=0)

with h5py.File(ROOT / "ccf" / "obs_zz.h5", "r") as h:
    t = h["t_sub"][:]
    nwin = h["nwin"][:]
    pairs = h.attrs["pairs"].split(";")
    fs, nfft, ml, sub = h.attrs["fs"], int(h.attrs["nfft"]), float(h.attrs["maxlag_s"]), float(h.attrs["substack_s"])
    SPEC = {b: (h[b]["spec"][:], h[b]["bins"][:]) for b in RNG}
dist = {p: G.inv(crd[p.split("-")[0]][1], crd[p.split("-")[0]][0], crd[p.split("-")[1]][1], crd[p.split("-")[1]][0])[2] / 1e3 for p in pairs}
starts = [i for i in range(0, len(t) - S + 1, STEP) if t[i + S - 1] - t[i] == (S - 1) * sub]
t_end = np.array([t[i] + S * sub for i in starts])  # back-stacked label
hour = np.round(t / sub).astype(np.int64)
rows, clock, summ = [], [], []
refs = {}
for b, (spec, bins) in SPEC.items():
    ref, lags = spectra_to_ccf(spec.sum(0), bins, nfft, fs, ml)
    refs[b] = (ref, lags)
    dt = lags[1] - lags[0]
    cs = np.concatenate([np.zeros((1,) + spec.shape[1:], spec.dtype), np.cumsum(spec, 0)])
    se = spec * (hour % 2 == 0)[:, None, None]
    so = spec * (hour % 2 == 1)[:, None, None]
    cse = np.concatenate([np.zeros((1,) + spec.shape[1:], spec.dtype), np.cumsum(se, 0)])
    cso = np.concatenate([np.zeros((1,) + spec.shape[1:], spec.dtype), np.cumsum(so, 0)])
    cur = spectra_to_ccf(np.stack([cs[i + S] - cs[i] for i in starts]), bins, nfft, fs, ml)[0]
    cue = spectra_to_ccf(np.stack([cse[i + S] - cse[i] for i in starts]), bins, nfft, fs, ml)[0]
    cuo = spectra_to_ccf(np.stack([cso[i + S] - cso[i] for i in starts]), bins, nfft, fs, ml)[0]
    for k, p in enumerate(pairs):
        e = np.abs(hilbert(ref[k]))
        sym = one_sided(e, lags, "pos") + one_sided(e, lags, "neg")
        tpk = np.argmax(sym[: int(min(ml - 1, max(10, dist[p] / 0.15)) / dt)]) * dt
        if WINDOW == "early":
            t1 = max(tpk, MIN_EARLY[b])
            t2 = min(max(2.5 * tpk, MIN_EARLY[b] + 20.0), ml - 10)
        else:
            t1 = max(2 * tpk, MIN_CODA[b])
            t2 = min(t1 + CODA_LEN[b], ml - 10)

        def measure(C):
            dv_s, w_s, cc_s = [], [], []
            for side in ("pos", "neg"):
                c = one_sided(C[:, k], lags, side)
                r = np.broadcast_to(one_sided(ref[k], lags, side), c.shape)
                st = stretch(c, r, dt, t1, t2, 0.01, 101, RNG[b])
                ok = (st["cc"] >= CC_MIN) & ~st["at_bound"]
                w = np.where(ok, 1 / np.maximum(st["err"], 1e-6) ** 2, 0.0)
                dv_s.append(np.where(ok, st["dv"], 0.0) * w)
                w_s.append(w)
                cc_s.append(st["cc"])
            W = w_s[0] + w_s[1]
            return np.where(W > 0, (dv_s[0] + dv_s[1]) / np.maximum(W, 1e-30), np.nan), np.where(W > 0, 1 / np.sqrt(np.maximum(W, 1e-30)), np.nan), np.nanmean(cc_s, 0)

        dv, err, cc = measure(cur)
        dve, _, _ = measure(cue)
        dvo, _, _ = measure(cuo)
        d = dve - dvo
        d = d[np.isfinite(d)]
        floor = 1.4826 * np.median(np.abs(d - np.median(d))) / 2 if len(d) > 5 else np.nan
        # clock check: lag of max cross-correlation between each daily stack and the reference (+-2 s)
        mlag = int(2.0 / dt)
        xc = np.array([[np.dot(np.roll(cur[s_, k], sh), ref[k]) for sh in range(-mlag, mlag + 1)] for s_ in range(0, len(starts), 24)])
        shift = (np.argmax(xc, 1) - mlag) * dt
        for s_, sh in zip(range(0, len(starts), 24), shift):
            clock.append(dict(pair=p, band=b, t=t_end[s_], shift_s=sh))
        rows.append(pd.DataFrame(dict(pair=p, band=b, t=t_end, dv=dv, err=err, cc=cc, floor=floor)))
        summ.append(dict(pair=p, band=b, dist_km=round(dist[p], 2), t_direct_s=round(tpk, 1), coda=f"{t1:.0f}-{t2:.0f} s",
                         cc_median=np.nanmedian(cc), kept=np.mean(np.isfinite(dv)), std_pct=100 * np.nanstd(dv), floor_pct=100 * floor,
                         std_over_floor=np.nanstd(dv) / floor if floor else np.nan, clock_shift_std_s=np.std(shift), clock_shift_maxabs_s=np.max(np.abs(shift))))
D = pd.concat(rows, ignore_index=True)
D.to_csv(ROOT / f"obs_dvv_24h{SUFFIX}.csv.gz", index=False)
SM = pd.DataFrame(summ)
SM.to_csv(ROOT / f"obs_dvv_summary{SUFFIX}.csv", index=False)
pd.DataFrame(clock).to_csv(ROOT / f"obs_clock_check{SUFFIX}.csv", index=False)
pd.set_option("display.width", 220)
print(SM.round(3).to_string(index=False))

# figures: dv/v per pair, and reference gathers
order = sorted(pairs, key=lambda p: dist[p])
fig, axs = plt.subplots(len(order) + 1, 1, figsize=(14, 1.7 * (len(order) + 1)), sharex=True)
for ax, p in zip(axs, order):
    for b, col in (("b1", "C0"), ("b2", "C3"), ("b3", "C2")):
        g = D[(D.pair == p) & (D.band == b)]
        td = pd.to_datetime(g.t, unit="s", utc=True)
        fl = g.floor.iloc[0]
        ax.fill_between(td, 100 * (g.dv - fl), 100 * (g.dv + fl), color=col, alpha=0.2)
        ax.plot(td, 100 * g.dv, color=col, lw=0.8, label=f"{b} {RNG[b]} Hz (floor {100 * fl:.3f}%, CC {np.nanmedian(g.cc):.2f})")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylabel(f"{p}\n{dist[p]:.1f} km\ndv/v %", fontsize=7)
    ax.legend(fontsize=6, loc="upper left", ncol=3)
axs[-1].plot(pd.to_datetime(env_df.index.values, unit="s", utc=True), env_df.Hs, "k-", lw=0.8)
axs[-1].set_ylabel("Hs (m)", fontsize=8)
fig.suptitle(f"OOI Hydrate Ridge seismometers (Z-Z): dv/v ({WINDOW} window), 24-h stacks moving 1 h, back-stacked (shaded = even/odd floor)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / f"obs_dvv_24h{SUFFIX}.png", dpi=110)
plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(16, 6))
for ax, b in zip(axs, RNG):
    ref, lags = refs[b]
    for j, p in enumerate(order):
        k = pairs.index(p)
        y = ref[k] / (np.abs(ref[k]).max() + 1e-30)
        ax.plot(lags, j + 0.45 * y, "k", lw=0.5)
        ax.text(lags[0], j, f"{p} {dist[p]:.1f} km", fontsize=7, va="bottom")
    ax.set(xlim=(-ml, ml) if b != "b3" else (-40, 40), yticks=[], xlabel="lag (s)", title=f"reference CCFs {b} {RNG[b]} Hz")
fig.tight_layout()
fig.savefig(FIG / "obs_reference_ccfs.png", dpi=110)
print("ok")
