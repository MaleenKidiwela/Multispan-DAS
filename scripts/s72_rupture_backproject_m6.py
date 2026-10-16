"""Item 8i: can multi-span DAS back-projection see the RUPTURE of the Jan 16 2026 M6.0 Blanco earthquake?

Time-resolved back-projection (as used to image large ruptures): both cables, all spans (188 bins), 100-m stream,
true times, absolute timing (D128 latency 11.8 s) and iasp91 P times - the same as the s70 location (25 km from the
catalogue), origin time from the catalogue (global networks). High-frequency energy: 0.4-0.9 Hz squared envelope, 1-s smoothing,
each bin normalised to its maximum in [P - 5 s, P + 40 s] (equal weight). Grid +-1.5 deg around the reference hypocentre, 0.02 deg, depth 10 km.
1) Reference hypocentre = the s70 onset back-projection (both cables, origin known). A first version took the maximum
   of the summed energy at tau 0-6 s instead; energy peaks 5-15 s after the onset, which put the reference 233 km away
   in a grid corner (rejected).
2) For each lag tau = -4..24 s (1 s): B_tau(node) = mean over bins of env_n(t0 + T_n(node) + tau), 2-s average;
   track the maximum (position relative to the reference, power).
3) Synthetic control: every bin gets the SAME source-time function (the observed beam envelope at the reference
   node, i.e. a point source) at its predicted arrival from the reference hypocentre, plus a random per-bin timing
   error (normal, sigma = observed residual scatter of the P onsets, s71/s70) and pre-event noise of that bin; the
   same time-resolved back-projection then shows the apparent migration of a point source ('swimming').
   20 realisations -> distribution of apparent migration; real rupture only if the observed migration exceeds it.
Blanco Transform strike ~297 deg (WNW-ESE).
Outputs: outputs_100m/beamform/rupture_m6.json, figs/rupture_m6.png
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from obspy.taup import TauPyModel
from pyproj import Geod
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from msdvv import load_config
from msdvv.geometry import read_route
from msdvv.interspan import make_bins
from msdvv.io import load_cfg

cfg = load_config(str(Path(__file__).resolve().parents[1] / "config_100m_v2.yaml"))
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
G = Geod(ellps="WGS84")
LAT = {256: 25.2, 128: 11.8}
DAS_HYSB1 = float(__import__("os").environ.get("DAS_HYSB1", "0"))  # s71 lag vs HYSB1 (3.75 s); 0 = timing of the s70 location
FSU = 4.0
STRIKE = 297.0
ev = pd.read_csv(io.StringIO(requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=60, params=dict(
    format="csv", starttime="2026-01-16T03:25:00", endtime="2026-01-16T03:27:00", minmagnitude=5.5)).text)).iloc[0]
T_EV = pd.Timestamp(ev.time).timestamp()
tg = np.arange(T_EV - 200, T_EV + 200, 1 / FSU)

B = []
for cab in ("north", "south"):
    b, bid = make_bins(pd.read_csv(OUT / f"channel_quality_{cab}.csv"), 1.0, 3)
    b = b[~((cab == "south") & (b.seg == "C3"))].copy()
    b["cable"] = cab
    b["chans"] = [np.where(bid == k)[0] for k in b.bin]
    B.append(b)
B = pd.concat(B, ignore_index=True)
inv = pd.read_csv(OUT / "inventory.csv")
ENV = np.zeros((len(B), len(tg)))
for cab in ("north", "south"):
    c = inv[(inv.cable == cab) & (inv.t0 <= tg[0]) & (inv.t1 >= tg[-1])].iloc[0]
    f = load_cfg(c.path, cfg)
    tt = f.times - LAT[int(c.D)] - DAS_HYSB1
    w = (tt >= tg[0] - 60) & (tt <= tg[-1] + 60)
    sos = butter(4, (0.4, 0.9), "bandpass", fs=f.fs, output="sos")
    for ix, r in B[B.cable == cab].iterrows():
        x = sosfiltfilt(sos, f.data[r.chans][:, w].astype(np.float64), axis=1).mean(0)
        ENV[ix] = np.interp(tg, tt[w], uniform_filter1d(np.abs(hilbert(x)) ** 2, int(f.fs) | 1))

# same model and timing as the s70 location that put the M6.0 25 km from the catalogue (iasp91, D128 latency 11.8 s)
mdl = TauPyModel("iasp91")
DG = np.arange(0, 6.01, 0.02)
TT = np.array([min(a.time for a in mdl.get_travel_times(10, d, ["P", "p", "Pn", "Pg"])) for d in DG])
REF = json.load(open(OUT / "beamform/backproject_m6_origin_known.json"))["results"]["both cables"]
H = (REF["lat"], REF["lon"])  # reference hypocentre = onset back-projection, both cables (s70, ORIGIN=known)


def ttime(lat_, lon_):
    _, _, d = G.inv(np.full(len(B), lon_), np.full(len(B), lat_), B.lon.values, B.lat.values)
    return np.interp(d / 111195.0, DG, TT)


# normalise each bin to its maximum in the P window predicted from the catalogue epicentre (equal weight per bin)
Tc = ttime(ev.latitude, ev.longitude)
pre = (tg >= T_EV - 180) & (tg <= T_EV - 40)
noise_med = np.median(ENV[:, pre], axis=1, keepdims=True)
for i in range(len(B)):
    m = (tg >= T_EV + Tc[i] - 5) & (tg <= T_EV + Tc[i] + 40)
    ENV[i] = np.clip((ENV[i] - noise_med[i]) / (ENV[i][m].max() - noise_med[i] + 1e-30), 0, None)
LATG = np.arange(H[0] - 1.5, H[0] + 1.5001, 0.02)
LONG = np.arange(H[1] - 2.0, H[1] + 2.0001, 0.02)
TAUS = np.arange(-4, 24.01, 1.0)
TTG = np.array([[ttime(la, lo) for lo in LONG] for la in LATG])  # (nlat, nlon, nbin)


def bp(E, taus, avg=2.0):
    """Time-resolved back-projection: (ntau, nlat, nlon) of the bin-mean envelope at t0 + T + tau (avg-s mean)."""
    Ec = uniform_filter1d(E, int(avg * FSU) | 1, axis=1)
    out = np.zeros((len(taus), len(LATG), len(LONG)))
    for k, tau in enumerate(taus):
        idx = np.clip(np.rint((T_EV + TTG + tau - tg[0]) * FSU).astype(int), 0, len(tg) - 1)  # (nlat, nlon, nbin)
        out[k] = Ec[np.arange(len(B))[None, None, :], idx].mean(-1)
    return out


def track(Bt):
    rows = []
    for k in range(len(Bt)):
        ia, io_ = np.unravel_index(np.argmax(Bt[k]), Bt[k].shape)
        rows.append((LATG[ia], LONG[io_], Bt[k].max()))
    return np.array(rows)


# 1) reference hypocentre fixed by the onset back-projection (energy peaks 5-15 s after the onset, so an energy-based
# reference is biased away from the array - first version, rejected)
Bobs = bp(ENV, TAUS)
ref = Bobs[(TAUS >= 0) & (TAUS <= 6)].sum(0)
ia, io_ = int(np.argmin(np.abs(LATG - H[0]))), int(np.argmin(np.abs(LONG - H[1])))
_, _, dH = G.inv(ev.longitude, ev.latitude, H[1], H[0])
print(f"reference hypocentre {H[0]:.2f} {H[1]:.2f}, {dH / 1e3:.0f} km from catalogue", flush=True)
TR = track(Bobs)


def rel(tr):
    az, _, d = G.inv(np.full(len(tr), H[1]), np.full(len(tr), H[0]), tr[:, 1], tr[:, 0])
    along = d / 1e3 * np.cos(np.radians(az - STRIKE))  # + = WNW along the transform
    across = d / 1e3 * np.sin(np.radians(az - STRIKE))
    return along, across


act = (TAUS >= 0) & (TAUS <= 12)  # rupture window (M6: ~5-10 s)
a_obs, c_obs = rel(TR)
w_obs = TR[:, 2] / TR[act, 2].max()
# migration = power-weighted slope of the along-strike position vs tau over the rupture window
def slope(a, w):
    m = act & (w > 0.5)
    return np.polyfit(TAUS[m], a[m], 1, w=w[m])[0] if m.sum() >= 3 else np.nan


s_obs = slope(a_obs, w_obs)
extent_obs = np.ptp(a_obs[act & (w_obs > 0.5)])

# 3) synthetic point sources
stf = Bobs[:, ia, io_]  # beam envelope at the reference node vs tau (point-source time function)
tau_fine = np.arange(-10, 40, 1 / FSU)
stf_f = np.interp(tau_fine, TAUS, stf, left=0, right=0)
Th = ttime(*H)
# per-bin timing scatter: residual of the envelope-onset time at the reference hypocentre
on = np.array([tg[np.argmax((tg >= T_EV + Th[i] - 10) & (ENV[i] >= 0.3))] - (T_EV + Th[i]) for i in range(len(B))])
sig = float(np.clip(1.4826 * np.median(np.abs(on - np.median(on))), 0.5, 5))
print(f"per-bin timing scatter (MAD-sigma) {sig:.2f} s", flush=True)
rng = np.random.default_rng(0)
syn = []
for r_ in range(20):
    S = np.zeros_like(ENV)
    for i in range(len(B)):
        S[i] = np.interp(tg - (T_EV + Th[i] + rng.normal(0, sig)), tau_fine, stf_f, left=0, right=0)
        j = rng.integers(0, pre.sum() - 1)
        nz = ENV[i][pre]
        S[i] += np.roll(nz, j)[: len(tg)] if len(nz) >= len(tg) else np.resize(np.roll(nz, j), len(tg))
    Ts = track(bp(S, TAUS))
    a_s, c_s = rel(Ts)
    w_s = Ts[:, 2] / Ts[act, 2].max()
    syn.append((slope(a_s, w_s), np.ptp(a_s[act & (w_s > 0.5)]), a_s, c_s))
syn_slopes = np.array([s[0] for s in syn])
syn_ext = np.array([s[1] for s in syn])
res = dict(reference_hypocentre=dict(lat=float(H[0]), lon=float(H[1]), dist_to_catalogue_km=round(dH / 1e3, 1)), timing_scatter_s=sig,
           observed=dict(along_strike_slope_km_per_s=float(s_obs), along_strike_extent_km=float(extent_obs)),
           synthetic_point_source=dict(slope_mean=float(np.nanmean(syn_slopes)), slope_std=float(np.nanstd(syn_slopes)),
                                       slope_95=[float(np.nanpercentile(syn_slopes, 2.5)), float(np.nanpercentile(syn_slopes, 97.5))],
                                       extent_median=float(np.nanmedian(syn_ext)), extent_95=float(np.nanpercentile(syn_ext, 95))))
json.dump(res, open(OUT / "beamform/rupture_m6.json", "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)

fig, axs = plt.subplots(1, 3, figsize=(19, 6))
ax = axs[0]
ax.pcolormesh(LONG, LATG, ref / ref.max(), vmin=0.5, vmax=1, cmap="magma", shading="auto")
sc = ax.scatter(TR[act, 1], TR[act, 0], c=TAUS[act], s=30 + 120 * w_obs[act], cmap="viridis", edgecolor="w", zorder=4)
ax.plot(ev.longitude, ev.latitude, "*", ms=16, color="w", mec="k", label="catalogue epicentre")
L = 0.6
ax.plot([H[1] - L * np.sin(np.radians(STRIKE)) / np.cos(np.radians(H[0])), H[1] + L * np.sin(np.radians(STRIKE)) / np.cos(np.radians(H[0]))],
        [H[0] - L * np.cos(np.radians(STRIKE)), H[0] + L * np.cos(np.radians(STRIKE))], "c--", lw=1.5, label="Blanco Transform strike (297°)")
fig.colorbar(sc, ax=ax, label="time after origin (s)")
ax.set(xlabel="longitude", ylabel="latitude", title="Beam maximum every 1 s (size = power), background: summed 0-6 s")
ax.set_aspect(1 / np.cos(np.radians(H[0])))
ax.legend(fontsize=7)
ax = axs[1]
ax.plot(TAUS, a_obs, "k-o", ms=4, label="observed: along strike (+ = WNW)")
ax.plot(TAUS, c_obs, "k:", label="observed: across strike")
for s_ in syn:
    ax.plot(TAUS, s_[2], "-", color="tab:red", alpha=0.15, lw=1)
ax.plot([], [], "-", color="tab:red", alpha=0.5, label="synthetic point sources (20)")
ax.axvspan(0, 12, color="0.9", zorder=0)
ax.set(xlabel="time after origin (s)", ylabel="km from reference hypocentre", title="Beam-maximum position vs time")
ax.legend(fontsize=7)
ax = axs[2]
ax.hist(syn_slopes, bins=12, color="tab:red", alpha=0.6, label="synthetic point source")
ax.axvline(s_obs, color="k", lw=2, label=f"observed {s_obs:+.2f} km/s")
ax.set(xlabel="along-strike migration speed (km/s), power-weighted fit 0-12 s", ylabel="count",
       title="Is the migration larger than for a point source?")
ax.legend(fontsize=8)
fig.suptitle("Jan 16 2026 M6.0 Blanco: time-resolved multi-span DAS back-projection (both cables, 0.4-0.9 Hz) vs synthetic point source")
fig.tight_layout()
fig.savefig(OUT / "figs/rupture_m6.png", dpi=100)
