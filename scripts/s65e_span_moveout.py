"""Item 8b (span beamforming): along-stretch moveout of the Jan 13 2026 ML 2.0 S envelope on each lit stretch of fiber.

For each stretch that records the event (s65b/s65c: north 68-100 km, south 99-115 km, south 35-45 km, north 25-35 km),
channel envelopes (2-3.6 Hz energy, 2-s smoothing, log) are cross-correlated against the stretch's own stack in
[0, 90] s after the PNSN origin (DAS time) -> relative delay per channel (iterated twice: realign, restack). A line fit
of delay vs along-route distance gives the apparent slowness p_app (s/km) along the cable; with an S velocity Vs,
cos(angle between propagation direction and cable) = p_app * Vs -> two mirror-image candidate directions per stretch
(cone ambiguity of a straight array). Intersecting the candidate rays of several stretches = multi-span triangulation.
Outputs: outputs_100m/das_location_jan13_moveout.csv, figs/das_location_jan13_moveout.png
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

from msdvv.geometry import read_route
from msdvv.io import load_file

G = Geod(ellps="WGS84")
T0 = pd.Timestamp("2026-01-13 05:12:47.25", tz="UTC").timestamp()
CAT = dict(lat=45.1618, lon=-124.390)
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
STRETCH = {"north": [(20, 30), (67, 100)], "south": [(20, 50), (98, 120)]}  # lit stretches from the s65g 10-km stacks (SNR > 50)
MAXSH = 8.0
rows, traces = [], {}
for cab in ("north", "south"):
    f = load_file(sorted(glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq20260113/nokia_DAS_{cab}_*"))[0])
    ch = f.channels()
    g = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    lat = np.interp(ch.opt_km, g.opt_km, g.lat)
    lon = np.interp(ch.opt_km, g.opt_km, g.lon)
    rkm = np.interp(ch.opt_km, g.opt_km, g.route_km) if "route_km" in g else ch.opt_km.values
    tt = f.times - T0
    w = (tt > -120) & (tt < 150)
    x = sosfiltfilt(butter(4, (2.0, 3.6), "bandpass", fs=f.fs, output="sos"), f.data[:, w].astype(np.float64), axis=1)
    tw = tt[w]
    env = uniform_filter1d(np.abs(hilbert(x, axis=1)) ** 2, int(round(2 * f.fs)), axis=1)
    e = np.log(env / np.median(env[:, (tw > -90) & (tw < -5)], axis=1, keepdims=True))
    ok = ch.block.notna().values & (ch.pos_in_block.values >= 3) & (ch.pos_in_block.values <= ch.block_len.values - 4)
    m = (tw >= -5) & (tw <= 90)
    dt = tw[1] - tw[0]
    ns = int(MAXSH / dt)
    for a, b in STRETCH[cab]:
        sel = np.where(ok & (ch.opt_km.values >= a) & (ch.opt_km.values <= b))[0]
        E = np.clip(e[sel][:, m], 0, None)
        sh = np.zeros(len(sel))
        for it in range(3):
            ref = np.mean([np.roll(E[i], -int(round(sh[i] / dt))) for i in range(len(sel))], axis=0)
            ref = ref - ref.mean()
            for i in range(len(sel)):
                u = E[i] - E[i].mean()
                c = np.array([np.dot(np.roll(u, -k), ref) for k in range(-ns, ns + 1)])
                k = int(np.argmax(c))
                dk = 0.5 * (c[k - 1] - c[k + 1]) / (c[k - 1] - 2 * c[k] + c[k + 1]) if 0 < k < len(c) - 1 else 0
                sh[i] = (k - ns + dk) * dt
        cc = np.array([np.corrcoef(np.roll(E[i], -int(round(sh[i] / dt))), ref)[0, 1] for i in range(len(sel))])
        good = cc > 0.5
        s_km = rkm[sel] - rkm[sel].min()
        A = np.vstack([s_km[good], np.ones(good.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, sh[good], rcond=None)
        r_ = sh[good] - A @ coef
        se = np.sqrt(np.sum(r_ ** 2) / max(good.sum() - 2, 1) / np.sum((s_km[good] - s_km[good].mean()) ** 2))
        # stretch azimuth (direction of increasing optical distance), centroid
        az = G.inv(lon[sel][0], lat[sel][0], lon[sel][-1], lat[sel][-1])[0]
        ref_on = tw[m][np.argmax(ref + ref.mean() >= 0.5 * (ref + ref.mean()).max())]
        rows.append(dict(cable=cab, km0=a, km1=b, n=int(good.sum()), lat=float(lat[sel].mean()), lon=float(lon[sel].mean()),
                         az_increasing_km=float(az % 360), p_app_s_per_km=float(coef[0]), p_se=float(se), cc_med=float(np.median(cc)),
                         t_rel_mean=float(np.mean(sh[good])), resid_rms=float(np.sqrt(np.mean(r_ ** 2)))))
        traces[f"{cab} {a}-{b}"] = (s_km, sh, good, coef, tw[m], E)
M = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(M.round(3).to_string(index=False))
M.to_csv(f"{OUT}/das_location_jan13_moveout.csv", index=False)

fig, axs = plt.subplots(2, len(traces), figsize=(5 * len(traces), 9))
for j, (k, (s_km, sh, good, coef, tm, E)) in enumerate(traces.items()):
    ax = axs[0, j]
    ax.pcolormesh(tm, s_km, E, vmin=0, vmax=3, cmap="magma", shading="auto")
    ax.plot(20 + sh, s_km, "c.", ms=3)
    ax.set(title=f"{k} km: envelopes (log SNR)", xlabel="s after PNSN origin (DAS time)", ylabel="km along stretch")
    ax = axs[1, j]
    ax.plot(s_km[good], sh[good], "k.", ms=4)
    ax.plot(s_km[~good], sh[~good], ".", color="0.7", ms=4)
    ax.plot(s_km, coef[0] * s_km + coef[1], "r-", label=f"p_app {coef[0]:+.3f} s/km")
    ax.set(xlabel="km along stretch", ylabel="relative delay (s)")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/figs/das_location_jan13_moveout.png", dpi=95)

# ---- multi-span wavefront location: fit every channel's relative delay within each well-correlated stretch
# (free constant per stretch -> no latency, no onset picks; only the wavefront shape across spans)
USE = [k for k, r in zip(traces, rows) if r["cc_med"] > 0.6]
pts = []
for cab in ("north", "south"):
    f_geom = pd.read_csv(f"{OUT}/geometry_{cab}.csv")
    for k in USE:
        if not k.startswith(cab):
            continue
        a, b = map(float, k.split()[1].split("-"))
        s_km, sh, good, *_ = traces[k]
        gg = f_geom[(f_geom.opt_km >= a) & (f_geom.opt_km <= b)]
        r0 = np.interp(a, f_geom.opt_km, f_geom.route_km)
        la = np.interp(s_km + r0, gg.route_km, gg.lat)
        lo = np.interp(s_km + r0, gg.route_km, gg.lon)
        for i in np.where(good)[0]:
            pts.append(dict(stretch=k, lat=la[i], lon=lo[i], d=sh[i]))
Q = pd.DataFrame(pts)
LATG = np.arange(44.80, 45.60, 0.01)
LONG = np.arange(-125.40, -123.85, 0.01)
mis = np.full((len(LATG), len(LONG)), np.inf)
best = (np.inf,)
grp = [Q.stretch.values == k for k in USE]
for ia, la_ in enumerate(LATG):
    _, _, d = G.inv(np.tile(Q.lon.values, len(LONG)), np.tile(Q.lat.values, len(LONG)), np.repeat(LONG, len(Q)), np.full(len(LONG) * len(Q), la_))
    d = d.reshape(len(LONG), len(Q)) / 1e3
    for z in (1, 5, 10, 15, 20):
        r = np.hypot(d, z)
        for v in np.arange(2.4, 4.21, 0.1):
            res = Q.d.values[None] - r / v
            for gm in grp:
                res[:, gm] -= np.median(res[:, gm], axis=1, keepdims=True)
            L1 = np.mean([np.mean(np.abs(res[:, gm]), axis=1) for gm in grp], axis=0)  # each stretch weighted equally
            k = np.argmin(L1)
            mis[ia] = np.minimum(mis[ia], L1)
            if L1[k] < best[0]:
                best = (float(L1[k]), float(la_), float(LONG[k]), z, float(v))
_, _, ds = G.inv(CAT["lon"], CAT["lat"], best[2], best[1])
ia, io = np.where(mis <= best[0] * 1.1)
print(f"wavefront fit over {USE}: misfit {best[0]:.3f} s at {best[1]:.2f}N {best[2]:.2f}W, z {best[3]} km, Vs {best[4]:.1f}; "
      f"{ds / 1e3:.1f} km from PNSN; region (misfit <= 1.1 x min) lat {LATG[ia].min():.2f}-{LATG[ia].max():.2f}, lon {LONG[io].min():.2f}-{LONG[io].max():.2f}")
pd.DataFrame([dict(misfit=best[0], lat=best[1], lon=best[2], depth_km=best[3], vs=best[4], dist_to_PNSN_km=ds / 1e3, stretches=";".join(USE))]).to_csv(
    f"{OUT}/das_location_jan13_wavefront.csv", index=False)
np.save(f"{OUT}/das_location_jan13_wavefront_misfit.npy", mis)
fig, ax = plt.subplots(figsize=(10, 7))
pc = ax.pcolormesh(LONG, LATG, mis / best[0], vmin=1, vmax=2, cmap="viridis_r", shading="auto")
ax.contour(LONG, LATG, mis / best[0], levels=[1.1], colors="w")
for c in ("north", "south"):
    rt = read_route(c)
    ax.plot(rt.lon, rt.lat, "-", color="0.6", lw=1)
for k in USE:
    q = Q[Q.stretch == k]
    ax.plot(q.lon, q.lat, ".", ms=5, label=f"{k} km")
ax.plot(CAT["lon"], CAT["lat"], "*", ms=16, color="0.85", mec="k", label="PNSN (onshore only; comparison)")
ax.plot(best[2], best[1], "*", ms=18, color="tab:red", mec="k", label=f"multi-span wavefront fit ({ds / 1e3:.0f} km from PNSN)")
ax.set(xlim=(LONG[0], LONG[-1]), ylim=(LATG[0], LATG[-1]), xlabel="longitude", ylabel="latitude",
       title=f"Jan 13 ML 2.0: location from the S-wavefront shape across {len(USE)} fiber stretches (no latency, no picks)\n"
             f"misfit/min (white: 1.1); z {best[3]} km, Vs {best[4]:.1f} km/s")
ax.set_aspect(1 / np.cos(np.radians(45.2)))
ax.legend(fontsize=7, loc="lower left")
fig.colorbar(pc, ax=ax, label="misfit / minimum")
fig.savefig(f"{OUT}/figs/das_location_jan13_wavefront.png", dpi=100, bbox_inches="tight")

# ---- stretch onsets from the aligned stack (AIC on the stacked energy) -> per-channel absolute DAS times
# t_i = stack onset + (delay_i - mean delay); both cables share one latency (s67: 5.80 s), so inter-stretch
# differences are pure travel-time differences.
on_rows = []
for k in USE:
    s_km, sh, good, coef, tm, E = traces[k]
    dt_ = tm[1] - tm[0]
    stack = np.mean([np.roll(np.exp(E[i]), -int(round((sh[i] - sh[good].mean()) / dt_))) for i in np.where(good)[0]], axis=0)
    lo_, hi_ = np.argmax(tm >= 0), np.argmax(stack)  # change point between origin and the stack maximum
    y = stack[lo_:hi_ + 1]
    n = len(y)
    kk = np.arange(3, n - 3)
    aic = np.array([j * np.log(np.var(y[:j]) + 1e-12) + (n - j) * np.log(np.var(y[j:]) + 1e-12) for j in kk])
    t_on = float(tm[lo_ + kk[np.argmin(aic)]])
    # bootstrap over channels for the stack-onset uncertainty
    rng = np.random.default_rng(1)
    bs = []
    gi = np.where(good)[0]
    for _ in range(100):
        pick = rng.choice(gi, len(gi))
        st = np.mean([np.roll(np.exp(E[i]), -int(round((sh[i] - sh[good].mean()) / dt_))) for i in pick], axis=0)
        h2 = np.argmax(st)
        y2 = st[lo_:h2 + 1]
        n2 = len(y2)
        k2 = np.arange(3, n2 - 3)
        a2 = np.array([j * np.log(np.var(y2[:j]) + 1e-12) + (n2 - j) * np.log(np.var(y2[j:]) + 1e-12) for j in k2])
        bs.append(tm[lo_ + k2[np.argmin(a2)]])
    q = Q[Q.stretch == k]
    bg = np.median(stack[(tm >= -5) & (tm <= 3)])
    thr = {p_: float(tm[lo_ + np.argmax(stack[lo_:] >= bg + p_ * (stack.max() - bg))]) for p_ in (0.1, 0.2, 0.5)}
    on_rows.append(dict(stretch=k, t_on_aic=t_on, t_on_aic_sd=float(np.std(bs)), t10=thr[0.1], t20=thr[0.2], t50=thr[0.5], n=len(q)))
    t_on = thr[0.2]  # S rise: 20 % of the stack maximum above background (AIC is unstable on the long north stretch)
    Q.loc[Q.stretch == k, "t_abs_das"] = t_on + (q.d - q.d.mean())
ON = pd.DataFrame(on_rows)
print(ON.round(2).to_string(index=False))
Q.to_csv(f"{OUT}/das_location_jan13_channel_times.csv", index=False)
