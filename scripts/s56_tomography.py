"""Static ambient-noise surface-wave (Scholte/Rayleigh) group-velocity tomography of the south-cable region (Hydrate Ridge).

Data: all-period (Dec 23 - Jan 28) reference cross-span CCFs (interspan_v4_ram: RAM, gap-fixed, 1-km bins, 3996 pairs,
bands b1 0.12-0.30 Hz and b0 0.06-0.15 Hz).
Group velocity per pair and period T: Gaussian narrow-band filter (alpha 25) of the cross-spectrum, envelope of the
CCF, positive and negative lag envelopes summed, peak in [d/1.05, d/0.50] s (reference-guided: the dominant regional
branch at ~0.7 km/s; the slow 0.4-0.5 km/s branch and zero-lag energy are excluded); picks on the window edge rejected. Kept if: d >= 3 lambda (lambda = U T),
SNR >= 8 (peak / RMS envelope at |lag| > d/0.25 + 30 s), and the two lag sides agree within 15 % when both have SNR >= 5.
Inversion: straight rays, 0.05 deg cells, slowness perturbation relative to the mean, damped least squares with
Laplacian smoothing (scipy lsqr); cells with < 5 rays masked. Checkerboard test (anomalies +-8 %, ~20 km) with the same
rays and Gaussian noise equal to the residual RMS.
Outputs: outputs_100m/tomography/{measurements.csv, maps_T<T>.npz}, figs/final/tomo_{maps,checker,coverage}.png
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.signal import hilbert
from scipy.sparse.linalg import lsqr
from pyproj import Geod

from msdvv import ROOT
from msdvv.geometry import read_route, repeaters
from msdvv.xcorr import spectra_to_ccf

B = Path("/data/wsd02/multispan_dvv")
CABLE = os.environ.get("CABLE", "south")
SRC = B / ("outputs_100m_v2/interspan_v4_ram" if CABLE == "south" else "outputs_100m_v2/interspan_v6_north")
OD = B / ("outputs_100m/tomography" if CABLE == "south" else "outputs_100m/tomography_north")
OD.mkdir(parents=True, exist_ok=True)
FD = B / "outputs_100m/figs/final"
G = Geod(ellps="WGS84")
PERIODS = {"b1": [4.0, 5.0, 6.0, 7.0], "b0": [8.0, 10.0, 12.0]}
ALPHA, ML = 25.0, 480.0
UMIN, UMAX = 0.50, 1.05  # search window around the dominant branch (~0.7 km/s); excludes zero-lag energy and the slow branch
P = pd.read_csv(SRC / "pairs.csv")
BN = pd.read_csv(SRC / "bins.csv").set_index("bin")
P["lat_i"], P["lon_i"] = BN.loc[P.bin_i, "lat"].values, BN.loc[P.bin_i, "lon"].values
P["lat_j"], P["lon_j"] = BN.loc[P.bin_j, "lat"].values, BN.loc[P.bin_j, "lon"].values
P = P[~P.seg_i.isin(["C3"]) & ~P.seg_j.isin(["C3"])].copy() if CABLE == "south" else P.copy()
keep = P.index.values

# ---------------- reference spectra and group-velocity measurements
rows = []
for band, Ts in PERIODS.items():
    S = None
    for f in sorted(SRC.glob("*.h5")):
        with h5py.File(f, "r") as h:
            s = h[band]["spec"][:].sum(0)
            S = s if S is None else S + s
            bins, fs, nfft = h[band]["bins"][:], h.attrs["fs"], int(h.attrs["nfft"])
    S = S[keep]
    fr = np.fft.rfftfreq(nfft, 1 / fs)[bins]
    for T in Ts:
        f0 = 1 / T
        w = np.exp(-ALPHA * ((fr - f0) / f0) ** 2).astype(np.float32)
        cc, lags = spectra_to_ccf(S * w, bins, nfft, fs, ML)
        env = np.abs(hilbert(cc, axis=-1))
        pos = lags >= 0
        ep, en = env[:, pos], env[:, : pos.sum()][:, ::-1]  # en: negative side (lags -ML..0) mirrored onto 0..+ML
        la = lags[pos]
        for k, (pi, r) in enumerate(P.iterrows()):
            d = r.dist_km
            win = (la >= d / UMAX) & (la <= d / UMIN)  # reference-guided: dominant regional branch only
            noise = la >= min(d / 0.25 + 30, ML - 40)
            if win.sum() < 5 or noise.sum() < 20:
                continue
            out = {}
            for nm, e in (("sum", ep[k] + en[k]), ("pos", ep[k]), ("neg", en[k])):
                j = np.where(win)[0][np.argmax(e[win])]
                out[nm] = (la[j], e[j] / (np.sqrt(np.mean(e[noise] ** 2)) + 1e-30))
            t, snr = out["sum"]
            U = d / t
            agree = True
            if out["pos"][1] >= 5 and out["neg"][1] >= 5:
                agree = abs(out["pos"][0] - out["neg"][0]) <= 0.15 * t
            edge = (t <= d / UMAX + 1.5 / fs) or (t >= d / UMIN - 1.5 / fs)
            ok = (snr >= 8) and agree and (d >= 3 * U * T) and not edge
            rows.append(dict(band=band, T=T, pair=pi, bin_i=r.bin_i, bin_j=r.bin_j, dist_km=d, t_s=t, U=U, snr=snr,
                             t_pos=out["pos"][0], t_neg=out["neg"][0], snr_pos=out["pos"][1], snr_neg=out["neg"][1], ok=ok,
                             lat_i=r.lat_i, lon_i=r.lon_i, lat_j=r.lat_j, lon_j=r.lon_j))
        print(f"T={T:g} s ({band}): {sum(x['ok'] for x in rows if x['T'] == T)} of {len(P)} pairs kept", flush=True)
M = pd.DataFrame(rows)
M["src"] = "DAS-DAS"
INCLUDE_OBS = os.environ.get("INCLUDE_OBS", "1") == "1" and CABLE == "south"  # seismometer paths exist for the south cable only
SUF = ("" if INCLUDE_OBS else "_dasonly") if CABLE == "south" else "_north"
if INCLUDE_OBS:
    # seismometer-DAS paths (HYSB1 slope base, HYS14 summit inside the loop) x every good south-cable bin: vertical
    # component (Scholte/Rayleigh), RAM, latency-corrected, timing-QC'd, gap-fixed; same picking rules as DAS-DAS
    from obspy import read_inventory
    from msdvv.obsdas import file_timing_qc, load_hourly_qc
    t_o, H_o, lags_o, Po = load_hourly_qc("Z", file_timing_qc("T"), prep="ram", maxlag=ML)
    ref_o = H_o.sum(0).astype(np.float64)
    del H_o
    inv_o = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
    crd_o = {st_.code: (st_.latitude, st_.longitude) for n_ in inv_o for st_ in n_}
    fs_o = 1 / (lags_o[1] - lags_o[0])
    Fo = np.fft.rfft(ref_o, axis=-1)
    fo = np.fft.rfftfreq(ref_o.shape[-1], 1 / fs_o)
    pos_o = lags_o >= 0
    la_o = lags_o[pos_o]
    orow = []
    for T in [t_ for t_ in PERIODS["b1"]]:
        f0 = 1 / T
        cc = np.fft.irfft(Fo * np.exp(-ALPHA * ((fo - f0) / f0) ** 2), n=ref_o.shape[-1], axis=-1)
        env = np.abs(hilbert(cc, axis=-1))
        ep, en = env[:, pos_o], env[:, : pos_o.sum()][:, ::-1]
        for k, r in Po.iterrows():
            d = r.dist_km
            win = (la_o >= d / UMAX) & (la_o <= d / UMIN)
            noise = la_o >= min(d / 0.25 + 30, ML - 40)
            if win.sum() < 5 or noise.sum() < 20 or d < 5:
                continue
            out = {}
            for nm, e in (("sum", ep[k] + en[k]), ("pos", ep[k]), ("neg", en[k])):
                j = np.where(win)[0][np.argmax(e[win])]
                out[nm] = (la_o[j], e[j] / (np.sqrt(np.mean(e[noise] ** 2)) + 1e-30))
            t, snr = out["sum"]
            Uo = d / t
            agree = True
            if out["pos"][1] >= 5 and out["neg"][1] >= 5:
                agree = abs(out["pos"][0] - out["neg"][0]) <= 0.15 * t
            edge = (t <= d / UMAX + 1.5 / fs_o) or (t >= d / UMIN - 1.5 / fs_o)
            ok = (snr >= 8) and agree and (d >= 3 * Uo * T) and not edge
            la_s, lo_s = crd_o[r.obs]
            orow.append(dict(band="b1", T=T, pair=-1, bin_i=r.obs, bin_j=r.bin, dist_km=d, t_s=t, U=Uo, snr=snr, t_pos=out["pos"][0], t_neg=out["neg"][0],
                             snr_pos=out["pos"][1], snr_neg=out["neg"][1], ok=ok, lat_i=la_s, lon_i=lo_s, lat_j=r.bin_lat, lon_j=r.bin_lon, src=f"{r.obs}-DAS"))
        print(f"T={T:g} s seismometer-DAS: {sum(x['ok'] for x in orow if x['T'] == T)} of {len(Po)} paths kept "
              f"(HYSB1 {sum(x['ok'] for x in orow if x['T'] == T and x['src'] == 'HYSB1-DAS')}, HYS14 {sum(x['ok'] for x in orow if x['T'] == T and x['src'] == 'HYS14-DAS')})", flush=True)
    M = pd.concat([M, pd.DataFrame(orow)], ignore_index=True)
M.to_csv(OD / f"measurements{SUF}.csv", index=False)

# ---------------- grid and rays
if CABLE == "south":
    LON = np.arange(-125.65, -124.15 + 1e-9, 0.05)
    LAT = np.arange(44.25, 45.35 + 1e-9, 0.05)
else:  # north cable: Pacific City -> Axial; bins span ~124.0-127.3 W, 45.2-45.9 N
    LON = np.arange(-127.45, -123.85 + 1e-9, 0.05)
    LAT = np.arange(44.95, 46.05 + 1e-9, 0.05)
nx, ny = len(LON) - 1, len(LAT) - 1
lonc, latc = (LON[:-1] + LON[1:]) / 2, (LAT[:-1] + LAT[1:]) / 2


def ray_rows(D):
    """Sparse G (nray x ncell) of path length (km) per cell, straight great-circle rays."""
    I, J, V = [], [], []
    for i, r in enumerate(D.itertuples()):
        pts = np.array(G.npts(r.lon_i, r.lat_i, r.lon_j, r.lat_j, 200, initial_idx=0, terminus_idx=0))
        seg = r.dist_km / (len(pts) - 1)
        ix = np.clip(np.searchsorted(LON, pts[:, 0]) - 1, 0, nx - 1)
        iy = np.clip(np.searchsorted(LAT, pts[:, 1]) - 1, 0, ny - 1)
        cell = iy * nx + ix
        u, c = np.unique(cell, return_counts=True)
        I += [i] * len(u)
        J += list(u)
        V += list(c * seg * (len(pts) - 1) / len(pts))
    return sp.csr_matrix((V, (I, J)), shape=(len(D), nx * ny))


def laplacian():
    rows_, cols, vals = [], [], []
    k = 0
    for iy in range(ny):
        for ix in range(nx):
            c = iy * nx + ix
            nb = [(iy + a) * nx + ix + b for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= iy + a < ny and 0 <= ix + b < nx]
            rows_ += [k] * (len(nb) + 1)
            cols += [c] + nb
            vals += [len(nb)] + [-1.0] * len(nb)
            k += 1
    return sp.csr_matrix((vals, (rows_, cols)), shape=(k, nx * ny))


L = laplacian()
import os
LAM_S, LAM_D = float(os.environ.get("LAM_S", "40")), float(os.environ.get("LAM_D", "4"))


def invert(Gm, dt):
    A = sp.vstack([Gm, LAM_S * L, LAM_D * sp.identity(nx * ny)])
    b = np.concatenate([dt, np.zeros(L.shape[0] + nx * ny)])
    return lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=5000)[0]


maps = {}
for T in sorted(M["T"].unique()):
    D = M[(M["T"] == T) & M.ok].reset_index(drop=True)
    if len(D) < 30:
        continue
    Gm = ray_rows(D)
    s0 = (D.t_s / D.dist_km).median()  # reference slowness (s/km)
    dt = D.t_s.values - s0 * D.dist_km.values
    # one pass of outlier rejection (|residual| > 3 MAD after a first inversion)
    m1 = invert(Gm, dt)
    res = dt - Gm @ m1
    good = np.abs(res - np.median(res)) <= 3 * 1.4826 * np.median(np.abs(res - np.median(res)))
    D, Gm, dt = D[good].reset_index(drop=True), Gm[good], dt[good]
    m = invert(Gm, dt)
    res = dt - Gm @ m
    hits = np.asarray((Gm > 0).sum(0)).ravel()
    U = 1 / (s0 + m)
    Umap = np.where(hits >= 5, U, np.nan).reshape(ny, nx)
    # checkerboard: +-8 % slowness, 4x4-cell squares (~20 km), same rays, noise = residual RMS
    chk = 0.08 * s0 * np.sign(np.sin(np.pi * (np.arange(nx)[None] // 4 + 0.5)) * np.sin(np.pi * (np.arange(ny)[:, None] // 4 + 0.5)))
    NC = int(os.environ.get("CHK_N", "4"))
    chk = np.where(((np.arange(ny)[:, None] // NC) + (np.arange(nx)[None] // NC)) % 2 == 0, 1, -1) * 0.08 * s0
    dtc = Gm @ chk.ravel() + np.random.default_rng(0).normal(0, np.std(res), len(dt))
    mc = invert(Gm, dtc)
    msk = hits >= 5
    r_chk = float(np.corrcoef(chk.ravel()[msk], mc[msk])[0, 1])
    amp_chk = float(np.std(mc[msk]) / np.std(chk.ravel()[msk]))
    maps[T] = dict(r_chk=r_chk, amp_chk=amp_chk, U=Umap, U0=1 / s0, hits=hits.reshape(ny, nx), chk_in=np.where(hits.reshape(ny, nx) >= 5, chk / s0, np.nan),
                   chk_out=np.where(hits.reshape(ny, nx) >= 5, mc.reshape(ny, nx) / s0, np.nan), n=len(D),
                   rms_before=float(np.std(dt)), rms_after=float(np.std(res)), D=D)
    np.savez(OD / f"maps_T{T:g}{SUF}.npz", lon=lonc, lat=latc, U=Umap, hits=hits.reshape(ny, nx), U0=1 / s0)
    print(f"T={T:g} s: {len(D)} rays ({int((D.src != 'DAS-DAS').sum())} seismometer-DAS), U0 {1 / s0:.3f} km/s, travel-time residual std {np.std(dt):.2f} -> {np.std(res):.2f} s, "
          f"U range {np.nanpercentile(Umap, 5):.3f}-{np.nanpercentile(Umap, 95):.3f} km/s; checker ({NC} cells) r {r_chk:.2f}, amplitude {amp_chk:.2f}", flush=True)

# ---------------- figures
fc = json.load(open(ROOT / "resources/qfaults_oregon_margin_strikeslip.geojson"))
rt, rp = read_route(CABLE), repeaters(CABLE)


def deco(ax):
    ax.plot(rt.lon, rt.lat, "-", color="k", lw=1.0)
    for f in fc["features"]:
        c = np.array(f["geometry"]["coordinates"])
        ax.plot(c[:, 0], c[:, 1], color="saddlebrown", lw=1.6)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(-125.148 + 5 / (111.2 * np.cos(np.radians(44.569))) * np.cos(th), 44.569 + 5 / 111.2 * np.sin(th), "--", color="goldenrod", lw=1.3)
    ax.plot(rp.lon, rp.lat, "k^", ms=6)
    ax.set(xlim=(LON[0], LON[-1]), ylim=(LAT[0], LAT[-1]))
    ax.set_aspect(1 / np.cos(np.radians(44.8)))


Ts = sorted(maps)
nc = 4
nr = int(np.ceil(len(Ts) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.6 * nr), squeeze=False, layout="constrained")
for ax, T in zip(axs.ravel(), Ts):
    mp = maps[T]
    dU = 100 * (mp["U"] - mp["U0"]) / mp["U0"]
    im = ax.pcolormesh(LON, LAT, dU, cmap="RdBu", vmin=-10, vmax=10, shading="flat")
    deco(ax)
    ax.set_title(f"T = {T:g} s: U0 = {mp['U0']:.2f} km/s, {mp['n']} paths\nresid. {mp['rms_before']:.1f} -> {mp['rms_after']:.1f} s", fontsize=9)
for ax in axs.ravel()[len(Ts):]:
    ax.axis("off")
fig.colorbar(im, ax=axs, shrink=0.6, label="group-velocity perturbation (%)  (red slow, blue fast)")
TT_ = "cross-span DAS + HYSB1/HYS14-DAS correlations" if INCLUDE_OBS else "cross-span DAS correlations only"
fig.suptitle(f"Ambient-noise Scholte/Rayleigh group-velocity tomography, {CABLE} cable{' / Hydrate Ridge' if CABLE == 'south' else ' (nearly straight cable: rays along it, i.e. an along-cable profile only)'} ({TT_}, Dec 23 - Jan 28)\n"
             "brown = Qfaults (Alvin Canyon, Daisy Bank), dashed circle = Southern Hydrate Ridge, black = cable; cells with < 5 rays blank", fontsize=11)
fig.savefig(FD / f"tomo_maps{SUF}.png", dpi=100)
plt.close(fig)

fig, axs = plt.subplots(2, len(Ts), figsize=(3.6 * len(Ts), 6.8), squeeze=False, layout="constrained")
for j, T in enumerate(Ts):
    mp = maps[T]
    for i, (key, lab) in enumerate((("chk_in", "input"), ("chk_out", "recovered"))):
        im = axs[i, j].pcolormesh(LON, LAT, -100 * mp[key], cmap="RdBu", vmin=-8, vmax=8, shading="flat")
        deco(axs[i, j])
        axs[i, j].set_title(f"T = {T:g} s, {lab}", fontsize=9)
fig.colorbar(im, ax=axs, shrink=0.6, label="velocity perturbation (%)")
fig.suptitle("Checkerboard test (+-8 %, 0.6-deg (~50-65 km) squares), same rays and noise level as the data", fontsize=11)
fig.savefig(FD / f"tomo_checker{SUF}.png", dpi=95)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained")
T = 5.0 if 5.0 in maps else Ts[0]
D = maps[T]["D"]
ax = axs[0]
for r in D.itertuples():
    c_ = {"DAS-DAS": "tab:blue", "HYSB1-DAS": "tab:green", "HYS14-DAS": "tab:red"}.get(r.src, "k")
    ax.plot([r.lon_i, r.lon_j], [r.lat_i, r.lat_j], color=c_, lw=0.25 if r.src == "DAS-DAS" else 0.8, alpha=0.4 if r.src == "DAS-DAS" else 0.8)
if INCLUDE_OBS:
    for nm_, c_ in (("HYSB1", "tab:green"), ("HYS14", "tab:red")):
        ax.plot(crd_o[nm_][1], crd_o[nm_][0], "v", color=c_, mec="k", ms=11)
        ax.annotate(nm_, (crd_o[nm_][1], crd_o[nm_][0]), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")
deco(ax)
ax.set_title(f"ray coverage, T = {T:g} s ({len(D)} paths: blue DAS-DAS, green HYSB1-DAS, red HYS14-DAS)", fontsize=10)
ax = axs[1]
for T_ in Ts:
    q = M[(M["T"] == T_) & M.ok]
    ax.scatter(q.dist_km, q.U, s=3, label=f"T = {T_:g} s (median {q.U.median():.2f} km/s)")
ax.set(xlabel="inter-bin distance (km)", ylabel="group velocity (km/s)", title="kept measurements")
ax.legend(fontsize=8, markerscale=3)
fig.savefig(FD / f"tomo_coverage{SUF}.png", dpi=100)
json.dump({f"{T:g}": dict(U0=maps[T]["U0"], n=maps[T]["n"], rms_before=maps[T]["rms_before"], rms_after=maps[T]["rms_after"]) for T in Ts},
          open(OD / f"summary{SUF}.json", "w"), indent=1)
print("done")
