"""Item 8c: what does MULTI-SPAN beamforming add over a conventional single-span DAS? Array-response comparison.

Configurations (1-km bins of good channels, as in s64; south C3 excluded as tonal):
  single span (conventional DAS reach): south A1, south A2, north A1
  one cable, all spans:                  south loop (A1 A2 B3 C4 D4), north (A1 A2 B3 C4)
  both cables:                           north + south (assumes the inter-cable latency difference is calibrated)
Synthetic noise-free plane Rayleigh wave (axial-strain sensitivity cos^2(baz - cable azimuth)) at f and slowness s0,
Bartlett beam with the same weighting (as s64). Metrics over true back-azimuths 0-356 (4 deg), noisy data: per-bin
amplitude SNR 0.5, 8 windows, CSDM without autocorrelations: back-azimuth error of
the beam maximum, highest sidelobe (beam power outside +-20 deg / +-0.1 s/km of the truth, relative to the peak),
and the 0.5-power width in back-azimuth at the true slowness.
Outputs: outputs_100m/beamform/array_response_metrics.csv, figs/array_response_{maps,metrics}.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod

from msdvv.interspan import make_bins

G = Geod(ellps="WGS84")
OUT = Path("/data/wsd02/multispan_dvv/outputs_100m")
FREQS = (0.1, 0.2, 0.5)
SNR = 0.5  # per-bin amplitude SNR for the error statistics (8 windows)
S0 = 0.35  # s/km (~2.9 km/s Rayleigh-like regional / microseism)
BAZ = np.radians(np.arange(0, 360, 2.0))
SLO = np.arange(0.0, 1.0001, 0.01)

B = []
for cab in ("north", "south"):
    q = pd.read_csv(OUT / f"channel_quality_{cab}.csv")
    b, _ = make_bins(q, 1.0, 3)
    b["cable"] = cab
    if cab == "south":
        b = b[b.seg != "C3"]
    # cable azimuth per bin from neighbours in the same segment
    azs = np.full(len(b), np.nan)
    for _, g in b.groupby("seg"):
        idx = g.index.values
        for i, ix in enumerate(idx):
            j0, j1 = idx[max(i - 2, 0)], idx[min(i + 2, len(idx) - 1)]
            azs[b.index.get_loc(ix)] = G.inv(b.lon[j0], b.lat[j0], b.lon[j1], b.lat[j1])[0]
    b["cab_az"] = azs
    B.append(b)
B = pd.concat(B, ignore_index=True)
lat0, lon0 = B.lat.mean(), B.lon.mean()
az, _, d = G.inv(np.full(len(B), lon0), np.full(len(B), lat0), B.lon.values, B.lat.values)
B["x"], B["y"] = d / 1e3 * np.sin(np.radians(az)), d / 1e3 * np.cos(np.radians(az))
CONF = {
    "south A1 (single span)": (B.cable == "south") & (B.seg == "A1"),
    "south A2 (single span)": (B.cable == "south") & (B.seg == "A2"),
    "north A1 (single span)": (B.cable == "north") & (B.seg == "A1"),
    "north, all spans": B.cable == "north",
    "south loop, all spans": B.cable == "south",
    "north + south, all spans": np.ones(len(B), bool),
}
print({k: int(v.sum()) for k, v in CONF.items()}, {k: sorted(B[v].seg.unique()) for k, v in CONF.items()})

KX = -np.sin(BAZ)[:, None] * SLO[None]  # wave propagates away from baz
KY = -np.cos(BAZ)[:, None] * SLO[None]


RNG = np.random.default_rng(0)


def beam(sel, f, baz0, s0=S0, snr=None, nreal=8):
    """Bartlett beam; with snr: CSDM of nreal windows of the plane wave (random phase) + independent complex noise per
    bin with amplitude = mean signal amplitude / snr."""
    x, y, ca = B.x.values[sel], B.y.values[sel], np.radians(B.cab_az.values[sel])
    d = np.cos(baz0 - ca) ** 2 * np.exp(-2j * np.pi * f * (-np.sin(baz0) * s0 * x - np.cos(baz0) * s0 * y))
    if snr is not None:
        sig = np.abs(d).mean() / snr
        D = d[None] * np.exp(2j * np.pi * RNG.random((nreal, 1))) + sig / np.sqrt(2) * (RNG.standard_normal((nreal, len(d))) + 1j * RNG.standard_normal((nreal, len(d))))
        A = np.cos(BAZ[:, None, None] - ca[None, None]) ** 2
        W = A * np.exp(-2j * np.pi * f * (KX[..., None] * x + KY[..., None] * y))
        # W^H C W with C = CSDM without its diagonal (as s64), computed as mean_r |W^H d_r|^2 - sum_n |W_n|^2 <|d_rn|^2>
        num = np.mean(np.abs(np.einsum("abn,rn->abr", np.conj(W), D)) ** 2, axis=-1) - (np.abs(W) ** 2) @ np.mean(np.abs(D) ** 2, 0)
        return num / (np.abs(W).sum(-1) ** 2 - (np.abs(W) ** 2).sum(-1))
    A = np.cos(BAZ[:, None, None] - ca[None, None]) ** 2  # (nbaz, 1, n)
    W = A * np.exp(-2j * np.pi * f * (KX[..., None] * x + KY[..., None] * y))
    num = np.abs(np.einsum("abn,n->ab", np.conj(W), d)) ** 2
    den = (np.abs(W) ** 2).sum(-1) * (np.abs(d) ** 2).sum()
    return num / den


rows = []
for f in FREQS:
    for name, sel in CONF.items():
        sel = np.asarray(sel)
        for b0 in np.radians(np.arange(0, 360, 4.0)):
            P = beam(sel, f, b0, snr=SNR)
            ia, js = np.unravel_index(np.argmax(P), P.shape)
            dB = np.degrees(np.angle(np.exp(1j * (BAZ - b0))))
            far = (np.abs(dB)[:, None] > 20) | (np.abs(SLO - S0)[None] > 0.1)
            js0 = np.argmin(np.abs(SLO - S0))
            prof = P[:, js0] / P.max()
            rows.append(dict(f=f, config=name, true_baz=round(np.degrees(b0)), baz_err=float(dB[ia]), slo_err=float(SLO[js] - S0),
                             sidelobe=float(P[far].max() / P.max()), width_deg=float(2 * np.sum(prof >= 0.5))))
M = pd.DataFrame(rows)
M.to_csv(OUT / "beamform/array_response_metrics.csv", index=False)
S = M.assign(abs_err=M.baz_err.abs(), fail=M.baz_err.abs() > 10).groupby(["f", "config"], sort=False).agg(
    median_abs_baz_err=("abs_err", "median"), frac_wrong_gt10deg=("fail", "mean"), median_sidelobe=("sidelobe", "median"),
    median_width_deg=("width_deg", "median"))
pd.set_option("display.width", 200)
print(S.round(2).to_string())

# figure 1: geometry + beam maps for one true direction (250 deg, the Blanco/microseism direction) at 0.2 Hz
show = ["south A1 (single span)", "north A1 (single span)", "south loop, all spans", "north + south, all spans"]
fig = plt.figure(figsize=(20, 9))
ax = fig.add_subplot(2, 5, (1, 6))
cols = dict(zip(sorted((B.cable + " " + B.seg).unique()), plt.cm.tab10.colors))
for k, g in B.groupby(B.cable + " " + B.seg):
    ax.plot(g.lon, g.lat, ".", color=cols[k], ms=4, label=k)
ax.set(xlabel="longitude", ylabel="latitude", title="1-km bins per span (C3 south excluded)")
ax.set_aspect(1 / np.cos(np.radians(lat0)))
ax.legend(fontsize=7, ncol=2)
for i, name in enumerate(show):
    for r, f in enumerate((0.2, 0.5)):
        a = fig.add_subplot(2, 5, 2 + i + 5 * r, projection="polar")
        P = beam(np.asarray(CONF[name]), f, np.radians(250))
        a.pcolormesh(BAZ, SLO, (P / P.max()).T, cmap="viridis", vmin=0, vmax=1, shading="auto")
        a.plot(np.radians(250), S0, "r+", ms=12, mew=2)
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)
        a.set_yticklabels([])
        a.set_title(f"{name}\n{f} Hz, true 250 deg (+)", fontsize=8)
fig.suptitle(f"Array response: single span vs multi-span vs both cables (plane Rayleigh wave, {S0} s/km, axial-strain cos^2 sensitivity)")
fig.tight_layout()
fig.savefig(OUT / "figs/array_response_maps.png", dpi=100)

# figure 2: back-azimuth error vs true back-azimuth per configuration
fig, axs = plt.subplots(1, len(FREQS), figsize=(18, 5), sharey=True)
for ax, f in zip(axs, FREQS):
    for name in CONF:
        q = M[(M.f == f) & (M.config == name)]
        ax.plot(q.true_baz, q.baz_err.abs(), lw=2 if "all" in name else 1, label=name)
    ax.set(xlabel="true back-azimuth (deg)", title=f"{f} Hz: |back-azimuth error| of the beam maximum", yscale="symlog")
axs[0].set_ylabel("deg")
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figs/array_response_metrics.png", dpi=100)
