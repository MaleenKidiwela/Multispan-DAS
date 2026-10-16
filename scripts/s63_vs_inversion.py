"""Item 5: shear-velocity depth structure from Scholte/Rayleigh group-velocity dispersion (env pyfetch: disba + pyproj).

Data per region (median of the measurements inside it, uncertainty = max(15 %, spread)):
  * along-cable short-offset group velocity (s12 scholte_velocity_profile.csv; 100-m stream): b2 0.3-0.7 Hz (T ~ 2 s)
    and b1 0.12-0.3 Hz (T ~ 5 s). The 1-2 Hz (8-Hz stream) picks are excluded: in deep water they are faster than
    the lower bands (0.7-2 km/s), i.e. water-borne / leaky-P energy, not the Scholte wave.
  * tomography group velocity (s56, fibre + seismometer paths, strong smoothing) at T = 5, 6, 7, 8 s, at the region's
    cells (resolution >= ~50 km; 4 s excluded as inconsistent).
Regions: south slope (span A2, route km 95-131, ~1.9 km water), abyssal base of Hydrate Ridge (span B3, route 201-249,
2.9 km), Hydrate Ridge summit area (tomography only, within 15 km of HYS14, ~0.8 km water), shelf (spans A1/C4,
~0.2-0.5 km water; T ~ 2 s only).
Model: water layer (Vp 1.5, rho 1.03) over sediments with Vs(z) = Vs0 + g z (capped at 2.5 km/s) down to 4 km, then a
half-space (Vs 3.2, Vp 5.6, rho 2.7); Vp = 1.16 Vs + 1.36 (Castagna mudrock line, >= 1.55), rho = 1.74 Vp^0.25 (Gardner,
>= 1.6). Grid search over Vs0 in [0.05, 1.6] km/s and g in [0.05, 2.0] (km/s)/km; likelihood exp(-chi2/2); posterior
mean and 5-95 % Vs(z). Fundamental-mode group velocity (disba, Dunkin).
Outputs: outputs_100m/vs_inversion/{data.csv, results.csv}, figs/final/vs_profiles.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from disba import GroupDispersion
from pyproj import Geod

B = Path("/data/wsd02/multispan_dvv")
OD = B / "outputs_100m/vs_inversion"
OD.mkdir(parents=True, exist_ok=True)
G = Geod(ellps="WGS84")
HYS14 = (44.5691, -125.1479)
prof = pd.read_csv(B / "outputs_100m/dvv/scholte_velocity_profile.csv")
prof = prof[prof.cable == "south"]
TOMO = {T: np.load(B / f"outputs_100m/tomography/maps_T{T}.npz") for T in (5, 6, 7, 8)}
REG = {
    "south slope (A2, ~1.9 km water)": dict(span=2, route=(95, 131), tomo=None),
    "base of Hydrate Ridge (B3, 2.9 km water)": dict(span=3, route=(201, 249), tomo="cable"),
    "Hydrate Ridge summit (tomography, ~0.8 km water)": dict(span=None, route=None, tomo="hys14", water=0.8),
    "shelf (A1/C4, ~0.2-0.5 km water)": dict(span=(1, 4), route=None, tomo=None),
}


def tomo_value(T, lat, lon, r_km):
    M = TOMO[T]
    LO, LA = np.meshgrid(M["lon"], M["lat"])
    _, _, d = G.inv(LO.ravel(), LA.ravel(), np.full(LO.size, lon), np.full(LO.size, lat))
    m = (d / 1e3 <= r_km) & np.isfinite(M["U"].ravel())
    return (float(np.median(M["U"].ravel()[m])), float(np.std(M["U"].ravel()[m]))) if m.any() else (np.nan, np.nan)


data = []
for name, R in REG.items():
    if R["span"] is not None:
        spans = R["span"] if isinstance(R["span"], tuple) else (R["span"],)
        q = prof[prof.span.isin(spans)]
        if R["route"]:
            q = q[q.route_km.between(*R["route"])]
        water = float(q.depth.median()) / 1e3
        for band, T in (("b2", 2.0), ("b1", 5.0)):
            v = q[q.band == band].v_kms
            if len(v) >= 3 and not (band == "b1" and water < 0.6):  # b1 at 0.4-3.2 km offsets on the shelf: < 1 wavelength
                data.append(dict(region=name, T=T, U=float(v.median()), err=max(0.15 * v.median(), float(v.std())), src=f"within-span {band} (n={len(v)})", water_km=water))
        if R["tomo"] == "cable":
            for T in TOMO:
                u, s = tomo_value(T, float(q.lat.median()), float(q.lon.median()), 12)
                if np.isfinite(u):
                    data.append(dict(region=name, T=float(T), U=u, err=max(0.15 * u, s), src="tomography", water_km=water))
    else:
        for T in TOMO:
            u, s = tomo_value(T, *HYS14, 15)
            if np.isfinite(u):
                data.append(dict(region=name, T=float(T), U=u, err=max(0.15 * u, s), src="tomography", water_km=R["water"]))
D = pd.DataFrame(data)
D.to_csv(OD / "data.csv", index=False)
print(D.round(3).to_string(index=False))

Z = np.arange(0, 4.0, 0.1)


def model(vs0, g, water):
    vs = np.minimum(vs0 + g * (Z + 0.05), 2.5)
    th = np.r_[water, np.full(len(Z), 0.1), 1.0]
    VS = np.r_[0.0, vs, 3.2]
    VP = np.r_[1.5, np.maximum(1.55, 1.16 * vs + 1.36), 5.6]
    RHO = np.r_[1.03, np.maximum(1.6, 1.74 * np.maximum(1.55, 1.16 * vs + 1.36) ** 0.25), 2.7]
    return th, VP, VS, RHO, vs


V0 = np.linspace(0.05, 1.6, 45)
GR = np.linspace(0.05, 2.0, 40)
res, post = [], {}
for name, g_ in D.groupby("region", sort=False):
    T = g_["T"].values
    o = np.argsort(T)
    T, U, E = T[o], g_.U.values[o], g_.err.values[o]
    water = float(g_.water_km.iloc[0])
    chi = np.full((len(V0), len(GR)), np.inf)
    for i, v0 in enumerate(V0):
        for j, gg in enumerate(GR):
            th, vp, vs, rho, _ = model(v0, gg, water)
            try:
                u = GroupDispersion(th, vp, vs, rho, algorithm="dunkin")(T, mode=0, wave="rayleigh").velocity
                if len(u) == len(T):
                    chi[i, j] = np.sum(((u - U) / E) ** 2)
            except Exception:  # noqa: BLE001
                pass
    L = np.exp(-(chi - np.nanmin(chi)) / 2)
    L /= L.sum()
    ib = np.unravel_index(np.nanargmin(chi), chi.shape)
    prof_all = np.array([[model(v0, gg, water)[4] for gg in GR] for v0 in V0])  # (nv0, ng, nz)
    w = L.ravel()
    P = prof_all.reshape(-1, len(Z))
    oo = np.argsort(P, axis=0)
    cw = np.cumsum(w[oo], axis=0)
    pct = lambda q: np.array([P[oo[np.searchsorted(cw[:, k], q), k], k] for k in range(len(Z))])  # noqa: E731
    th, vp, vs, rho, vsb = model(V0[ib[0]], GR[ib[1]], water)
    ub = GroupDispersion(th, vp, vs, rho, algorithm="dunkin")(np.linspace(1.5, 9, 40), mode=0, wave="rayleigh")
    post[name] = dict(T=T, U=U, E=E, src=g_.src.values[o], mean=(w[:, None] * P).sum(0), p5=pct(0.05), p95=pct(0.95), best=vsb, curve=(ub.period, ub.velocity), water=water)
    res.append(dict(region=name, n_data=len(T), water_km=water, vs0_best=V0[ib[0]], grad_best=GR[ib[1]], chi2_best=float(np.nanmin(chi)),
                    vs_100m=float(np.interp(0.1, Z, post[name]["mean"])), vs_500m=float(np.interp(0.5, Z, post[name]["mean"])),
                    vs_1km=float(np.interp(1.0, Z, post[name]["mean"])), vs_2km=float(np.interp(2.0, Z, post[name]["mean"])),
                    vs_1km_5_95=f"{np.interp(1.0, Z, post[name]['p5']):.2f}-{np.interp(1.0, Z, post[name]['p95']):.2f}"))
RS = pd.DataFrame(res)
RS.to_csv(OD / "results.csv", index=False)
pd.set_option("display.width", 220)
print(RS.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
cols = plt.get_cmap("tab10").colors
for k, (name, p) in enumerate(post.items()):
    c = cols[k]
    axs[0].errorbar(p["T"], p["U"], yerr=p["E"], fmt="o", color=c, capsize=3)
    axs[0].plot(*p["curve"], color=c, lw=1.2, label=name)
    axs[1].fill_betweenx(Z, p["p5"], p["p95"], color=c, alpha=0.15)
    axs[1].plot(p["mean"], Z, color=c, lw=2, label=name)
axs[0].set(xlabel="period (s)", ylabel="group velocity (km/s)", title="Scholte/Rayleigh group-velocity dispersion: data (points) and best-fit models")
axs[0].legend(fontsize=8)
axs[1].set(xlabel="Vs (km/s)", ylabel="depth below seafloor (km)", ylim=(3, 0), xlim=(0, 2.6),
           title="Vs(z): posterior mean and 5-95 % (linear-gradient sediment model)")
axs[1].legend(fontsize=8, loc="lower left")
fig.suptitle("Item 5: shear-velocity structure along the south cable (within-span + tomography group velocities; 15 % errors)", fontsize=11)
fig.tight_layout()
fig.savefig(B / "outputs_100m/figs/final/vs_profiles.png", dpi=105)
print("done")
