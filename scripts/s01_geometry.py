"""Step 0b: repeater picks and channel geometry for each cable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.geometry import amplitude_profile, build_geometry, read_route, repeaters
from msdvv.io import load_cfg

cfg = load_config()
out = Path(cfg["output_dir"])
(out / "figs").mkdir(parents=True, exist_ok=True)
inv = pd.read_csv(out / "inventory.csv")
PROFILE_S = 20 * 60  # 20 min per file is enough for a median-amplitude profile


def profile(path):
    f = load_cfg(path, cfg, tslice=slice(0, int(PROFILE_S * inv.fs[inv.path == path].iloc[0])))
    return amplitude_profile(f.data)


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for i, (cab, g) in enumerate(inv.groupby("cable")):
    g = g.iloc[np.linspace(0, len(g) - 1, min(len(g), 24)).astype(int)]  # spread sample of files
    profs = Parallel(n_jobs=len(g))(delayed(profile)(p) for p in g.path)
    lm = np.median(np.vstack(profs), axis=0)
    chans = load_cfg(g.path.iloc[0], cfg, tslice=slice(0, 1024)).channels()
    geo, picks, (a, b) = build_geometry(lm, chans, cab, cfg["geometry"]["edge_pad"], cfg["geometry"]["repeater_search_km"],
                                        cfg["geometry"].get("repeater_opt_km", {}).get(cab))
    geo["log_med_amp"] = lm
    geo.to_csv(out / f"geometry_{cab}.csv", index=False)
    picks.to_csv(out / f"repeater_picks_{cab}.csv", index=False)
    print(f"== {cab}: route = {a:.4f} * optical + {b:.2f} km")
    print(picks.to_string())
    print(geo.groupby(["block", "span"]).agg(ch0=("ch", "min"), ch1=("ch", "max"), opt0=("opt_km", "min"), opt1=("opt_km", "max"),
                                             route0=("route_km", "min"), route1=("route_km", "max"), dmin=("depth", "min"), dmax=("depth", "max")).round(1).to_string())
    ax = axes[0, i]
    live = geo.ch > 0  # channel 0 is dead (all zeros)
    ax.plot(geo.opt_km[live], lm[live], "k.", ms=2)
    for p in picks.itertuples():
        ax.axvline(p.opt_km, color="r", lw=0.8)
    ax.set(title=f"{cab}: median |strain rate| (log10)", xlabel="optical distance (km)")
    ax = axes[1, i]
    route = read_route(cab)
    ax.plot(route.lon, route.lat, "-", color="0.7")
    sc = ax.scatter(geo.lon, geo.lat, c=geo.depth, s=3, cmap="viridis_r")
    rep = repeaters(cab)
    ax.plot(rep.lon, rep.lat, "r>", ms=7)
    ax.plot(geo.lon[geo.ch.isin(picks.ch)], geo.lat[geo.ch.isin(picks.ch)], "kx", ms=8)
    plt.colorbar(sc, ax=ax, label="depth (m)")
    ax.set(title=f"{cab}: channel locations (x = picked repeater)", xlabel="lon", ylabel="lat")
fig.tight_layout()
fig.savefig(out / "figs" / "geometry.png", dpi=130)
