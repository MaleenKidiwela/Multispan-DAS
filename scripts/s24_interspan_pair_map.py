"""Map of the inter-span bin pairs used for the across-feature dv/v (same selection as s23:
converging pairs, CC >= 0.3, <= 70 km, excluded segments dropped), coloured by the features they cross.
Saves <interspan dir>/pairs_used_<band>_<SH>h.csv.   Run with MSDVV_INTERSPAN=interspan_v3."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.signal import hilbert

from msdvv import ROOT, interspan_cfg, load_config
from msdvv.dvv import one_sided
from msdvv.geometry import read_route, repeaters
from msdvv.xcorr import spectra_to_ccf

BAND, SH = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("b1", 168)
cfg = load_config()
xs = interspan_cfg(cfg)
out = Path(cfg["output_dir"])
od = out / xs.get("out_subdir", "interspan")
P = pd.read_csv(od / "pairs.csv")
B = pd.read_csv(od / "bins.csv").set_index("bin")
FEAT = [c for c in P if c.startswith("x_")]
EXCL = set(xs.get("exclude_segs", []))
base_ok = (~P.seg_i.isin(EXCL) & ~P.seg_j.isin(EXCL) & (P.dist_km <= 70)).values

t_all, specs = [], []
for f in sorted(od.glob("*.h5")):
    with h5py.File(f, "r") as h:
        t_all.append(h["t_sub"][:])
        specs.append(h[BAND]["spec"][:])
        bins, fs, nfft, ml = h[BAND]["bins"][:], h.attrs["fs"], h.attrs["nfft"], h.attrs["maxlag_s"]
t = np.concatenate(t_all)
ut, inv = np.unique(t, return_inverse=True)
S = np.zeros((len(ut),) + specs[0].shape[1:], np.complex64)
np.add.at(S, inv, np.concatenate(specs))
del specs
ref, lags = spectra_to_ccf(S.sum(0), bins, nfft, fs, ml)
dt = lags[1] - lags[0]
env = np.abs(hilbert(ref, axis=-1))
tpk = np.argmax(one_sided(env, lags, "pos") + one_sided(env, lags, "neg"), axis=-1) * dt
vpk = P.dist_km.values / np.maximum(tpk, 1e-6)
vg = np.median(vpk[(tpk > 2) & (vpk > 0.3) & (vpk < 5) & base_ok])
t1 = np.maximum(1.3 * P.dist_km.values / vg, 10.0)
t2 = ml - 10.0
sub = xs["substack_s"]
STEP = dict(xs.get("stacks_h", [[24, 6], [168, 24]]))
starts = [i for i in range(0, len(ut) - SH + 1, STEP.get(SH, 24)) if ut[i + SH - 1] - ut[i] == (SH - 1) * sub]
cs = np.concatenate([np.zeros((1,) + S.shape[1:], S.dtype), np.cumsum(S, 0)])
cur, _ = spectra_to_ccf(np.stack([cs[i + SH] - cs[i] for i in starts]), bins, nfft, fs, ml)
la = np.abs(lags)
conv = np.full(len(P), np.nan)
for p in np.where(base_ok & (t1 < t2 - 30))[0]:
    m = (la >= t1[p]) & (la <= t2)
    a = cur[:, p][:, m] - cur[:, p][:, m].mean(1, keepdims=True)
    r = (ref[p][None] - cur[:, p])[:, m]
    r = r - r.mean(1, keepdims=True)
    conv[p] = np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30))
P["conv_cc"] = conv
U = P[conv >= 0.3].copy()
U.to_csv(od / f"pairs_used_{BAND}_{SH}h.csv", index=False)
print(f"{len(U)} pairs used ({BAND}, {SH}-h stacks), group velocity {vg:.2f} km/s")
print(U[FEAT].sum().to_string())


def cls(r):
    f = [c[2:].replace(" fault", "") for c in FEAT if r[c]]
    return " + ".join(f) if f else "control (no feature)"


U["cls"] = U.apply(cls, axis=1)
print(U.cls.value_counts().to_string())

fig, ax = plt.subplots(figsize=(13, 10))
rt = read_route("south")
ax.plot(rt.lon, rt.lat, "-", color="0.75", lw=2, zorder=1)
colors = {"control (no feature)": "0.55", "Alvin Canyon": "tab:red", "Daisy Bank": "tab:blue",
          "Alvin Canyon + Daisy Bank": "tab:purple", "SHR": "goldenrod"}
order = ["control (no feature)"] + [c for c in U.cls.value_counts().index if c != "control (no feature)"]
for k in order:
    g = U[U.cls == k]
    if g.empty:
        continue
    segs = [[(B.lon[a], B.lat[a]), (B.lon[b], B.lat[b])] for a, b in zip(g.bin_i, g.bin_j)]
    col = colors.get(k, plt.cm.tab10(len(colors) % 10))
    colors.setdefault(k, col)
    ax.add_collection(LineCollection(segs, colors=[col], linewidths=0.35 if k.startswith("control") else 0.5,
                                     alpha=0.35 if k.startswith("control") else 0.55, zorder=2, label=f"{k} ({len(g)})"))
used_bins = np.unique(np.r_[U.bin_i, U.bin_j])
SPC = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}
for sp, g in B.loc[used_bins].groupby("span"):
    ax.scatter(g.lon, g.lat, s=18, color=SPC.get(sp, "k"), edgecolor="k", lw=0.3, zorder=4)
for seg, g in B.loc[used_bins].groupby("seg"):
    m = g.iloc[len(g) // 2]
    ax.annotate(f"S-{seg}", (m.lon, m.lat), xytext=(6, 6), textcoords="offset points", fontsize=9, fontweight="bold")
fc = json.load(open(ROOT / xs["features_geojson"]))
for f in fc["features"]:
    c = np.array(f["geometry"]["coordinates"])
    ax.plot(c[:, 0], c[:, 1], color="saddlebrown", lw=2.2, zorder=3)
for name, (la_, lo_, r) in xs["points"].items():
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(lo_ + r / (111.2 * np.cos(np.radians(la_))) * np.cos(th), la_ + r / 111.2 * np.sin(th), "--", color="goldenrod", lw=1.5, zorder=3)
    ax.plot(lo_, la_, "D", color="gold", mec="k", ms=9, zorder=5)
    ax.annotate("S. Hydrate Ridge", (lo_, la_), xytext=(8, 4), textcoords="offset points", fontsize=9)
rp = repeaters("south")
ax.plot(rp.lon, rp.lat, "k^", ms=8, zorder=5)
for nm, (la_, lo_) in {"Pacific City": (45.202, -123.967), "Newport": (44.637, -124.053)}.items():
    ax.plot(lo_, la_, "ks", ms=4)
    ax.annotate(nm, (lo_, la_), xytext=(4, -3), textcoords="offset points", fontsize=8)
ax.text(-125.0, 44.47, "Alvin Canyon fault", color="saddlebrown", fontsize=8, rotation=-12)
ax.text(-125.1, 44.80, "Daisy Bank fault", color="saddlebrown", fontsize=8, rotation=-14)
ax.set(xlim=(-125.8, -123.85), ylim=(44.25, 45.3), xlabel="longitude", ylabel="latitude",
       title=f"South-cable inter-span bin pairs used for across-feature dv/v ({BAND} {xs['bands'][BAND]} Hz, {SH}-h stacks, "
             f"CC ≥ 0.3, ≤ 70 km): {len(U)} pairs, {len(used_bins)} bins")
ax.set_aspect(1 / np.cos(np.radians(44.8)))
leg = ax.legend(fontsize=8, loc="lower left", title="paths crossing (number of pairs)")
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_linewidth(2)
fig.tight_layout()
fn = out / "figs" / f"interspan_pairs_map_{BAND}_{SH}h.png"
fig.savefig(fn, dpi=120)
print(fn)
