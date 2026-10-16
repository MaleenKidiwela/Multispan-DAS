"""Step 3: convergence of short stacks toward a leave-one-run-out reference.

For each cable x band x spatial level (single pair, 1/2/5 km bins) and stack length T,
CC(stack_T, reference) on the coda window and on the direct-wave window, per (segment, k, bin).
Outputs: outputs/convergence/<cable>_<band>_<level>.npz and summary.csv; figs/convergence.png.
"""
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
from msdvv.stack import (aggregate, bin_table, coda_mask, contiguous_segments, direct_mask, distance_bins, load_aggregated,
                         load_runs, pair_mask, window_cc)

cfg = load_config()
out = Path(cfg["output_dir"])
cd = out / "convergence"
cd.mkdir(exist_ok=True)
CONV = cfg.get("convergence", {})
T_SUBS = CONV.get("T_subs", [1, 2, 3, 6, 12])  # in substacks (pilot: x 10 min)
SUB_MIN = cfg["xcorr"]["substack_s"] / 60
LEVELS = ["pair"] + [f"{b}km" for b in cfg["dvv"]["bin_km"]]


def run_job(cab, band, level):
    geo = pd.read_csv(out / f"channel_quality_{cab}.csv")
    good = geo.good.values
    seg_mode = CONV.get("ref_exclude", "run") == "segment"
    if level == "pair":
        sub_every = CONV.get("pair_subsample", 1) if seg_mode else 1  # bound memory on long records
        src = np.where(good)[0][::sub_every]
        bins_ch = np.full(len(geo), -1)
        bins_ch[src] = np.arange(len(src))
        nbin = None if sub_every == 1 and not seg_mode else len(src)
        btab = geo.iloc[src][["block", "span", "opt_km", "route_km", "depth"]].assign(bin=np.arange(len(src)))
        if nbin is None:
            bins_ch = np.arange(len(geo))
            btab = geo[["ch", "block", "span", "opt_km", "route_km", "depth"]].rename(columns={"ch": "bin"})
    else:
        bins_ch = distance_bins(geo, float(level[:-2]))
        bins_ch[~good] = -1
        nbin = int(bins_ch.max()) + 1
        btab = bin_table(geo.assign(good=geo.good), bins_ch)
    res = {T: dict(coda=[], direct=[], run=[], t=[]) for T in T_SUBS}
    if seg_mode:
        # long continuous records: spectra aggregated per bin while streaming run files; reference =
        # everything except the tested segment (CCFs are linear: ref = C(total) - C(segment)).
        R = load_aggregated(out / "ccf", cab, band, good, bins_ch, nbin)
        r0 = R
        offs = R.ks * R.dx_km
        Ctot, lags = R.ccf(R.spec.sum(0), workers=4)
        has = np.abs(Ctot).sum(-1) > 0
        cm = [coda_mask(lags, o, cfg["dvv"], band) for o in offs]
        dm = [direct_mask(lags, o, cfg["dvv"], band) for o in offs]
        for T in T_SUBS:
            st = contiguous_segments(R.t_sub, T, R.substack_s)
            if len(st) == 0:
                continue
            seg = np.stack([R.spec[s : s + T].sum(0) for s in st])
            cur, _ = R.ccf(seg, workers=4)
            ref = Ctot[None] - cur
            cc_c = np.stack([window_cc(cur[:, kk], ref[:, kk], cm[kk]) for kk in range(len(offs))], 1)
            cc_d = np.stack([window_cc(cur[:, kk], ref[:, kk], dm[kk]) for kk in range(len(offs))], 1)
            cc_c[:, ~has] = np.nan
            cc_d[:, ~has] = np.nan
            res[T]["coda"].append(cc_c.astype(np.float32))
            res[T]["direct"].append(cc_d.astype(np.float32))
            res[T]["run"] += [R.run] * len(st)
            res[T]["t"].append(R.t_sub[st])
            del seg, cur, ref
        rs = []
    else:
        rs = load_runs(out / "ccf", cab, band)
        r0 = rs[0]
        pm = pair_mask(r0.valid, r0.ks, good)
        offs = r0.ks * r0.dx_km
        total = sum(r.spec.sum(0) for r in rs)
    for r in rs:
        ref_spec = aggregate(total - r.spec.sum(0), pm, bins_ch, nbin)
        ref, lags = r0.ccf(ref_spec, workers=2)  # (nk, nbin, nlag)
        cm = [coda_mask(lags, o, cfg["dvv"], band) for o in offs]
        dm = [direct_mask(lags, o, cfg["dvv"], band) for o in offs]
        for T in T_SUBS:
            nseg = len(r.t_sub) // T
            if nseg == 0:
                continue
            seg = r.spec[: nseg * T].reshape((nseg, T) + r.spec.shape[1:]).sum(1)
            cur, _ = r0.ccf(aggregate(seg, pm, bins_ch, nbin), workers=2)  # (nseg, nk, nbin, nlag)
            cc_c = np.stack([window_cc(cur[:, kk], ref[kk][None], cm[kk]) for kk in range(len(offs))], 1)
            cc_d = np.stack([window_cc(cur[:, kk], ref[kk][None], dm[kk]) for kk in range(len(offs))], 1)
            # pairs/bins without data -> NaN
            has = np.abs(ref).sum(-1) > 0  # (nk, nbin)
            cc_c[:, ~has] = np.nan
            cc_d[:, ~has] = np.nan
            res[T]["coda"].append(cc_c.astype(np.float32))
            res[T]["direct"].append(cc_d.astype(np.float32))
            res[T]["run"] += [r.run] * nseg
            res[T]["t"].append(r.t_sub[: nseg * T : T])
    save = {}
    for T in [T for T in T_SUBS if res[T]["coda"]]:
        save[f"coda_{T}"] = np.concatenate(res[T]["coda"])
        save[f"direct_{T}"] = np.concatenate(res[T]["direct"])
        save[f"run_{T}"] = np.array(res[T]["run"])
        save[f"t_{T}"] = np.concatenate(res[T]["t"])
    np.savez_compressed(cd / f"{cab}_{band}_{level}.npz", ks=r0.ks, **save)
    btab.to_csv(cd / f"bins_{cab}_{level}.csv", index=False)
    rows = []
    for T in [T for T in T_SUBS if f"coda_{T}" in save]:
        for win in ("coda", "direct"):
            a = save[f"{win}_{T}"]  # (nseg, nk, nbin)
            per_bin = np.nanmedian(a, axis=(0, 1))
            for (blk, span), g in btab.groupby(["block", "span"]):
                idx = g.bin.values
                v = per_bin[idx]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                rows.append(dict(cable=cab, band=band, level=level, T_min=SUB_MIN * T, window=win, block=blk, span=span,
                                 n_bins=len(v), cc_median=np.median(v), frac_cc_gt_0p8=np.mean(v > 0.8),
                                 frac_cc_gt_0p5=np.mean(v > 0.5)))
    return rows


jobs = [(c, b, l) for c in ("north", "south") for b in cfg["xcorr"]["bands"] for l in LEVELS]
rows = Parallel(n_jobs=min(len(jobs), CONV.get("n_jobs", 16)), verbose=5)(delayed(run_job)(*j) for j in jobs)
summ = pd.DataFrame([r for rr in rows for r in rr])
summ.to_csv(cd / "summary.csv", index=False)

# --- figure: median CC vs stack length, coda window, per block/span, all levels
fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
for i, cab in enumerate(("north", "south")):
    for j, band in enumerate(cfg["xcorr"]["bands"]):
        for jj, win in enumerate(("coda", "direct")):
            ax = axs[i, 2 * j + jj]
            s = summ[(summ.cable == cab) & (summ.band == band) & (summ.window == win)]
            for (blk, span), g in s.groupby(["block", "span"]):
                for level, ls in zip(LEVELS, ["-", "--", "-.", ":"]):
                    gg = g[g.level == level].sort_values("T_min")
                    if len(gg) and gg.n_bins.iloc[0] >= 1:
                        ax.plot(gg.T_min, gg.cc_median, ls, color=f"C{ord(blk) - 65 + 4 * (span - 1)}",
                                label=f"{blk}{span} {level}" if level in ("pair", "5km") else None)
            tk = [SUB_MIN * t for t in T_SUBS]
            ax.set(xscale="log", xticks=tk, xticklabels=[f"{t:g}" for t in tk],
                   title=f"{cab} {band} {win}", xlabel="stack length (min)", ylim=(0, 1))
            ax.axhline(0.8, color="k", lw=0.5)
        axs[i, 0].set_ylabel("median CC with reference")
    axs[i, -1].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(out / "figs" / "convergence.png", dpi=120)
print(summ[(summ.window == "coda") & (summ.T_min.isin(CONV.get("print_T_min", [30, 60])))].pivot_table(
    index=["cable", "block", "span", "band"], columns=["level", "T_min"], values="cc_median").round(2).to_string())
