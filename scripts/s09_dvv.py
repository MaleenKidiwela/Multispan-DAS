"""Step 4: dv/v time series and even/odd null test.

Stage 1 (per cable x band x level): all-January reference CCFs per (offset k, distance bin).
Stage 2 (per cable x band x level x run, parallel): 30-min moving stacks -> stretching per
(segment, k, bin, window, side); WCC second method at the wcc level; even/odd null test.
Combination: inverse-variance mean over k of measurements with cc >= cc_min and not at the
stretch-grid bound -> per (segment, bin, window, side); 'both' = weighted mean of pos & neg.
Outputs: outputs/dvv/dvv.parquet-like csv.gz files, null.csv.gz, refs/*.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.dvv import one_sided, stretch, wcc, weighted_mean
from msdvv.stack import (aggregate, bin_table, contiguous_segments, distance_bins, lag_windows, load_aggregated,
                         load_run, load_runs, pair_mask)

cfg = load_config()
D = cfg["dvv"]
out = Path(cfg["output_dir"])
dd = out / D["out_subdir"]
(dd / "refs").mkdir(parents=True, exist_ok=True)
MERGE = D.get("merge_runs", False)  # long continuous records: one time series per cable
N_CHUNKS = D.get("n_chunks", 6)     # time chunks per (cable, band, level) in merged mode
SIDES = ("pos", "neg")


def setup_bins(geo, level):
    bins_ch = distance_bins(geo, float(level[:-2]))
    bins_ch[~geo.good.values] = -1
    return bins_ch, int(bins_ch.max()) + 1


def make_ref(cab, band, level):
    geo = pd.read_csv(out / f"channel_quality_{cab}.csv")
    if MERGE:  # long records: stream run files, aggregate per bin on the fly
        bins_ch, nbin = setup_bins(geo, level)
        R = load_aggregated(out / "ccf", cab, band, geo.good.values, bins_ch, nbin)
        ref, lags = R.ccf(R.spec.sum(0), workers=4)
        np.savez(dd / "refs" / f"{cab}_{band}_{level}.npz", ref=ref, lags=lags, ks=R.ks, bins_ch=bins_ch)
        bin_table(geo, bins_ch).to_csv(dd / "refs" / f"bins_{cab}_{level}.csv", index=False)
        sub, _ = R.ccf(R.spec, workers=4)  # (nsub, nk, nbin, nlag)
        np.save(dd / "refs" / f"sub_{cab}_{band}_{level}.npy", sub.astype(np.float32))
        np.savez(dd / "refs" / f"submeta_{cab}_{band}_{level}.npz", t_sub=R.t_sub, setup=np.array(R.setup),
                 dx_km=R.dx_km, substack_s=R.substack_s,
                 logpow=R.power if R.power is not None else np.full((len(R.t_sub), nbin), np.nan, np.float32))
        return [f"{R.run}:{i}" for i in range(N_CHUNKS)]
    rs = load_runs(out / "ccf", cab, band)
    r0 = rs[0]
    pm = pair_mask(r0.valid, r0.ks, geo.good.values)
    bins_ch, nbin = setup_bins(geo, level)
    ref, lags = r0.ccf(aggregate(sum(r.spec.sum(0) for r in rs), pm, bins_ch, nbin), workers=4)
    np.savez(dd / "refs" / f"{cab}_{band}_{level}.npz", ref=ref, lags=lags, ks=r0.ks, bins_ch=bins_ch)
    bin_table(geo, bins_ch).to_csv(dd / "refs" / f"bins_{cab}_{level}.csv", index=False)
    return [r.run for r in rs]


def measure(cur, ref, lags, ks, band, do_wcc, dx_km):
    """cur (nseg, nk, nbin, nlag), ref (nk, nbin, nlag) -> long DataFrame of per-k measurements."""
    rng = cfg["xcorr"]["bands"][band]
    dt = lags[1] - lags[0]
    nseg, nk, nbin, _ = cur.shape
    has = np.abs(ref).sum(-1) > 0
    recs = []
    for kk, k in enumerate(ks):
        wins = lag_windows(k * dx_km, D, band)
        bsel = np.where(has[kk])[0]
        if len(bsel) == 0:
            continue
        for side in SIDES:
            c = one_sided(cur[:, kk][:, bsel], lags, side).reshape(nseg * len(bsel), -1)
            r = np.broadcast_to(one_sided(ref[kk][bsel], lags, side), (nseg, len(bsel), c.shape[-1])).reshape(c.shape)
            for wname in D["windows"]:
                t1, t2 = wins[wname]
                res = stretch(c, r, dt, t1, t2, D["dv_range"], D["nbtrial"], rng)
                rec = dict(seg=np.repeat(np.arange(nseg), len(bsel)), bin=np.tile(bsel, nseg), k=k, side=side,
                           window=wname, dv=res["dv"], cc=res["cc"], cc0=res["cc0"], err=res["err"],
                           at_bound=res["at_bound"])
                if do_wcc:
                    wl = 2.5 / rng[0] if wname != "direct" else min(2.5 / rng[0], max(t2 - t1, 2 * dt) * 0.9)
                    rec["dv_wcc"] = wcc(c, r, dt, t1, t2, wl, wl / 4)["dv"]
                recs.append(pd.DataFrame(rec))
    return pd.concat(recs, ignore_index=True)


def combine(m: pd.DataFrame, keys):
    """Weighted mean over k (and over sides when 'side' not in keys)."""
    ok = (m.cc >= D["cc_min"]) & ~m.at_bound
    m = m.assign(dv=m.dv.where(ok), err=m.err.where(ok))
    if "dv_wcc" in m:
        m = m.assign(dv_wcc=m.dv_wcc.where(ok))
    g = m.groupby(keys)
    rows = []
    for key, x in g:
        mu, e, n = weighted_mean(x.dv.values, x.err.values, 0)
        row = dict(zip(keys, key), dv=mu, err=e, n=n, cc=x.cc[x.dv.notna()].mean())
        if "dv_wcc" in x:
            w = np.isfinite(x.dv_wcc) & np.isfinite(x.err)
            row["dv_wcc"] = np.average(x.dv_wcc[w], weights=1 / x.err[w] ** 2) if w.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_job_merged(cab, band, level, chunk):
    """One time chunk of the merged series: moving S-substack stacks and even/odd null blocks."""
    R = np.load(dd / "refs" / f"{cab}_{band}_{level}.npz")
    ref, lags, ks = R["ref"], R["lags"], R["ks"]
    M = np.load(dd / "refs" / f"submeta_{cab}_{band}_{level}.npz")
    t_sub, setup_all, dx_km, sub_s = M["t_sub"], M["setup"], float(M["dx_km"]), float(M["substack_s"])
    sub = np.load(dd / "refs" / f"sub_{cab}_{band}_{level}.npy", mmap_mode="r")
    S, step = D["stack_subs"], D["step_subs"]
    run = f"{cab[0].upper()}ALL"
    st_all = contiguous_segments(t_sub, S, sub_s, step)
    st = np.array_split(st_all, N_CHUNKS)[chunk]
    res = null = None
    if len(st):
        cur = np.stack([np.asarray(sub[s : s + S]).sum(0) for s in st])
        m = measure(cur, ref, lags, ks, band, level == D["wcc_level"], dx_km)
        m["seg"] += np.searchsorted(st_all, st[0])  # globally unique segment ids
        side_res = combine(m, ["seg", "bin", "window", "side"])
        both = combine(m, ["seg", "bin", "window"]).assign(side="both")
        res = pd.concat([side_res, both], ignore_index=True)
        tc = dict(zip(np.searchsorted(st_all, st), t_sub[st] + S * sub_s))  # back-stacked: label = end of window
        su = dict(zip(np.searchsorted(st_all, st), setup_all[st]))
        res["t"] = res.seg.map(tc)
        res["setup"] = res.seg.map(su)
        lp = M["logpow"]  # (nsub, nbin): mean log10 in-band noise power over the stack window
        lps = {(g, b): v for g, s0 in zip(np.searchsorted(st_all, st), st)
               for b, v in enumerate(np.nanmean(lp[s0 : s0 + S], axis=0))}
        res["logpow"] = [lps.get((g, b), np.nan) for g, b in zip(res.seg, res.bin)]
        res = res.assign(cable=cab, band=band, level=level, run=run)
    B = 2 * S
    bk_all = contiguous_segments(t_sub, B, sub_s, B)
    bk = np.array_split(bk_all, N_CHUNKS)[chunk]
    if len(bk):
        blk = np.stack([np.asarray(sub[s : s + B]) for s in bk])
        ev, od = blk[:, 0::2].sum(1), blk[:, 1::2].sum(1)
        me = combine(measure(ev, ref, lags, ks, band, False, dx_km), ["seg", "bin", "window"])
        mo = combine(measure(od, ref, lags, ks, band, False, dx_km), ["seg", "bin", "window"])
        null = me.merge(mo, on=["seg", "bin", "window"], suffixes=("_even", "_odd"))
        null["seg"] += np.searchsorted(bk_all, bk[0])
        null = null.assign(cable=cab, band=band, level=level, run=run)
    return res, null


def run_job(cab, band, level, run):
    if MERGE:
        return run_job_merged(cab, band, level, int(run.split(":")[1]))
    R = np.load(dd / "refs" / f"{cab}_{band}_{level}.npz")
    ref, lags, ks, bins_ch = R["ref"], R["lags"], R["ks"], R["bins_ch"]
    geo = pd.read_csv(out / f"channel_quality_{cab}.csv")
    r = load_run(out / "ccf" / f"{run}.h5", band)
    pm = pair_mask(r.valid, r.ks, geo.good.values)
    nbin = int(bins_ch.max()) + 1
    S, step = D["stack_subs"], D["step_subs"]
    sub, _ = r.ccf(aggregate(r.spec, pm, bins_ch, nbin), workers=1)  # (nsub, nk, nbin, nlag)
    starts = np.arange(0, len(r.t_sub) - S + 1, step)
    cs = np.cumsum(np.concatenate([np.zeros((1,) + sub.shape[1:], sub.dtype), sub]), 0)
    cur = cs[starts + S] - cs[starts]
    do_wcc = level == D["wcc_level"]
    m = measure(cur, ref, lags, ks, band, do_wcc, r.dx_km)
    t_c = r.t_sub[starts] + S * r.substack_s  # back-stacked: label = end of window
    setup = np.array(r.setup)[starts]
    side_res = combine(m, ["seg", "bin", "window", "side"])
    both = combine(m, ["seg", "bin", "window"]).assign(side="both")
    res = pd.concat([side_res, both], ignore_index=True)
    res["t"] = t_c[res.seg.values]
    res["setup"] = setup[res.seg.values]
    res = res.assign(cable=cab, band=band, level=level, run=run)
    # even/odd null test on blocks of 2*S substacks: interleaved halves (each = one stack length) vs the reference
    B = 2 * S
    nbk = len(r.t_sub) // B
    null = None
    if nbk:
        blk = sub[: nbk * B].reshape((nbk, B) + sub.shape[1:])
        ev, od = blk[:, 0::2].sum(1), blk[:, 1::2].sum(1)
        me = combine(measure(ev, ref, lags, ks, band, False, r.dx_km), ["seg", "bin", "window"])
        mo = combine(measure(od, ref, lags, ks, band, False, r.dx_km), ["seg", "bin", "window"])
        null = me.merge(mo, on=["seg", "bin", "window"], suffixes=("_even", "_odd")).assign(
            cable=cab, band=band, level=level, run=run)
    return res, null


if __name__ == "__main__":
    combos = [(c, b, l) for c in ("north", "south") for b in cfg["xcorr"]["bands"] for l in D["levels"]]
    run_lists = Parallel(n_jobs=min(len(combos), D.get("n_jobs_ref", 8)))(delayed(make_ref)(*c) for c in combos)
    jobs = [(c, b, l, run) for (c, b, l), runs in zip(combos, run_lists) for run in runs]
    print(f"{len(jobs)} dv/v jobs")
    res = Parallel(n_jobs=min(len(jobs), D.get("n_jobs", 46)), verbose=5)(delayed(run_job)(*j) for j in jobs)
    dv = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
    null = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    dv["tlabel"] = "end"  # t = end of the (trailing) stack window
    dv.to_csv(dd / "dvv.csv.gz", index=False)
    null.to_csv(dd / "null.csv.gz", index=False)
    print(dv.groupby(["cable", "band", "level", "window", "side"]).agg(
        n_meas=("dv", lambda x: x.notna().sum()), med_err=("err", "median"), std_dv=("dv", "std")).to_string())
