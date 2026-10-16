import sys; sys.path.insert(0, "/home/seismic/Multispan_DV"); sys.path.insert(0, "/tmp/claude-39217/-home-seismic-Multispan-DV/bdcf973f-3351-4060-bd01-4915969ab0b9/scratchpad")
import os; os.environ["MSDVV_CONFIG"] = "config_100m.yaml"
import numpy as np, pandas as pd
import clock_track as C
from msdvv.raw import read_header, chunk_start_times, chunk_start_times_counter
for when in sys.argv[1:]:
    r = C.inv.iloc[(np.abs(C.inv.t0 - pd.Timestamp(when, tz="UTC").timestamp())).argmin()]
    h = read_header(r.path); m = h["m"]; fs = h["attrs"]["new_fs"]
    for nm, fn in (("proc_start", chunk_start_times), ("counter", chunk_start_times_counter)):
        ct = fn(r.path, h); tt = (ct[:, None] + np.arange(m)[None] / fs).ravel()
        W, tc, lags, f = C.window_ccfs(r.path, lambda f_, tt=tt: tt[: f_.nt])
        sm = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, "same"), 1, W)
        ok = W.sum(1) > 0; pk = lags[np.argmax(sm, 1)][ok]; snr = (sm.max(1) / np.median(sm, 1))[ok]
        g = snr > 1.6
        A = np.vstack([tc[ok][g], np.ones(g.sum())]).T; sl, ic = np.linalg.lstsq(A, pk[g], rcond=None)[0]
        print(f"{when} {nm:10s}: lag median {np.median(pk[g]):+.1f} s, trend {sl:+.2f} s/h, scatter (MAD) {1.4826 * np.median(np.abs(pk[g] - np.median(pk[g]))):.1f} s, n={g.sum()}", flush=True)
