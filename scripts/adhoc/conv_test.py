import sys; sys.path.insert(0, "/home/seismic/Multispan_DV")
import numpy as np, pandas as pd, h5py
from pathlib import Path
from scipy.signal import hilbert
from msdvv.obsdas import OUT, load_hourly, neighbour_stack, moving, moving_windows, early_window, shift_lag, best_shift
QC = pd.read_csv(OUT / "obsdas_combined" / "per_file_shift_T.csv")
CL = pd.read_csv(OUT / "obsdas_combined" / "pairs.csv")
CLUSTERS = [c for c in CL.cluster.dropna().unique() if c]
# hour -> (file ok?, shift)
hour_q = {}
for f in sorted((OUT / "obsdas").glob("*.h5")):
    r = QC[QC.file == f.name[:-3]]
    if not len(r): continue
    r = r.iloc[0]
    ok0, okj = r.cc0 >= 0.6, (r.cc >= 0.6) and abs(r.shift_s) > 10
    with h5py.File(f) as h:
        for x in h["t_sub"][:]: hour_q[x] = ("good", 0.0) if ok0 else (("jump", r.shift_s) if okj else ("bad", 0.0))
rows = []
for comp in ["T", "R", "Z"]:
    t, H, lags, P = load_hourly(comp)
    dt = lags[1] - lags[0]
    q = [hour_q.get(x, ("bad", 0.0)) for x in t]
    keep = np.array([k != "bad" for k, _ in q])
    for i, (k, sh) in enumerate(q):
        if k == "jump": H[i] = shift_lag(H[i], sh, dt)
    for qc_name, use in (("all files", np.ones(len(t), bool)), ("good files", keep)):
        Hq = neighbour_stack(H[use], P, dt, 2)
        tq = t[use]
        tend, i0, i1 = moving_windows(tq)
        C = moving(Hq, i0, i1); ref = Hq.sum(0)
        for g in CLUSTERS:
            idx = np.where(CL.cluster.values == g)[0]
            # ballistic arrival from the reference envelope (|lag| < 2.5 d / 0.5 km/s + 10)
            env = np.abs(hilbert(ref[idx], axis=-1))
            tb = []
            for j, p in enumerate(idx):
                lim = np.abs(lags) < max(15, 2.5 * P.dist_km[p] / 0.5 + 10)
                tb.append(lags[np.where(lim)[0][np.argmax(env[j][lim])]])
            tb = np.array(tb)
            def cc_of(X, R, m):
                a = X[:, m] - X[:, m].mean(1, keepdims=True); r = (R[None] - X)[:, m]; r = r - r.mean(1, keepdims=True)
                return np.nanmedian((a * r).sum(1) / np.sqrt((a**2).sum(1) * (r**2).sum(1) + 1e-30))
            res = {}
            for wname in ("ballistic", "coda"):
                cps, cst = [], None
                # cluster stack: align each pair on the cluster's median-arrival pair via reference xcorr
                pm = idx[np.argsort(np.abs(tb - np.median(tb)))[0]]
                S = np.zeros((C.shape[0], len(lags)), np.float32); Rs = np.zeros(len(lags), np.float32)
                for j, p in enumerate(idx):
                    sgn = np.sign(tb[j]) if abs(tb[j]) > 3 else 1.0
                    if wname == "ballistic":
                        m = (np.abs(lags - tb[j]) <= 12)
                    else:
                        m = (sgn * lags >= abs(tb[j]) + 15) & (sgn * lags <= abs(tb[j]) + 90)
                    cps.append(cc_of(C[:, p], ref[p], m))
                    sh = best_shift(ref[p], ref[pm], dt, 10.0)
                    nrm = np.abs(ref[p]).max() + 1e-30
                    S += shift_lag(C[:, p], sh, dt) / nrm; Rs += shift_lag(ref[p][None], sh, dt)[0] / nrm
                tbm = tb[list(idx).index(pm)]; sgn = np.sign(tbm) if abs(tbm) > 3 else 1.0
                m = (np.abs(lags - tbm) <= 12) if wname == "ballistic" else ((sgn * lags >= abs(tbm) + 15) & (sgn * lags <= abs(tbm) + 90))
                res[wname] = (np.median(cps), cc_of(S, Rs, m))
            t1, t2 = early_window(np.median(P.dist_km[idx]), lags[-1])
            rows.append(dict(comp=comp, qc=qc_name, cluster=g, arrival_s=f"{np.median(tb):+.0f} ({tb.min():+.0f}..{tb.max():+.0f})",
                             early_win=f"{t1:.0f}-{t2:.0f}", ball_pair=res["ballistic"][0], ball_stack=res["ballistic"][1],
                             coda_pair=res["coda"][0], coda_stack=res["coda"][1]))
        print(comp, qc_name, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(OUT / "obsdas_combined" / "convergence_options.csv", index=False)
pd.set_option("display.width", 250); print(D.round(2).to_string(index=False))
