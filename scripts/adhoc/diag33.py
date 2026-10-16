import sys; sys.path.insert(0, "/home/seismic/Multispan_DV")
import h5py, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import hilbert
from msdvv.xcorr import spectra_to_ccf
od = Path("/data/wsd02/multispan_dvv/outputs_100m/obsdas")
P = pd.read_csv(od / "paths.csv")
band = "b1"
T, SP, SET = [], [], []
for f in sorted(od.glob("*.h5")):
    with h5py.File(f, "r") as h:
        s = h[band]["spec"][:]
        if s.shape[0] == 0: continue
        T.append(h["t_sub"][:]); SP.append(s.reshape(s.shape[0], -1, s.shape[3])); SET += [h.attrs["setup"]] * s.shape[0]
        bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], h.attrs["nfft"], h.attrs["maxlag_s"]
t = np.concatenate(T); S = np.concatenate(SP); SET = np.array(SET)
print("subs", len(t), "unique", len(np.unique(t)), pd.Series(SET).value_counts().to_dict())
print("time range", pd.to_datetime(t.min(), unit="s"), pd.to_datetime(t.max(), unit="s"))
for su in np.unique(SET):
    ref, lags = spectra_to_ccf(S[SET == su].sum(0), bins, nfft, fs, ml)
    env = np.abs(hilbert(ref, axis=-1))
    noise = np.median(env[:, np.abs(lags) > 150], 1)
    for o in ["HYSB1", "HYS14"]:
        idx = P.index[(P.obs == o)].values
        idx = idx[np.argsort(P.dist_km.values[idx])][:12]
        print(su, o)
        for p in idx:
            k = np.argmax(env[p]); print(f"   d={P.dist_km[p]:6.2f} seg={P.seg[p]} peak lag {lags[k]:+7.2f} s  snr {env[p,k]/noise[p]:5.1f}")
ut, c = np.unique(t, return_counts=True)
d = ut[c > 1]
print("dup hours", len(d))
for x in d[:8]:
    print(pd.to_datetime(x, unit="s"), SET[t == x])
print("mixed-setup dups", sum(len(set(SET[t == x])) > 1 for x in d))
