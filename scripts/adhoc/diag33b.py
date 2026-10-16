import sys; sys.path.insert(0, "/home/seismic/Multispan_DV")
import h5py, numpy as np, pandas as pd
from pathlib import Path
from msdvv.xcorr import spectra_to_ccf
od = Path("/data/wsd02/multispan_dvv/outputs_100m/obsdas")
P = pd.read_csv(od / "paths.csv")
band = "b1"; LAT = {"D128_g256_native": 25.233 - 13.42, "D256_g256_native": 25.233}
T, SP = [], []
for f in sorted(od.glob("*.h5")):
    with h5py.File(f, "r") as h:
        s = h[band]["spec"][:]
        if s.shape[0] == 0: continue
        bins, fs, nfft, ml = h[band]["bins"][:], h.attrs["fs"], h.attrs["nfft"], h.attrs["maxlag_s"]
        fr = np.fft.rfftfreq(nfft, 1 / fs)[bins]
        T.append(h["t_sub"][:]); SP.append(s.reshape(s.shape[0], -1, s.shape[3]) * np.exp(-2j*np.pi*fr*LAT[h.attrs["setup"]]))
t = np.concatenate(T); ut, inv = np.unique(t, return_inverse=True)
S = np.zeros((len(ut),) + SP[0].shape[1:], np.complex64); np.add.at(S, inv, np.concatenate(SP))
ref, lags = spectra_to_ccf(S.sum(0), bins, nfft, fs, ml)
day = (ut // 86400).astype(int)
for L, lab in ((24, "1 d"), (168, "7 d")):
    st = [i for i in range(0, len(ut) - L, L)]
    C = spectra_to_ccf(np.stack([S[i:i+L].sum(0) for i in st]), bins, nfft, fs, ml)[0]
    for o, dmax in (("HYSB1", 3), ("HYSB1", 10), ("HYSB1", 40), ("HYS14", 25), ("HYS14", 40)):
        idx = P.index[(P.obs == o) & (P.dist_km < dmax)].values
        for wlab, m in (("|lag|<40", np.abs(lags) < 40), ("40-190", (np.abs(lags) > 40) & (np.abs(lags) < 190))):
            a = C[:, idx][..., m]; r = (ref[None, idx] - C[:, idx])[..., m]
            cc = (a*r).sum(-1)/np.sqrt((a**2).sum(-1)*(r**2).sum(-1))
            print(f"{lab} {o}<{dmax}km n={len(idx)} {wlab}: median CC {np.median(cc):.2f}")
