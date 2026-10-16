import sys; sys.path.insert(0, "/home/seismic/Multispan_DV")
import numpy as np, pandas as pd
from obspy.taup import TauPyModel
from pyproj import Geod
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
from msdvv import load_config
from msdvv.interspan import make_bins
from msdvv.io import load_cfg
cfg = load_config("/home/seismic/Multispan_DV/config_100m_v2.yaml")
OUT = "/data/wsd02/multispan_dvv/outputs_100m"
G = Geod(ellps="WGS84")
T_EV, LA, LO = pd.Timestamp("2026-01-16T03:25:53.034Z").timestamp(), 43.686, -128.057
M = {"iasp91": TauPyModel("iasp91"), "oceanic": TauPyModel("/tmp/claude-39217/-home-seismic-Multispan-DV/bdcf973f-3351-4060-bd01-4915969ab0b9/scratchpad/taup/ocean6.npz")}
inv = pd.read_csv(f"{OUT}/inventory.csv")
rows = []
for cab in ("north", "south"):
    b, bid = make_bins(pd.read_csv(f"{OUT}/channel_quality_{cab}.csv"), 1.0, 3)
    c = inv[(inv.cable == cab) & (inv.t0 <= T_EV - 300) & (inv.t1 >= T_EV + 300)].iloc[0]
    f = load_cfg(c.path, cfg)
    tt = f.times - T_EV  # DAS stamp relative to catalogue origin (latency NOT removed)
    w = (tt > -300) & (tt < 300)
    sos = butter(4, (0.4, 0.9), "bandpass", fs=f.fs, output="sos")
    for _, r in b.iterrows():
        if cab == "south" and r.seg == "C3":
            continue
        x = sosfiltfilt(sos, f.data[np.where(bid == r.bin)[0]][:, w].astype(np.float64), axis=1).mean(0)
        e = uniform_filter1d(np.abs(hilbert(x)) ** 2, 3)
        t = tt[w]
        nz = np.median(e[(t > -250) & (t < -30)])
        m = (t > 0) & (t < 150)
        k = np.argmax(e[m] > 10 * nz) if (e[m] > 10 * nz).any() else -1  # first exceedance of 10x noise energy
        if k < 0:
            continue
        d = G.inv(LO, LA, r.lon, r.lat)[2] / 111195.0
        pr = {n: min(a.time for a in mm.get_travel_times(10, d, ["P", "p", "Pn", "Pg"])) for n, mm in M.items()}
        rows.append(dict(cable=cab, seg=r.seg, dist_deg=d, t_on=t[m][k], **{f"pred_{n}": v for n, v in pr.items()}))
R = pd.DataFrame(rows)
for n in M:
    R[f"lat_{n}"] = R.t_on - R[f"pred_{n}"]
print(len(R), "bins with onset")
print(R.groupby("cable")[["dist_deg", "t_on", "pred_iasp91", "pred_oceanic", "lat_iasp91", "lat_oceanic"]].median().round(2))
print("overall median apparent latency (onset - predicted P):", R.lat_iasp91.median().round(2), "(iasp91)", R.lat_oceanic.median().round(2), "(oceanic); MAD",
      (R.lat_oceanic - R.lat_oceanic.median()).abs().median().round(2))
