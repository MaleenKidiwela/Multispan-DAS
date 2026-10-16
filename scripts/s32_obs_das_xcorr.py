"""Seismometer-DAS correlations: OO.HYSB1 / OO.HYS14 (Z, or horizontals rotated per bin) against every good 1-km bin
of one cable (OBSDAS_CABLE, OBSDAS_COMP; default south, Z)
(100-m stream, 1.86 Hz), hourly substacks, per DAS file.

The OBS (20 Hz, clean clocks) are low-passed and interpolated onto the DAS sample times (DAS proc_start + i/fs, which
lag true UTC by an unknown, setup-dependent latency); the latency therefore appears as a constant lag shift per
acquisition setup and is calibrated in s33. Same pre-processing as the DAS inter-span correlations (one-bit, whitening).
Output: <output_dir>/obsdas/<file>.h5 with spec (nsub, nobs, nbin, nb) per band, plus bins.csv / paths.csv.
"""
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
import scipy.fft as sfft
from joblib import Parallel, delayed
from obspy import UTCDateTime, read, read_inventory
from pyproj import Geod
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from msdvv import ROOT, load_config
from msdvv.interspan import load_features, make_bins
from msdvv.io import load_cfg
from msdvv.xcorr import band_bins, window_starts

cfg = load_config()
out = Path(cfg["output_dir"])
OBSDIR = Path("/data/wsd02/multispan_dvv/obs/continuous")
OBS = ["HYSB1", "HYS14"]
# OBSDAS_CABLE: south (default) | north.  OBSDAS_COMP: Z (default) | R | T  (horizontals rotated per bin to the
# station->bin azimuth, R, or 90 deg from it, T; nearest of 12 sectors of 15 deg)
CAB = os.environ.get("OBSDAS_CABLE", "south")
COMP = os.environ.get("OBSDAS_COMP", "Z")
TRUE_TIMES = os.environ.get("OBSDAS_TRUE_TIMES", "1") == "1"  # 0: old behaviour (t0 + i / fs)
# OBSDAS_PREP: onebit (default) | ram (running-absolute-mean temporal normalisation, RAM_S window, instead of one-bit;
# applied identically to OBS and DAS before the per-window spectral whitening)
PREP = os.environ.get("OBSDAS_PREP", "onebit")
if os.environ.get("OBSDAS_REGRID", "0") == "1":  # dropped DAS chunks -> zero gaps on the true time grid (as the DAS-DAS v2 runs)
    cfg["regrid_drops"] = True
RAM_S = float(os.environ.get("RAM_S", "20"))
od = out / ("obsdas" + ("" if CAB == "south" else f"_{CAB}") + ("" if COMP == "Z" else f"_{COMP}") + ("" if PREP == "onebit" else f"_{PREP}"))
od.mkdir(exist_ok=True)
NSEC = 12
WIN, STEP, MAXLAG, NFFT, SUB = 600.0, 300.0, 200.0, 2048, 3600
BANDS = {"b1": (0.12, 0.30), "b2": (0.30, 0.70)}
G = Geod(ellps="WGS84")
inv_obs = read_inventory(str(OBSDIR / "OO_response.xml"))
crd = {s.code: (s.latitude, s.longitude) for n in inv_obs for s in n}
q = pd.read_csv(out / f"channel_quality_{CAB}.csv")
bins, bid = make_bins(q, 1.0, 3)
bins.to_csv(od / "bins.csv", index=False)

# path table: every OBS x bin, distance, feature crossings (same feature definitions as the DAS inter-span work)
xs = cfg["interspan"]
lines, points = load_features(ROOT / xs["features_geojson"], {k: tuple(v) for k, v in xs["points"].items()})
from msdvv.interspan import _local_xy, _seg_intersect  # noqa: E402

lat0, lon0 = bins.lat.mean(), bins.lon.mean()
rows = []
for o in OBS:
    ox, oy = _local_xy(crd[o][0], crd[o][1], lat0, lon0)
    for b in bins.itertuples():
        bx, by = _local_xy(b.lat, b.lon, lat0, lon0)
        _, _, d = G.inv(crd[o][1], crd[o][0], b.lon, b.lat)
        p1, p2 = np.array([ox, oy]), np.array([bx, by])
        az = G.inv(crd[o][1], crd[o][0], b.lon, b.lat)[0] % 360
        r = dict(obs=o, bin=b.bin, seg=b.seg, dist_km=d / 1e3, bin_lat=b.lat, bin_lon=b.lon, az=az)
        for n, ls in lines.items():
            r[f"x_{n}"] = bool(any(np.any(_seg_intersect(p1, p2, np.c_[_local_xy(L[:-1, 1], L[:-1, 0], lat0, lon0)],
                                                         np.c_[_local_xy(L[1:, 1], L[1:, 0], lat0, lon0)])) for L in ls))
        for n, (la, lo, rad) in points.items():
            c = np.array(_local_xy(la, lo, lat0, lon0))
            v, w = p2 - p1, c - p1
            tt = np.clip(np.dot(w, v) / max(np.dot(v, v), 1e-12), 0, 1)
            r[f"d_{n}"] = float(np.linalg.norm(c - (p1 + tt * v)))
            r[f"x_{n}"] = r[f"d_{n}"] <= rad or (o == "HYS14" and n == "SHR")  # HYS14 sits on the summit
        rows.append(r)
paths = pd.DataFrame(rows)
paths["cable"] = CAB
paths["comp"] = COMP
rot = (paths.az + (90.0 if COMP == "T" else 0.0)) % 180.0  # component azimuth (polarity irrelevant after one-bit)
paths["sector"] = np.round(rot / (180.0 / NSEC)).astype(int) % NSEC
SEC = paths.sector.values.reshape(len(OBS), -1)  # (nobs, nbin)
paths.to_csv(od / "paths.csv", index=False)
print(f"{len(bins)} bins x {len(OBS)} OBS = {len(paths)} paths; distance range {paths.dist_km.min():.1f}-{paths.dist_km.max():.1f} km", flush=True)
inv = pd.read_csv(out / "inventory.csv")
inv = inv[inv.cable == CAB]
ev = pd.read_csv(out / "events.csv", parse_dates=["win0", "win1"])
EVW = [(a.timestamp(), b.timestamp()) for a, b in zip(ev.win0, ev.win1)]


def obs_on(times, sta, fs_target, comp="Z"):
    """OBS component (20 Hz) low-passed below the DAS Nyquist and interpolated onto `times` (POSIX s). NaN in gaps."""
    t0, t1 = times[0] - 120, times[-1] + 120
    y = np.full(len(times), np.nan)
    days = pd.date_range(pd.Timestamp(t0, unit="s").floor("D"), pd.Timestamp(t1, unit="s").floor("D"), freq="D")
    sos = butter(6, 0.45 * fs_target, "lowpass", fs=20.0, output="sos")
    for d in days:
        for fn in glob.glob(str(OBSDIR / f"{d:%Y%m%d}" / f"OO.{sta}.*.mseed")):
            for tr in read(fn).select(component=comp):
                if tr.stats.npts < 400:
                    continue
                tt = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) / tr.stats.sampling_rate
                m = (times >= tt[0] + 10) & (times <= tt[-1] - 10)
                if m.any():
                    y[m] = np.interp(times[m], tt, sosfiltfilt(sos, tr.data.astype(np.float64)))
    return y


def ram(x, fs):
    """Running-absolute-mean normalisation along the last axis (window RAM_S s), clipped at 10x to tame spikes."""
    from scipy.ndimage import uniform_filter1d
    w = max(3, int(round(RAM_S * fs)) | 1)
    a = uniform_filter1d(np.abs(x), w, axis=-1) + 1e-30
    y = x / a
    return np.clip(y, -10, 10).astype(np.float32)


def one(path):
    name = od / (Path(path).name + ".h5")
    if name.exists():
        return "exists"
    f = load_cfg(path, cfg)
    use = np.where(bid >= 0)[0]
    nbin = int(bid.max()) + 1
    times = f.times if (TRUE_TIMES and f.times is not None) else f.t0 + np.arange(f.nt) / f.fs  # per-chunk proc_start (dropped chunks)
    sos = butter(4, [0.08, 0.85], "bandpass", fs=f.fs, output="sos")
    x = sosfiltfilt(sos, f.data[use], axis=1).astype(np.float32)
    if PREP == "ram":
        x = ram(x, f.fs)
    if COMP == "Z":
        ob = np.vstack([obs_on(times, s, f.fs) for s in OBS])[:, None]  # (nobs, 1, nt)
    else:  # rotated horizontals: component at azimuth th = cos(th) N + sin(th) E, NSEC sectors over 0-180 deg
        th = np.radians(np.arange(NSEC) * 180.0 / NSEC)
        NE = [(obs_on(times, s, f.fs, "N"), obs_on(times, s, f.fs, "E")) for s in OBS]
        ob = np.stack([np.cos(th)[:, None] * n_[None] + np.sin(th)[:, None] * e_[None] for n_, e_ in NE])  # (nobs, nsec, nt)
    okobs = ~np.isnan(ob).any(1)
    ob = np.nan_to_num(ob)
    ob = sosfiltfilt(sos, ob, axis=-1)
    if PREP == "ram":
        ob = ram(ob, f.fs) * okobs[:, None]
    starts, n = window_starts(f.nt, f.fs, WIN, STEP)
    taper = tukey(n, 0.05).astype(np.float32)
    bb = {b: band_bins(f.fs, NFFT, r, 0.03) for b, r in BANDS.items()}
    M = np.zeros((len(use), nbin), np.float32)
    M[np.arange(len(use)), bid[use]] = 1.0
    centres = f.t0 + (starts + n / 2) / f.fs
    sid = np.floor(centres / SUB).astype(np.int64)
    usub = np.unique(sid)
    acc = {b: np.zeros((len(usub), len(OBS), nbin, len(idx)), np.complex64) for b, (idx, _) in bb.items()}
    nwin = np.zeros((len(usub), len(OBS)), np.int32)
    gap_ok = np.ones(len(starts), bool)
    if f.valid is not None:  # regridded: reject windows with > 10 % zero-filled (dropped-chunk) samples
        cv = np.concatenate([[0], np.cumsum(f.valid)])
        gap_ok = (cv[starts + n] - cv[starts]) >= 0.9 * n
    for w, s0 in enumerate(starts):
        if not gap_ok[w]:
            continue
        a, e = times[s0], times[min(s0 + n, f.nt - 1)]
        if any(a < e1 and e > e0 for e0, e1 in EVW):
            continue
        j = np.searchsorted(usub, sid[w])
        seg = x[:, s0 : s0 + n]
        seg = seg - seg.mean(1, keepdims=True)
        seg = (np.sign(seg) if PREP == "onebit" else seg) * taper
        X = sfft.rfft(seg, n=NFFT, axis=1)
        oseg = ob[..., s0 : s0 + n]
        oseg = oseg - oseg.mean(-1, keepdims=True)
        oseg = (np.sign(oseg) if PREP == "onebit" else oseg) * taper
        Xo = sfft.rfft(oseg, n=NFFT, axis=-1)  # (nobs, nsec, nf)
        for b, (idx, wt) in bb.items():
            Xb = X[:, idx]
            Xb = Xb / (np.abs(Xb) + 1e-20) * wt
            Sb = M.T @ Xb  # (nbin, nb)
            Ob = Xo[..., idx]
            Ob = Ob / (np.abs(Ob) + 1e-20) * wt
            for k in range(len(OBS)):
                if okobs[k, s0 : s0 + n].all():
                    Ok = Ob[k, 0][None, :] if COMP == "Z" else Ob[k, SEC[k]]  # (1|nbin, nb)
                    acc[b][j, k] += (Ok * np.conj(Sb)).astype(np.complex64)  # OBS (first) x DAS bin
        for k in range(len(OBS)):
            nwin[j, k] += int(okobs[k, s0 : s0 + n].all())
    keep = nwin.sum(1) > 0
    with h5py.File(name, "w") as h:
        h.attrs.update(dict(cable=CAB, comp=COMP, prep=PREP, fs=f.fs, nfft=NFFT, maxlag_s=MAXLAG, substack_s=SUB, setup=f.setup, obs=",".join(OBS),
                            note="positive lag = energy arriving at the DAS bin after the OBS (before latency correction)"))
        h["t_sub"], h["nwin"] = usub[keep] * SUB, nwin[keep]
        for b, (idx, _) in bb.items():
            g = h.create_group(b)
            g["bins"] = idx
            g.create_dataset("spec", data=acc[b][keep], compression="lzf")
    return "ok"


res = Parallel(n_jobs=12, verbose=5)(delayed(one)(p) for p in inv.path)
print(pd.Series(res).value_counts().to_string())
