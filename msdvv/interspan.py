"""Inter-span (cross-feature) correlations: bin-to-bin noise correlations between good channels in
different repeater spans of the same cable, whose straight paths cross a target feature.

Bin-pair stack: for disjoint bins I, J the sum over all channel pairs i in I, j in J of the whitened
cross-spectra X_i X_j* equals S_I S_J* with S_I = sum_i X_i, so one window costs one FFT per
channel plus one product per bin pair.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.fft as sfft
from pyproj import Geod
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from .io import load_cfg
from .xcorr import band_bins, reject_windows, window_starts, window_times

GEOD = Geod(ellps="WGS84")
KM_PER_DEG_LAT = 111.2


def make_bins(q: pd.DataFrame, bin_km: float, min_good: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    """1-D bins along each (block, span) from good channels. Returns bin table and bin id per channel (-1 = unused)."""
    bid = np.full(len(q), -1)
    rows, nb = [], 0
    for (blk, sp), g in q[q.good].groupby(["block", "span"]):
        k = np.floor((g.route_km - g.route_km.min()) / bin_km).astype(int)
        for kk, gg in g.groupby(k):
            if len(gg) < min_good:
                continue
            bid[gg.index] = nb
            rows.append(dict(bin=nb, block=blk, span=sp, seg=f"{blk}{sp}", n_ch=len(gg), route_km=gg.route_km.mean(),
                             lat=gg.lat.mean(), lon=gg.lon.mean(), depth=gg.depth.mean()))
            nb += 1
    return pd.DataFrame(rows), bid


def _local_xy(lat, lon, lat0, lon0):
    return (np.asarray(lon) - lon0) * KM_PER_DEG_LAT * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * KM_PER_DEG_LAT


def _seg_intersect(p1, p2, q1, q2):
    """Do segments p1-p2 and q1-q2 (2-D) intersect? Vectorised over q (arrays (n, 2))."""
    def orient(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    p1, p2 = np.broadcast_to(p1, q1.shape), np.broadcast_to(p2, q1.shape)
    return (orient(p1, p2, q1) != orient(p1, p2, q2)) & (orient(q1, q2, p1) != orient(q1, q2, p2))


def load_features(geojson_path, points: dict):
    """Fault polylines from a GeoJSON (fault_name property) plus named point features {name: (lat, lon, radius_km)}."""
    fc = json.load(open(geojson_path))
    lines = {}
    for f in fc["features"]:
        lines.setdefault(f["properties"]["fault_name"], []).append(np.array(f["geometry"]["coordinates"]))
    return lines, points


def pair_table(bins: pd.DataFrame, lines: dict, points: dict, same_span: bool = False, min_km: float = 5.0) -> pd.DataFrame:
    """All bin pairs on different spans (or the same span if same_span), with path length and feature crossings."""
    lat0, lon0 = bins.lat.mean(), bins.lon.mean()
    bx, by = _local_xy(bins.lat, bins.lon, lat0, lon0)
    segs = {n: [np.c_[_local_xy(L[:, 1], L[:, 0], lat0, lon0)] for L in ls] for n, ls in lines.items()}
    pts = {n: (np.array(_local_xy(la, lo, lat0, lon0)), r) for n, (la, lo, r) in points.items()}
    rows = []
    b = bins.reset_index(drop=True)
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            if not same_span and b.seg[i] == b.seg[j]:
                continue
            _, _, d = GEOD.inv(b.lon[i], b.lat[i], b.lon[j], b.lat[j])
            if d / 1e3 < min_km:
                continue
            p1, p2 = np.array([bx[i], by[i]]), np.array([bx[j], by[j]])
            row = dict(bin_i=int(b.bin[i]), bin_j=int(b.bin[j]), seg_i=b.seg[i], seg_j=b.seg[j], dist_km=d / 1e3,
                       mid_lat=(b.lat[i] + b.lat[j]) / 2, mid_lon=(b.lon[i] + b.lon[j]) / 2)
            for n, polys in segs.items():
                row[f"x_{n}"] = bool(any(np.any(_seg_intersect(p1, p2, P[:-1], P[1:])) for P in polys))
            for n, (c, r) in pts.items():
                # distance from point to the path segment
                v, w = p2 - p1, c - p1
                t = np.clip(np.dot(w, v) / max(np.dot(v, v), 1e-12), 0, 1)
                row[f"d_{n}"] = float(np.linalg.norm(c - (p1 + t * v)))
                row[f"x_{n}"] = row[f"d_{n}"] <= r
            rows.append(row)
    return pd.DataFrame(rows)


def correlate_file_interspan(path, cfg, bid: np.ndarray, pairs: np.ndarray, events=None, workers=4):
    """Hourly substacks of bin-pair cross-spectra for one file.

    bid: bin id per channel (-1 unused); pairs: (npair, 2) bin ids. Returns dict like correlate_file.
    """
    xs = cfg["interspan"]
    f = load_cfg(path, cfg)
    chans = f.channels()
    use = np.where(bid >= 0)[0]
    nbin = int(bid.max()) + 1
    x = f.data[use]
    sos = butter(4, xs["prefilt"], btype="bandpass", fs=f.fs, output="sos")
    x = sosfiltfilt(sos, x, axis=1).astype(np.float32)
    if xs.get("normalisation", "onebit") == "ram":
        # running-absolute-mean temporal normalisation (instead of one-bit), window ram_s
        from scipy.ndimage import uniform_filter1d
        w = max(3, int(round(xs.get("ram_s", 20.0) * f.fs)) | 1)
        x = np.clip(x / (uniform_filter1d(np.abs(x), w, axis=1) + 1e-30), -10, 10).astype(np.float32)
    if xs.get("remove_common_mode", False):
        # signal common to the whole cable (zero-lag stripe in the inter-span gathers): subtract the
        # median over all used channels at each sample (propagating waves are not common at these offsets)
        x -= np.median(x, axis=0, keepdims=True)
    starts, n = window_starts(f.nt, f.fs, xs["win_s"], xs["step_s"])
    sub_ch = chans.iloc[use].reset_index(drop=True).assign(ch=np.arange(len(use)))
    w0, w1, wc, drop = window_times(f, starts, n)
    bad = reject_windows(x, starts, n, sub_ch, f.t0, f.fs, events, cfg["noise"]["reject_mad"], times=(w0, w1)) | drop
    nfft = xs["nfft"]
    taper = tukey(n, 0.05).astype(np.float32)
    bands = {b: band_bins(f.fs, nfft, rng, xs["whiten_taper_hz"]) for b, rng in xs["bands"].items()}
    M = np.zeros((len(use), nbin), np.float32)
    M[np.arange(len(use)), bid[use]] = 1.0
    sub = xs["substack_s"]
    sid = np.floor(wc / sub).astype(np.int64)  # true window-centre time (dropped chunks accounted for)
    usub = np.unique(sid)
    acc = {b: np.zeros((len(usub), len(pairs), len(idx)), np.complex64) for b, (idx, _) in bands.items()}
    nwin = np.zeros(len(usub), np.int32)
    for w, s0 in enumerate(starts):
        if bad[w]:
            continue
        j = np.searchsorted(usub, sid[w])
        seg = x[:, s0 : s0 + n]
        seg = seg - seg.mean(1, keepdims=True)
        seg = (np.sign(seg) if xs.get("normalisation", "onebit") == "onebit" else seg) * taper
        X = sfft.rfft(seg, n=nfft, axis=1, workers=workers)
        for b, (idx, wt) in bands.items():
            Xb = X[:, idx]
            Xb = (Xb / (np.abs(Xb) + 1e-20) * wt).astype(np.complex64)
            S = M.T @ Xb  # (nbin, nb) bin sums
            acc[b][j] += S[pairs[:, 0]] * np.conj(S[pairs[:, 1]])
        nwin[j] += 1
    keep = nwin > 0
    return dict(path=path, cable=f.cable, t_sub=usub[keep] * sub, nwin=nwin[keep], setup=f.setup,
                spec={b: a[keep] for b, a in acc.items()}, bins={b: idx for b, (idx, _) in bands.items()},
                fs=f.fs, nfft=nfft, n_rejected=int(bad.sum()), n_windows=len(starts), n_drop_windows=int(drop.sum()))
