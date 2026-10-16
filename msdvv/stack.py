"""Reading CCF substacks, pair selection, stacking, convergence metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .xcorr import spectra_to_ccf


@dataclass
class RunSpectra:
    run: str
    cable: str
    band: str
    spec: np.ndarray  # (nsub, nk, nch, nb) complex64, summed over windows
    nwin: np.ndarray  # (nsub,)
    t_sub: np.ndarray  # (nsub,) POSIX s
    file_idx: np.ndarray
    setup: list
    ks: np.ndarray
    valid: np.ndarray  # (nk, nch)
    bins: np.ndarray
    fs: float
    nfft: int
    maxlag_s: float
    dx_km: float = 0.2117155  # pilot files predate the attribute
    substack_s: float = 600.0
    power: np.ndarray | None = None  # (nsub, nch) mean in-band noise power (or (nsub, nbin) log10 when aggregated)

    def ccf(self, spec=None, workers=-1):
        return spectra_to_ccf(self.spec if spec is None else spec, self.bins, self.nfft, self.fs, self.maxlag_s, workers)


def load_run(h5path: Path, band: str) -> RunSpectra:
    with h5py.File(h5path, "r") as h:
        fi = h["file_idx"][:]
        setups = [s.decode() for s in h["setup"][:]]
        t_sub, spec, nwin = h["t_sub"][:], h[band]["spec"][:], h["nwin"][:]
        power = h[band]["power"][:] if "power" in h[band] else None
        # a substack straddling two consecutive files appears once per file: merge
        ut, first, inv = np.unique(t_sub, return_index=True, return_inverse=True)
        if len(ut) < len(t_sub):
            merged = np.zeros((len(ut),) + spec.shape[1:], spec.dtype)
            np.add.at(merged, inv, spec)
            if power is not None:
                pw = np.zeros((len(ut),) + power.shape[1:])
                np.add.at(pw, inv, power * nwin[:, None])
            nwin = np.bincount(inv, weights=nwin).astype(np.int32)
            spec = merged
            if power is not None:
                power = (pw / np.maximum(nwin, 1)[:, None]).astype(np.float32)
            fi = fi[first]
        return RunSpectra(
            run=h.attrs["run"], cable=h.attrs["cable"], band=band,
            spec=spec, nwin=nwin, t_sub=ut, file_idx=fi,
            setup=[setups[i] for i in fi], ks=h["ks"][:], valid=h["valid"][:], bins=h[band]["bins"][:],
            fs=float(h.attrs["fs"]), nfft=int(h.attrs["nfft"]), maxlag_s=float(h.attrs["maxlag_s"]),
            dx_km=float(h.attrs.get("dx_km", 0.2117155)), substack_s=float(h.attrs.get("substack_s", 600.0)),
            power=power,
        )


def load_runs(ccf_dir: Path, cable: str, band: str, runs=None):
    paths = sorted(Path(ccf_dir).glob(f"{cable[0].upper()}*.h5"))
    out = [load_run(p, band) for p in paths if runs is None or p.stem in runs]
    return out


def merge_runs(rs: list, name: str | None = None) -> RunSpectra:
    """Concatenate runs of one cable on the common (clock-aligned) substack grid.

    Substacks shared by two files are summed. Returns one RunSpectra in time order; use
    ``contiguous_segments`` to avoid stacking across real gaps.
    """
    t = np.concatenate([r.t_sub for r in rs])
    ut, inv = np.unique(t, return_inverse=True)
    spec = np.zeros((len(ut),) + rs[0].spec.shape[1:], rs[0].spec.dtype)
    nwin = np.zeros(len(ut), np.int64)
    has_pw = all(r.power is not None for r in rs)
    pw = np.zeros((len(ut),) + rs[0].power.shape[1:]) if has_pw else None
    setup = [""] * len(ut)
    off = 0
    for r in rs:
        idx = inv[off : off + len(r.t_sub)]
        np.add.at(spec, idx, r.spec)
        np.add.at(nwin, idx, r.nwin)
        if has_pw:
            np.add.at(pw, idx, r.power * r.nwin[:, None])
        for i, s in zip(idx, r.setup):
            setup[i] = s
        off += len(r.t_sub)
    r0 = rs[0]
    return RunSpectra(run=name or f"{r0.cable[0].upper()}ALL", cable=r0.cable, band=r0.band, spec=spec, nwin=nwin.astype(np.int32),
                      t_sub=ut, file_idx=np.zeros(len(ut), int), setup=setup, ks=r0.ks, valid=r0.valid, bins=r0.bins,
                      fs=r0.fs, nfft=r0.nfft, maxlag_s=r0.maxlag_s, dx_km=r0.dx_km, substack_s=r0.substack_s,
                      power=(pw / np.maximum(nwin, 1)[:, None]).astype(np.float32) if has_pw else None)


def run_paths(ccf_dir: Path, cable: str):
    return sorted(Path(ccf_dir).glob(f"{cable[0].upper()}[0-9]*.h5"))


def total_spectra(ccf_dir: Path, cable: str, band: str):
    """Sum of all substacks, streaming over run files. Returns (sum (nk, nch, nb), first RunSpectra meta)."""
    tot, r0 = None, None
    for p in run_paths(ccf_dir, cable):
        r = load_run(p, band)
        tot = r.spec.sum(0) if tot is None else tot + r.spec.sum(0)
        if r0 is None:
            r0 = r
        r.spec = None
    return tot, r0


def load_aggregated(ccf_dir: Path, cable: str, band: str, good: np.ndarray, bins_ch: np.ndarray, nbin: int) -> RunSpectra:
    """Merged hourly series of bin-aggregated spectra, streaming one run file at a time.

    spec -> (nsub, nk, nbin, nb); power -> (nsub, nbin) mean log10 in-band power of the bin's good channels.
    """
    parts = []
    for p in run_paths(ccf_dir, cable):
        r = load_run(p, band)
        pm = pair_mask(r.valid, r.ks, good)
        r.spec = aggregate(r.spec, pm, bins_ch, nbin)
        if r.power is not None:
            M = np.zeros((len(good), nbin), np.float32)
            ok = good & (bins_ch >= 0)
            M[np.where(ok)[0], bins_ch[ok]] = 1.0
            r.power = (np.log10(r.power.astype(np.float64) + 1e-40) @ M / np.maximum(M.sum(0), 1)).astype(np.float32)
        parts.append(r)
    return merge_runs(parts)


def contiguous_segments(t_sub: np.ndarray, T: int, sub_s: float, step: int | None = None) -> np.ndarray:
    """Start indices of windows of T consecutive substacks with no missing substack."""
    step = step or T
    gap = np.r_[True, np.diff(t_sub) > 1.5 * sub_s]  # True where a new contiguous piece starts
    piece = np.cumsum(gap)
    starts = []
    for p in np.unique(piece):
        idx = np.where(piece == p)[0]
        starts += list(range(idx[0], idx[-1] - T + 2, step))
    return np.array(starts, int)


def pair_mask(valid: np.ndarray, ks: np.ndarray, good: np.ndarray) -> np.ndarray:
    """valid pairs whose two channels are both 'good' -> (nk, nch) bool."""
    nch = valid.shape[1]
    m = valid.copy()
    for j, k in enumerate(ks):
        m[j, : nch - k] &= good[: nch - k] & good[k:]
        m[j, nch - k :] = False
    return m


def normalize(cc: np.ndarray, axis=-1):
    return cc / (np.sqrt(np.sum(cc**2, axis=axis, keepdims=True)) + 1e-30)


def window_cc(a: np.ndarray, b: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Zero-lag correlation coefficient of a and b over lag samples sel (broadcast over leading dims)."""
    x, y = a[..., sel], b[..., sel]
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    return np.sum(x * y, -1) / (np.sqrt(np.sum(x**2, -1) * np.sum(y**2, -1)) + 1e-30)


def coda_mask(lags: np.ndarray, offset_km: float, cfg_dvv: dict, band: str, side: str = "both"):
    t1 = max(cfg_dvv["min_coda_s"][band], cfg_dvv["coda_start_mult"] * offset_km / cfg_dvv["v_direct"])
    t2 = cfg_dvv["coda_end_s"][band]
    a = np.abs(lags)
    m = (a >= t1) & (a <= t2)
    if side == "pos":
        m &= lags > 0
    elif side == "neg":
        m &= lags < 0
    return m


def lag_windows(offset_km: float, cfg_dvv: dict, band: str) -> dict:
    """One-sided (t1, t2) lag windows [s] for dv/v measurement.

    direct : from the fastest plausible arrival (offset / v_fast) to the slow direct-wave time t_d
    early  : early coda, t_d .. early_coda_mult * t_d   (t_d = max(offset / v_direct, t_direct_min_s))
    coda   : late coda as in the convergence analysis
    """
    td = max(offset_km / cfg_dvv["v_direct"], cfg_dvv["t_direct_min_s"])
    t1c = max(cfg_dvv["min_coda_s"][band], cfg_dvv["coda_start_mult"] * offset_km / cfg_dvv["v_direct"])
    return dict(
        direct=(max(cfg_dvv["direct_min_s"], offset_km / cfg_dvv["v_fast"]), td),
        early=(td, cfg_dvv["early_coda_mult"] * td),
        coda=(t1c, cfg_dvv["coda_end_s"][band]),
    )


def direct_mask(lags: np.ndarray, offset_km: float, cfg_dvv: dict, band: str):
    t1 = max(cfg_dvv["min_coda_s"][band], cfg_dvv["coda_start_mult"] * offset_km / cfg_dvv["v_direct"])
    return np.abs(lags) < t1


def aggregate(spec: np.ndarray, pm: np.ndarray, bins_ch: np.ndarray, nbin: int | None) -> np.ndarray:
    """Sum in-band cross-spectra over the pairs of each distance bin.

    spec (..., nk, nch, nb); pm (nk, nch) pair mask; bins_ch (nch,) bin id per source channel (-1 = none).
    nbin=None -> single-pair level (masked, no aggregation). Returns (..., nk, nbin, nb).
    """
    if nbin is None:
        return spec * pm[..., None]
    out = np.zeros(spec.shape[:-2] + (nbin, spec.shape[-1]), spec.dtype)
    for kk in range(pm.shape[0]):
        M = np.zeros((spec.shape[-2], nbin), np.float32)
        ok = pm[kk] & (bins_ch >= 0)
        M[np.where(ok)[0], bins_ch[ok]] = 1.0
        out[..., kk, :, :] = np.swapaxes(np.swapaxes(spec[..., kk, :, :], -1, -2) @ M, -1, -2)
    return out


def bin_table(geo: pd.DataFrame, bins_ch: np.ndarray) -> pd.DataFrame:
    g = geo.assign(bin=bins_ch)
    g = g[g.bin >= 0]
    return g.groupby("bin").agg(block=("block", "first"), span=("span", "first"), opt_km=("opt_km", "mean"),
                                route_km=("route_km", "mean"), depth=("depth", "mean"), lat=("lat", "mean"),
                                lon=("lon", "mean"), n_good=("good", "sum")).reset_index()


def distance_bins(geo: pd.DataFrame, bin_km: float) -> np.ndarray:
    """Bin id per channel: bins never straddle a mask block or a repeater span."""
    key = geo.block.astype(str) + geo.span.astype(str)
    out = np.full(len(geo), -1)
    nb = 0
    for _, g in geo.groupby(key, sort=False):
        b = np.floor((g.opt_km - g.opt_km.min()) / bin_km).astype(int)
        out[g.index] = b + nb
        nb += b.max() + 1
    return out
