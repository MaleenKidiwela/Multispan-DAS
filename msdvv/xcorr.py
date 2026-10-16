"""Short-offset ambient-noise cross-correlation within kept mask blocks.

Pipeline per file: bandpass (prefilt) -> sliding windows -> reject noisy/event windows ->
one-bit -> rFFT -> per band: spectral whitening -> cross-spectra X_i conj(X_{i+k}) for
k = k_min..k_max (both channels in the same mask block) -> sum per substack.

Only the in-band FFT bins are stored (cross-spectra are zero outside the whitened band),
which keeps the substacks small; ``spectra_to_ccf`` converts back to lag time.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.fft as sfft
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from .io import load_cfg
from .noise import remove_common_mode


def band_bins(fs, nfft, band, taper_hz):
    f = sfft.rfftfreq(nfft, 1 / fs)
    lo, hi = band[0] - taper_hz, band[1] + taper_hz
    idx = np.where((f >= lo) & (f <= hi))[0]
    fb = f[idx]
    w = np.ones_like(fb)
    up = fb < band[0]
    w[up] = 0.5 * (1 - np.cos(np.pi * (fb[up] - lo) / taper_hz))
    dn = fb > band[1]
    w[dn] = 0.5 * (1 + np.cos(np.pi * (fb[dn] - band[1]) / taper_hz))
    return idx, w.astype(np.float32)


def pair_index(chans: pd.DataFrame, ks) -> np.ndarray:
    """valid[k_i, ch] = True if ch and ch+k lie in the same block."""
    nch = len(chans)
    valid = np.zeros((len(ks), nch), bool)
    blk = chans.block.values
    for j, k in enumerate(ks):
        valid[j, : nch - k] = blk[: nch - k] == blk[k:]
    return valid


def window_starts(nt, fs, win_s, step_s):
    n, s = int(round(win_s * fs)), int(round(step_s * fs))
    return np.arange(0, nt - n + 1, s), n


MAX_GAP_FRAC = 0.10  # regridded files: windows with > 10 % zero-filled (dropped-chunk) samples are rejected


def window_times(f, starts, n):
    """True start, end and centre time of each window, and whether it spans a dropped chunk.

    Uses the per-sample times of raw files (DasFile.times, from the per-chunk proc_start); the acquisition drops whole
    chunks without leaving a gap in the array, so a window whose true duration exceeds n/fs by > 1 s splices two
    records and is flagged. Regridded files (load_file regrid=True) have the gaps zero-filled at their true place; then
    only windows with more than MAX_GAP_FRAC missing samples are flagged. Files without per-chunk times (h5) fall back
    to t0 + i/fs and are never flagged."""
    tt = f.times if getattr(f, "times", None) is not None else f.t0 + np.arange(f.nt) / f.fs
    w0 = tt[starts]
    w1 = tt[starts + n - 1] + 1.0 / f.fs
    centre = tt[starts + n // 2]
    drop = (w1 - w0) > n / f.fs + 1.0
    if getattr(f, "valid", None) is not None:  # regridded file: dropped chunks are zero gaps; reject only mostly-empty windows
        cv = np.concatenate([[0], np.cumsum(f.valid)])
        drop |= (cv[starts + n] - cv[starts]) < (1 - MAX_GAP_FRAC) * n
    return w0, w1, centre, drop


def reject_windows(x: np.ndarray, starts, n, chans, t0, fs, events: pd.DataFrame | None, mad_k: float, times=None):
    """Window is rejected if it overlaps an event window, or its block-median RMS is an outlier
    (median + mad_k * MAD over the file) in any block. times: optional (w0, w1) true window start/end (window_times)."""
    rms = np.stack([np.sqrt(np.mean(x[:, s : s + n] ** 2, axis=1)) for s in starts], axis=1)  # (nch, nw)
    bad = np.zeros(len(starts), bool)
    for _, g in chans.groupby("block"):
        r = np.median(rms[g.ch.values], axis=0)
        med = np.median(r)
        mad = np.median(np.abs(r - med)) * 1.4826
        bad |= r > med + mad_k * mad
    if events is not None and len(events):
        w0, w1 = times if times is not None else (t0 + starts / fs, t0 + starts / fs + n / fs)
        for e0, e1 in zip(events.win0.astype("int64") / 1e9, events.win1.astype("int64") / 1e9):
            bad |= (w0 < e1) & (w1 > e0)
    return bad


def correlate_file(path, cfg, events=None, workers=4, good=None):
    """Return dict with per-substack cross-spectra for each band and metadata.

    good: optional bool (nch,) channel-quality mask; pairs with a bad channel are left at zero
    (they compress to nothing on disk).
    """
    xc = cfg["xcorr"]
    f = load_cfg(path, cfg)
    chans = f.channels()
    x = f.data
    if xc["remove_common_mode"]:
        x = remove_common_mode(x, chans)
    sos = butter(4, xc["prefilt"], btype="bandpass", fs=f.fs, output="sos")
    x = sosfiltfilt(sos, x, axis=1).astype(np.float32)
    starts, n = window_starts(f.nt, f.fs, xc["win_s"], xc["step_s"])
    w0, w1, wc, drop = window_times(f, starts, n)
    bad = reject_windows(x, starts, n, chans, f.t0, f.fs, events, cfg["noise"]["reject_mad"], times=(w0, w1)) | drop
    if xc.get("normalisation", "onebit") == "ram":
        from scipy.ndimage import uniform_filter1d
        wr = max(3, int(round(xc.get("ram_s", 20.0) * f.fs)) | 1)
        xr = np.clip(x / (uniform_filter1d(np.abs(x), wr, axis=1) + 1e-30), -10, 10).astype(np.float32)
    nfft = xc["nfft"]
    ks = np.arange(xc["k_min"], xc["k_max"] + 1, xc.get("k_step", 1))
    valid = pair_index(chans, ks)
    keep_pair = valid.copy()
    if good is not None:
        for kk, k in enumerate(ks):
            keep_pair[kk, : f.nch - k] &= good[: f.nch - k] & good[k:]
            keep_pair[kk, f.nch - k :] = False
    kp = keep_pair.astype(np.float32)
    taper = tukey(n, 0.05).astype(np.float32)
    bands = {b: band_bins(f.fs, nfft, rng, xc["whiten_taper_hz"]) for b, rng in xc["bands"].items()}
    # substack assignment by window centre, relative to file start, aligned to absolute clock
    sub = xc["substack_s"]
    sid = np.floor(wc / sub).astype(np.int64)  # true window-centre time (dropped chunks accounted for)
    usub = np.unique(sid)
    acc = {b: np.zeros((len(usub), len(ks), f.nch, len(idx)), np.complex64) for b, (idx, _) in bands.items()}
    # mean in-band noise power per channel and substack (before one-bit), for source diagnostics
    powr = {b: np.zeros((len(usub), f.nch), np.float64) for b in bands}
    pnorm = 2.0 / (f.fs * np.sum(taper.astype(np.float64) ** 2))
    nwin = np.zeros(len(usub), np.int32)
    for w, s0 in enumerate(starts):
        if bad[w]:
            continue
        j = np.searchsorted(usub, sid[w])
        seg = x[:, s0 : s0 + n]
        seg = seg - seg.mean(1, keepdims=True)
        Xr = sfft.rfft(seg * taper, n=nfft, axis=1, workers=workers)
        for b, (idx, _) in bands.items():
            powr[b][j] += np.mean(np.abs(Xr[:, idx]) ** 2, axis=1) * pnorm
        if xc.get("normalisation", "onebit") == "onebit":
            seg = np.sign(seg) * taper  # one-bit
        else:  # "ram": running-absolute-mean normalised series (computed below, before windowing)
            seg = xr[:, s0 : s0 + n] - xr[:, s0 : s0 + n].mean(1, keepdims=True)
            seg = seg * taper
        X = sfft.rfft(seg, n=nfft, axis=1, workers=workers)
        for b, (idx, wt) in bands.items():
            Xb = X[:, idx]
            Xb = (Xb / (np.abs(Xb) + 1e-20) * wt).astype(np.complex64)  # whitening within band
            for kk, k in enumerate(ks):
                acc[b][j, kk, : f.nch - k] += Xb[: f.nch - k] * np.conj(Xb[k:]) * kp[kk, : f.nch - k, None]
        nwin[j] += 1
    keep = nwin > 0
    return dict(
        path=path,
        cable=f.cable,
        t_sub=usub[keep] * sub,
        nwin=nwin[keep],
        spec={b: a[keep] for b, a in acc.items()},
        power={b: (p[keep] / nwin[keep, None]).astype(np.float32) for b, p in powr.items()},
        bins={b: idx for b, (idx, _) in bands.items()},
        ks=ks,
        valid=valid,
        fs=f.fs,
        dx_km=f.dx / 1e3,
        nfft=nfft,
        setup=f.setup,
        n_rejected=int(bad.sum()),
        n_drop_windows=int(drop.sum()),
        n_windows=len(starts),
    )


def write_run(h5path: Path, results: list, cfg, run: str):
    """Write the substacks of all files in a run to one h5 file (one group per band)."""
    results = sorted(results, key=lambda r: r["t_sub"][0] if len(r["t_sub"]) else 0)
    results = [r for r in results if len(r["t_sub"])]
    if not results:
        return
    r0 = results[0]
    if cfg["xcorr"].get("merge_substacks", False):
        # short files (Nov 6-26: 17 min) give several partial substacks per hour: sum them (spectra are window sums,
        # power is a per-window mean -> nwin-weighted). Keeps storage at one substack per hour.
        t_all = np.concatenate([r["t_sub"] for r in results])
        ut, inv_ = np.unique(t_all, return_inverse=True)
        nw = np.concatenate([r["nwin"] for r in results]).astype(np.int64)
        merged = dict(r0)
        merged["t_sub"], merged["nwin"] = ut, np.bincount(inv_, weights=nw, minlength=len(ut)).astype(np.int32)
        merged["spec"], merged["power"] = {}, {}
        for b in r0["spec"]:
            S = np.concatenate([r["spec"][b] for r in results])
            out_ = np.zeros((len(ut),) + S.shape[1:], S.dtype)
            np.add.at(out_, inv_, S)
            merged["spec"][b] = out_
            if "power" in r0:
                Pw = np.concatenate([r["power"][b] for r in results]).astype(np.float64) * nw[:, None]
                acc = np.zeros((len(ut), Pw.shape[1]))
                np.add.at(acc, inv_, Pw)
                merged["power"][b] = (acc / merged["nwin"][:, None]).astype(np.float32)
        merged["path"] = ";".join(r["path"] for r in results)
        first_file = np.zeros(len(ut), np.int64)
        for i, r in enumerate(results[::-1]):  # file index of the first file contributing to each hour
            first_file[np.searchsorted(ut, r["t_sub"])] = len(results) - 1 - i
        paths, setups = [r["path"] for r in results], [r["setup"] for r in results]
        results, r0 = [merged], merged
    with h5py.File(h5path, "w") as h:
        h.attrs.update(dict(run=run, cable=r0["cable"], fs=r0["fs"], nfft=r0["nfft"], maxlag_s=cfg["xcorr"]["maxlag_s"],
                            dx_km=r0["dx_km"], substack_s=cfg["xcorr"]["substack_s"]))
        h["ks"] = r0["ks"]
        h["valid"] = r0["valid"]
        h["t_sub"] = np.concatenate([r["t_sub"] for r in results])
        h["nwin"] = np.concatenate([r["nwin"] for r in results])
        if cfg["xcorr"].get("merge_substacks", False):
            h["file_idx"] = first_file
            h["files"] = np.array(paths, dtype="S")
            h["setup"] = np.array(setups, dtype="S")
        else:
            h["file_idx"] = np.concatenate([np.full(len(r["t_sub"]), i) for i, r in enumerate(results)])
            h["files"] = np.array([r["path"] for r in results], dtype="S")
            h["setup"] = np.array([r["setup"] for r in results], dtype="S")
        for b in r0["spec"]:
            g = h.create_group(b)
            g["bins"] = r0["bins"][b]
            g.create_dataset("spec", data=np.concatenate([r["spec"][b] for r in results]), chunks=True, compression="lzf")
            if "power" in r0:
                g["power"] = np.concatenate([r["power"][b] for r in results])


def spectra_to_ccf(spec: np.ndarray, bins: np.ndarray, nfft: int, fs: float, maxlag_s: float, workers=-1):
    """In-band cross-spectra (..., nb) -> CCF (..., nlag) with lags -maxlag..maxlag (s).

    Convention: positive lag = energy travelling from channel i to channel i+k (increasing distance).
    """
    full = np.zeros(spec.shape[:-1] + (nfft // 2 + 1,), np.complex64)
    full[..., bins] = spec
    # X_i conj(X_j) -> irfft gives c(tau) = sum x_i(t+tau) x_j(t); flip so +lag = i -> j
    cc = sfft.irfft(np.conj(full), n=nfft, axis=-1, workers=workers)
    m = int(round(maxlag_s * fs))
    cc = np.concatenate([cc[..., -m:], cc[..., : m + 1]], axis=-1)
    lags = np.arange(-m, m + 1) / fs
    return cc.astype(np.float32), lags
