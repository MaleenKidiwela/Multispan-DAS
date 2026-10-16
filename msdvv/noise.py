"""Noise characterisation: PSD, inter-channel coherence vs offset, f-k, window statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.fft as sfft
from scipy.signal import detrend
from scipy.signal.windows import tukey


def remove_common_mode(data: np.ndarray, chans: pd.DataFrame) -> np.ndarray:
    """Subtract the per-block median across channels at each time sample."""
    out = data.copy()
    for _, g in chans.groupby("block"):
        sl = slice(g.ch.min(), g.ch.max() + 1)
        out[sl] -= np.median(data[sl], axis=0, keepdims=True)
    return out


def segment_fft(data: np.ndarray, fs: float, seg_s: float, workers: int = -1):
    """Non-overlapping tapered segments -> rfft. Returns X (nch, nseg, nf) complex64, freqs."""
    n = int(round(seg_s * fs))
    nseg = data.shape[1] // n
    seg = data[:, : nseg * n].reshape(data.shape[0], nseg, n)
    seg = detrend(seg, axis=-1, type="linear")
    seg *= tukey(n, 0.1)[None, None, :].astype(np.float32)
    X = sfft.rfft(seg, axis=-1, workers=workers).astype(np.complex64)
    return X, sfft.rfftfreq(n, 1 / fs)


def psd_from_fft(X: np.ndarray, fs: float, n: int) -> np.ndarray:
    """One-sided PSD averaged over segments, (nch, nf)."""
    w = tukey(n, 0.1)
    p = np.mean(np.abs(X) ** 2, axis=1) * 2 / (fs * np.sum(w**2))
    return p


def coherence_vs_offset(X: np.ndarray, chans: pd.DataFrame, ks) -> np.ndarray:
    """Magnitude-squared coherence between channel i and i+k (same block only).

    Returns C (nch, len(ks), nf); NaN where i+k is outside the block.
    """
    nch, _, nf = X.shape
    C = np.full((nch, len(ks), nf), np.nan, dtype=np.float32)
    pxx = np.mean(np.abs(X) ** 2, axis=1)
    for _, g in chans.groupby("block"):
        a, b = g.ch.min(), g.ch.max() + 1
        for j, k in enumerate(ks):
            if b - a <= k:
                continue
            i = np.arange(a, b - k)
            sxy = np.mean(X[i] * np.conj(X[i + k]), axis=1)
            C[i, j] = np.abs(sxy) ** 2 / (pxx[i] * pxx[i + k] + 1e-40)
    return C


def fk_power(block: np.ndarray, fs: float, dx_m: float, seg_s: float = 120.0, workers: int = -1):
    """Average f-k power of a (nch, nt) block. Returns P (nk, nf) with k fft-shifted, k [1/km], f [Hz]."""
    n = int(round(seg_s * fs))
    nseg = block.shape[1] // n
    nch = block.shape[0]
    wx = tukey(nch, 0.1)[:, None]
    wt = tukey(n, 0.1)[None, :]
    P = 0
    for s in range(nseg):
        x = detrend(block[:, s * n : (s + 1) * n], axis=1) * wx * wt
        F = sfft.fftshift(sfft.fft(sfft.rfft(x, axis=1, workers=workers), axis=0, workers=workers), axes=0)
        P = P + np.abs(F) ** 2
    k = sfft.fftshift(sfft.fftfreq(nch, dx_m / 1e3))
    f = sfft.rfftfreq(n, 1 / fs)
    return P / max(nseg, 1), k, f


def window_rms(data: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    n = int(round(win_s * fs))
    nw = data.shape[1] // n
    x = data[:, : nw * n].reshape(data.shape[0], nw, n)
    x = x - x.mean(-1, keepdims=True)
    return np.sqrt(np.mean(x**2, axis=-1))


def band_mean(arr: np.ndarray, f: np.ndarray, band) -> np.ndarray:
    m = (f >= band[0]) & (f <= band[1])
    return np.nanmean(arr[..., m], axis=-1)
