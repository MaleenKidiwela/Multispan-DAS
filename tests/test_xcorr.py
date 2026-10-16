"""Synthetic checks of the correlation engine (run: python tests/test_xcorr.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.fft as sfft

from msdvv.xcorr import band_bins, pair_index, spectra_to_ccf


def test_lag_sign_and_band_roundtrip():
    fs, nfft, n = 7.45, 4096, 1788
    rng = np.random.default_rng(0)
    src = rng.standard_normal(n + 200)
    delay = 25  # samples: channel 1 sees the wave 25 samples after channel 0 (i -> i+1 propagation)
    x0, x1 = src[100 : 100 + n], src[100 - delay : 100 - delay + n]
    X0, X1 = sfft.rfft(x0, nfft), sfft.rfft(x1, nfft)
    idx, w = band_bins(fs, nfft, (0.12, 0.6), 0.03)
    spec = (X0[idx] / np.abs(X0[idx]) * w) * np.conj(X1[idx] / np.abs(X1[idx]) * w)
    cc, lags = spectra_to_ccf(spec[None], idx, nfft, fs, 90)
    peak = lags[np.argmax(cc[0])]
    assert abs(peak - delay / fs) < 0.5 / fs, (peak, delay / fs)


def test_pair_index_blocks():
    ch = pd.DataFrame(dict(ch=np.arange(10), block=list("AAAAABBBBB")))
    v = pair_index(ch, [2])
    assert v[0].tolist() == [True, True, True, False, False, True, True, True, False, False]


if __name__ == "__main__":
    test_lag_sign_and_band_roundtrip()
    test_pair_index_blocks()
    print("xcorr tests passed")
