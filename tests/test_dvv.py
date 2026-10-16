"""Synthetic stretching tests: known dv/v recovered by the vectorised code, noisepy stretching
and the independent windowed cross-correlation method (noisepy MWCS/WCC are unstable for these short low-frequency windows) (run: python tests/test_dvv.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from noisepy.monitoring.monitoring_methods import stretching
from scipy.signal import butter, sosfiltfilt

from msdvv.dvv import stretch, wcc

FS = 7.4506
DT = 1 / FS


def synth(dvv, noise, n=600, band=(0.12, 0.3), seed=0):
    """Band-limited random 'coda' ref(t) and cur(t) = ref(t (1 + dvv)) (+ noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    sos = butter(4, band, "bandpass", fs=FS, output="sos")
    tf = np.arange(4 * n) * DT / 4  # fine grid for exact resampling
    base = sosfiltfilt(sos, rng.standard_normal(4 * n), padlen=200)
    base = np.interp(tf, tf, base)
    ref = np.interp(t, tf, base)
    cur = np.interp(t * (1 + dvv), tf, base)
    sc = ref.std()
    ref = ref + noise * sc * sosfiltfilt(sos, rng.standard_normal(n))
    cur = cur + noise * sc * sosfiltfilt(sos, rng.standard_normal(n))
    return t, ref, cur


def test_recover(dvv=2e-3, t1=5.0, t2=60.0, band=(0.12, 0.3), rtol=0.1):
    """rtol: allowed relative bias; short windows (direct-wave case) resolve dv/v only to ~20 %."""
    vals = []
    for seed in range(20):
        t, ref, cur = synth(dvv, 0.05, band=band, seed=seed)
        r = stretch(cur[None], ref[None], DT, t1, t2, 0.01, 101, band)
        para = dict(t=t, twin=[t1, t2], freq=band, dt=DT)
        iw = np.arange(int((t1 - t.min()) / DT) + 1, int((t2 - t.min()) / DT) + 1)
        dv_np, err_np, cc_np, _ = stretching(ref[iw], cur[iw], 0.01, 101, para)
        win = 2.5 / band[0]  # ~2.5 periods of the lowest frequency
        dv_mw = 100 * wcc(cur[None], ref[None], DT, t1, t2, win, win / 4)["dv"][0]
        vals.append((r["dv"][0], dv_np / 100, dv_mw / 100, r["cc"][0], cc_np))
    v = np.array(vals)
    med = np.median(v, axis=0)
    print(f"true {dvv:.2e}: vectorised {med[0]:.3e}  noisepy-stretch {med[1]:.3e}  wcc {med[2]:.3e}  "
          f"cc {med[3]:.3f}/{med[4]:.3f}  max|vec-noisepy| {np.max(np.abs(v[:, 0] - v[:, 1])):.1e}")
    assert abs(med[0] - dvv) < rtol * abs(dvv) + 5e-5
    assert np.max(np.abs(v[:, 0] - v[:, 1])) < 2e-4
    assert abs(med[2] - dvv) < 2.5 * rtol * abs(dvv) + 1e-4


if __name__ == "__main__":
    for d in (2e-3, -1e-3, 5e-4):
        test_recover(d)
    test_recover(1e-3, 2.0, 15.0, (0.3, 0.7), rtol=0.25)  # direct-wave-like window
    print("dvv tests passed")
