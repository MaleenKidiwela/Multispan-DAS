"""Sensitive DAS check for a local event: envelope (2-3.6 Hz) of channels near the epicentre, aligned on the
predicted S (and P) arrival from the catalogue hypocentre and stacked; a real arrival shows a peak at a lag equal
to the (unknown, ~5-15 s) DAS clock delay. Compares with the same stack at random times (empirical null)."""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from msdvv.io import load_file

G = Geod(ellps="WGS84")
GEO = {c: pd.read_csv(f"/data/wsd02/multispan_dvv/outputs_100m/geometry_{c}.csv") for c in ("north", "south")}  # opt_km -> lat/lon
EV = {
    "20260113": dict(t="2026-01-13 05:12:47", lat=45.162, lon=-124.390, z=1.0, label="Jan 13 ML 2.0"),
    "20251213": dict(t="2025-12-13 15:40:16", lat=44.623, lon=-124.356, z=24.6, label="Dec 13 ML 1.6"),
}
VP, VS, MAXKM, BAND = 6.0, 3.45, 30.0, (2.0, 3.6)
import os
TRUE_TIMES = os.environ.get("TRUE_TIMES", "1") == "1"  # 0: old behaviour (t0 + i/fs; wrong by up to 164 s on Jan 13)
res = []
fig, axs = plt.subplots(len(EV), 2, figsize=(15, 4.5 * len(EV)), squeeze=False)
for r_, (key, e) in enumerate(EV.items()):
    t_eq = pd.Timestamp(e["t"], tz="UTC").timestamp()
    for c_, cab in enumerate(("south", "north")):
        ax = axs[r_, c_]
        files = sorted(p for p in glob.glob(f"/data/wsd02/multispan_dvv/raw_hf/eq{key}/nokia*") if f"_{cab}_" in p)
        if not files:
            ax.set_title(f"{e['label']} {cab}: no file")
            continue
        parts, t0s = [], []
        for p in files:
            f = load_file(p)
            parts.append(f)
        fs, ch = parts[0].fs, parts[0].channels()
        data = np.concatenate([f.data for f in parts], axis=1)
        if TRUE_TIMES:  # per-chunk true sample times (dropped chunks accounted for; msdvv.raw.sample_times)
            tt = np.concatenate([f.times if f.times is not None else f.t0 + np.arange(f.nt) / fs for f in parts]) - t_eq
            o_ = np.argsort(tt)
            tt, data = tt[o_], data[:, o_]
        else:
            tt = parts[0].t0 + np.arange(data.shape[1]) / fs - t_eq
        g = GEO[cab]
        lat = np.interp(ch.opt_km, g.opt_km, g.lat)
        lon = np.interp(ch.opt_km, g.opt_km, g.lon)
        _, _, d = G.inv(lon, lat, np.full(len(ch), e["lon"]), np.full(len(ch), e["lat"]))
        hyp = np.hypot(d / 1e3, e["z"])
        sel = np.where(hyp <= np.min(hyp) + MAXKM)[0]
        sel = sel[ch.pos_in_block.values[sel] > 3]
        w = (tt > -900) & (tt < 300)
        x = sosfiltfilt(butter(4, BAND, "bandpass", fs=fs, output="sos"), data[sel][:, w], axis=1)
        tw = tt[w]
        env = uniform_filter1d(np.abs(hilbert(x, axis=1)), int(1.0 * fs), axis=1)
        env /= np.median(env, axis=1, keepdims=True)
        lags = np.arange(-60, 90, 1 / fs)

        def stack(ref):  # align each channel on its predicted S time (+ ref offset) and average
            out = np.zeros(len(lags))
            for i, h in enumerate(hyp[sel]):
                out += np.interp(ref + h / VS + lags, tw, env[i])
            return out / len(sel)

        s_ev = stack(0.0)
        null_refs = np.arange(-850, -150, 20.0)
        null = np.array([stack(r0).max() for r0 in null_refs])
        pk = lags[np.argmax(s_ev)]
        z = (s_ev.max() - np.median(null)) / (1.4826 * np.median(np.abs(null - np.median(null))) + 1e-9)
        res.append(dict(event=e["label"], cable=cab, dx_m=round(parts[0].dx, 1), n_ch=len(sel), min_dist_km=round(hyp.min(), 1),
                        peak=round(s_ev.max(), 2), peak_lag_s=round(pk, 1), null_median=round(np.median(null), 2), null_max=round(null.max(), 2), robust_z=round(z, 1)))
        ax.plot(lags, s_ev, "k", lw=1, label="aligned on predicted S (event)")
        for r0 in null_refs[::5]:
            ax.plot(lags, stack(r0), color="0.8", lw=0.5)
        ax.axvspan(5, 15, color="tab:green", alpha=0.15, label="expected DAS clock lag")
        ax.set(xlabel="lag after predicted S arrival (s)", ylabel="mean envelope / median",
               title=f"{e['label']} - {cab} cable, {parts[0].dx:.0f} m channels, {len(sel)} ch within {MAXKM:.0f} km of nearest (min {hyp.min():.1f} km)")
        ax.legend(fontsize=7)
fig.suptitle(f"DAS detection test, {BAND[0]}-{BAND[1]} Hz envelopes (grey: same stack at random earlier times)", fontsize=10)
fig.tight_layout()
fig.savefig("/data/wsd02/multispan_dvv/outputs_100m/figs/das_local_event_detection" + ("" if TRUE_TIMES else "_nominal_times") + ".png", dpi=110)
print(pd.DataFrame(res).to_string(index=False))
