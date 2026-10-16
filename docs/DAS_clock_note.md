# Timing of the OOI multispan DAS records (Nokia, Nov 2025 – Jan 2026): dropped chunks and setup latency

*Technical note, 2026-10-15. Prepared from the Multispan_DV analysis (msdvv package) for the OOI / UW fiber lab / Nokia teams.
Evidence figure: `outputs_100m/figs/clock_note_evidence.png`. Per-file table: `outputs_100m/clock_drops_per_file.csv`.
Scripts: `s58_clock_evidence.py`, `msdvv/raw.py` (`chunk_start_times`, `chunk_start_times_counter`, `sample_times`).*

## Summary

1. **Dropped chunks.** The acquisition sometimes drops whole data chunks without leaving a gap in the stored array: 32 samples = 17.18 s on the 100-m stream, and 4.29 s or 8.59 s on other setups. Readers that time samples as `stream.proc_start + i / fs` (as in the example notebook) accumulate errors of up to about 16 minutes by the end of a file. Every drop is recorded in the file trailer, so the true time of every sample can be recovered.
2. **Latency per setup.** The absolute time stamps lag UTC by an amount that depends on the acquisition setup. Against the cabled seismometer OO.HYSB1 (colocated with the south cable), the lag is about 25.2 s for D256 (1.86 Hz) and about 11.8 s for D128 (3.73 Hz), in the 0.12–0.3 Hz band. The two differ by exactly 13.42 s = 2²⁴ / 1.25 MHz. This agrees with the dataset readme ("~7 s at 7.45 Hz, ~12–13 s at 3.73 Hz") once dropped chunks are accounted for.

## 1. Where the information is

Each raw OFDR file stores its data as `nchunk` chunks of shape `[1, nch, m]`. After the trailing `bytes_written` object, the msgpack trailer holds two lists:
- **write info** per chunk: `idx`, `offset`, `write_time`;
- **chunk metadata** per chunk (empty for chunk 0): `stream.proc_start` and `stream.gpu_idx`.

`gpu_idx` is a sample counter. It normally advances by a fixed step per chunk (256 on the 100-m stream). A **dropped chunk** shows as a step of twice that (514–518), and `stream.proc_start` jumps by two chunk lengths (34.4 s). The stored array has no gap there; the two neighbouring chunks are simply concatenated.

## 2. How common the drops are (raw files on disk, Dec 23 – Jan 28; plus the 8-Hz stream samples)

| Stream | Cable | Setup (D) | Files | Files with drops | Drops | Hours | Max drift by end of file |
|---|---|---|---|---|---|---|---|
| 100-m | north | 256 | 57 | 34 | 449 | 666 | 958 s |
| 100-m | north | 128 | 31 | 16 | 44 | 192 | 84 s |
| 100-m | south | 256 | 91 | 53 | 448 | 666 | 872 s |
| 100-m | south | 128 | 51 | 21 | 44 | 192 | 71 s |
| 8-Hz | north / south | 512 (Jan 12 files) | 2 | 2 | 46 / 31 | 10 | 243 / 164 s |
| 8-Hz | both | 8, 16, 64 (Dec 23 – Jan 5) | 58 | 2 | 2 | 28 | 4 s |

**When:** drops are rare before Jan 6, then frequent from about Jan 6 18:00 to Jan 13 and from Jan 21 18:00 to Jan 28 (1–5 drops per hour). That pattern suggests load on the acquisition or processing system. The lower-rate 3.73 Hz period (Jan 13–21) had far fewer drops.

**Example** (south 100-m file, Jan 8 00:34): 8 dropped chunks → +167 s by the end of the 7.5-h file. The error grows as a staircase, one 17.18-s step per drop.

## 3. Consequences if the drops are ignored

- **Absolute time:** comparisons with any other instrument (seismometers, earthquake catalogues, T phases) are wrong by up to minutes. Correlating the DAS against the colocated seismometer HYSB1 collapses from CC 0.8–0.9 to about 0 on every drop-affected file (bottom panel of the figure).
- **DAS–DAS processing:** windows that span a drop splice two records with a missing 17-s piece. Hourly labels and earthquake exclusion windows shift by up to about 15 min. In our dv/v processing this produced a spurious "acquisition-setup step" and a spurious correlation of dv/v with swell period. Both disappeared once the drops were handled.

## 4. Recovering true time

1. **Per-chunk proc_start (used in the analysis):** sample j of chunk k is at `proc_start_k + j / fs`.
2. **Sample counter (recommended):** `t_k = proc_start_0 + (gpu_idx_k − gpu_idx_0) / step · chunk_s`.
   - During the drop periods, `proc_start` also creeps slowly between drops (median 3–10 s/h), while `gpu_idx` shows no missing samples.
   - Against HYSB1, both models give a constant lag through each file (trends ≤ 1 s/h, identical medians), so the creep is below about 1 s at the scale tested. The counter is still the more principled choice.
3. **Gaps:** after timing, place samples on a regular grid of true time and leave dropped chunks as flagged gaps, so that processing can reject damaged windows. We zero-fill and reject windows with more than 10 % missing samples.
4. **Latency:** subtract the setup-dependent latency (section 5) for absolute time.

Code (Python):

```python
from msdvv.raw import read_header, chunk_start_times_counter
h = read_header(path)
t_chunk = chunk_start_times_counter(path, h)                      # true start of every chunk
t = (t_chunk[:, None] + np.arange(h["m"])[None] / h["attrs"]["new_fs"]).ravel()   # every sample
```

## 5. Latency (DAS time stamp − UTC), from the colocated seismometer HYSB1

| Setup | Sampling | 0.12–0.3 Hz | 0.3–0.7 Hz | Readme |
|---|---|---|---|---|
| D256, g256 (100-m stream) | 1.86 Hz | +25.2 s | +38.7 s* | – |
| D128, g256 (Jan 13–21) | 3.73 Hz | +11.8 s | +25.2 s* | ~12–13 s |
| D512 (8-Hz stream, Jan 13) | 7.45 Hz | **+5.80 s** on both cables (D256 value − 19.40 s; see below) | – | ~7 s at 7.45 Hz |

*The higher band converges poorly on the colocated pairs; prefer the 0.12–0.3 Hz values.

**D512 from the simultaneous 100-m stream (s67_stream_latency.py).** On Jan 13 2026 the 8-Hz stream (D512) and the 100-m stream (D256) recorded the same fibers at the same time: north 01:43–08:03, south 01:49–05:33 UTC.
- **Method:** cross-correlate the same fiber positions in the microseism band, 0.1–0.5 Hz, in 20-min windows.
- **Result:** the 8-Hz stamps are 19.408 s (north, 29 positions, median CC 0.91, scatter 5 ms) and 19.397 s (south, 25 positions, CC 0.81, scatter 8 ms) earlier than the 100-m stamps for the same signal.
- **D512 latency:** 25.2 − 19.40 = 5.8 s, identical on both cables to within 10 ms. The absolute value carries the ±~3 s uncertainty of the D256 calibration; the difference between setups is exact.
- **Correction:** an earlier estimate of 13–14 s came from aligning the Jan 13 ML 2.0 on its PNSN origin time and location. That method is circular, PNSN offshore locations are poorly constrained, and the estimate is superseded.

- **Method:** the envelope peak of the stacked colocated HYSB1 × south-cable correlations (bins < 3 km, travel time < 1 s). It includes about ±3 s of uncertainty from the near-field arrival.
- **Setup difference:** D256 − D128 is exactly 13.42 s in both bands, so the latency changes deterministically with the setup, as the readme states.
- **Not yet calibrated:** the November setups (D64, 7.45 Hz, gauges 256 and 128).

## 6. Suggestions for the data providers

- Document the drops and the trailer fields, and update the example notebook to time samples from `gpu_idx` (or per-chunk `proc_start`).
- For the h5 files that were downsampled by hand (Nov 6–26), the per-chunk metadata were not carried over, so drops in those files cannot be recovered. If the original raw files still exist, the chunk metadata could be exported.
- Consider publishing the latency per setup, calibrated against OO.HYSB1 or the cabled broadband stations.

## 7. Related: north-cable polarity (found in the beamforming work)

- **Finding:** when both cables are beamformed together on teleseismic Rayleigh waves (0.04–0.07 Hz, `s69_surface_wave_beam.py`), the north-cable data must be sign-flipped:
  - with the flip, the both-cable beam power rises in 7 of 9 events;
  - back-azimuth errors drop from 40–66° to 3–18°.
- **Likely cause:** a sign convention in the phase-to-strain conversion or in the optical setup of one interrogator. Please confirm.
- **Scope:** this does not affect single-cable processing, but it does affect any product that combines the two cables.
