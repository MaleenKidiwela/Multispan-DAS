# Project log: technical notes, step by step

This is the original working README of the project: every step, test and decision, in the order they were made (January pilot, then the continuous 100-m stream, final products, follow-ups, November, beamforming). The showcase README is in the repository root.

## Original pilot (January 2026 8-Hz snapshots, `config.yaml`)

- Input: `/data/whd02/arose_data/multispan_8hz` (+ `filtered_invest/`).

### Run

```bash
./run_all.sh                     # ~10 min on 48 cores; env: anaconda3/envs/noisepy2
python scripts/make_notebooks.py # notebooks/01_noise, 02_convergence, 03_dvv (views over outputs/)
```

All parameters live in `config.yaml`. Outputs go to `outputs/` (about 9 GB, almost all of it CCF substacks in `outputs/ccf/`).

| Step | Script | Output |
|---|---|---|
| Inventory (Jan 2026; dedup; runs) | `s00_inventory` | `inventory.csv` |
| Repeater picks → channel lat/lon/depth | `s01_geometry` | `geometry_*.csv`, `repeater_picks_*.csv` |
| Earthquake exclusion windows (ComCat) | `s02_events` | `events.csv` |
| PSD, coherence vs offset, f–k | `s03_noise`, `s04_noise_figs` | `noise/`, `figs/psd_*`, `coherence_*`, `fk_*` |
| Physical / non-physical channel classification | `s05_quality` | `channel_quality_*.csv` |
| Cross-correlation (in-band cross-spectra, 10-min substacks) | `s06_xcorr` | `ccf/<run>.h5` |
| Reference gathers | `s07_gathers` | `figs/gathers_*` |
| Convergence vs stack length and spatial stacking; coda remedies | `s08_convergence`, `s08b_coda_fixes` | `convergence/` |
| dv/v (stretching + WCC), even/odd null test | `s09_dvv` | `dvv/dvv.csv.gz`, `dvv/null.csv.gz` |
| Span averages, noise floor, tides, run offsets | `s10_analysis` | `dvv/*.csv`, `figs/dvv_*` |

Tests: `tests/test_xcorr.py` checks the lag sign and the band round-trip. `tests/test_dvv.py` checks recovery of a synthetic dv/v. In that test the vectorised stretching matches noisepy to within 1e-5.

## Continuous 100-m stream (`config_100m.yaml`)

The archive at piweb.ooirsn.uw.edu/das25 is mirrored to `/data/wsd02/multispan_dvv/raw/YYYY/MM/DD/<stream>/`. `scripts/d00_download.py` downloads at about 120 MB/s, roughly 3 GB per day for both cables. It writes to `.part` files and checks that each file ends with the expected marker bytes before renaming it.

The raw binaries are read by `msdvv/raw.py`, a port of the official reader. Outputs go to `/data/wsd02/multispan_dvv/outputs_100m`.

```bash
MSDVV_CONFIG=config_100m.yaml ./run_all.sh     # or run any single step with MSDVV_CONFIG set
```

Differences from the pilot setup:
- 105.9 m channels at 1.86 Hz.
- 1-h substacks.
- Offsets k = 2, 4 … 30 (0.2–3.2 km).
- Pairs with a bad channel are zeroed before the compressed write. A week takes about 11 GB of CCFs for both cables.
- Each cable is merged into one hourly series. dv/v uses 6-h stacks moving by 1 h, primarily on the coda window.
- The convergence reference is everything except the tested segment.

**Test week, Jan 6–12 2026 (153 h north, 158 h south):**
- The late coda converges with long stacks. With 5-km bins the median CC is 0.44–0.85 at 6 h and 0.6–0.94 at 24 h. In the pilot it was 0.1–0.5 at 1 h.
- The 6-h, span-averaged coda dv/v floor (even/odd null test) is 0.02–0.13%, with most spans at 0.03–0.09%.
- No tidal signal (2 of 132 tests with |z| > 2).
- Coda dv/v shows slow swings of ±0.2–0.4% over 1–3 days that are coherent between nearby spans. That is well above the floor, and the cause (environmental or noise source) has not been tested yet.

### Full continuous run, Dec 23 2025 to Jan 28 2026

The run covers 858 hours per cable: all 230 files, 110 GB raw, with no missing days. Jan 13–21 was recorded at 3.73 Hz with internal decimation D = 128. It is decimated by 2 to 1.86 Hz and labelled as a separate setup period.

Processing took about 67 minutes with `/data/wsd02/multispan_dvv/run_full.sh`. CCFs take 58 GB in `outputs_100m/ccf`.

- **Pairs:** about 10,800 on the north cable and 9,050 on the south.
- **Convergence:** median coda CC with 5-km bins is 0.44–0.76 at 6 h, 0.59–0.89 at 1 day and 0.70–0.96 at 1 week.
- **6-h span-averaged coda noise floor:** 0.035–0.16%, mostly 0.04–0.09%. This comes from 53–71 even/odd null blocks.
- **Tides:** none detected (1 of 120 tests with |z| > 2). The 1σ error is about 2–3 × 10⁻⁴ per m.
- **Variability:** the standard deviation of dv/v over 5 weeks is 0.11–0.25%, which is 2–4× the noise floor.
  - **Before Jan 13:** about 0.1–0.17%, and largely independent between spans (the first principal component explains 35% of the variance).
  - **From Jan 13:** about 0.13–0.25%, and largely common to both cables (61% of the variance; north–south r = 0.46).
- **Ocean conditions (`s11_environment`, NDBC buoys 46050 and 46089):**
  - DAS in-band noise power follows significant wave height (r = 0.77–0.83). There is no step at the Jan 13 or Jan 21 setup changes, so the drop in noise power from Jan 13 is real: the seas calmed from Hs 3–7 m to about 1–1.5 m.
  - The common-mode dv/v swings occur in this calm period and coincide with long-period swell arrivals (dominant period 15–17 s). Higher noise power goes with lower dv/v (r ≈ −0.2 to −0.3 on several spans).
  - The multi-day dv/v variability is therefore most likely driven by changes in the noise field, not by structural velocity change. No coseismic step from the Jan 16 M6.0 stands out above it.

### Structure: seafloor surface-wave velocity (`s12_velocity_profile`)

These are group velocities picked from the 5-week reference correlations in 2-km bins, using offsets of 0.4–3.2 km. They are output to `dvv/scholte_velocity_profile.csv` and `figs/scholte_velocity_profile.png`.

- **0.3–0.7 Hz (b2):** 0.2–0.7 km/s. **0.12–0.3 Hz (b1):** 0.3–1.5 km/s.
  - Velocity rises with period, as expected for sediments that get stiffer with depth.
  - b1 values above ~1 km/s are poorly constrained: the wavelengths exceed the 3.2 km maximum offset and the picks are only a few samples.
- **The north and south cables run 4–6 km apart on the shelf, and they independently show the same b2 profile:**
  - about 0.5–0.55 km/s at 20–25 km;
  - a slow zone of about 0.18–0.2 km/s at 30–35 km along the route (about 300 m water depth);
  - recovery to about 0.3–0.36 km/s by 40–47 km.
  - A tentative reading is softer sediment there. It should be checked against shelf sediment and bathymetry maps.
- **Velocity increases with water depth,** strongly for b1 (r = 0.68) and weakly for b2 (r = 0.36). Abyssal-plain spans give faster long-period velocities, perhaps from thinner or stiffer sediment over oceanic crust. This is a hypothesis.

### Inter-span (cross-feature) dv/v, south cable (`s20`–`s22`, `msdvv/interspan.py`, config section `interspan`)

Good channels are grouped into 1-km bins per span. Bins in different spans are correlated, so the wave path between them crosses the targets: the Alvin Canyon fault, the Daisy Bank fault (USGS Qfaults traces) and Southern Hydrate Ridge (within 5 km of the summit). The stack over all channel pairs between two bins equals the product of the bins' summed whitened spectra.

Processing used 10-min windows, ±200 s lags, hourly substacks and 24-h dv/v stacks moving by 6 h.

- **Pairs:** 3996 bin pairs, 5–140 km apart.
- **0.12–0.3 Hz:**
  - Group velocity is about 0.83 km/s.
  - Propagating arrivals are clear to about 25 km and weak by about 60 km.
  - The median daily-stack convergence CC is only 0.27.
- **0.3–0.7 Hz:** does not converge (CC 0.04).
- **Artefacts:** a zero-lag stripe common to all distances (observatory or cross-phase-modulation signal, excluded by the coda window), and tonal artefacts on a few bins.
- **Feature dv/v:** each 24-h step is fitted as dv/v = b(t) + Σ x_pf δ_f(t).
  - Null test against 115 random NW-striking pseudo-faults, using the 825 pairs that converge (≤ 70 km).
  - None of the features stands out: δ_f variability is 0.024–0.056% against 0.041% (median) and 0.064% (p95) for the random lines.
  - The formal errors are about 2.8× too small, because pairs share bins.
  - **No feature-specific dv/v is detectable** in these 5 weeks. Variations smaller than about 0.06% per day can't be separated from path-to-path noise.

### Case study: relocating uw62216067 (catalogue "ML 2.2, 62 km W of Depoe Bay", 2026-01-26 09:09:21 UTC)

- **PNSN solution:** 36 P picks only, all at land stations 225–327 km to the NE (St Helens / Rainier networks). Azimuthal gap 333°, horizontal error ±21 km.
- **OOI seafloor stations (OO.HYS11–14, OO.HYSB1; data from EarthScope FDSN):**
  - Impulsive P arrivals and clear S arrivals.
  - S−P is 20.8 s at HYSB1 and 23.3–23.6 s at the summit stations. The catalogue location predicts 5–7 s.
  - P reaches HYSB1 3.4 s before the summit stations, so the source lies to the WSW.
- **Joint grid search** (OBS P+S plus PNSN P, iasp91, depth fixed at 10 km):
  - Epicentre 43.58°N 126.82°W, origin 09:08:56.8 UTC (24.6 s earlier than catalogued), weighted RMS 1.1 s.
  - That is 211 km from the catalogue epicentre and 15–26 km from the active cluster about 190–200 km W of Bandon (M4.1 Nov 5, M3.9 Nov 18, M2.9 Jan 1).
- **DAS (7.45 Hz, 1–3.5 Hz):** a 3–4 min arrival appears on both cables, earliest on the offshore spans. No arrival appears at the times the catalogue location predicts.
- **Files:** `/data/wsd02/multispan_dvv/obs/` and `outputs_100m/uw62216067_*`; figures in `figs/uw62216067_*.png`.
- **Not yet done:** a 1-D oceanic velocity model, a free depth, DAS picks in the inversion, and Axial T-phase timing.

### Stable across-feature dv/v (`s23_interspan_feature_series.py`, `MSDVV_INTERSPAN=interspan_v3`)

**Setup.**
- **Correlations:** the v3 inter-span correlations: no common-mode subtraction (the v2 median subtraction injected a zero-lag term), bands b0 (0.06–0.15 Hz) and b1 (0.12–0.3 Hz), and the tonal S-C3 bins excluded.
- **Pairs:** only converging pairs (coda CC ≥ 0.3) up to 70 km apart.
- **Groups:** for each feature, the pairs whose paths cross it, plus a control group (no feature) and all paths. Each group's dv/v is the inverse-variance mean over its pairs.
- **Noise floor:** an even/odd-hour split of every stack, floor = 1.4826·MAD(even − odd)/2.

**Results, b1:**

| Stack | Floor | Time-series std | Std / floor |
|---|---|---|---|
| 7-day | 0.004–0.011% | ±0.03% | 3–8 |
| 1-day | 0.016–0.044% | — | — |

**Results, b0:** noise-limited.

**Interpretation.** All path groups follow the same regional variation, which is likely noise-source driven. The difference between feature and control paths is small, as the pseudo-fault null test found (`s22`).

**Method for the Nov–Dec extension:** b1, 7-day stacks, cross-span bin pairs ≤ 70 km. Each acquisition setup needs its own reference (or a step check).

### Noise-source correction (`s25_source_correction.py`)

**Method.**
- **Regressors:** within-span dv/v (coda, 5-km bins, 6-h stacks) is regressed on source proxies measured from the correlations themselves:
  - positive/negative lag-energy asymmetry in the coda;
  - the same asymmetry in the direct-wave window;
  - log noise power.
- **Hs kept out:** wave height is the physical forcing, so it is deliberately not a regressor.

**What the correction removes.**
- **Most spans:** the proxies explain only 2–18% of the variance.
- **Shelf spans N-A1 and S-A1 at 0.3–0.7 Hz:** 28–29%, and there they also remove the raw dv/v–Hs anticorrelation (r ≈ −0.35 → ≈ 0).
- **Caveat:** noise power and Hs are collinear (r ≈ 0.8), so a source effect cannot be separated from wave loading this way.

**After the correction.**
- **Jan 8–9 storm:** the storm-time dip remains on the shelf spans in b2 (−0.15 to −0.27%, z −0.8 to −1.9; mean z −1.34 shelf vs −0.55 deep). That is suggestive, not significant.
- **Discriminating tests:**
  - more storms (Nov–Dec);
  - the 1–2 Hz fine-spacing record, which covers the Dec 25–27 and Jan 3–4 storms with shallower sensitivity.

### 24-h back-stacked dv/v (`MSDVV_DVV=dvv_24h`; all stacks are now labelled at window END)

**Stability.** 24-h stacks moving by 1 h, 5-week 100-m run, coda window, 5-km bins:

| | 6-h stacks | 24-h stacks |
|---|---|---|
| Median stretching CC | 0.71 | 0.79 |
| Span-averaged floor | 0.062% | 0.037% (0.022–0.117%) |

**Storms.** Test: dv(stack ending at onset + 24 h) − dv(stack ending at onset), source-corrected, against the same statistic at other times.
- Shelf spans give inconsistent signs between storms:
  - Dec 25: +0.09 to +0.24% at 0.3–0.7 Hz;
  - Jan 3: −0.07 to −0.13%;
  - Jan 8: −0.06 to −0.10%.
- Shelf |z| ≤ 1.1 throughout.
- **No storm signal.** The earlier Jan 8 "shelf dip" came partly from 6-h centre-labelled stacks.

**1–2 Hz run (Dec 23–Jan 5) agrees.** Source proxies explain 56–71% of the variance on shelf and slope spans. Storm responses are small and positive, the opposite sign to the lower band, and consistent with residual source effects.

### Seafloor seismometers (`s30`–`s33`; OO.HYS11–14, OO.HYSB1, vertical)

**OBS–OBS** (`s30`, `s31`; `OBS_WINDOW=coda|early`):
- 10 pairs; 24-h stacks moving 1 h, back-stacked.
- HYS14 chronfix Δt = 0 for this period, and daily clock shifts are ≤ 0.15 s.
- Early window: CC 0.77–0.98, nearly every stack kept, floor 0.07–0.21%. The late coda is more precise where it is kept, but is often rejected.
- Variations stay mostly at the noise level.

**Seismometer–DAS** (`s32`, `s33`; HYSB1/HYS14 × every good 1-km south-cable bin, b1/b2):
- **DAS time-stamp latency** comes from the envelope peak of the colocated HYSB1 bins (< 3 km). The DAS is late relative to UTC:

  | Setup | b1 latency | b2 latency |
  |---|---|---|
  | D256 (100-m stream) | 25.2 s | 20.9 s |
  | D128 (Jan 13–21) | 11.8 s | 7.5 s |

  - The difference between the setups is exactly 13.42 s (= 2^24 / 1.25 MHz) in both bands, which looks like a counter offset.
  - The ~4 s difference between bands is travel time and phase in the colocated peak, so treat the absolute value as ±3 s.
  - Corrected per row before merging; the residual shift between setups is 0.
  - `obsdas/latency_calibration.json`.
- **Convergence** of 24-h stacks against a leave-out reference:
  - b1 early window (|lag| < 40 s): CC 0.80 at < 3 km from HYSB1 and 0.45 within 25 km of HYS14.
  - The late coda reaches only 0.16–0.22 (0.3–0.5 at 7 days), the same behaviour as the OBS pairs.
  - b2 does not converge.
  - Default window is therefore early: [max(0.7 d/vg, 3 s), max(2.5 d/vg, t1 + 30 s)] with vg = 0.80 km/s. `OBSDAS_WINDOW=coda` is kept as an option.
- **Stacks:** windows are [t − 24 h, t), accepted with ≥ 20 h of data.
- **OBS gaps:** whole-day FDSN requests for HYSB1/HYS14 came back truncated on some days. `d04b_fill_obs_gaps.py` re-requests the missing parts in 2-h chunks, raising coverage from ~86% to 91–92%. The remaining gaps are real network outages, simultaneous on all five stations; the DAS has no gaps. Series coverage went from 542 to 717 hourly points.
- **dv/v** (b1, 34 converging paths of 202: 17 from HYSB1 at 0.5–38 km, 17 from HYS14 at 21–30 km; map `obsdas_all_pairs_map.png`, `s35`):

  | Group | Period | Paths per stack | Std | Typical hourly change |
  |---|---|---|---|---|
  | HYS14 | Dec 23–Jan 8 | 17 | 0.17% | 0.02% |
  | HYS14 | Jan 9–28 | ~4 | 0.35% | 0.11% |
  | HYSB1 | Dec 23–Jan 8 | 13 | 0.22% | 0.05% |
  | HYSB1 | Jan 9–28 | ~3 | 0.54% | 0.13% |

  - After Jan 9 (calm seas) too few paths pass CC ≥ 0.4, so the series is unreliable.
  - Whole-period even/odd floor: 0.13% (HYS14) and 0.16% (HYSB1), 4–8× worse than DAS–DAS inter-span.
  - b2 does not converge (2 paths); its latency estimate (38.7 s) is not meaningful.
- **Feature test is impossible with these two stations.** Every HYS14 path counts as crossing SHR and the Alvin Canyon fault, and every HYSB1 path is a control. "Feature" and "station" are therefore the same split.
- **Files:** `outputs_100m/obsdas/` and `figs/obsdas/` (`obsdas_gather_b1.png`, `obsdas_series_24h.png`, `obsdas_paths_map_b1.png`).

## Data conventions (easy to get wrong)
- The data start time is `stream.proc_start`; filename times are unreliable. Absolute times lag real time by about 7 s.
- `masked` gives the **kept** optical intervals, concatenated in the array. Use `new_fs`, not `fs2`.
- Repeaters show up as a one-channel amplitude spike followed by a drop:
  - North: 67.1 / 172.1 / 276.9 km optical (route ≈ 0.963 × optical).
  - South: 98.0 / 206.0 / 258.1 km optical (route ≈ 0.980 × optical − 1.9 km).

## Pilot findings
1. **Usable channels.** About 66% of the north channels and 30% of the south channels are coherent at 0.4 km offset; the rest is non-physical noise.
   - Coherence is highest just past each repeater and decays along the span.
   - The coherence classifier reproduces the readme's good and bad stretches.
2. **Bands.** Coherent energy lies only between 0.12 and 0.7 Hz (microseism and Scholte waves at ~0.3 km/s). Above 0.7 Hz the data are incoherent even 0.2 km apart. Common-mode removal makes no difference.
3. **Convergence.** These figures are for 1–5 km spatial stacks.
   - Direct waves reach a median CC of 0.9 or more within 10–30 min on most spans.
   - Early coda (1–2.5× the slow direct time) reaches 0.65–0.90 within 60 min.
   - **Late coda (≥15 s) does not converge within 2 h** (0.1–0.5). SVD denoising helps a little and stacking across offsets hurts.
4. **dv/v precision (30-min stacks, even/odd null test).**
   - Per 5-km bin: 0.1–0.4%.
   - Averaged over a span: about 0.04–0.08% on the best north spans (b2, early coda).
   - The scatter is unbiased (median difference ≈ 0).
5. **Tides.** None detected. Of 142 regressions, 9 have |z| > 2, against ≈ 6.5 expected by chance. On the north spans the 2σ bound is roughly 3–7 × 10⁻⁴ per m of tide.
6. **Run-to-run offsets.** Offsets of ±0.2–0.4% often differ between frequency bands. That points to changes in the noise source rather than in velocity. There is no step at the Jan 21 acquisition change.
7. **Second method.** Windowed cross-correlation agrees with stretching at r = 0.5–0.7 on coda windows, but only r = 0.2–0.4 on the short early-coda window.

## Recommendations for the full archive (piweb.ooirsn.uw.edu/das25)
- Use stretches with constant acquisition settings. 211.7 m spacing is available from Jan 6–28; the sampling rate changes on Jan 13 and Jan 21.
- Use the first ~20–40 km past each repeater, the 0.12–0.7 Hz band, offsets of 0.4–3 km, and 2–5 km spatial stacks.
- Day-long stacks should make the late coda usable. Use the coda for dv/v, and treat direct and early-coda results as source-sensitive.
- If the noise is white, about 3 months of continuous data improves tidal sensitivity by ~10×, to ~3–7 × 10⁻⁵ per m. That approaches plausible tidal loading signals. This is a rough extrapolation.

### Denoising the seismometer–DAS correlations (`s36_obsdas_denoise_test.py`; `s32` now takes `OBSDAS_COMP=Z|R|T`)

**Setup.** b1, early window, 24-h back-stacked stacks moving 1 h. Each method is applied identically to the full, even-hour and odd-hour stacks.

**Methods.**
- **Selective stacking:** keep hours whose CCF correlates with the reference above the pair's median.
- **SVD + Wiener:** rank 6 plus a 5×5 Wiener filter on each pair's 24-h series.
- **Neighbour stacking:** each bin plus ±2 bins on the same segment, statically aligned on the reference.
- **R/T:** horizontals rotated per bin to the station→bin azimuth (radial) or 90° from it (transverse), in 12 sectors.

**Bias check.** A known stretch (0.2%, 0.5%) is injected into every hourly CCF after 2026-01-01. The recovered step is measured relative to the uninjected run, over the 5 days before vs days 1–6 after.

**Converging pairs by component and method** (of 202):

| | base | nbr | select | svd |
|---|---|---|---|---|
| Z | 34 | 55 | 202 | 136 |
| R | 39 | 60 | – | – |
| T | 47 | **71** (28 at 30–60 km) | – | – |

**Injection recovery (Z):**
- base: 41–86%;
- neighbour stacking: 80–90%;
- selective stacking: 36–47%;
- SVD: 39–54%.

Selective stacking and SVD appear to cut the floor to 0.03%, but they roughly halve real dv/v changes, so they are **not used**. Neighbour stacking is unbiased within the test scatter.

**Noise floor.** Whole-period even/odd floors are 0.10–0.13% for T and R, against 0.13–0.16% for Z.

**Recommendation.** Transverse (+R, +Z) with neighbour stacking, combining components per pair.

**Production run: combined components** (`s37_obsdas_combined.py`, helpers in `msdvv/obsdas.py`; outputs `outputs_100m/obsdas_combined/`, figures `obsdas_combined_*.png`).
- **Pairs:** Z+R+T, neighbour-stacked (±2 bins), combined per pair by 1/err². 80 of 202 pairs converge in at least one component.

| Group | Pairs | Floor | Std, Dec 23–Jan 8 | Pairs per stack, Jan 9–28 |
|---|---|---|---|---|
| All pairs | 80 | **0.069%** | 0.07% | 25 |
| HYS14 × ACF | 30 | 0.097% | – | 13 |
| HYSB1 control | 28 | 0.117% | – | 9 |
| HYS14 × Daisy Bank (not ACF) | 10 | 0.090% | – | 3 |

- **Injection recovery:** 78–90% for every group (0.2% and 0.5% at Jan 1).
- **Feature contrast** (HYS14 × ACF minus HYS14 × Daisy Bank, same station):
  - Dec 23–Jan 8: std 0.13%, equal to the combined floor (0.13%), and only 4% of stacks exceed 2× floor. **No fault-specific signal.**
  - Jan 9–28: the Daisy Bank group has about 3 pairs per stack, so the contrast is not reliable there.
- **Behaviour over time:** stable to ±0.1% in Dec 23–Jan 8. The largest excursions, about ±0.5%, follow the Jan 7–10 storm (Hs 6 m) on every group, which points to a noise-source effect rather than a medium change.

**Split series** (`obsdas_split_series_24h.png`, `obsdas_split_pairs_map.png`). The clusters are separated at the gaps between used bins along the cable.

| Cluster | Pairs | Distance | Floor | Injection recovery |
|---|---|---|---|---|
| Red, HYS14 × ACF → SB3 | 16 | 20–24 km | 0.114% | 88% |
| Red, HYS14 × ACF → SC4 west (lon < −124.885) | 9 | 28–29 km | 0.117% | 87% |
| Red, HYS14 × ACF → SC4 east | 5 | 30–31 km | 0.088% | 99% |
| Blue, HYSB1 control → SB3 | 16 | 1–9 km | 0.164% | **48%** |
| Blue, HYSB1 control → SC4 | 12 | 39–46 km | 0.107% | 90% |
| Orange, HYS14 × Daisy Bank | 10 | 44–56 km | 0.090% | 83% |

- **Blue SB3 is unreliable.** It is near-colocated, so the early window starts at lag ≈ 0 where stretching has little leverage, and the ±3 s latency uncertainty adds to that. Its amplitude is unreliable, about 2× low.
- **Opposite signs on the same station.** In the calm period (Jan 13–19) the red SB3 and red SC4-west series move in opposite directions (+0.5 vs −0.4%). Both are HYS14 paths at different azimuths, which fits a change in noise directivity rather than a medium change.

### DAS timing problems found with the colocated seismometer (`s38_obsdas_cluster_ccfs.py`; per-file check `obsdas_combined/per_file_shift_T.csv`)

The cluster correlation functions (`obsdas_cluster_ccfs_ref.png`, `obsdas_cluster_ccfs_24h.png`) look stable on Dec 23–Jan 6 and Jan 13–21, and smeared on Jan 7–13 and Jan 22–28. The cause is timing on the DAS side.

**Per-file check** (HYSB1 T component × SB3 bins < 4 km, CC against the Dec 26–Jan 6 reference, ±150 s):
- **Dec 23–Jan 6:** CC 0.8–0.9 at zero shift. Six files are shifted as a whole by +17.2 s or +34.9–36 s and still correlate at CC 0.7–0.86, so they can be corrected per file.
- **Jan 6 18:20–Jan 13 17:00 and Jan 21 18:00–Jan 28:** CC 0.1–0.2 at every lag. A direct ±30-min OBS–DAS correlation of whole files shows no peak either.
  - 20-min windows show a **progressive drift of about −30 s/h within a file** (Jan 8 file: +25 s → −118 s over 5 h). Samples are lost without being marked.
  - These periods coincide with 65–385 s gaps between files.
- **Jan 13–21 (D128):** mostly good, but about a third of the files are shifted or drifting.

**Consequences:**
- DAS–DAS products (within-span, cross-span) share one clock and are not affected.
- Seismometer–DAS dv/v outside the good files is invalid. The "calm-period noise" after Jan 9 and the Jan 13–19 opposite-sign excursions are timing artefacts, not noise-source effects.
- Absolute-time DAS work (earthquake arrivals) needs the same per-file check.

**Fix:** keep only files that pass the colocated check (CC at zero shift ≥ 0.6, after correcting whole-file jumps). Optionally, re-time the drifting files by tracking the colocated lag in 20-min windows.

### Coda-only seismometer–DAS dv/v (`s40_obsdas_coda.py`; timing QC and loaders in `msdvv/obsdas.py`; outputs `outputs_100m/obsdas_coda/`)

**Data.** Only files that pass the per-file DAS timing check (`file_timing_qc`): 59 good, 5 jump-corrected, 76 dropped, leaving 392 h. In practice that is Dec 23–Jan 6, with only a few 24-h windows in Jan 13–21.

**Windows.**
- **Ballistic pick** per pair, component and lag side: the envelope peak within [d/2.0, d/0.35 + 10] s (±10 s for d < 3 km), on the good-timing reference.
- Picks with SNR < 4 are filled from the cluster's good picks plus a moveout term at 0.6 km/s.
- **Coda window** = pick +15 s to +90 s on each side (+15 to +60 s gives higher convergence but a higher floor).

**Stacking.** Cluster CCF = each side of each pair aligned on its ballistic pick, normalised, summed. Stretching is applied to both sides and to Z/R/T, combined by 1/err². 24-h back-stacked, 1-h step.

**Results:**

| Cluster | Floor | Std | Injection recovery |
|---|---|---|---|
| Red SB3 | 0.15% | 0.13% | 82% |
| Red SC4 west | 0.13% | 0.14% | 77% |
| Red SC4 east | 0.08% | 0.13% | 102% |
| Blue SB3 | 0.10% | 0.18% | 92% |
| Blue SC4 | 0.10% | 0.13% | 85% |
| Orange | 0.12% | 0.15% | 86% |

- Coda convergence CC is 0.3–0.74, depending on side and component.
- **Pick quality:** ballistic arrivals are clear on the SB3 clusters (10–16 of 16 pairs picked per side and component). On SC4 and orange few pairs give clean picks (0–4); those CCFs ring without a distinct arrival, so their coda windows may still contain slow (~0.35–0.4 km/s) ballistic energy.

### DAS clock fix: per-chunk time stamps (`msdvv.raw.sample_times`, `DasFile.times`)

**The problem.** The raw OFDR trailer stores, for every 32-sample chunk (17.18 s at 1.86 Hz), its own `stream.proc_start` and `stream.gpu_idx`.
- The acquisition sometimes drops a whole chunk: gpu_idx steps by 514 instead of 256, and proc_start by 34.4 s instead of 17.2 s. Nothing marks the drop in the data array.
- Timing every sample as `t0 + i/fs` therefore accumulates +17.2 s per drop. Examples: 8 drops (+167 s) in the Jan 8 file, 5 in Jan 24, 1 in Dec 25 (the "whole-file 17.2-s jumps").
- This is what broke the seismometer–DAS correlations on Jan 7–13 and Jan 22–28.

**The fix.** `load_file` now attaches the true per-sample times from the per-chunk proc_start, and `s32` interpolates the OBS onto them.
- After the fix, 137 of 140 files pass the colocated check (the 3 failures are 1–2-h fragments), so 910 of 914 h are usable.
- The old correlations are kept in `outputs_100m/obsdas_nominal_times/`.

**Coda-only seismometer–DAS dv/v, clock-fixed** (`s40`; Dec 23–Jan 28, 735 24-h stacks per cluster):

| Cluster | Floor | Std | Injection recovery |
|---|---|---|---|
| Red SB3 | 0.105% | 0.149% | 98% |
| Red SC4 west | 0.114% | 0.147% | 86% |
| Red SC4 east | 0.141% | 0.185% | 98% |
| Blue SB3 | 0.158% | 0.222% | 81% |
| Blue SC4 | 0.130% | 0.130% | 98% |
| Orange | 0.129% | 0.188% | 79% |

- Correlation with wave height is |r| ≤ 0.18 for every cluster: the coda measurement no longer tracks the noise sources the way the ballistic one did.
- The remaining gaps (Jan 10–11, Jan 18) are seismometer outages.

**DAS–DAS products** (within-span, cross-span) are unaffected apart from substack labels off by at most a few minutes. Earthquake timing on the DAS should use `f.times`.

### Raising the seismometer–DAS CC: tests (`s40` variants, compared by `s41_obsdas_variants.py`)

**How the tests were run.**
- Each variant was run twice: on the data, and with a 0.5% stretch injected in every hourly CCF from 2026-01-01.
- **Stretching CC:** median over Z/R/T × both lag sides of each 24-h stack's stretching CC.
- **Kept:** share of stacks with stretching CC ≥ 0.4.
- **Recovery:** injected-minus-clean step, 5 days before vs days 1–6 after.
- All values are medians over the 6 clusters.
- Outputs: `outputs_100m/obsdas_coda/variants/<name>/`, with the summary in `variants/variants_comparison.{csv,md}` and the figure `figs/obsdas/obsdas_variants.png`.
- Rerun with `/data/wsd02/multispan_dvv/run_obsdas_variants.sh` and `run_obsdas_variants2.sh`.

| Variant | Setting | Stretching CC | Convergence | Kept | Floor | Std | Recovery |
|---|---|---|---|---|---|---|---|
| ref | one-bit, 0.12–0.30 Hz, 24 h, coda pick+15..+90 s | 0.60 | 0.51 | 85% | 0.130% | 0.167% | 92% |
| **ram** | running-absolute-mean (20 s) instead of one-bit, on OBS and DAS (`s32 OBSDAS_PREP=ram`) | **0.66** | **0.59** | **94%** | **0.108%** | 0.154% | 92% |
| band15_25 | analysis band 0.15–0.25 Hz | 0.67 | 0.61 | 90% | 0.126% | 0.160% | 91% |
| combo | ram + 0.15–0.25 Hz | 0.67 | 0.60 | 95% | 0.121% | 0.161% | 93% |
| coda60 | coda pick+15..+60 s | 0.66 | 0.58 | 90% | 0.139% | 0.202% | 96% |
| reject3 / reject2 | drop hours with CCF norm > median + 3 / 2 MAD | 0.60 / 0.60 | 0.51 | 85% | 0.130 / 0.133% | – | 92% |
| stack48 | 48-h stacks (check only) | 0.71 | 0.63 | 98% | 0.093% | 0.140% | 93% |

**Conclusions:**
- **Running-absolute-mean (ram) is adopted** as the default in `s40` and the production run. It lowers the floor in 5 of 6 clusters and raises CC by about 0.06 without bias (recovery 83–109% per cluster).
- **The narrower band and the shorter coda raise CC, but not the floor:** less bandwidth or fewer cycles cost precision. The combo is not better than ram alone.
- **Rejecting hours by amplitude does nothing.** Earthquake windows are already removed upstream, and whitened one-bit hourly CCFs have no amplitude outliers: k = 2 rejects 1.6% of hours, k = 3 none.
- **48-h stacks** would reach CC 0.71 and a 0.09% floor, if a 2-day standard were ever acceptable.
- The cluster CCF figures (`s38`) still show the one-bit correlations.

### Cross-span (DAS–DAS) correlation functions (`s42_interspan_ccfs.py`; `MSDVV_INTERSPAN=interspan_v3`)

> 2026-10-16: the one-bit `interspan_v3` correlation files were deleted as superseded (see the cleanup log in docs/NEXT_STEPS.md). The figures below remain. The report's cross-span CCF figure comes from the final RAM product `interspan_v4_ram`.

**Figures:**
- `figs/interspan_v3_gathers_{b0,b1}.png`: reference CCFs of the converging pairs per group, in 2-km distance bins.
- `figs/interspan_v3_ccf24h_{b0,b1}.png`: 24-h CCFs over time for the 3 best pairs per group.
- `figs/interspan_v3_bestpair_{b0,b1}.png`: daily 24-h stacks over the reference, plus the hourly CC to the reference, for the best pair of each group.

**Best pairs at b1** (median coda-window CC, 10–90% range):

| Group | Pair | Distance | Median CC | 10–90% |
|---|---|---|---|---|
| Control | SB3 67 – SC4 73 | 33 km | 0.75 | 0.55–0.81 |
| ACF | SA2 48 – SB3 63 | 69 km | 0.64 | 0.42–0.76 |
| Daisy Bank | SA2 49 – SB3 60 | 65 km | 0.62 | 0.38–0.73 |
| SHR | SB3 55 – SD4 88 | 68 km | 0.56 | 0.40–0.75 |

The CC dips coincide with the Jan 1–3, Jan 6–10 and Jan 12–13 high-wave periods.

**Findings:**
- **Slow arrivals inside the coda window.** The s23 coda window starts at 1.3 d/vg with vg = 0.80 km/s (the median envelope-peak velocity). The best pairs show strong slow arrivals (~0.45–0.55 km/s) inside that window:
  - control: +65 s at 33 km;
  - ACF: +135 s at 69 km;
  - Daisy Bank: +115 s at 65 km.
  So part of the cross-span "coda" is ballistic. This is the same issue fixed for the seismometer–DAS work; the fix is per-pair, per-side ballistic picks with the coda after the strongest arrival.
- **Zero-lag energy.** The ACF and Daisy Bank gathers carry strong energy near zero lag (±10 s) at all distances (34–56 km), which does not move out with distance. It is either broadside waves or a common-mode term. It lies outside the coda window, so it does not enter dv/v.

**Added test: ram_stack48** (running-absolute-mean + 48-h back-stacked stacks, 1-h step):
- stretching CC 0.76, convergence 0.68, 99.8% of stacks kept;
- floor **0.071%** (0.057–0.087% per cluster), std 0.126%;
- injection recovery 92%.

This is the most stable seismometer–DAS configuration. The cost is 2-day temporal resolution: a step change ramps in over 48 h. The project standard is still 24 h / 1 h pending the user's decision.

**Production setting for seismometer–DAS (chosen by the user, 2026-10-14):** running-absolute-mean normalisation, **48-h back-stacked stacks moving 1 h** (`s40` and `s38` defaults; outputs `obsdas_coda/series_48h.csv`).

| Cluster | Stretching CC | Floor | Std | Injection recovery |
|---|---|---|---|---|
| Red SB3 | 0.77 | 0.070% | 0.10% | 92% |
| Red SC4 west | 0.77 | 0.057% | 0.13% | 87% |
| Red SC4 east | 0.75 | 0.087% | 0.13% | 91% |
| Blue SB3 | 0.80 | 0.080% | 0.15% | 95% |
| Blue SC4 | 0.70 | 0.072% | 0.10% | 98% |
| Orange | 0.70 | 0.063% | 0.13% | 106% |

All other products keep 24 h / 1 h. The s41 variant tests are pinned to 24 h (`STACK_H=24` in the run scripts) except the stack48 variants.

## Final coda-only dv/v: three products (2026-10-15; plan `docs/PLAN_final_dvv.md`)

### Processing common to all products
- **DAS clock.**
  - The acquisition drops whole chunks (17.18 s on the 100-m stream; about 490 per cable; drift up to 16 min per file).
  - All loaders use the true per-sample times from the per-chunk `proc_start` (`msdvv.raw.sample_times`).
  - The DAS–DAS correlators reject windows that span a drop, and use true times for hourly labels and earthquake windows (`msdvv.xcorr.window_times`).
  - DAS–station additionally applies the setup latency and the per-file colocated timing check.
  - Re-correlated outputs: `outputs_100m_v2` (one-bit) and `outputs_100m_v2ram` (running-absolute-mean) for within-span and cross-span.
- **Coda only, never ballistic** (`msdvv/coda.py`):
  - Coherent ballistic branches are found by stacking reference envelopes along lag = d/v. This is done per segment pair (cross-span) or span (within-span) and lag side, relative to the late-coda level; the duration of the slowest packet is measured.
  - Coda start c0 = max(d/v_b + packet + 10 s, d/U_floor + 20 s), with U_floor = 0.40 km/s across spans and DAS–station, and 0.15 km/s within spans (shelf Scholte waves).
  - Window length 90 s (60 s within spans). Lags extend to 400 / 300 / 150 s.
  - Window figures: `outputs_100m_v2*/interspan_coda/*/windows_check_*.png`, `outputs_100m_v2ram/within_coda/figs/within_windows_*.png`, `figs/obsdas/obsdas_coda_picks.png`.
- **Rejected window rules** (documented failures):
  - Per-pair strongest envelope peaks: they pick coda bumps.
  - Envelope ≥ 2× late background: real coda also exceeds the background, so the windows came out far too late.
- **Stacks:** 24 h and 48 h, back-stacked, 1-h step. Floor = even/odd split. Injection of a 0.5% stretch after Jan 1; injection runs reuse the clean run's windows and selection.
- **Acquisition setup step** (`s48_setup_correction.py`):
  - The D128 period (Jan 13 18:18 – Jan 21 18:22) reads 0.03–0.3% "slow". The steps are equal and opposite at the two changes (`outputs_100m/setup_steps.csv`), so this is an acquisition effect.
  - Mixed-setup stacks are dropped. DAS–DAS products get one D128 offset per unit, estimated from both changes.
  - For DAS–station the D128 period is excluded, because too few D128 stacks pass the timing check.

### Results (medians over units; `outputs_100m/final_comparison.{csv,md}`, `s46`)

| Product | Variant | Stretching CC | Convergence | Floor | Injection recovery |
|---|---|---|---|---|---|
| Within-span | **RAM, 0.3–0.7 Hz, 48 h** | **0.80** | 0.76 | **0.020%** | 96% |
| Within-span | RAM, 0.3–0.7 Hz, 24 h | 0.71 | 0.66 | 0.026% | 98% |
| Within-span | RAM, 0.12–0.3 Hz, 48 h / 24 h | 0.77 / 0.67 | 0.73 / 0.63 | 0.024 / 0.029% | 91 / 87% |
| Within-span | one-bit, 0.3–0.7 Hz, 48 h / 24 h | 0.74 / 0.63 | – | 0.029 / 0.035% | 93 / 93% |
| Cross-span | **RAM, 0.12–0.3 Hz, 48 h** | 0.52 | 0.39 | **0.017%** | 78% |
| Cross-span | RAM, 0.12–0.3 Hz, 24 h | 0.47 | 0.37 | 0.027% | 84% |
| Cross-span | one-bit, 48 h / 24 h | 0.53 / 0.48 | – | 0.019 / 0.035% | 78 / 83% |
| Cross-span | 2-km super-pairs (stacked before stretching) | 0.50–0.55 | – | 0.028–0.057% | 81–87% |
| DAS–station | **RAM, Z+R+T, 48 h** | 0.66 | 0.55 | 0.073% | 91% |
| DAS–station | RAM, Z+R+T, 24 h | 0.55 | 0.45 | 0.122% | 90% |

- **Running-absolute-mean beats one-bit** in every product.
- **48 h beats 24 h** on CC and floor everywhere.
- **Super-pairs do not help cross-span:** coda from different paths does not add coherently, and fewer independent units raise the floor.
- **Strict coda windows lower CC** against the earlier, partly ballistic windows. DAS–station drops from 0.76 to 0.66 at 48 h; cross-span stays at about 0.5.

### What the dv/v shows (Dec 23 – Jan 28; figures `outputs_100m/figs/final/final_{within_b2,within_b1,crossspan_b1,das_station}.png`)
- **Within-span:**
  - All spans on both cables share a slow ±0.1% variation; north and south cable averages correlate at r = 0.92–0.94.
  - It correlates with the swell period (Tp at NDBC 46050, r = +0.52 at 24 h / +0.66 at 48 h) and with noise power (r = −0.45 / −0.52), and matches the earlier common mode (r = 0.84).
  - It is a noise-source effect, not a medium change.
- **Cross-span:** fault groups minus control have std 0.027–0.052% against a combined floor of 0.023–0.041%, and exceed 2× the floor in 8–11% of stacks (hourly-stepped stacks are strongly autocorrelated). **No fault-specific signal.**
- **DAS–station:** HYS14 × ACF minus HYS14 × Daisy Bank has std 0.16% (48 h) against a combined floor of 0.12%. **No fault-specific signal.**
- **Open:** OBS horizontal orientation is unresolved (colocated-cable and teleseismic Rayleigh tests inconclusive). It does not bias dv/v: each component is stretched against its own reference.

**Scripts:** s40 (DAS–station), s43 (cross-span), s45 (within-span), s46 (comparison), s47 (within-span figures), s48 (setup correction), s49 (final figures).
**Rerun scripts** (`/data/wsd02/multispan_dvv`): `run_within_v2.sh`, `run_interspan_coda.sh`, `run_interspan_super.sh`, `run_within_coda.sh`, `run_obsdas_final.sh`.

### Final rebuild (2026-10-15): dropped-chunk gaps filled
- **Gap filling.** `load_file(regrid=True)` (config key `regrid_drops: true` in config_100m_v2*.yaml) places every raw sample on a regular grid of true time. Dropped chunks become zero-filled gaps flagged in `DasFile.valid`, and only windows with > 10% missing samples are rejected (`msdvv.xcorr.MAX_GAP_FRAC`). Before, every window spanning a drop was rejected, losing up to ~50% of the windows on Jan 7–13 and Jan 22–28.
- **The setup step was mostly an artefact.** After the fix, the D128 steps of the DAS–DAS products become small and no longer equal-and-opposite (`setup_steps.csv`):
  - within-span (RAM, 0.3–0.7 Hz, 24 h): −0.02 / +0.07%;
  - cross-span: ≈ 0.
  The earlier "step" came largely from the drop-related window loss next to the setup changes, so **DAS–DAS series are no longer setup-corrected** (and no mixed stacks are dropped). The DAS–station step remains (−0.21 / +0.28%), and its D128 period stays excluded.
- **Final numbers** (`final_comparison.md`):

| Product | Variant | Stretching CC | Floor | Std | Injection recovery |
|---|---|---|---|---|---|
| Within-span | RAM, 0.3–0.7 Hz, 48 h | 0.80 | 0.019% | 0.037% | 98% |
| Within-span | RAM, 0.3–0.7 Hz, 24 h | 0.71 | 0.027% | 0.048% | 99% |
| Cross-span | RAM, 0.12–0.3 Hz, 48 h | 0.52 | 0.017% | 0.017% | 81% |
| Cross-span | RAM, 0.12–0.3 Hz, 24 h | 0.46 | 0.027% | 0.031% | 86% |
| DAS–station | RAM, Z+R+T, 48 h | 0.66 | 0.073% | 0.12% | 91% |
| DAS–station | RAM, Z+R+T, 24 h | 0.55 | 0.122% | 0.17% | 90% |

- **Final figures:**
  - dv/v: `outputs_100m/figs/final/final_{within_b2,within_b1,crossspan_b1,das_station}.png`;
  - maps: `final_maps.png`;
  - correlation functions: `outputs_100m_v2ram/within_coda/figs/within_windows_{b2,b1}.png`, `outputs_100m_v2/interspan_coda/ram_b1_48h/windows_check_b1_48h.png`, `outputs_100m/figs/obsdas/obsdas_cluster_ccfs_{ref,24h}.png` and `obsdas_coda_picks.png`.

### DAS–station rebuilt with the gap fix (s32 `OBSDAS_REGRID=1`) and final decisions
- **The setup step is no longer systematic for DAS–station.** After regridding, its D128 steps are mostly below 1–2× the floor or have the same sign at both changes. Only blue HYSB1 → SC4 keeps an equal-and-opposite step (−0.26 / +0.27%). The D128 period is therefore **included** for all clusters, with blue SC4 flagged.
- **DAS–station final numbers:**
  - 48 h: CC 0.65, floor 0.071%, std 0.10%, injection recovery 94%;
  - 24 h: CC 0.53, floor 0.112%, std 0.15%, recovery 94%.
- **Remaining DAS–station gaps:** real seismometer outages (about Dec 24–26, Jan 18–19) and 3 short DAS fragments that fail the timing check.

### Cross-span CC tests (`outputs_100m/crossspan_cc_tests.csv`; all groups, RAM)

| Variant | Pairs | Stretching CC | Floor | Std | Injection recovery |
|---|---|---|---|---|---|
| b1, 1-km bins, 24 h / 48 h (**final**) | 533 / 840 | 0.46 / 0.53 | 0.018 / 0.013% | 0.019 / 0.011% | 85 / 81% |
| b1, 2-km bins, 24 h / 48 h | 216 / 303 | 0.48 / 0.55 | 0.023 / 0.015% | – | 86 / 82% |
| b0 (0.06–0.15 Hz), 24 h / 48 h | 560 / 805 | 0.56 / 0.61 | 0.037 / 0.023% | – | 83 / 81% |
| b1, 150-s coda, 24 h / 48 h | 359 / 728 | 0.44 / 0.50 | 0.020 / 0.012% | – | 84 / 77% |
| b1, 96 h / 168 h stacks | 1134 / 1269 | 0.61 / 0.70 | 0.009 / 0.006% | 0.006 / 0.004% | 73 / 68% |

- **2-km bins:** CC +0.02 but fewer pairs and a slightly higher floor.
- **0.06–0.15 Hz:** CC +0.08 but a worse floor.
- **150-s coda:** CC lower.
- **Long stacks:** the only route to CC ≥ 0.6–0.7, but they smooth the response, so a change is under-recovered (~70%).

The final cross-span product stays at b1 with 1-km bins (24 h and 48 h). The 96-h and 168-h series are available as low-time-resolution products.

### Correction (final gap-filled data, 2026-10-15)
- **Within-span (RAM, 0.3–0.7 Hz):** the earlier statements that the common variation "follows the swell period (r = +0.52/+0.66)" and that north and south correlate at r = 0.92–0.94 came from the pre-gap-fix series. On the final series:
  - north–south correlation is r = 0.64–0.69;
  - correlations with Hs, Tp and noise power are |r| ≤ 0.24.
- **Two regimes:** dv/v is very stable on Dec 23 – Jan 13 (cable-average std 0.009–0.019%), and more variable on Jan 13–28 (0.027–0.038%). That later period starts with the D128 acquisition change and the calm-sea period; its cause is unresolved and is not explained by the wave or noise-power proxies.
- **DAS–station:** r(dv/v, Hs) ranges from −0.05 to +0.38 across clusters.
- **Feature contrasts on the final series:**
  - cross-span 48 h: ACF − control std 0.029% vs floor 0.026%; Daisy Bank − control 0.027% vs 0.024%;
  - DAS–station 48 h: HYS14 × ACF → SB3 − HYS14 × Daisy Bank: std 0.12% vs floor 0.12%.
  None of these is a feature-specific signal.

### uw62216067 re-picked with deep-learning pickers (`s51_ml_picks.py`, env `pyfetch`; `s52_relocate_ml.py`)

**Picks.** PhaseNet and EQTransformer were run with OBS-trained ("obs", PickBlue) and land-trained (instance, original) weights on the 200-Hz records of HYSB1, HYS11–14 and the Axial Seamount stations.
- **P:** all models agree to within 0.1 s, with probabilities up to 0.95:

  | Station | P | S | S−P |
  |---|---|---|---|
  | HYSB1 | −2.04 s | +18.9 s | 20.9 s |
  | Summit stations (HYS11–14) | +1.21–1.31 s | +24.2–24.6 s | 23.0–23.3 s |

  Times are relative to the catalogue origin.
- **S:** probabilities 0.2–0.64. The sharp horizontal arrival near +13.5 s is not picked as S by any model.
- **Axial Seamount:** picks there belong to separate local Axial microquakes (P and S 0.3–0.7 s apart); the event was not recorded at Axial.
- **Conclusion:** the original manual picks were essentially correct.

**T-phase.** A strong T-phase arrives at all five Hydrate Ridge stations (envelope peaks +120 to +132 s; earliest at HYSB1). That is impossible for the catalogue epicentre (predicted at about +25 s) and confirms a distant source to the WSW.

**Relocation** (consensus OBS P+S, 36 land P down-weighted, depth grid 5/10/20 km):

| Velocity model | Location | Origin | rms | From catalogue | T-phase residual |
|---|---|---|---|---|---|
| iasp91 | 43.60N 126.80W | 09:08:56.4 | 1.15 s | 207 km | +40 s |
| oceanic (3 + 4 km crust over 8.0 km/s mantle) | **43.35N 127.19W** | **09:08:56.6** | **1.03 s** | 249 km | **+11 s** (peak vs onset) |

The oceanic model is preferred: lower rms and a consistent T-phase. That places the event near the Blanco Transform – Gorda Ridge junction. No other agency catalogued it, and the PNSN location (one-sided land network over 225 km away) is wrong by about 250 km.
Outputs: `obs/uw62216067_ml_picks.csv`, `obs/uw62216067_relocation_ml.json`, `figs/uw62216067_ml_picks.png`, `figs/uw62216067_relocation_ml.png`.

**S-wave check (the +13.5-s arrival).** The alternative S (horizontal STA/LTA onset at +13.9 to +14.4 s on the summit stations, +5.4 s at HYSB1) was tested by relocating both hypotheses with the oceanic model:

| S hypothesis | rms, OBS only | rms, with land P | T-phase residual |
|---|---|---|---|
| A: S at +24 s | 0.27 s | 1.03 s | +3 to +11 s |
| B: S at +13.5 s | 1.11 s | 1.29 s | +65 to +73 s (physically impossible) |

S at +24 s (consensus deep-learning and original manual picks) is kept. The +13.5-s energy is a short burst in the P coda, probably converted or scattered, not the direct S. Figure: `figs/uw62216067_seismograms_picks.png` (`s53_event_seismograms.py`).

### Structure: static surface-wave group-velocity tomography (`s56_tomography.py`; outputs `outputs_100m/tomography/`, figures `figs/final/tomo_{maps,checker,coverage}.png`)

**Data.** All-period reference cross-span CCFs (RAM, gap-fixed, south cable, 1-km bins, 3800 pairs without C3).

**Picks.** Narrow-band Gaussian filters (α = 25) at T = 4–12 s, envelope of both lag sides summed. Picking is reference-guided: the peak is searched in [d/1.05, d/0.50] s, around the dominant regional branch at about 0.72 km/s.
- An unguided search mixed the slow 0.4–0.5 km/s branch, the ~0.8 km/s branch and zero-lag energy, and the inversion was unstable (negative velocities).
- **Kept:** SNR ≥ 8, both sides consistent, d ≥ 3λ, no edge picks.
- **Kept per period:** 255–940 paths at 4–8 s; 10 and 12 s have too few.

**Inversion.** Straight rays on 0.05° cells, damped Laplacian-smoothed least squares (λ_s = 150, λ_d = 10), one 3-MAD outlier pass.
- **Fit:** travel-time residual std 17–22 s before, 14–19 s after. Single-path pick scatter is about ±15%.
- **Resolution:** a ±8% checkerboard with 0.2° (~20 km) squares is not recovered (r ≤ 0.22). With 0.6° (~50–65 km) squares it is recovered at r = 0.54–0.76 and 50–70% of the amplitude. Only features ≥ 50 km are interpretable.
- **Stability:** the 5–8 s maps correlate with each other at r = 0.45–0.91; 4 s is inconsistent with the rest.

**Result (5–8 s, reference U0 = 0.71–0.72 km/s):**
- A broad **fast** zone (+5 to +10%) covers Hydrate Ridge and the Alvin Canyon fault corridor in the southern half of the loop (about 44.45–44.8°N).
- **Slower** (−5 to −10%) material lies to the north and northwest (about 44.9–45.2°N, mid-slope basins).
- Interpretation (tentative): uplifted, more consolidated accreted sediments of the Hydrate Ridge thrust anticline against thicker, softer slope-basin fill. Rays sample mostly the region inside the cable loop, so values right at the loop edges are least reliable.
- **No time-lapse imaging:** dv/v is at the noise level, so it would only map noise.

**Seismometer paths added to the tomography** (`s56` default `INCLUDE_OBS=1`; `INCLUDE_OBS=0` gives the DAS-only maps `*_dasonly`).
- **Data:** HYSB1/HYS14 × south-cable bins, Z component (Scholte/Rayleigh), RAM, latency-corrected, timing-QC'd, gap-fixed, same picking rules.
- **Kept paths:** 17 / 34 / 42 / 57 at 4 / 5 / 6 / 7 s (HYSB1 6–32, HYS14 11–25), out of 202. The OBS–DAS correlations beyond about 30 km are weak, and the short HYSB1 paths fail d ≥ 3λ. There is no 8–12 s, because the OBS–DAS data hold only 0.12–0.3 Hz.
- **Effect:** about 4–6% more rays. The overall checkerboard recovery is unchanged (r 0.54–0.76). Within 25 km of HYS14 the hit count rises 10–20% and the fast anomaly is reinforced (mean +7 to +13% vs +6 to +12% without), with local changes ≤ 8%.
- **Conclusion:** the seismometers confirm the fast Hydrate Ridge anomaly but do not change resolution; the fiber–fiber pairs dominate.

## Follow-up items (docs/NEXT_STEPS.md)

### Item 1: DAS channels at the fault crossings (`s57_fault_crossings.py`; `outputs_100m/fault_crossings{,_summary}.csv`, `figs/fault_crossings_coherence.png`)

**Crossings** (Qfaults × south cable route): Alvin Canyon fault at route km 274.5 (optical 282 km, inside mask block C); Daisy Bank fault at route km 295.6 (optical 303.5 km, in the 300–310 km mask gap). Both are on the shelf at about 270 m.

**Coherence** (0.42 km offset, 0.12–0.6 Hz; physical if ≥ 0.15), median within ±3 km of each crossing:

| Configuration | Alvin Canyon | Daisy Bank | Reference C4 (route km 255–267) |
|---|---|---|---|
| Nov 5, 105.9 m, g256, full fibre | 0.50 | 0.52 | 0.51 |
| Nov 20, 105.9 m, g256, full fibre | 0.16 | 0.12 | 0.21 |
| Nov 27 / Dec 1, 52.9 m, g128, full fibre | 0.27 / 0.25 | 0.25 / 0.25 | 0.31 / 0.27 |
| Dec 5 / Dec 15, 52.9 m, g128, masked | 0.08 / 0.09 | masked | 0.16 / 0.33 |
| Dec 23–29, 13.2 m and Jan 1–4, 26.5 m (8-Hz stream) | not recorded | not recorded | 0.07–0.17 |
| Jan, 100-m stream (s05) | 0.06 | masked | 0.25 |

**f–k test** (share of 0.12–0.6 Hz energy at apparent velocity 0.15–4 km/s, ±3 km windows):

| File | Alvin Canyon | Daisy Bank | Good references | Far fibre (optical 600 km) |
|---|---|---|---|---|
| Nov 27 | 44% | 24% | 55–71% | 25% |
| Dec 1 | 23–26% everywhere | | | |
| Nov 5 | 51–64% everywhere | | | |

**Conclusion (provisional):**
- In the masked configurations (Dec 5 onward, including all Dec–Jan products) the crossings carry no physical signal.
- In the unmasked Nov configurations the coherence metric does not discriminate (high everywhere, including far along the fibre), and 10-min samples are too short for an f–k verdict. The Alvin Canyon crossing on Nov 27 is the only borderline case.
- **Next:** download the full unmasked Nov 26 – Dec 2 south record at native 52.9 m around both crossings (optical 270–310 km, part of item 7) and test whether noise correlations across the crossings converge with physical moveout.

### Item 2: DAS clock technical note (`docs/DAS_clock_note.md`; evidence `s58_clock_evidence.py`, `figs/clock_note_evidence.png`, `outputs_100m/clock_drops_per_file.csv`)
- **Drops:** the 100-m stream lost about 450 chunks per cable at D256 (17.18 s each; drift up to 958 s by the end of a file) and about 44 at D128. They are concentrated on Jan 6 18:00 – Jan 13 and Jan 21 18:00 – Jan 28; the 8-Hz stream Jan 12 files lost 31–46 chunks.
- **Proc_start creep:** between drops, `proc_start` creeps 3–10 s/h in those periods while `gpu_idx` shows no missing samples. Against HYSB1, the per-chunk proc_start model and the new sample-counter model (`msdvv.raw.chunk_start_times_counter`) agree within about 1 s (trends ≤ 1 s/h), so the pipeline results are unaffected. The counter model is recommended in the note.
- **Latency vs UTC:** D256 +25.2 s, D128 +11.8 s (difference exactly 2²⁴ / 1.25 MHz = 13.42 s).
- The note lists suggestions for the data providers. It has not been sent; the user decides whether to share it.

### Item 3: earthquakes with the corrected DAS clock
- **3a, Jan 13 2026 05:12:47 ML 2.0** (32 km W of Neskowin; about 1 km from south A1). `s17_das_event_detect.py` now uses true per-sample times (`TRUE_TIMES=1`). The 8-Hz stream files around the event (D512, 211.7 m) had lost 31 (south) and 46 (north) chunks, so the earlier "no detection" used times wrong by up to 164 s.
  - With true times, the S-aligned 2–3.6 Hz envelope stack peaks at lag 14.0 s (south, robust z = 5.4) and 12.7 s (north, z = 10.6), against z = 0.2 / 0.5 before. **The event is detected on both cables.** The lag gives the DAS latency of the D512 setup: about 13–14 s.
  - Record section `figs/das_jan13_record_section.png` (`s59_jan13_das_section.py`): S arrivals from the nearest channels (about 1 km) out to about 12 km on the south cable, and 8–12 km (plus site-amplified energy at 30–35 km) on the north cable.
  - Dec 13 ML 1.6 (depth 25 km, nearest good channel 26 km): still not detected (z = −1.5).
- **3c, magnitude of the relocated uw62216067** (`s60_magnitude_uw62216067.py`): 2–8 Hz horizontal S amplitude at HYSB1 and HYS11–14, with a Hutton–Boore distance term and station constants from 7 Blanco reference events (M 2.7–3.4, Dec 23 – Jan 28).
  - Leave-one-out on the references: bias −0.01, MAD 0.17.
  - **Result: M 3.09 ± 0.08 (5 stations), not ML 2.2.** The catalogue magnitude was computed for the wrong distance.
  - Figure `figs/uw62216067_magnitude.png`; result `obs/uw62216067_magnitude.json`.
- **3b, deep-learning scan of the Hydrate Ridge OBS:** see below once complete (`s61_obs_dl_scan.py`).

### Item 4: drivers of the within-span variability after Jan 13 (`s62_ocean_drivers.py`; `outputs_100m/ocean_drivers/`, `figs/ocean_drivers.png`)

**Candidates:** OOI tsunameters at HYS14 (777 m) and HYSB1 (2907 m) via EarthScope FDSN: de-tided bottom pressure (8 constituents) and temperature, for the full period. The Nobska current meters have only 2–6 days in this period and are unusable. Also Hs, Tp, DAS noise power, DAS dropped chunks per hour, and the D128 setup flag. All are trailing means over the same 24- or 48-h window as the dv/v.

**Result:** none of the correlations is significant once the strong autocorrelation of overlapping 24–48-h stacks is accounted for. With 36 days there are only about 5–8 independent samples, so a correlation needs |r| > 0.67–0.82 to be significant (AR(1)-corrected 95 %).

| Candidate | dv/v target | Period | r |
|---|---|---|---|
| HYS14 temperature | within-span north / south | Jan 13–28 | −0.48 / −0.52 |
| HYS14 de-tided pressure | within-span north | whole period | +0.37 |
| DAS drops per hour | within-span south | whole period | +0.52 |
| DAS drops per hour | cross-span | whole period | −0.44 |
| Tp | within-span south | Jan 13–28 | −0.51 |

The cause of the larger variations after Jan 13 is therefore not identified. The November extension (item 7) roughly doubles the record and is the real test of these candidates.

### Item 5: shear-velocity depth structure (`s63_vs_inversion.py`, env pyfetch/disba; `outputs_100m/vs_inversion/{data,results}.csv`, `figs/final/vs_profiles.png`)

**Data:** Scholte/Rayleigh group velocities per region:
- within-span short-offset values from the 100-m stream: 0.3–0.7 Hz (T ≈ 2 s) and 0.12–0.3 Hz (T ≈ 5 s);
- tomography values at 5–8 s (fibre + seismometer paths).

The 1–2 Hz picks are excluded: in deep water they are faster (0.7–2 km/s) than the lower bands, so they are water-borne or leaky-P energy, not the Scholte wave.

**Model:** water layer over sediments with Vs = Vs0 + g·z (capped at 2.5 km/s) to 4 km depth, then a half-space; Vp and density from the mudrock line and Gardner's relation. Grid search with 15 % errors; posterior mean and 5–95 %.

| Region | Data | Vs at 0.1 / 0.5 / 1 / 2 km | Vs at 1 km, 5–95 % | Fit |
|---|---|---|---|---|
| Base of Hydrate Ridge (B3, 2.9 km water) | 6 (T 2–8 s) | 0.63 / 0.85 / 1.13 / 1.67 km/s | 1.01–1.24 | χ² 0.5 (good) |
| South slope (A2, 1.9 km) | 2 | 0.71 / 1.01 / 1.39 / 1.92 | 0.81–2.12 | underdetermined |
| Shelf (A1/C4, 0.2–0.5 km) | 1 | 0.51 / 0.89 / 1.35 / 1.93 | 0.51–2.20 | underdetermined |
| Hydrate Ridge summit (tomography only, 0.8 km water) | 4 (T 5–8 s) | 1.14 / 1.16 / 1.20 / 1.26 | 1.02–1.26 | χ² 7.9 (poor) |

**Conclusion:**
- Only the base of Hydrate Ridge is well constrained: a normal sediment gradient from about 0.6 km/s at the seafloor to about 1.7 km/s at 2 km.
- The summit asks for markedly faster material from the seafloor down. That is consistent with the fast tomography anomaly and with consolidated, hydrate- and carbonate-bearing accreted sediments of the ridge, but its dispersion is flat and fits poorly, and the tomography values average over at least 50 km. Treat it as qualitative.
- Better depth resolution would need per-bin dispersion over a wide period range (multi-band FTAN on the cross-span pairs) or the 52.9-m November data at short periods.

### Item 6: north-cable cross-span dv/v and tomography (config `interspan_v6_north` in config_100m_v2.yaml; `s20`, `s43`, `s56 CABLE=north`)

**Correlations:** RAM, gap-fixed, 89 1-km bins, 2850 pairs (A1–A2–B3–C4; 16–245 km). No mapped feature is crossed, so the whole product is a control.

**Coda dv/v** (b1, coda after the slowest coherent ballistic packet, ≤ 70 km; outputs `outputs_100m_v2/interspan_coda/north_ram_b1_{24,48}h/`):

| Stack | Pairs | Stretching CC | Floor | Std | Injection recovery |
|---|---|---|---|---|---|
| 24 h | 594 | 0.46 | 0.021 % | 0.038 % | 84 % |
| 48 h | 791 | 0.53 | 0.011 % | 0.019 % | 81 % |

- The quality is the same as the south cable.
- North and south cross-span series are **uncorrelated** (r = 0.07 at 24 h, −0.12 at 48 h). There is no common regional variation between the cables at the 0.01–0.02 % level, and each cable's variations are close to its noise floor (std ≈ 1.7–1.8× the floor).

**Tomography** (`outputs_100m/tomography_north/`, `figs/final/tomo_{maps,checker,coverage}_north.png`):
- 118–880 paths at 4–10 s, U0 0.66–0.69 km/s, 12-cell checkerboard r 0.49–0.72 at 5–10 s (4 s unusable).
- The cable is nearly straight, so all rays run along it and the result is an **along-cable profile**, not a map.
- The 5–8 s periods agree: slower (−5 to −10 %) on the outer abyssal section (about 126.0–126.7°W) and near 124.8°W (upper slope / shelf break), faster (+5 to +10 %) between about 125.2 and 125.9°W (mid-slope).

### Disk clean-up (2026-10-15, approved by the user)
- **Deleted:** superseded correlation `.h5` files, about 140 GB in total.
  - `outputs_100m/ccf`: first within-span run, before the clock and gap fixes.
  - `outputs_100m/interspan`, `interspan_v2`: early cross-span versions.
  - `outputs_100m/interspan_v3`, `interspan_v4_ram`: before the gap fix.
  - `outputs_100m/obsdas_old`, `obsdas_nominal_times`, `obsdas_noregrid_backup`: DAS–station before the latency and clock fixes.
- **Kept:** the CSV results inside those folders (pair tables, feature series, timing-QC tables), so the documented numbers and figures stay reproducible.
- **Final correlations kept:**
  - within-span: `outputs_100m_v2ram/ccf` (final) and `outputs_100m_v2/ccf` (one-bit comparison);
  - cross-span: `outputs_100m_v2/interspan_v4_ram` (final south), `interspan_v3` (one-bit comparison), `interspan_v6_north`;
  - DAS–station: `outputs_100m/obsdas_{,R_,T_}ram` and the one-bit `obsdas{,_R,_T}`;
  - all raw data.
- The deleted files can be regenerated from `raw/` with the old configs.
- Free space on /data/wsd02 went from 60 GB to 200 GB.

**3b, deep-learning scan, first pass** (`s61_obs_dl_scan.py`: PhaseNet "obs", P ≥ 0.5, S ≥ 0.4, ≥ 3 stations within 4 s; `obs/dl_scan_events.csv`):
- **Output:** 260 picks and 23 associations. 10 of them are artefacts at the daily-file boundary (23:59:59).
- **Found:** catalogued Jan 1 M2.9, Jan 16 M6.0, Jan 20 M3.4 and Jan 25 M2.8 (Blanco), and the relocated uw62216067.
- **Missed:** several M2.7–3.1 Blanco events and the Jan 13 local ML 2.0.
- **Sensitivity:** about M3 at Blanco distance. The archived OBS data are 20 Hz, so the picker gets no energy above 10 Hz.
- **Uncatalogued candidates:**
  - 2025-12-31 09:53:38 and 2026-01-07 20:25:09: 4–5 stations, S−P 9.5–9.9 s (about 75–80 km), HYSB1 first (west/southwest);
  - 2026-01-18 21:49:13: S−P 21.9 s, Blanco distance.
- **Second pass running** (`TAG=_v2`): P/S ≥ 0.3, 6-s association window, day-edge picks removed.

**3b, second pass** (`TAG=_v2`: P/S ≥ 0.3, 6-s association, day-edge picks removed; `obs/dl_scan_events_v2.csv`, `figs/obs_dl_scan_v2.png`):
- **Output:** 843 picks and 27 associated events, with no day-boundary artefacts left.
- **Catalogued events found (8):** all Blanco events of M ≥ 2.7 in ComCat for the period (Jan 1 M2.9, Jan 16 M6.0 and M3.1, Jan 20 M3.4, Jan 25 M2.8, Jan 26 M3.0 and M2.7) and uw62216067.
- **Missed:** the Jan 13 local ML 2.0 (about 90 km NE of HYS14). Its high-frequency energy is lost in the 20-Hz archive.
- **Uncatalogued events:**
  - **local (S−P 9.5–9.9 s, about 75–80 km, HYSB1 first by 3.2–3.3 s, i.e. to the WSW):** 2025-12-31 09:53:38 (5 stations) and 2026-01-07 20:25:09 (5 stations);
  - **Blanco-distance (S−P 21.6–27.4 s):** 2025-12-31 21:29, 2026-01-17 04:28, 2026-01-18 21:49 and 23:12, 2026-01-28 01:37;
  - 10 more without an S−P estimate (3–5 stations).
- **Conclusion:** the seafloor array detects Blanco events below the land-network catalogue threshold, and two moderate local events about 80 km WSW of Hydrate Ridge that no network catalogued.

### Item 8 (user's idea): beamforming across spans (`s64_beamform.py`; `outputs_100m/beamform/`, `figs/beamform_*.png`)

**Idea:** one interrogator records each cable, so all its spans are on exactly the same clock, and dropped chunks remove every channel at once. The south cable's loop (about 70 × 90 km, 101 good 1-km bins) is a 2-D array.

**Validation with earthquakes:**
- **Frequency-domain Bartlett beamforming** (whitened bin spectra, plane wave, 0.05–0.25 Hz) **failed** for the Jan 16 M6.0 Blanco (true back-azimuth 244°) and the Dec 27 M6.6 Taiwan teleseism (303°).
  - The Taiwan surface waves are not above the DAS noise at all (event/noise power ≈ 1 at 0.05–0.5 Hz), so that test is void.
  - For the M6.0 the event/noise power is 80–450 at 0.05–0.12 Hz, but the low-frequency record section shows no coherent moveout across bins.
- **Time-domain onset picks** (0.2–0.8 Hz STA/LTA per bin + plane-wave fit) **work for the M6.0**: back-azimuth 250° (true 244°), apparent velocity 6.3 km/s (P), residual 1.8 s, 85 of 101 bins.
- **M2.9–3.4 Blanco events** (including the relocated uw62216067) are too weak at 1.86 Hz sampling: the picks are noise, with residuals of about 45–50 s. The 7.45-Hz 8-Hz stream would be needed.
- **Noise:** the hourly frequency-domain beamforming of the ambient noise (0.12–0.25 Hz, 300-s windows, unit weights) over Dec 23 – Jan 28 is running, to test whether changes in noise direction track the within-span dv/v variability.

## Data quality and corrections (`docs/DATA_QUALITY.md`)

The checklist of known issues with the multispan DAS records (Nov 2025 – Jan 2026) and how to correct them. It covers:

- **Timing:** dropped chunks, missing chunk metadata in the Nov 6–26 files, latency per acquisition setup, unreliable filename times.
- **Channels and geometry:** non-physical channels (including at both fault crossings), tonal segments, edges, the north-cable polarity flip, single-component axial strain.
- **Setups, files and coverage:** setup changes, short files, outages, sampling limits, fiber coupling.
- **External references:** offshore catalogue locations.

It also gives a processing recipe and recommendations for the data providers. The same section is in the report artifact ("Data quality and corrections").

## Reproducing the report (`docs/REPRODUCE.md`)

The guide maps each report section to its scripts, settings and run order. It also lists the conda environments (`docs/environment_*.yml`), the run chains (`runs/`), the report source and its build steps (`docs/report/`), and the known reproducibility gaps.
