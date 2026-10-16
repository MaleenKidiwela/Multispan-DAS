# Plan to a stopping point: three dv/v products with the best CC and stable correlation functions and dv/v

**Products:**
- **A. Within-span:** DAS–DAS, short offsets 0.2–3.2 km, both cables.
- **B. Across-span:** DAS–DAS bin pairs across features, south cable.
- **C. DAS–station:** HYSB1/HYS14 × south-cable bins.

**Stopping point:** for each product,
1. processing optimised by documented, injection-checked tests;
2. figures of stable correlation functions;
3. the dv/v series with noise floor;
4. a metrics table.

Everything is written up in `README.md`, with a short final report.

---

## 0. Shared foundation (all three products)

### 0.1 DAS timing (clock)
**Problem.** The acquisition drops whole 32-sample chunks without leaving a gap:
- 100-m stream: 17.18-s chunks; about 490 drops per cable over Dec 23–Jan 28 (north 50/88 files, south 74/142); in-file drift up to about 16 min.
- 8-Hz stream: 4.29-s chunks; for example 31–46 drops in the Jan 12 files.

True per-sample times come from the per-chunk `proc_start` (`msdvv.raw.sample_times`, `DasFile.times`). The downsampled h5 files of Nov 6–26 have no per-chunk metadata.

**Actions:**
1. **Window QC at drops.** In the DAS–DAS correlators (`xcorr.correlate_file`, `interspan.correlate_file_interspan`), reject every window that contains a chunk drop, i.e. a time jump > 1.5 chunk lengths. This removes the splice artefacts (about 4% of windows).
2. **Substack labels.** Assign windows to hourly substacks with the true times.
3. **Event exclusion.** Use the true times for the earthquake exclusion windows, which are now misplaced by up to about 15 min.
4. **DAS latency per setup** (DAS–station only): already calibrated from the colocated HYSB1 bins (D256 25.2 s, D128 11.8 s). The per-file QC check stays in the pipeline.
5. **November / h5 data:** no chunk times. Use only for DAS–DAS (relative timing, after window QC), or QC per file with the colocated HYSB1 check, as for C.

### 0.2 Seismometer side (C only)
- HYS14 chronfix Δt = 0 for this period; OBS–OBS daily clock shifts ≤ 0.15 s.
- Gaps filled (`d04b`); the remaining gaps are real outages.
- **Horizontal orientation check (new).** The StationXML gives nominal 0°/90° for HHN/HHE. Verify by rotating the horizontals and maximising the colocated HYSB1 × SB3 correlation (and HYS14 × SB3). Correct R/T if a misorientation is found.

### 0.3 Acquisition setup changes
- 100-m stream: D256 → D128 (Jan 13 18:00) → D256 (Jan 21 18:00).
- Test for steps at the transitions: dv/v of the stacks just before vs just after, against the even/odd floor.
- If a step exceeds 2× the floor, use per-setup references or a step correction.

### 0.4 Data selection
- Channel quality (`s05`).
- Repeater and mask edges (±6 channels).
- C3 excluded (tonal).
- ComCat events (M ≥ 5 global, M ≥ 2.5 regional), with true times.

### 0.5 Common test and validation toolkit
The same as for C (`s40`/`s41`). Each test variant is run twice: on the data, and with an injected 0.5% step.

**Metrics:**
- stretching CC (median);
- convergence CC (24-h stack vs reference without it);
- share of stacks kept (CC ≥ 0.4);
- even/odd floor;
- std;
- injection recovery (target 85–115%);
- correlation with wave height and source proxies.

**Standard figures per product:**
1. Reference gathers with ballistic picks and coda windows.
2. Best pair / cluster stability: daily stacks over the reference, plus the hourly CC to the reference.
3. dv/v series with the floor.
4. Map of the pairs used.
5. Variants comparison.

**Window rule for all products:** coda only, i.e. ballistic picked per pair and lag side (strongest envelope peak at 0.35–2 km/s, weak picks filled from the moveout), coda from pick +15 s.

**Stacking:** back-stacked, 1-h step. C uses 48 h (user decision). A and B are tested at 24 h and 48 h; the user chooses.

---

## A. Within-span (DAS–DAS, short offsets)

**Status.**
- One-bit + whitening, 24 h / 1 h, coda window from about 2 t_d.
- 0.12–0.7 Hz: CC 0.79, floor about 0.04%.
- 1–2 Hz (Dec 23–Jan 5): CC 0.86, floor about 0.015%.
- Dominated by noise-source changes.

**Steps:**
1. **Re-correlate the 100-m stream** with the drop-window QC and true-time labels, in two versions: one-bit and running-absolute-mean (20 s). Raw data are local (111 GB); a few hours with 48 cores.
2. **Re-correlate the 1–2 Hz fine-spacing record** (Dec 23–Jan 5) the same way.
3. **Ballistic picks per offset bin and segment.** Shelf Scholte waves are slow (0.18–0.2 km/s), so a 3.2-km pair has its ballistic at about 16–18 s. Start the coda after the strongest arrival, with the same pick/fill rule.
4. **Tests** (injection-checked):
   - normalisation: one-bit vs RAM;
   - band: b1, b2, and 1–2 Hz where available;
   - spatial stack: 1, 2 or 5 km bins;
   - stack length: 24 h vs 48 h;
   - coda length: +60 or +90 s.
5. **Setup-step check** at Jan 13 and Jan 21.
6. **Source-effect diagnostic** (`s25`): report how much of the dv/v the source proxies explain; it is not used as a correction unless agreed.
7. **Figures:** standard set per cable and span (shelf, slope, abyssal spans; north and south).

## B. Across-span (DAS–DAS bin pairs, south cable)

**Status.**
- v3: one-bit, b0/b1, coda from 1.3 d/vg with vg = 0.8 km/s.
- 715 converging pairs in daily stacks, floor 0.017–0.045%.
- No feature-specific signal against pseudo-fault nulls.
- **Problem:** strong slow arrivals (0.45–0.55 km/s) fall inside that coda window. The first per-pair-pick run (`s43`) gave implausibly fast fill velocities (1.2–1.7 km/s), so the picks need fixing.

**Steps:**
1. **Re-correlate** (`s20`, about 1.5 min per variant) with the drop-window QC and true-time labels: one-bit and RAM.
2. **Fix the ballistic picks.**
   - Restrict the search to plausible group velocities (0.35–1.2 km/s).
   - Exclude |lag| < 15 s.
   - Inspect the picks on gathers and on the best pairs.
   - Start the coda after the slowest strong arrival. Long pairs whose coda would run past the 200-s maximum lag are dropped, or the correlation maximum lag is extended to 300 s (nfft 2048 at 1.86 Hz allows it).
3. **Diagnose the zero-lag energy** (±10 s at 34–56 km on the fault pairs):
   - Compare with same-span pairs and test its moveout to tell broadside waves from a common-mode signal.
   - If it is common-mode, remove it with a per-block k ≈ 0 notch before binning; not with median subtraction, which injected a zero-lag term in v2.
   - It is outside the coda window either way.
4. **Group stacking.** Stack the CCFs of a segment pair (e.g. SA2–SB3, SB3–SC4, SB3–SD4), aligned on the ballistic picks, before stretching, as in the C clusters. Then report the feature groups.
5. **Tests:**
   - normalisation: one-bit vs RAM;
   - band: b0, b1, 0.15–0.25 Hz;
   - stack length: 24 h vs 48 h;
   - coda length;
   - bin size: 1 vs 2 km.
6. **Feature contrast:** fault groups vs control vs pseudo-fault nulls, using the final settings.
7. **Figures:** standard set, plus segment-pair gathers.

## C. DAS–station (HYSB1/HYS14 × south bins)

**Status (production).**
- Per-chunk DAS times, setup latency, per-file timing QC (137 of 140 files good).
- RAM normalisation; Z+R+T; ballistic picks per pair, side and component; coda pick +15 to +90 s.
- 6 clusters (red SB3 / SC4 west / SC4 east, blue SB3 / SC4, orange).
- 48 h / 1 h: stretching CC 0.70–0.80, floor 0.057–0.087%, injection recovery 87–106%.

**Steps:**
1. **Horizontal orientation check** (0.2); recompute R/T if needed.
2. **Re-time or drop the 3 failing fragments**; confirm the latency is unchanged after acquisition restarts (per-file QC).
3. **Window check** for red SC4 east Z+ (ballistic tail inside the coda): coda start after d/0.35 km/s for that side.
4. **Final figures:** reference gathers with picks, cluster CCFs over time (48 h), dv/v, pair map, variants table.
5. **Feature contrast:** HYS14 × ACF vs HYS14 × Daisy Bank (same station), and control vs features, against the even/odd floor.

---

## Order and effort

| Step | Product | Work | Compute |
|---|---|---|---|
| 1 | all | 0.1 window QC at chunk drops + true-time labels in both DAS–DAS correlators; 0.2 orientation check | code ~1 h |
| 2 | B | re-correlate (one-bit and RAM), fix picks, zero-lag diagnosis, group stacking, tests, figures | ~2–3 h |
| 3 | A | re-correlate 100-m stream (one-bit and RAM) + 1–2 Hz record, picks, tests, figures | ~3–4 h (mostly compute) |
| 4 | C | orientation, fragments, window fix, final figures | ~1 h |
| 5 | all | 24 h vs 48 h decision for A and B (user), final runs, README, one-page report | ~1 h |

**Decision points for the user:**
1. 24 h vs 48 h for A and B, after the tests.
2. Whether the source-proxy correction is applied to A.
3. Whether November is processed after the stopping point. It needs window QC; Nov 6–26 h5 files have no chunk times.

**Acceptance targets (24 h unless 48 h is chosen):**
- median stretching CC ≥ 0.7 (A, C), ≥ 0.6 (B);
- injection recovery 85–115%;
- std/floor ≈ 1–2;
- stable reference waveforms in the best-pair figures.


---
## Status (2026-10-15)

| Step | Status |
|---|---|
| 0.1 timing | done |
| 0.2 orientation | tested, unresolved; harmless for dv/v |
| 0.3 setup step | found and corrected (DAS–station: excluded) |
| A, B, C | final 24 h + 48 h products with injection checks and figures |

Results are in the README section "Final coda-only dv/v: three products".

**Not met:** cross-span CC ≥ 0.6 (0.47–0.53 with strict coda windows) and cross-span injection recovery ≥ 85% at 48 h (78%).

**Possible next steps:**
- longer coda windows for cross-span (the lags allow up to ~490 s);
- pairing the cross-span result with within-span bins at the path ends;
- November (after the user's go-ahead).
