# Reproducing the "Multispan dv/v" report

The report source is `docs/report/index.html` (published as https://claude.ai/artifact/48SnQS3Q72PfLvNgSrBeUW). This guide maps each report section to the code, settings and order of runs that produce it.

## Setup

| What | Where |
|---|---|
| Code | this project: package `msdvv/`, `scripts/` (s00–s80, d00–d06), configs `config_*.yaml` |
| Run chains | `runs/run_*.sh`. These are copies; the originals live in `/data/wsd02/multispan_dvv/`, and the chains write there |
| Conda environments | `docs/environment_noisepy2.yml` (everything except the deep-learning steps) and `docs/environment_pyfetch.yml` (SeisBench: s51, s54, s61; disba: s63) |
| Data and outputs | `/data/wsd02/multispan_dvv/` (raw/, raw_nov/, raw_hf/, obs/, outputs_*). Raw DAS data come from the OOI archive (`piweb.ooirsn.uw.edu/das25`; d00, d05); seismometer data come from EarthScope/IRIS FDSN (d04, d04b) |
| Oceanic travel-time model | `resources/taup/ocean6.npz` (built by s52 if missing) |
| Choosing a config | `MSDVV_CONFIG=<config>.yaml` |

Caveats:
- **Live online queries:** USGS ComCat, IRIS and the archive are queried at run time, and catalogues can be revised, so event lists may change slightly.
- **Deleted one-bit correlations:** the one-bit correlation files were deleted as superseded (cleanup log in `docs/NEXT_STEPS.md`). The one-bit comparison rows need s06/s20/s32 to be rerun with the one-bit configs first.

## Main products: Dec 23, 2025 – Jan 28, 2026 (report sections Data, Processing, Within-span, Cross-span, DAS–seismometer)

1. **Download and inventory:** `d00_download.py`, then `s00_inventory.py`, `s01_geometry.py`, `s02_events.py`, `s03_noise.py`, `s05_quality.py`. Run with `config_100m_v2ram.yaml`; the one-bit comparison uses `config_100m_v2.yaml`.
2. **Correlations, then coda dv/v (24/48 h, 0.5 % injection):** `runs/run_final_rebuild.sh`
   - within-span: s06 → s45;
   - cross-span: s20 (`MSDVV_INTERSPAN=interspan_v4_ram`; north `interspan_v6_north`) → s43;
   - setup check s48, comparison table s46, figures s49 / s47.
3. **Cross-span CC tests and long stacks:** `runs/run_crossspan_cc_tests.sh`, `runs/run_crossspan_long.sh`.
4. **DAS–seismometer:** seismometer data via `d04_download_obs.py` + `d04b_fill_obs_gaps.py`; then `runs/run_obsdas_regrid.sh`, `run_obsdas_final.sh`, `run_obsdas_combined.sh`. That is s32 Z/R/T (`OBSDAS_PREP=ram`, `OBSDAS_REGRID=1`) → s33 latency → s37 clusters → s40 coda (`STACK_H=48`; 24 h as `VARIANT=final_ram_24h`). Variant tests: s41.
5. **Report figures:** maps s50 (defaults `WITHIN=ram_b2_48h`, `CROSS=ram_b1_48h`); correlation-function figures s38 (DAS–seismometer), s42 (cross-span; `MSDVV_INTERSPAN=interspan_v4_ram`), s47 (within-span).

## Structure (report section Structure)

- **Tomography:** s56. South cable: `CABLE=south INCLUDE_OBS=1 LAM_S=150 LAM_D=10`. North cable: `CABLE=north`. Checkerboard size: `CHK_N`. The script defaults (`LAM_S=40`, `LAM_D=4`) are **not** the final values.
- **Vs(z):** s63 (pyfetch).

## Follow-up analyses (report section Follow-up analyses)

| Item | Scripts and settings |
|---|---|
| 1 Fault crossings | s57; November earthquake test s75 |
| 2 DAS clock | s58 (evidence figure). Note: `docs/DAS_clock_note.md`. 8-Hz latency s67. Diagnostics: `scripts/adhoc/clock_models.py`, `chunk_times.py`, `clock_track.py` |
| 3 Earthquakes | s59 (Jan 13 section). uw62216067: s51 (pyfetch picks) → s52 (relocation) → s53 / s54 (seismograms, pyfetch) → s55 (map) → s60 (magnitude). Seismometer scan: s61 (pyfetch; first pass default thresholds, second pass `TAG=_v2 P_THR=0.3 S_THR=0.3`. The second-pass thresholds were not logged; they are recovered from the minimum pick probabilities in `obs/dl_scan_picks_v2.csv`. `ASSOC_S` was likely left at its default (4 s)) |
| 4 Drivers | s62 |
| 5 Vs(z) | s63 |
| 6 North cross-span | as the main chain with `MSDVV_INTERSPAN=interspan_v6_north` |
| 7 November | see below |
| 8 Beamforming | see below |

### Item 8: multi-span beamforming

| Report part | Run |
|---|---|
| 8a array response | `s66_array_response.py` |
| 8b noise direction | `TAG=hourly s64_beamform.py`; vs dv/v: `s76_noise_direction_vs_dvv.py` |
| 8c M6.0 P beams | `FEATURE=stalta ONLY=2026-01-16T03 s68_event_beam_compare.py`, then `s68b_m6_beam_map.py`; all events: `FEATURE=stalta s68_event_beam_compare.py`. Frequency-domain M6.0 beams: `EVENT="2026-01-16 03:25:53" TAG=jan16_M6 s64_beamform.py` (also `WEIGHT=none` / `rayleigh`) |
| 8d Rayleigh waves | `NORTH_SIGN=-1 TAG=_sign-1 s69_surface_wave_beam.py` (and `NORTH_SIGN=1 TAG=_sign1` for the polarity comparison) |
| 8e 8-Hz latency | `s67_stream_latency.py` |
| 8f Jan 13 ML 2.0 | `s65b_jan13_envelopes.py`, `s65g_span_stacks.py`, `s65e_span_moveout.py` (writes the channel times), `s65f_jan13_final_locate.py`, `s65h_southloop_stack.py`; rejected versions: s65, s65c, s65d |
| Amity M3.2 | `s73_amity_section.py`, `s74_amity_locate.py` (`MODE=wavefront` / `common`), direction `s77_small_event_direction.py` (needs the s74 `amity_bin_times.csv`), fault crossings `s75`. Data: the four 8-Hz files in `raw_hf/eq20251123` (two after cleanup; the event is in the earlier files) |
| 8g M6.0 back-projection | `ORIGIN=known s70_backproject_m6.py` and `ORIGIN=free s70_backproject_m6.py` |
| 8g+ joint with seismometers | `s71_backproject_m6_joint.py` |
| 8g++ rupture test | `s72_rupture_backproject_m6.py` |
| Polarity residual check | `scripts/adhoc/m6_residuals.py` |

### Item 7: November (Nov 6 – Dec 17, 2025)

1. **Download and reduce:** `d05_stream_reduce.py` with `config_nov.yaml` (writes `raw_nov/reduced`, `raw_nov/crossings`).
2. **Within-span and cross-span:** `runs/run_nov.sh` with `config_nov_ram.yaml` (`merge_substacks: true`). Then:
   - `runs/run_nov_short.sh`: joins the 9.6-min Nov 26 – Dec 2 files with `d06_merge_short_files.py`, runs cross-span on them with `config_nov_ram_short.yaml`, compacts with `s20b_compact_interspan.py`, and redoes s43.
   - **Injection test inside one setup:** `scripts/adhoc/inj_nov20.sh`. It needs copies of the clean variant directories first, as in the session log.
3. **DAS–seismometer:** first `d04_download_obs.py 2025-11-06 2025-12-17`. Then `runs/run_nov_obsdas.sh`, `run_nov_obsdas2.sh`, `run_nov_obsdas3.sh`, with `config_nov_obsdas.yaml` and:
   - `MSDVV_OBSDAS_OUT=/data/wsd02/multispan_dvv/outputs_nov_obsdas`
   - `MSDVV_QC_REF=2025-11-19,2025-11-26`
   - `MSDVV_QC_BLOCK_H=24`

   The input folder `raw_nov/obsdas_input` holds links to the reduced files, with the short files replaced by the joined pieces; see the one-off command in the session.
4. **Comparisons and figures:** `s78_nov_compare.py`, `s79_nov_obsdas_compare.py`, `s80_nov_crossspan_figs.py` (also makes the January figures in the November layout).

## Report assembly

The report was built by editing `docs/report/index.html` in steps, embedding figures as JPEG data URIs. The step scripts are in `docs/report/build/` (`embed_helpers.py` holds the image embedding).
- They expect to run from a folder containing `report/index.html`, and read figures from `/data/wsd02/multispan_dvv/outputs_100m/figs/`.
- `index.html` itself is the source of truth; the step scripts document how each section was added.
- The page is about 15.6 MB, against a 16 MB limit.

## Known reproducibility gaps

- **No version control:** the project is not a git repository, and scripts were edited in place. The current versions reproduce the current results.
- **Variant runs:** a few one-off variant runs were started from the command line. Their settings are listed above and in each script's docstring.
