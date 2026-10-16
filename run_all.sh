#!/usr/bin/env bash
# Full pipeline (~10 min on 48 cores for the Jan snapshot pilot). Uses the noisepy2 env (base env h5py is broken).
# Continuous 100-m stream:  MSDVV_CONFIG=config_100m.yaml ./run_all.sh   (downloads first)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/seismic/anaconda3/envs/noisepy2/bin/python}
$PY tests/test_xcorr.py
$PY tests/test_dvv.py
if [[ "${MSDVV_CONFIG:-}" == *100m* ]]; then $PY scripts/d00_download.py; fi
for s in s00_inventory s01_geometry s02_events s03_noise s04_noise_figs s05_quality \
         s06_xcorr s07_gathers s08_convergence s08b_coda_fixes s09_dvv s10_analysis s11_environment s12_velocity_profile; do
  # s08b (coda remedies) is a snapshot-pilot test; s11 needs the long merged series (100-m config)
  [[ "$s" == s08b_coda_fixes && "${MSDVV_CONFIG:-}" == *100m* ]] && continue
  [[ ( "$s" == s11_environment || "$s" == s12_velocity_profile ) && "${MSDVV_CONFIG:-}" != *100m* ]] && continue
  echo "=== $s"; $PY scripts/$s.py
done
