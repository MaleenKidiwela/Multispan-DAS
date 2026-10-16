#!/usr/bin/env bash
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
for s in s01_geometry s02_events s03_noise s04_noise_figs s05_quality s06_xcorr s07_gathers s08_convergence s09_dvv s10_analysis s11_environment s12_velocity_profile; do
  echo "=== $s $(date -u +%H:%M:%S)"
  $PY scripts/$s.py > /data/wsd02/multispan_dvv/logs/$s.log 2>&1
  rc=$?
  grep -v -E "Warning|warn\(|nanmean|nanmedian|ret = |return _m|x = um|avg = |^\[Parallel" /data/wsd02/multispan_dvv/logs/$s.log | tail -40
  if [ $rc -ne 0 ]; then echo "=== FAILED $s (rc=$rc)"; exit $rc; fi
done
echo "=== ALL DONE $(date -u +%H:%M:%S)"
