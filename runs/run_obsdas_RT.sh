#!/usr/bin/env bash
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
for c in R T; do
  echo "=== s32 comp $c $(date -u +%H:%M:%S)"
  OBSDAS_COMP=$c $PY scripts/s32_obs_das_xcorr.py > /data/wsd02/multispan_dvv/logs/obsdas_xcorr_$c.log 2>&1 || { echo "=== FAILED $c"; exit 1; }
done
echo "=== ALL DONE $(date -u +%H:%M:%S)"
