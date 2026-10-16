#!/usr/bin/env bash
# within-span re-correlation with the DAS timing fix: one-bit (v2) then running-absolute-mean (v2ram)
set -uo pipefail
cd /home/seismic/Multispan_DV
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
for c in config_100m_v2.yaml config_100m_v2ram.yaml; do
  echo "=== s06 $c $(date -u +%H:%M:%S)"
  MSDVV_CONFIG=$c $PY scripts/s06_xcorr.py > $L/s06_${c%.yaml}.log 2>&1 || { echo "=== FAILED s06 $c"; tail -5 $L/s06_${c%.yaml}.log; exit 1; }
  tail -2 $L/s06_${c%.yaml}.log
done
echo "=== ALL DONE $(date -u +%H:%M:%S)"
