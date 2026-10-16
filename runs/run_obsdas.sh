#!/usr/bin/env bash
# OBS-DAS only (HYSB1/HYS14 x south fibre): s32 correlations, then s33 dv/v (early window)
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
echo "=== s32 $(date -u +%H:%M:%S)"; $PY scripts/s32_obs_das_xcorr.py > $L/obsdas_xcorr.log 2>&1 || { echo "=== FAILED s32"; exit 1; }
echo "=== s33 $(date -u +%H:%M:%S)"; OBSDAS_WINDOW=early $PY scripts/s33_obs_das_dvv.py > $L/obsdas_dvv.log 2>&1 || { echo "=== FAILED s33"; exit 1; }
echo "=== ALL DONE $(date -u +%H:%M:%S)"
