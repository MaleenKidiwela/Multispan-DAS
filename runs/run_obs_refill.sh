#!/usr/bin/env bash
# after d04b gap fill: OBS-OBS (s30, s31 coda+early) then OBS-DAS (s32, s33 early)
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
run() { echo "=== $1 $(date -u +%H:%M:%S)"; shift; "$@"; rc=$?; [ $rc -ne 0 ] && { echo "=== FAILED rc=$rc"; exit $rc; }; }
run s30 sh -c "$PY scripts/s30_obs_xcorr.py > $L/obs_xcorr.log 2>&1"
run s31_coda sh -c "OBS_WINDOW=coda $PY scripts/s31_obs_dvv.py > $L/obs_dvv.log 2>&1"
run s31_early sh -c "OBS_WINDOW=early $PY scripts/s31_obs_dvv.py > $L/obs_dvv_early.log 2>&1"
run s32 sh -c "$PY scripts/s32_obs_das_xcorr.py > $L/obsdas_xcorr.log 2>&1"
run s33 sh -c "OBSDAS_WINDOW=early $PY scripts/s33_obs_das_dvv.py > $L/obsdas_dvv.log 2>&1"
echo "=== ALL DONE $(date -u +%H:%M:%S)"
