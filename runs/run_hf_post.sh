#!/usr/bin/env bash
# waits for the streaming correlation, then runs the analysis steps for the 1-2 Hz config
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_hf.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
while kill -0 838291 2>/dev/null; do sleep 30; done
echo "=== stream finished $(date -u +%H:%M:%S)"; tail -2 /data/wsd02/multispan_dvv/logs/hf_stream.log
for s in s07_gathers s08_convergence s09_dvv s10_analysis s11_environment s12_velocity_profile; do
  echo "=== $s $(date -u +%H:%M:%S)"
  $PY scripts/$s.py > /data/wsd02/multispan_dvv/logs/hf_$s.log 2>&1; rc=$?
  grep -v -E "Warning|warn\(|nanmean|nanmedian|ret = |return _m|x = um|avg = |^\[Parallel" /data/wsd02/multispan_dvv/logs/hf_$s.log | tail -40
  if [ $rc -ne 0 ]; then echo "=== FAILED $s (rc=$rc)"; exit $rc; fi
done
echo "=== ALL DONE $(date -u +%H:%M:%S)"
