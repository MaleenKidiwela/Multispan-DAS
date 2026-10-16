#!/usr/bin/env bash
# OBS-DAS correlations with true per-chunk DAS times, then timing QC, coda dv/v (+ injection check)
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
for c in Z R T; do
  echo "=== s32 $c $(date -u +%H:%M:%S)"; OBSDAS_COMP=$c $PY scripts/s32_obs_das_xcorr.py > $L/obsdas_xcorr_true_$c.log 2>&1 || { echo "=== FAILED s32 $c"; exit 1; }
done
echo "=== QC $(date -u +%H:%M:%S)"
$PY -c "
import sys; sys.path.insert(0, '.')
from msdvv.obsdas import file_timing_qc
q = file_timing_qc('T'); print(q.status.value_counts().to_dict()); print('good+jump hours', int(q[q.status != 'bad'].nh.sum()), 'of', int(q.nh.sum()))
print(q[q.status != 'good'][['file', 'nh', 'setup', 'shift_s', 'cc', 'cc0', 'status']].to_string())
" 2>&1 | grep -v Warn
echo "=== s40 $(date -u +%H:%M:%S)"; $PY scripts/s40_obsdas_coda.py > $L/obsdas_coda_true.log 2>&1 || { echo "=== FAILED s40"; exit 1; }
echo "=== s40 inject $(date -u +%H:%M:%S)"; INJECT=0.005 $PY scripts/s40_obsdas_coda.py > $L/obsdas_coda_true_inj.log 2>&1 || { echo "=== FAILED inject"; exit 1; }
echo "=== ALL DONE $(date -u +%H:%M:%S)"
