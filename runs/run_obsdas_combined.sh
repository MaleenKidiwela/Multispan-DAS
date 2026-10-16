#!/usr/bin/env bash
# production Z+R+T neighbour-stacked OBS-DAS dv/v, then injection checks (0.2 %, 0.5 % at Jan 1)
set -uo pipefail
cd /home/seismic/Multispan_DV
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
echo "=== s37 $(date -u +%H:%M:%S)"; $PY scripts/s37_obsdas_combined.py > $L/obsdas_combined.log 2>&1 || { echo "=== FAILED s37"; exit 1; }
for e in 0.002 0.005; do
  echo "=== s37 inject $e $(date -u +%H:%M:%S)"; INJECT=$e $PY scripts/s37_obsdas_combined.py > $L/obsdas_combined_inj$e.log 2>&1 || { echo "=== FAILED inject $e"; exit 1; }
done
echo "=== ALL DONE $(date -u +%H:%M:%S)"
