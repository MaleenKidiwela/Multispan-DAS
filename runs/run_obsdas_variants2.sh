#!/usr/bin/env bash
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
variant() { n=$1; shift; for inj in 0 0.005; do echo "=== s40 $n inject=$inj $(date -u +%H:%M:%S)"; env STACK_H=24 VARIANT=$n INJECT=$inj "$@" $PY scripts/s40_obsdas_coda.py > $L/obsdas_var_${n}_inj$inj.log 2>&1 || { echo "=== FAILED $n $inj"; tail -5 $L/obsdas_var_${n}_inj$inj.log; }; done; }
variant coda60 CODA_LEN=60
variant reject2 HOUR_REJECT=2
variant combo OBSDAS_PREP=ram SUB_BAND=0.15,0.25
echo "=== s41 $(date -u +%H:%M:%S)"; $PY scripts/s41_obsdas_variants.py 2>&1 | grep -v Warn
echo "=== ALL DONE $(date -u +%H:%M:%S)"
