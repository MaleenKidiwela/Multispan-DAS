#!/usr/bin/env bash
# CC-improvement tests for the seismometer-DAS coda dv/v (README "Raising the seismometer-DAS CC")
set -uo pipefail
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
for c in Z R T; do
  echo "=== s32 ram $c $(date -u +%H:%M:%S)"; OBSDAS_PREP=ram OBSDAS_COMP=$c $PY scripts/s32_obs_das_xcorr.py > $L/obsdas_xcorr_ram_$c.log 2>&1 || { echo "=== FAILED s32 ram $c"; exit 1; }
done
variant() {  # name, env...
  n=$1; shift
  for inj in 0 0.005; do
    echo "=== s40 $n inject=$inj $(date -u +%H:%M:%S)"
    env STACK_H=24 VARIANT=$n INJECT=$inj "$@" $PY scripts/s40_obsdas_coda.py > $L/obsdas_var_${n}_inj$inj.log 2>&1 || { echo "=== FAILED $n $inj"; tail -5 $L/obsdas_var_${n}_inj$inj.log; }
  done
}
variant ref
variant ram OBSDAS_PREP=ram
variant reject3 HOUR_REJECT=3
variant band15_25 SUB_BAND=0.15,0.25
variant coda60 CODA_LEN=60
variant stack48 STACK_H=48
echo "=== s41 $(date -u +%H:%M:%S)"; $PY scripts/s41_obsdas_variants.py 2>&1 | grep -v Warn
echo "=== ALL DONE $(date -u +%H:%M:%S)"
