#!/usr/bin/env bash
# DAS-station coda dv/v with the floor rule: production (RAM) at 48 h and 24 h, each + injection; then CCF figures
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
for sh in 48 24; do for inj in 0 0.005; do
  echo "=== s40 ram ${sh}h inject=$inj $(date -u +%H:%M:%S)"
  VARIANT=$([ $sh = 48 ] && echo "" || echo final_ram_24h) OBSDAS_PREP=ram STACK_H=$sh INJECT=$inj $PY scripts/s40_obsdas_coda.py > $L/s40_final_${sh}h_inj$inj.log 2>&1 || { echo FAILED; tail -3 $L/s40_final_${sh}h_inj$inj.log; }
done; done
echo "=== s38 $(date -u +%H:%M:%S)"; $PY scripts/s38_obsdas_cluster_ccfs.py > $L/s38_final.log 2>&1
echo "=== ALL DONE $(date -u +%H:%M:%S)"
