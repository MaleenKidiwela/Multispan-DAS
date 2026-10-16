#!/usr/bin/env bash
# DAS-station with the dropped-chunk gap fix: s32 (RAM, Z/R/T, regrid) -> s40 48 h & 24 h (+ injection) -> s38 figures
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
for c in Z R T; do OBSDAS_REGRID=1 OBSDAS_PREP=ram OBSDAS_COMP=$c $PY scripts/s32_obs_das_xcorr.py > $L/rg_s32_$c.log 2>&1; echo "s32 $c $(tail -n 1 $L/rg_s32_$c.log)"; done
for sh in 48 24; do for inj in 0 0.005; do
  VARIANT=$([ $sh = 48 ] && echo "" || echo final_ram_24h) OBSDAS_PREP=ram STACK_H=$sh INJECT=$inj $PY scripts/s40_obsdas_coda.py > $L/rg_s40_${sh}h_inj$inj.log 2>&1
done; done
$PY scripts/s38_obsdas_cluster_ccfs.py > $L/rg_s38.log 2>&1
echo "=== ALL DONE $(date -u +%H:%M:%S)"
