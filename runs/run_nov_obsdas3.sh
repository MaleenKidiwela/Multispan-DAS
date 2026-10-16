#!/usr/bin/env bash
# continuation 2: per-file timing QC replaced by 24-h block QC (MSDVV_QC_BLOCK_H=24; the 17-min Nov files are too short),
# then s40 coda dv/v 48 / 24 h, injection 0.5 % at Dec 12 (inside one setup)
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_nov_obsdas.yaml MSDVV_OBSDAS_OUT=/data/wsd02/multispan_dvv/outputs_nov_obsdas MSDVV_QC_REF=2025-11-19,2025-11-26 MSDVV_QC_BLOCK_H=24
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs/nov
for sh in 48 24; do for inj in 0 0.005; do
  echo "=== s40 ${sh}h inj $inj $(date -u +%H:%M:%S)"
  STACK_H=$sh INJECT=$inj INJECT_AT=2025-12-12 VARIANT=final_ram_${sh}h $PY scripts/s40_obsdas_coda.py > $L/od_s40_${sh}h_inj$inj.log 2>&1 || { echo "=== FAILED s40 $sh $inj"; tail -5 $L/od_s40_${sh}h_inj$inj.log; exit 1; }
done; done
grep -v "Warning\|warn" $L/od_s40_48h_inj0.log | tail -14
echo "=== ALL DONE $(date -u +%H:%M:%S)"
