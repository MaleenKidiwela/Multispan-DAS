#!/usr/bin/env bash
# continuation of run_nov_obsdas.sh: s37 needs the one-bit R and T correlations too (as in January)
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_nov_obsdas.yaml MSDVV_OBSDAS_OUT=/data/wsd02/multispan_dvv/outputs_nov_obsdas MSDVV_QC_REF=2025-11-19,2025-11-26
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs/nov
guard() { local f=$(df -BG /data/wsd02 | tail -1 | awk '{print $4}' | tr -d G); [ "$f" -lt 15 ] && { echo "=== DISK GUARD: ${f} GB free - stopping"; exit 2; }; }
run() { guard; echo "=== $1 $(date -u +%H:%M:%S) disk $(df -h /data/wsd02 | tail -1 | awk '{print $4}') free"; shift; "$@"; rc=$?; [ $rc -ne 0 ] && { echo "=== FAILED rc=$rc"; exit $rc; }; }
for c in R T; do run s32_${c}_onebit sh -c "OBSDAS_COMP=$c OBSDAS_PREP=onebit $PY scripts/s32_obs_das_xcorr.py > $L/od_s32_${c}_onebit.log 2>&1"; done
run s37_clusters sh -c "$PY scripts/s37_obsdas_combined.py > $L/od_s37.log 2>&1"
grep -v "Warning\|warn" $L/od_s37.log | grep "converging\|cluster" | head -8
for sh in 48 24; do for inj in 0 0.005; do
  run s40_${sh}h_inj$inj sh -c "STACK_H=$sh INJECT=$inj INJECT_AT=2025-12-12 VARIANT=final_ram_${sh}h $PY scripts/s40_obsdas_coda.py > $L/od_s40_${sh}h_inj$inj.log 2>&1"
done; done
grep -v "Warning\|warn" $L/od_s40_48h_inj0.log | tail -14
echo "=== ALL DONE $(date -u +%H:%M:%S)"
