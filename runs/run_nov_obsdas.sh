#!/usr/bin/env bash
# Item 7, third product: November DAS-seismometer (HYSB1, HYS14) coda dv/v with the January pipeline
# (s32 Z one-bit for the latency calibration + Z/R/T RAM -> s33 latency per setup -> timing QC (Nov 19-26 reference)
#  -> s37 clusters -> s40 coda dv/v 24 / 48 h, injection 0.5 % at Dec 12 inside one setup)
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_nov_obsdas.yaml MSDVV_OBSDAS_OUT=/data/wsd02/multispan_dvv/outputs_nov_obsdas MSDVV_QC_REF=2025-11-19,2025-11-26
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs/nov
guard() { local f=$(df -BG /data/wsd02 | tail -1 | awk '{print $4}' | tr -d G); [ "$f" -lt 15 ] && { echo "=== DISK GUARD: ${f} GB free - stopping"; exit 2; }; }
run() { guard; echo "=== $1 $(date -u +%H:%M:%S) disk $(df -h /data/wsd02 | tail -1 | awk '{print $4}') free"; shift; "$@"; rc=$?; [ $rc -ne 0 ] && { echo "=== FAILED rc=$rc"; exit $rc; }; }
run s00 sh -c "$PY scripts/s00_inventory.py > $L/od_s00.log 2>&1"
cp /data/wsd02/multispan_dvv/outputs_nov_ram/events.csv $MSDVV_OBSDAS_OUT/events.csv
run s32_Z_onebit sh -c "OBSDAS_COMP=Z OBSDAS_PREP=onebit $PY scripts/s32_obs_das_xcorr.py > $L/od_s32_Z_onebit.log 2>&1"
for c in Z R T; do run s32_${c}_ram sh -c "OBSDAS_COMP=$c OBSDAS_PREP=ram $PY scripts/s32_obs_das_xcorr.py > $L/od_s32_${c}_ram.log 2>&1"; done
run s33_latency sh -c "$PY scripts/s33_obs_das_dvv.py > $L/od_s33.log 2>&1"
grep "DAS latency per setup" $L/od_s33.log
run s37_clusters sh -c "$PY scripts/s37_obsdas_combined.py > $L/od_s37.log 2>&1"
for sh in 48 24; do for inj in 0 0.005; do
  run s40_${sh}h_inj$inj sh -c "STACK_H=$sh INJECT=$inj INJECT_AT=2025-12-12 VARIANT=final_ram_${sh}h $PY scripts/s40_obsdas_coda.py > $L/od_s40_${sh}h_inj$inj.log 2>&1"
done; done
grep -v "Warning\|warn" $L/od_s40_48h_inj0.log | tail -14
echo "=== ALL DONE $(date -u +%H:%M:%S)"
