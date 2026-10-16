#!/usr/bin/env bash
# cross-span CC tests, after the final rebuild: 2-km bins, b0 band, 150-s coda (each 24 h + 48 h, clean + injection)
cd /home/seismic/Multispan_DV
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
until grep -q "ALL DONE" $L/final_rebuild_chain.log; do sleep 60; done
echo "=== s20 2-km bins $(date -u +%H:%M:%S)"
MSDVV_CONFIG=config_100m_v2.yaml MSDVV_INTERSPAN=interspan_v5_ram2km $PY scripts/s20_interspan_xcorr.py > $L/cc_s20_2km.log 2>&1; tail -n 1 $L/cc_s20_2km.log
job() { # interspan band coda_len tag sh
  for inj in 0 0.005; do MSDVV_CONFIG=config_100m_v2.yaml MSDVV_INTERSPAN=$1 BAND=$2 CODA_LEN=$3 VARIANT=$4_${5}h STACK_H=$5 INJECT=$inj $PY scripts/s43_interspan_coda.py > $L/cc_s43_$4_${5}h_inj$inj.log 2>&1; done
  echo "done $4 ${5}h $(date -u +%H:%M)"; grep "all paths" $L/cc_s43_$4_${5}h_inj0.log; }
export -f job; export PY L
printf '%s\n' "interspan_v5_ram2km b1 90 ram2km_b1 24" "interspan_v5_ram2km b1 90 ram2km_b1 48" \
              "interspan_v4_ram b0 90 ram_b0 24" "interspan_v4_ram b0 90 ram_b0 48" \
              "interspan_v4_ram b1 150 ram_b1_coda150 24" "interspan_v4_ram b1 150 ram_b1_coda150 48" | xargs -P 3 -I{} bash -c 'job {}'
echo "=== ALL DONE $(date -u +%H:%M:%S)"
