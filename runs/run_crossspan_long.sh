#!/usr/bin/env bash
# cross-span coda dv/v with longer back-stacked windows (96 h, 168 h; step 1 h), RAM b1, clean + injection; after the CC tests
cd /home/seismic/Multispan_DV
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
until grep -q "ALL DONE" $L/crossspan_cc_tests.log; do sleep 60; done
for sh in 96 168; do for inj in 0 0.005; do
  MSDVV_CONFIG=config_100m_v2.yaml MSDVV_INTERSPAN=interspan_v4_ram BAND=b1 VARIANT=ram_b1_${sh}h STACK_H=$sh INJECT=$inj $PY scripts/s43_interspan_coda.py > $L/long_s43_${sh}h_inj$inj.log 2>&1
done; echo "done ${sh}h $(date -u +%H:%M)"; grep "all paths" $L/long_s43_${sh}h_inj0.log; done
echo "=== ALL DONE $(date -u +%H:%M:%S)"
