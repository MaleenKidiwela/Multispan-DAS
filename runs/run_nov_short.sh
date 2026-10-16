#!/usr/bin/env bash
# item 7 fix: cross-span correlations for the joined Nov 26 - Dec 2 pieces (d06), merge into outputs_nov_ram, compact, redo s43
cd /home/seismic/Multispan_DV
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs/nov
O=/data/wsd02/multispan_dvv
set -e
MSDVV_CONFIG=config_nov_ram_short.yaml $PY scripts/s00_inventory.py > $L/short_s00.log 2>&1
cp $O/outputs_nov_ram/events.csv $O/outputs_nov_ram_short/events.csv
for v in interspan_v4_ram interspan_v6_north; do
  MSDVV_CONFIG=config_nov_ram_short.yaml MSDVV_INTERSPAN=$v $PY scripts/s20_interspan_xcorr.py > $L/short_s20_$v.log 2>&1
  tail -n 1 $L/short_s20_$v.log
  n=$(ls $O/outputs_nov_ram_short/$v/*.h5 | wc -l); echo "$v: $n pieces"
  mv $O/outputs_nov_ram_short/$v/*.h5 $O/outputs_nov_ram/$v/
  MSDVV_CONFIG=config_nov_ram.yaml MSDVV_INTERSPAN=$v $PY scripts/s20b_compact_interspan.py > $L/short_compact_$v.log 2>&1; tail -n 1 $L/short_compact_$v.log
done
set +e
xjob() { for inj in 0 0.005; do MSDVV_CONFIG=config_nov_ram.yaml MSDVV_INTERSPAN=$1 VARIANT=${2}_b1_${3}h STACK_H=$3 INJECT=$inj INJECT_AT=2025-11-27 BAND=b1 $PY scripts/s43_interspan_coda.py > $L/s43_${2}_${3}h_inj$inj.log 2>&1 || echo "FAILED s43 $2 $3 $inj"; done; echo "cross $2 $3 h done $(date -u +%H:%M)"; }
export -f xjob; export PY L
printf '%s\n' 'interspan_v4_ram ram 24' 'interspan_v4_ram ram 48' 'interspan_v6_north north 24' 'interspan_v6_north north 48' | xargs -P 2 -I{} bash -c 'xjob {}'
echo "=== ALL DONE $(date -u +%H:%M:%S)"
