#!/usr/bin/env bash
# final rebuild with regridded DAS (dropped chunks -> zero gaps): re-correlate within- and cross-span, coda dv/v (24/48 h,
# one-bit and RAM, + injection), setup correction, comparison, figures
cd /home/seismic/Multispan_DV
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
echo "=== correlations $(date -u +%H:%M:%S)"
MSDVV_CONFIG=config_100m_v2.yaml $PY scripts/s06_xcorr.py > $L/rb_s06_v2.log 2>&1 &
MSDVV_CONFIG=config_100m_v2ram.yaml $PY scripts/s06_xcorr.py > $L/rb_s06_v2ram.log 2>&1 &
(for v in interspan_v3 interspan_v4_ram; do MSDVV_CONFIG=config_100m_v2.yaml MSDVV_INTERSPAN=$v $PY scripts/s20_interspan_xcorr.py > $L/rb_s20_$v.log 2>&1; done) &
wait
tail -n 1 $L/rb_s06_v2.log $L/rb_s06_v2ram.log $L/rb_s20_interspan_v3.log $L/rb_s20_interspan_v4_ram.log
echo "=== cross-span coda $(date -u +%H:%M:%S)"
rm -rf /data/wsd02/multispan_dvv/outputs_100m_v2/interspan_coda/*
xjob() { for inj in 0 0.005; do MSDVV_CONFIG=config_100m_v2.yaml MSDVV_INTERSPAN=$1 VARIANT=${2}_b1_${3}h STACK_H=$3 INJECT=$inj BAND=b1 $PY scripts/s43_interspan_coda.py > $L/rb_s43_${2}_${3}h_inj$inj.log 2>&1; done; }
wjob() { for inj in 0 0.005; do MSDVV_CONFIG=$1 VARIANT=${2}_${3}_${4}h BAND=$3 STACK_H=$4 INJECT=$inj $PY scripts/s45_within_coda.py > $L/rb_s45_${2}_${3}_${4}h_inj$inj.log 2>&1; done; }
export -f xjob wjob; export PY L
rm -rf /data/wsd02/multispan_dvv/outputs_100m_v2/within_coda/* /data/wsd02/multispan_dvv/outputs_100m_v2ram/within_coda/*
( printf '%s\n' "interspan_v3 onebit 24" "interspan_v3 onebit 48" "interspan_v4_ram ram 24" "interspan_v4_ram ram 48" | xargs -P 2 -I{} bash -c 'xjob {}' ) &
( for c in "config_100m_v2.yaml onebit" "config_100m_v2ram.yaml ram"; do for b in b1 b2; do for sh in 24 48; do echo "$c $b $sh"; done; done; done | xargs -P 3 -I{} bash -c 'wjob {}' ) &
wait
echo "=== post $(date -u +%H:%M:%S)"
$PY scripts/s48_setup_correction.py > $L/rb_s48.log 2>&1; tail -n 25 $L/rb_s48.log
$PY scripts/s46_final_compare.py > $L/rb_s46.log 2>&1; cat $L/rb_s46.log
$PY scripts/s49_final_figs.py > $L/rb_s49.log 2>&1
for b in b2 b1; do MSDVV_CONFIG=config_100m_v2ram.yaml NORM=ram BAND=$b $PY scripts/s47_within_figs.py > $L/rb_s47_$b.log 2>&1; done
echo "=== ALL DONE $(date -u +%H:%M:%S)"
