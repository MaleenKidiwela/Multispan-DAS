#!/usr/bin/env bash
# within-span coda dv/v (s45): one-bit vs running-absolute-mean x b1/b2 x 24/48 h; each clean run, then injection (reusing its selection)
cd /home/seismic/Multispan_DV
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
job() { # config tag band stack_h
  for inj in 0 0.005; do
    MSDVV_CONFIG=$1 VARIANT=${2}_${3}_${4}h BAND=$3 STACK_H=$4 INJECT=$inj $PY scripts/s45_within_coda.py > $L/s45_${2}_${3}_${4}h_inj$inj.log 2>&1
  done; echo "done $2 $3 ${4}h $(date -u +%H:%M)"; }
export -f job; export PY L
for c in "config_100m_v2.yaml onebit" "config_100m_v2ram.yaml ram"; do for b in b1 b2; do for sh in 24 48; do echo "$c $b $sh"; done; done; done | xargs -P 3 -I{} bash -c 'job {}'
echo "=== ALL DONE $(date -u +%H:%M:%S)"
