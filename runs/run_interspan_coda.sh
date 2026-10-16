#!/usr/bin/env bash
# cross-span coda dv/v (s43): one-bit vs running-absolute-mean x 24/48 h; each clean run, then injection (reusing its selection)
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_100m_v2.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs
job() { # interspan variant tag stack_h
  for inj in 0 0.005; do
    MSDVV_INTERSPAN=$1 VARIANT=${2}_b1_${3}h STACK_H=$3 INJECT=$inj BAND=b1 $PY scripts/s43_interspan_coda.py > $L/s43_${2}_b1_${3}h_inj$inj.log 2>&1
  done; echo "done $2 ${3}h $(date -u +%H:%M)"; }
export -f job; export PY L
printf '%s\n' "interspan_v3 onebit 24" "interspan_v3 onebit 48" "interspan_v4_ram ram 24" "interspan_v4_ram ram 48" | xargs -P 2 -I{} bash -c 'job {}'
echo "=== ALL DONE $(date -u +%H:%M:%S)"
