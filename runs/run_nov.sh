#!/usr/bin/env bash
# Item 7: November - December 2025 (Nov 6 - Dec 17) with the final January settings (config_nov_ram.yaml).
# v2 (after the first attempt filled the disk): within-span substacks merged per hour (merge_substacks), cross-span
# resumes and skips files without usable windows, disk guard (stop below 15 GB free), stop on any failure.
cd /home/seismic/Multispan_DV
export MSDVV_CONFIG=config_nov_ram.yaml
PY=/home/seismic/anaconda3/envs/noisepy2/bin/python
L=/data/wsd02/multispan_dvv/logs/nov
mkdir -p $L
guard() { local f=$(df -BG /data/wsd02 | tail -1 | awk '{print $4}' | tr -d G); [ "$f" -lt 15 ] && { echo "=== DISK GUARD: only ${f} GB free - stopping"; exit 2; }; }
run() { guard; echo "=== $1 $(date -u +%H:%M:%S) disk $(df -h /data/wsd02 | tail -1 | awk '{print $4}') free"; shift; "$@"; rc=$?; [ $rc -ne 0 ] && { echo "=== FAILED rc=$rc"; exit $rc; }; }
# disk watcher: kill this chain's python jobs if free space falls below 10 GB
( while sleep 60; do f=$(df -BG /data/wsd02 | tail -1 | awk '{print $4}' | tr -d G); if [ "$f" -lt 10 ]; then echo "=== WATCHER: ${f} GB free - killing nov jobs"; pkill -f "MSDVV_NOV_CHAIN" ; ps aux | grep "[c]onfig_nov_ram\|[s]06_xcorr.py\|[s]20_interspan_xcorr.py\|[s]4[35]_" | awk '{print $2}' | xargs -r kill; exit; fi; done ) &
WATCH=$!
run s06_within sh -c "$PY scripts/s06_xcorr.py > $L/s06.log 2>&1"
run s20_south sh -c "RESUME=1 MSDVV_INTERSPAN=interspan_v4_ram $PY scripts/s20_interspan_xcorr.py > $L/s20_interspan_v4_ram.log 2>&1"
run s20_north sh -c "RESUME=1 MSDVV_INTERSPAN=interspan_v6_north $PY scripts/s20_interspan_xcorr.py > $L/s20_interspan_v6_north.log 2>&1"
tail -n 3 $L/s06.log $L/s20_interspan_v4_ram.log $L/s20_interspan_v6_north.log
wjob() { for inj in 0 0.005; do VARIANT=ram_${1}_${2}h BAND=$1 STACK_H=$2 INJECT=$inj INJECT_AT=2025-11-27 $PY scripts/s45_within_coda.py > $L/s45_${1}_${2}h_inj$inj.log 2>&1 || echo "FAILED s45 $1 $2 $inj"; done; echo "within $1 $2 h done $(date -u +%H:%M)"; }
xjob() { for inj in 0 0.005; do MSDVV_INTERSPAN=$1 VARIANT=${2}_b1_${3}h STACK_H=$3 INJECT=$inj INJECT_AT=2025-11-27 BAND=b1 $PY scripts/s43_interspan_coda.py > $L/s43_${2}_${3}h_inj$inj.log 2>&1 || echo "FAILED s43 $2 $3 $inj"; done; echo "cross $2 $3 h done $(date -u +%H:%M)"; }
export -f wjob xjob; export PY L
run coda sh -c "( printf '%s\n' 'b1 24' 'b1 48' 'b2 24' 'b2 48' | xargs -P 2 -I{} bash -c 'wjob {}' ) & ( printf '%s\n' 'interspan_v4_ram ram 24' 'interspan_v4_ram ram 48' 'interspan_v6_north north 24' 'interspan_v6_north north 48' | xargs -P 2 -I{} bash -c 'xjob {}' ) & wait"
for f in $L/s45_*_inj0.log $L/s43_*_inj0.log; do echo "--- $f"; tail -n 4 $f; done
kill $WATCH 2>/dev/null
echo "=== ALL DONE $(date -u +%H:%M:%S) disk $(df -h /data/wsd02 | tail -1 | awk '{print $4}') free"
