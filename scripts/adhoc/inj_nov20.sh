cd /home/seismic/Multispan_DV
for v in b1_24 b1_48 b2_48; do b=${v%_*}; sh=${v#*_}
  MSDVV_CONFIG=config_nov_ram.yaml VARIANT=ram_${b}_${sh}h_injNov20 BAND=$b STACK_H=$sh INJECT=0.005 INJECT_AT=2025-11-20 /home/seismic/anaconda3/envs/noisepy2/bin/python scripts/s45_within_coda.py > /data/wsd02/multispan_dvv/logs/nov/s45_${b}_${sh}h_injNov20.log 2>&1 &
done; wait; echo done
