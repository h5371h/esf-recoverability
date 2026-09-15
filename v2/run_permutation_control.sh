#!/bin/bash
# Usage: run_permutation_control.sh <backbone> [n_perm]  -> results/permutation_control_<backbone>.txt
BB="$1"; N="${2:-10}"; source ~/venv/bin/activate; cd ~/v2; OUT=results/permutation_control_$BB.txt; : > $OUT
for i in $(seq 1 $N); do
  python train_head.py --tuab ~/tuab/edf --features features_$BB --probe mlp --seed $((100+i)) --permute-labels --fixed-scale-n 5 --out /tmp/perm_${BB}_$i.pt > /tmp/perm_${BB}_$i.log 2>&1
  python eval_probe.py features_$BB /tmp/perm_${BB}_$i.pt 2>/dev/null | tail -1 >> $OUT
done
python train_head.py --tuab ~/tuab/edf --features features_$BB --probe linear --seed 101 --permute-labels --fixed-scale-n 5 --out /tmp/perm_${BB}_lin.pt > /tmp/perm_${BB}_lin.log 2>&1
python eval_probe.py features_$BB /tmp/perm_${BB}_lin.pt 2>/dev/null | tail -1 >> $OUT
echo "PERMUTATION_CONTROL_DONE $BB" >> $OUT; cat $OUT
