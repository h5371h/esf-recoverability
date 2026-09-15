#!/bin/bash
# Usage: chain_backbone.sh <backbone> [axes]   — train features -> 4 probes -> cell cache -> 4 scorings
BB="$1"; AXES="${2:-}"; source ~/venv/bin/activate; cd ~/v2
F() { grep -v "FutureWarning\|make_standard_montage\|HF_TOKEN\|UserWarning\|_VF.stft"; }
echo "STAGE $BB train_features $(date -u +%T)"
python extract_features.py --tuab ~/tuab/edf --split train --workers 24 --backbone $BB --out features_$BB 2>&1 | F | tail -2
echo "STAGE $BB eval_features $(date -u +%T)"
python extract_features.py --tuab ~/tuab/edf --split eval --workers 24 --backbone $BB --out features_$BB 2>&1 | F | tail -1
for cfg in "mlp 7" "mlp 11" "mlp 13" "linear 7"; do set -- $cfg
  echo "STAGE $BB head_$1_s$2 $(date -u +%T)"
  python train_head.py --tuab ~/tuab/edf --features features_$BB --probe $1 --seed $2 --out head_${BB}_$1_s$2.pt 2>&1 | F | grep "best:\|train [0-9]"
done
echo "STAGE $BB cells $(date -u +%T)"
python extract_cells.py --tuab ~/tuab/edf --head head_${BB}_mlp_s7.pt --workers 24 --backbone $BB --out cells_$BB ${AXES:+--axes $AXES} 2>&1 | F | grep -v "^cell " | tail -3
for cfg in "mlp 7" "mlp 11" "mlp 13" "linear 7"; do set -- $cfg
  python score_cells.py --cells cells_$BB --head head_${BB}_$1_s$2.pt --tag ${BB}_$1_s$2 --out results 2>&1 | F | tail -1
done
echo "STAGE $BB DONE $(date -u +%T)"
