#!/usr/bin/env bash
# Minimal end-to-end check: GINE train/eval fold 0 + all explainers (best fold) + mask PNGs.
#
# Usage:
#   ./scripts/mprov3/smoke_gine_explainer.sh
# Env: SKIP_SYNC=1, NUM_FOLDS (default 5), GNN_TRAIN_EPOCHS (default 1)

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

if [[ $# -gt 0 ]]; then
  echo "Usage: $0" >&2
  exit 1
fi

mprov3_maybe_uv_sync

nf="${NUM_FOLDS:-5}"
fold=0

echo "==> build_dataset.py"
run_gine_py build_dataset.py

echo "==> check_PyG_data_format.py (fold_index=$fold)"
run_gine_py check_PyG_data_format.py --num_folds "$nf" --fold_index "$fold"

echo "==> visualize_graphs.py --num-graphs-by-fold 1"
run_gine_py visualize_graphs.py --num-graphs-by-fold 1

echo "==> train.py (fold_index=$fold, epochs=${GNN_TRAIN_EPOCHS:-1})"
run_gine_py train.py --num_folds "$nf" --fold_index "$fold" --epochs "${GNN_TRAIN_EPOCHS:-1}"

echo "==> classify.py (fold_index=$fold)"
run_gine_py classify.py --num_folds "$nf" --fold_index "$fold"

echo "==> create_classification_report.py"
run_gine_py create_classification_report.py

echo "==> run_explanations.py (best fold = test accuracy from classify.py summary)"
run_mex_py scripts/run_explanations.py --results_root "$GNN_DIR/results"

echo "==> generate_visualizations.py"
run_mex_py scripts/generate_visualizations.py
