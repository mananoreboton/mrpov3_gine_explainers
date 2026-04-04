#!/usr/bin/env bash
# Full mprov3_gine chain (check_all.sh §3) for one CV fold: raw check → build → PyG check →
# visualize one graph → train → evaluate → HTML report.
#
# Usage:
#   ./scripts/mprov3/run_gine_fold.sh <fold_index>
# Env: SKIP_SYNC=1, NUM_FOLDS (default 5), GNN_TRAIN_EPOCHS (default 1)

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

fold="${1:?Usage: $0 <fold_index>}"

mprov3_maybe_uv_sync

nf="${NUM_FOLDS:-5}"

# --- TEMP: check_raw_data_format.py (expected to fail briefly; uncomment to re-enable) ---
# echo "==> check_raw_data_format.py (num_folds=$nf)"
# run_gine_py check_raw_data_format.py --num_folds "$nf"
# --- end TEMP ---

echo "==> build_dataset.py"
run_gine_py build_dataset.py

echo "==> check_PyG_data_format.py (fold_index=$fold)"
run_gine_py check_PyG_data_format.py --num_folds "$nf" --fold_index "$fold"

echo "==> visualize_graphs.py --num_graphs 1"
run_gine_py visualize_graphs.py --num_graphs 1

echo "==> train.py (fold_index=$fold, epochs=${GNN_TRAIN_EPOCHS:-1})"
run_gine_py train.py --num_folds "$nf" --fold_index "$fold" --epochs "${GNN_TRAIN_EPOCHS:-1}"

echo "==> evaluate.py (fold_index=$fold)"
run_gine_py evaluate.py --num_folds "$nf" --fold_index "$fold"

echo "==> create_evaluation_report.py"
run_gine_py create_evaluation_report.py
