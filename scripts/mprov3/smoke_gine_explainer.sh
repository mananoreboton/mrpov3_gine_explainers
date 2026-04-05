#!/usr/bin/env bash
# Minimal end-to-end check: full GINE §3 for fold 0 (tiny train epochs) + one explainer graph + viz.
# Mirrors check_all.sh steps 3–4 (mprov3_gine + mprov3_explainer), without the defaults package check.
#
# Usage:
#   ./scripts/mprov3/smoke_gine_explainer.sh [-m|--include-misclassified]
# Env: SKIP_SYNC=1, NUM_FOLDS (default 5), GNN_TRAIN_EPOCHS (default 1), EXPLAINERS (default GNNEXPL)

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

MPROV3_ARGS=()
mprov3_strip_include_misclassified_flags "$@"
mprov3_set_positional_from_mprov3_args

if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [-m|--include-misclassified]" >&2
  exit 1
fi

mprov3_maybe_uv_sync

nf="${NUM_FOLDS:-5}"
fold=0

# --- TEMP: check_raw_data_format.py (expected to fail briefly; uncomment to re-enable) ---
# echo "==> check_raw_data_format.py (num_folds=$nf)"
# run_gine_py check_raw_data_format.py --num_folds "$nf"
# --- end TEMP ---

echo "==> build_dataset.py"
run_gine_py build_dataset.py

echo "==> check_PyG_data_format.py (fold_index=$fold)"
run_gine_py check_PyG_data_format.py --num_folds "$nf" --fold_index "$fold"

echo "==> visualize_graphs.py --num-graphs-by-fold 1"
run_gine_py visualize_graphs.py --num-graphs-by-fold 1

echo "==> train.py (fold_index=$fold, epochs=${GNN_TRAIN_EPOCHS:-1})"
run_gine_py train.py --num_folds "$nf" --fold_index "$fold" --epochs "${GNN_TRAIN_EPOCHS:-1}"

echo "==> evaluate.py (fold_index=$fold)"
run_gine_py evaluate.py --num_folds "$nf" --fold_index "$fold"

echo "==> create_evaluation_report.py"
run_gine_py create_evaluation_report.py

mprov3_build_explain_cli
MEX_MISCLASS_ARGS=()
mprov3_misclassified_arg

echo "==> run_explanations.py --max_graphs 1 (smoke)"
run_mex_py scripts/run_explanations.py \
  "${EXPLAIN_CLI_ARGS[@]}" \
  ${MEX_MISCLASS_ARGS[@]+"${MEX_MISCLASS_ARGS[@]}"} \
  --results_root "$GNN_DIR/results" \
  --num_folds "$nf" \
  --fold_index "$fold" \
  --max_graphs 1

echo "==> generate_visualizations.py"
run_mex_py scripts/generate_visualizations.py "${EXPLAIN_CLI_ARGS[@]}"
