#!/usr/bin/env bash
# mprov3_explainer: run all explainers on the best CV fold (from GNN summaries), then mask PNGs.
# Checkpoint and dataset are read from mprov3_gine/results; explanations under
# mprov3_explainer/results/folds/fold_<k>/explanations/.
#
# Usage:
#   ./scripts/mprov3/run_explainer_fold.sh
# Env: SKIP_SYNC=1, FOLD_METRIC=test_accuracy|train_accuracy (default test_accuracy)

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

mprov3_maybe_uv_sync

fold_metric="${FOLD_METRIC:-test_accuracy}"
ra=(--results_root "$GNN_DIR/results")
if [[ "$fold_metric" != "test_accuracy" ]]; then
  ra+=(--fold_metric "$fold_metric")
fi

echo "==> run_explanations.py (--fold_metric $fold_metric)"
run_mex_py scripts/run_explanations.py "${ra[@]}"

echo "==> generate_visualizations.py"
run_mex_py scripts/generate_visualizations.py
