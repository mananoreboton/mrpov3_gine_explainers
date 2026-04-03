#!/usr/bin/env bash
# One fold: full GINE chain (run_gine_fold.sh) then explainer chain (run_explainer_fold.sh)
# with the training timestamp captured so the explainer matches this fold's checkpoint.
#
# Usage:
#   ./scripts/mprov3/run_gine_explainer_fold.sh [-m|--include-misclassified] <fold_index>

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

mprov3_strip_include_misclassified_flags "$@"
set -- "${MPROV3_ARGS[@]}"

fold="${1:?Usage: $0 [-m|--include-misclassified] <fold_index>}"

"$MPROV3_SCRIPT_DIR/run_gine_fold.sh" "$fold"
TRAIN_TS="$(mprov3_latest_training_ts)"
echo "Using trainings_timestamp=$TRAIN_TS for explainer"
"$MPROV3_SCRIPT_DIR/run_explainer_fold.sh" "$fold" "$TRAIN_TS"
