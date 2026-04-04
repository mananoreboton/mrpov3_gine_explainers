#!/usr/bin/env bash
# Run full GINE + explainer flow for every fold index in 0 .. NUM_FOLDS-1.
#
# Usage:
#   ./scripts/mprov3/run_all_folds.sh [-m|--include-misclassified]
# Env: SKIP_SYNC=1, NUM_FOLDS (default 5), GNN_TRAIN_EPOCHS, EXPLAINERS

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

nf="${NUM_FOLDS:-5}"
last=$((nf - 1))

for fold in $(seq 0 "$last"); do
  echo "################################################################################"
  echo "# Fold $fold / $last"
  echo "################################################################################"
  "$MPROV3_SCRIPT_DIR/run_gine_explainer_fold.sh" "$fold"
done
