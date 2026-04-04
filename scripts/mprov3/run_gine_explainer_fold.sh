#!/usr/bin/env bash
# One fold: full GINE chain (run_gine_fold.sh) then explainer chain (run_explainer_fold.sh).
#
# Usage:
#   ./scripts/mprov3/run_gine_explainer_fold.sh [-m|--include-misclassified] <fold_index>

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

MPROV3_ARGS=()
mprov3_strip_include_misclassified_flags "$@"
mprov3_set_positional_from_mprov3_args

fold="${1:?Usage: $0 [-m|--include-misclassified] <fold_index>}"

"$MPROV3_SCRIPT_DIR/run_gine_fold.sh" "$fold"
"$MPROV3_SCRIPT_DIR/run_explainer_fold.sh" "$fold"
