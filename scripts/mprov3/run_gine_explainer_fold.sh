#!/usr/bin/env bash
# One fold: full GINE chain (run_gine_fold.sh) then explainer on the best CV fold from summaries.
#
# Usage:
#   ./scripts/mprov3/run_gine_explainer_fold.sh <fold_index>

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

fold="${1:?Usage: $0 <fold_index>}"

"$MPROV3_SCRIPT_DIR/run_gine_fold.sh" "$fold"
"$MPROV3_SCRIPT_DIR/run_explainer_fold.sh"
