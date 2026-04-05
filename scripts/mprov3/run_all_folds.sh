#!/usr/bin/env bash
# Train and evaluate every fold (0 .. NUM_FOLDS-1), then run explainers once for the best fold.
#
# Usage:
#   ./scripts/mprov3/run_all_folds.sh
# Env: SKIP_SYNC=1, NUM_FOLDS (default 5), GNN_TRAIN_EPOCHS

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
last=$((nf - 1))

for fold in $(seq 0 "$last"); do
  echo "################################################################################"
  echo "# GINE fold $fold / $last"
  echo "################################################################################"
  "$MPROV3_SCRIPT_DIR/run_gine_fold.sh" "$fold"
done

echo "################################################################################"
echo "# Explainer (best fold from classification / training summaries)"
echo "################################################################################"
"$MPROV3_SCRIPT_DIR/run_explainer_fold.sh"
