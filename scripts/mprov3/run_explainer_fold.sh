#!/usr/bin/env bash
# mprov3_explainer for one fold: run_explanations.py (full test set) then generate_visualizations.py.
# Checkpoint: latest trainings/ run, or optional trainings_timestamp (folder name under results/trainings/).
#
# Usage:
#   ./scripts/mprov3/run_explainer_fold.sh [-m|--include-misclassified] <fold_index> [trainings_timestamp]
# Env: SKIP_SYNC=1, NUM_FOLDS (default 5), EXPLAINERS (default GNNEXPL; space-separated for several)
# Note: run_explanations loads checkpoint/dataset from mprov3_gine/results; explanations are written
# under mprov3_explainer/results/explanations/.

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "$MPROV3_SCRIPT_DIR/lib_common.sh"

MPROV3_ARGS=()
mprov3_strip_include_misclassified_flags "$@"
mprov3_set_positional_from_mprov3_args

fold="${1:?Usage: $0 [-m|--include-misclassified] <fold_index> [trainings_timestamp]}"
train_ts="${2:-}"

mprov3_maybe_uv_sync

nf="${NUM_FOLDS:-5}"
mprov3_build_explain_cli
MEX_MISCLASS_ARGS=()
mprov3_misclassified_arg

args=(--results_root "$GNN_DIR/results" --num_folds "$nf" --fold_index "$fold")
if [[ -n "$train_ts" ]]; then
  args+=(--trainings_timestamp "$train_ts")
fi

echo "==> run_explanations.py"
mprov3_run_explanations_capture_ts "${EXPLAIN_CLI_ARGS[@]}" "${MEX_MISCLASS_ARGS[@]}" "${args[@]}"

echo "==> generate_visualizations.py --timestamp $EXPL_TS"
run_mex_py scripts/generate_visualizations.py --timestamp "$EXPL_TS" "${EXPLAIN_CLI_ARGS[@]}"
