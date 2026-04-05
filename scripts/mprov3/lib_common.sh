#!/usr/bin/env bash
# Shared helpers for MProV3 GINE + explainer orchestration (sourced by other scripts).
# shellcheck disable=SC2034  # many vars are used by callers after source

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$MPROV3_SCRIPT_DIR/../.." && pwd)"
GNN_DIR="$ROOT/mprov3_gine"
MEX_DIR="$ROOT/mprov3_explainer"

# Optional env: SKIP_SYNC=1, GNN_TRAIN_EPOCHS, NUM_FOLDS (default 5), FOLD_METRIC for run_explainer_fold.sh.

mprov3_maybe_uv_sync() {
  if [[ "${SKIP_SYNC:-}" == "1" ]]; then
    echo "[mprov3] SKIP_SYNC=1 — skipping uv sync"
    return 0
  fi
  local pkg
  for pkg in mprov3_gine_explainer_defaults mprov3_gine mprov3_explainer; do
    echo "[mprov3] uv sync in $pkg"
    (cd "$ROOT/$pkg" && uv sync)
  done
}

run_gine_py() {
  (cd "$GNN_DIR" && uv run python "$@")
}

run_mex_py() {
  (cd "$MEX_DIR" && uv run python "$@")
}

mprov3_fold_cli() {
  local nf="${NUM_FOLDS:-5}"
  echo --num_folds "$nf" --fold_index "${1:?fold index required}"
}
