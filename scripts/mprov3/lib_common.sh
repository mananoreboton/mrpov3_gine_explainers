#!/usr/bin/env bash
# Shared helpers for MProV3 GINE + explainer orchestration (sourced by other scripts).
# shellcheck disable=SC2034  # many vars are used by callers after source

set -euo pipefail

MPROV3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$MPROV3_SCRIPT_DIR/../.." && pwd)"
GNN_DIR="$ROOT/mprov3_gine"
MEX_DIR="$ROOT/mprov3_explainer"

# Optional env: SKIP_SYNC=1, GNN_TRAIN_EPOCHS, NUM_FOLDS (default 5), EXPLAINERS (default GNNEXPL),
# INCLUDE_MISCLASSIFIED=1 or pass -m / --include-misclassified before positional args.

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

# Sets INCLUDE_MISCLASSIFIED from -m / --include-misclassified; appends remaining args to MPROV3_ARGS.
# Caller must run MPROV3_ARGS=() immediately before calling (global array; avoids nounset and bash
# scoping quirks when MPROV3_ARGS=() would run inside the function).
mprov3_strip_include_misclassified_flags() {
  INCLUDE_MISCLASSIFIED="${INCLUDE_MISCLASSIFIED:-0}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m | --include-misclassified)
        INCLUDE_MISCLASSIFIED=1
        export INCLUDE_MISCLASSIFIED
        shift
        ;;
      *)
        MPROV3_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

# Apply positional args after strip; safe with set -u and an empty remainder.
mprov3_set_positional_from_mprov3_args() {
  set -- ${MPROV3_ARGS[@]+"${MPROV3_ARGS[@]}"}
}

mprov3_fold_cli() {
  local nf="${NUM_FOLDS:-5}"
  echo --num_folds "$nf" --fold_index "${1:?fold index required}"
}

# Latest trainings/<timestamp>/ name under default GINE results (mtime).
mprov3_latest_training_ts() {
  local out
  out="$(
    cd "$GNN_DIR" && uv run python -c "
from pathlib import Path
from mprov3_gine_explainer_defaults import DEFAULT_RESULTS_ROOT, RESULTS_TRAININGS, get_latest_timestamp_dir
latest = get_latest_timestamp_dir(Path(DEFAULT_RESULTS_ROOT) / RESULTS_TRAININGS)
if latest is None:
    raise SystemExit('No training run found under results/trainings')
print(latest.name, end='')
"
  )"
  echo "$out"
}

# Sets EXPLAIN_CLI_ARGS for run_explanations / generate_visualizations (bash 3.2–safe).
mprov3_build_explain_cli() {
  EXPLAIN_CLI_ARGS=()
  local ex="${EXPLAINERS:-GNNEXPL}"
  # shellcheck disable=SC2206
  local parts=($ex)
  if [[ ${#parts[@]} -eq 1 ]]; then
    EXPLAIN_CLI_ARGS=(--explainer "${parts[0]}")
  else
    EXPLAIN_CLI_ARGS=(--explainers "${parts[@]}")
  fi
}

# Caller must run MEX_MISCLASS_ARGS=() immediately before (global array; same nounset/scoping
# pattern as MPROV3_ARGS).
mprov3_misclassified_arg() {
  if [[ "${INCLUDE_MISCLASSIFIED:-0}" == "1" ]]; then
    MEX_MISCLASS_ARGS+=(--no_correct_class_only)
  fi
}

# Runs run_explanations.py; sets EXPL_TS from stdout; preserves exit status. Uses pipefail.
mprov3_run_explanations_capture_ts() {
  local log ec
  log="$(mktemp)"
  set -o pipefail
  (cd "$MEX_DIR" && uv run python scripts/run_explanations.py "$@") 2>&1 | tee "$log"
  ec="${PIPESTATUS[0]}"
  EXPL_TS="$(grep 'Run timestamp: ' "$log" | tail -1 | sed 's/.*Run timestamp: //' | tr -d '\r' || true)"
  rm -f "$log"
  if [[ "$ec" -ne 0 ]]; then
    return "$ec"
  fi
  if [[ -z "${EXPL_TS:-}" ]]; then
    echo "ERROR: could not parse explanation run timestamp from run_explanations.py output" >&2
    return 1
  fi
  return 0
}
