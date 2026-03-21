#!/usr/bin/env bash
# End-to-end check (strict sequence). v2 runs last.
#
# From repository root:
#   ./check_all.sh
#   SKIP_SYNC=1 ./check_all.sh           # skip all `uv sync`
#   GNN_TRAIN_EPOCHS=5 ./check_all.sh    # default train epochs is 1
#
# Order:
#   1. uv sync — mprov3_gine_explainer_defaults, mprov3_gine, mprov3_explainer, v2 (last)
#   2. mprov3_gine_explainer_defaults — hardcoded constant names (see Python REQUIRED_NAMES below)
#   3. mprov3_gine — README §0–§4.1 (config defaults)
#   4. mprov3_explainer — run_explanations.py --explainer GNNExplainer --max_graphs 1, then generate_visualizations.py
#   5. v2 — uv run python compare_explainers.py (default args)
#
# Requires: MPro snapshot at mprov3_gine/config.DEFAULT_DATA_ROOT.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED=0

GNN_TRAIN_EPOCHS="${GNN_TRAIN_EPOCHS:-1}"
GNN_DIR="$ROOT/mprov3_gine"
MEX_DIR="$ROOT/mprov3_explainer"

fail() {
  echo "ERROR: $*"
  FAILED=1
}

section() {
  echo ""
  echo "================================================================================"
  echo "$*"
  echo "================================================================================"
}

# Run `uv run python` with args as separate words (avoids a broken `python` token if a line wraps mid-word).
run_uv_python() {
  local workdir="$1"
  local label="$2"
  shift 2
  (cd "$workdir" && uv run python "$@") || fail "$label"
}

# =============================================================================
# 1. uv sync — v2 last (matches execution order)
# =============================================================================
if [[ "${SKIP_SYNC:-}" == "1" ]]; then
  section "Skipping uv sync (SKIP_SYNC=1)"
else
  section "1. uv sync — mprov3_gine_explainer_defaults, mprov3_gine, mprov3_explainer, v2"
  for pkg in mprov3_gine_explainer_defaults mprov3_gine mprov3_explainer v2; do
    echo ""
    echo "--> $pkg"
    (cd "$ROOT/$pkg" && uv sync) || fail "uv sync failed in $pkg"
  done
fi

# =============================================================================
# 2. mprov3_gine_explainer_defaults — validate hardcoded public constants (keep in sync with package __init__)
# =============================================================================
section "2. mprov3_gine_explainer_defaults — required constants (hardcoded names)"
(
  cd "$ROOT/mprov3_gine_explainer_defaults" && uv run python -c "
import mprov3_gine_explainer_defaults as m

# Hardcoded list — update when adding/removing exports in mprov3_gine_explainer_defaults/__init__.py
REQUIRED_NAMES = (
    'DEFAULT_IN_CHANNELS',
    'DEFAULT_HIDDEN_CHANNELS',
    'DEFAULT_NUM_LAYERS',
    'DEFAULT_DROPOUT',
    'DEFAULT_OUT_CLASSES',
    'DEFAULT_POOL',
    'DEFAULT_EDGE_DIM',
    'DEFAULT_EXPLANATION_TYPE',
    'DEFAULT_MODEL_CONFIG',
    'NODE_MASK_ATTRIBUTES',
    'EDGE_MASK_OBJECT',
    'DEFAULT_GNN_EXPLAINER_EPOCHS',
    'DEFAULT_GNN_EXPLAINER_LR',
    'DEFAULT_PG_EXPLAINER_EPOCHS',
    'DEFAULT_PG_EXPLAINER_LR',
    'DEFAULT_IG_N_STEPS',
    'DEFAULT_IG_INTERNAL_BATCH_SIZE',
    'DEFAULT_PGM_NUM_SAMPLES',
    'DEFAULT_TRAINING_EPOCHS',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_TRAINING_LR',
    'DEFAULT_SEED',
    'DEFAULT_NUM_FOLDS',
    'DEFAULT_FOLD_INDEX',
    'DEFAULT_MPRO_SNAPSHOT_DIR_NAME',
    'MPRO_INFO_CSV',
    'MPRO_SPLITS_DIR',
    'MPRO_LIGAND_DIR',
    'MPRO_LIGAND_SDF_SUBDIR',
    'DEFAULT_TRAIN_SPLIT_FILE',
    'DEFAULT_VAL_SPLIT_FILE',
    'DEFAULT_TEST_SPLIT_FILE',
    'RESULTS_DIR_NAME',
    'RESULTS_TRAININGS',
    'RESULTS_DATASETS',
    'RESULTS_CLASSIFICATIONS',
    'RESULTS_VISUALIZATIONS',
    'RESULTS_EXPLANATIONS',
    'RESULTS_CHECK_FORMAT',
    'CHECK_FORMAT_DATASETS_SUBDIR',
    'CHECK_FORMAT_RAW_DATA_SUBDIR',
    'PYG_DATA_FILENAME',
    'PYG_PDB_ORDER_FILENAME',
    'DEFAULT_TRAINING_CHECKPOINT_FILENAME',
    'DEFAULT_PYG_DATASET_NAME',
)

missing = [n for n in REQUIRED_NAMES if not hasattr(m, n)]
if missing:
    raise SystemExit('missing: ' + ', '.join(missing))
for n in REQUIRED_NAMES:
    getattr(m, n)
print('OK:', len(REQUIRED_NAMES), 'constants')
"
) || fail "mprov3_gine_explainer_defaults constants check"

# =============================================================================
# 3. mprov3_gine — README §0–§4.1 (default MPRO_DATA_ROOT from config)
# =============================================================================
section "3. mprov3_gine — §0 check_raw_data_format.py (defaults)"
run_uv_python "$GNN_DIR" "§0 check_raw_data_format.py" check_raw_data_format.py

section "3. mprov3_gine — §1 build_dataset.py (defaults)"
run_uv_python "$GNN_DIR" "§1 build_dataset.py" build_dataset.py

section "3. mprov3_gine — §1.1 check_PyG_data_format.py (defaults)"
run_uv_python "$GNN_DIR" "§1.1 check_PyG_data_format.py" check_PyG_data_format.py

section "3. mprov3_gine — §2 visualize_graphs.py --num_graphs 1"
run_uv_python "$GNN_DIR" "§2 visualize_graphs.py" visualize_graphs.py --num_graphs 1

section "3. mprov3_gine — §3 train.py (epochs=$GNN_TRAIN_EPOCHS, defaults otherwise)"
run_uv_python "$GNN_DIR" "§3 train.py" train.py --epochs "$GNN_TRAIN_EPOCHS"

section "3. mprov3_gine — §4 evaluate.py (defaults)"
run_uv_python "$GNN_DIR" "§4 evaluate.py" evaluate.py

section "3. mprov3_gine — §4.1 create_evaluation_report.py (defaults)"
run_uv_python "$GNN_DIR" "§4.1 create_evaluation_report.py" create_evaluation_report.py

# =============================================================================
# 4. mprov3_explainer — explanations then visualizations (order required)
# =============================================================================
section "4. mprov3_explainer — scripts/run_explanations.py --explainer GNNExplainer --max_graphs 1"
run_uv_python "$MEX_DIR" "run_explanations.py" scripts/run_explanations.py --explainer GNNExplainer --max_graphs 1

section "4. mprov3_explainer — scripts/generate_visualizations.py --explainers GNNExplainer"
run_uv_python "$MEX_DIR" "generate_visualizations.py" scripts/generate_visualizations.py --explainers GNNExplainer

# =============================================================================
# 5. v2 — compare_explainers.py (default parameters), last
# =============================================================================
section "5. v2 — uv run python compare_explainers.py (last)"
run_uv_python "$ROOT/v2" "v2 compare_explainers.py" compare_explainers.py

echo ""
if [[ "$FAILED" -eq 0 ]]; then
  echo "All steps passed."
  exit 0
else
  echo "One or more steps failed (see ERROR lines above)."
  exit 1
fi
