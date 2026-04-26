#!/usr/bin/env bash
# Run README pipeline steps 1–7 with default CLI flags (no arguments passed to Python).
# Prerequisites: `uv sync` in this directory; raw MPro snapshot at DEFAULT_DATA_ROOT.
# Step 1 is advisory: a non-zero exit is logged and the pipeline continues. Steps 2–7 abort on failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

banner() {
  echo ""
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

banner "1/7 check_raw_data_format.py"
set +e
uv run python check_raw_data_format.py
step1_status=$?
set -e
if [[ "$step1_status" -ne 0 ]]; then
  echo "[WARN] Step 1 exited with status $step1_status; continuing with steps 2–7."
fi

banner "2/7 build_dataset.py"
uv run python build_dataset.py

banner "3/7 check_PyG_data_format.py"
uv run python check_PyG_data_format.py

banner "4/7 visualize_graphs.py"
uv run python visualize_graphs.py

banner "5/7 train.py"
uv run python train.py --seed 42

banner "6/7 classify.py"
uv run python classify.py

banner "7/7 create_classification_report.py"
uv run python create_classification_report.py

echo ""
echo "Pipeline finished."
