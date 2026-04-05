# MProV3 shell orchestration (`scripts/mprov3/`)

Bash helpers that run the **mprov3_gine** and **mprov3_explainer** pipelines in the same order as [`check_all.sh`](../../check_all.sh) §3–§4 (end-to-end checks), with optional **cross-validation folds**, **small smoke settings**, and a flag to **include misclassified** graphs in explainer preprocessing.

Run everything from the **repository root**:

```bash
./scripts/mprov3/smoke_gine_explainer.sh
```

## Order and parity with `check_all.sh`

**GINE** (per fold, when applicable):

1. `check_raw_data_format.py` → `build_dataset.py` → `check_PyG_data_format.py` → `visualize_graphs.py --num-graphs-by-fold 1` → `train.py` → `evaluate.py` → `create_evaluation_report.py`

For **`visualize_graphs.py`**, `--num-graphs-by-fold 1` caps to **at most one index row (and first draw) per (fold, train|val|test) bucket** (fast smoke); omit the flag for the full split-ordered plan (**one PNG per unique PDB**).

**Explainer** (always in this order):

1. `mprov3_explainer/scripts/run_explanations.py` (checkpoint and dataset resolved from **`mprov3_gine/results`** via `--results_root`)
2. `mprov3_explainer/scripts/generate_visualizations.py` (reads explanations under **`mprov3_explainer/results/explanations/`**)

These scripts do **not** run `check_all.sh` §2 (defaults package constant check).

**Note:** The `check_raw_data_format.py` step is **temporarily commented out** in [`run_gine_fold.sh`](run_gine_fold.sh) and [`smoke_gine_explainer.sh`](smoke_gine_explainer.sh) (marked `TEMP` / `end TEMP`). Uncomment that block in those files to run it again. [`check_all.sh`](../../check_all.sh) still runs §0 as usual.

## Scripts

| Script | Purpose |
|--------|---------|
| [`smoke_gine_explainer.sh`](smoke_gine_explainer.sh) | Full GINE chain for **fold 0**, **1** training epoch by default, **`run_explanations.py --max_graphs 1`**, then **`generate_visualizations.py`** (flat `results/` paths; same idea as `check_all.sh`). |
| [`run_gine_fold.sh`](run_gine_fold.sh) | Full GINE chain for **`fold_index`** (writes under fixed `mprov3_gine/results/` paths). |
| [`run_explainer_fold.sh`](run_explainer_fold.sh) | **`run_explanations.py`** on the **full test set** for **`fold_index`**, then **`generate_visualizations.py`**. |
| [`run_gine_explainer_fold.sh`](run_gine_explainer_fold.sh) | Runs **`run_gine_fold.sh`** then **`run_explainer_fold.sh`** for the same fold. |
| [`run_all_folds.sh`](run_all_folds.sh) | Loops **`run_gine_explainer_fold.sh`** for folds **`0 .. NUM_FOLDS-1`**. |

Shared logic lives in [`lib_common.sh`](lib_common.sh) (sourced, not executed alone).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SKIP_SYNC` | (unset) | If `1`, skip `uv sync` in `mprov3_gine_explainer_defaults`, `mprov3_gine`, and `mprov3_explainer`. |
| `NUM_FOLDS` | `5` | Passed as `--num_folds` where supported. |
| `GNN_TRAIN_EPOCHS` | `1` in smoke / per-fold GINE scripts | `train.py --epochs`. |
| `EXPLAINERS` | `GNNEXPL` | Space-separated explainer names (registry names, e.g. `GNNEXPL IGNODE`). Single name uses `--explainer`; multiple uses `--explainers`. |
| `INCLUDE_MISCLASSIFIED` | `0` | If `1`, explainer runs add **`--no_correct_class_only`** so preprocessing does not restrict metrics to correctly classified graphs. Same effect as **`-m` / `--include-misclassified`** on the scripts that invoke explanations. |

## Misclassified samples (`-m`)

Explainer-only preprocessing: pass **`-m`** or **`--include-misclassified`** before positional arguments on:

- `smoke_gine_explainer.sh`
- `run_explainer_fold.sh`
- `run_gine_explainer_fold.sh`
- `run_all_folds.sh`

Or set **`INCLUDE_MISCLASSIFIED=1`** in the environment. The flag is forwarded to **`run_explanations.py`** only (`--no_correct_class_only`).

## Examples

```bash
# Quick validation (skip sync if dependencies are already installed)
SKIP_SYNC=1 ./scripts/mprov3/smoke_gine_explainer.sh

# Smoke including misclassified graphs in explainer metrics
./scripts/mprov3/smoke_gine_explainer.sh -m

# One fold: GINE only
./scripts/mprov3/run_gine_fold.sh 2

# One fold: explainer only (checkpoint: mprov3_gine/results/trainings/fold_<k>/best_gnn.pt)
./scripts/mprov3/run_explainer_fold.sh 2

# Full GINE + explainer for fold 2
./scripts/mprov3/run_gine_explainer_fold.sh 2

# All folds (heavy: runs full GINE + full test explainer per fold)
NUM_FOLDS=5 ./scripts/mprov3/run_all_folds.sh
```

## Python flags used by the explainer step

- **`--results_root`** points at **`mprov3_gine/results`** so checkpoints (`trainings/fold_<k>/best_gnn.pt`) and datasets (`datasets/data.pt`) match training.

Explainer **artifacts** (reports, masks, comparison HTML) are under **`mprov3_explainer/results/`**, not under GINE results.

**Migration:** Older runs used timestamped subfolders under `results/`; those paths are no longer read automatically. Move or rebuild artifacts into the flat layout (see **`mprov3_gine/README.md`** and **`mprov3_explainer/README.md`**).

## Related docs

- [mprov3_gine README](../../mprov3_gine/README.md) — individual Python CLIs and `results/` layout.
- [mprov3_explainer README](../../mprov3_explainer/README.md) — explainers, `--no_correct_class_only`, outputs.
