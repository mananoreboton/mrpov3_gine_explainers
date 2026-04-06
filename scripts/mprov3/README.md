# MProV3 shell orchestration (`scripts/mprov3/`)

Bash helpers that run **mprov3_gine** and **mprov3_explainer** in a consistent order (aligned with [`check_all.sh`](../../check_all.sh) where applicable).

Run from the **repository root**:

```bash
./scripts/mprov3/smoke_gine_explainer.sh
```

## Order and parity with `check_all.sh`

**GINE** (per fold when applicable):

1. `build_dataset.py` → `check_PyG_data_format.py` → `visualize_graphs.py --num-graphs-by-fold 1` → `train.py` → `classify.py` → `create_classification_report.py`

For **`visualize_graphs.py`**, `--num-graphs-by-fold 1` caps smoke-style previews; omit the flag for a full split-ordered run.

**Explainer** (after GINE has written **`classification_summary.json`** for test-based fold choice):

1. **`mprov3_explainer/scripts/run_explanations.py`** – all explainers, **best fold** from `mprov3_gine/results` summaries (`--results_root` points at GINE results).
2. **`mprov3_explainer/scripts/generate_visualizations.py`** – RDKit **PNG** masks under **`mprov3_explainer/results/folds/fold_<k>/visualizations/`**.

These scripts do **not** run `check_all.sh` §2 (defaults package constant check).

**Note:** `check_raw_data_format.py` is **commented out** in [`run_gine_fold.sh`](run_gine_fold.sh) and [`smoke_gine_explainer.sh`](smoke_gine_explainer.sh). [`check_all.sh`](../../check_all.sh) still runs §0 as usual.

## Scripts

| Script | Purpose |
|--------|---------|
| [`smoke_gine_explainer.sh`](smoke_gine_explainer.sh) | GINE for **fold 0** (1 epoch default), then explainer **best fold** + PNGs. |
| [`run_gine_fold.sh`](run_gine_fold.sh) | Full GINE chain for **`fold_index`**. |
| [`run_explainer_fold.sh`](run_explainer_fold.sh) | **`run_explanations.py`** + **`generate_visualizations.py`** (no fold argument; fold from summaries). |
| [`run_gine_explainer_fold.sh`](run_gine_explainer_fold.sh) | **`run_gine_fold.sh`** then **`run_explainer_fold.sh`**. |
| [`run_all_folds.sh`](run_all_folds.sh) | **`run_gine_fold.sh`** for folds **0 .. NUM_FOLDS−1**, then **one** explainer run (best fold). |

Shared logic: [`lib_common.sh`](lib_common.sh).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SKIP_SYNC` | (unset) | If `1`, skip `uv sync` in defaults package, `mprov3_gine`, and `mprov3_explainer`. |
| `NUM_FOLDS` | `5` | Passed to GINE scripts as `--num_folds`. |
| `GNN_TRAIN_EPOCHS` | `1` in smoke | `train.py --epochs`. |
| `FOLD_METRIC` | `test_accuracy` | For **`run_explainer_fold.sh`**: forwarded as `test_accuracy` (default) or set to `train_accuracy` for **`run_explanations.py --fold_metric`**. |

## Examples

```bash
SKIP_SYNC=1 ./scripts/mprov3/smoke_gine_explainer.sh

./scripts/mprov3/run_gine_fold.sh 2

./scripts/mprov3/run_explainer_fold.sh

FOLD_METRIC=train_accuracy ./scripts/mprov3/run_explainer_fold.sh

./scripts/mprov3/run_gine_explainer_fold.sh 2

NUM_FOLDS=5 ./scripts/mprov3/run_all_folds.sh
```

## Python flags (explainer)

- **`run_explanations.py`:** `--results_root` (GINE), `--data_root`, `--fold_metric`, `--no_mask_spread_filter`.
- **`generate_visualizations.py`:** `--results_root` (explainer), `--data_root`.

Explainer **artifacts** live under **`mprov3_explainer/results/folds/`**, not under GINE results.

## Related docs

- [mprov3_gine README](../../mprov3_gine/README.md)
- [mprov3_explainer README](../../mprov3_explainer/README.md)
