# mprov3-explainer

MPro-GINE Explainer – PyTorch Geometric pipeline for graph-level explanations of the trained GINE classifier. Loads the model and dataset from **mprov3_gine/results**, runs **all** registered explainers on the **single best CV fold** (chosen from summaries produced by **evaluate.py** or **train.py**), and follows the **common representation** from *"Explaining the Explainers in Graph Neural Networks: a Comparative Study"* (Longa et al.): **(1) explanation masks** and **(2) preprocessing** (filtering, normalization) before metrics.

Artifacts go under **`mprov3_explainer/results/folds/fold_<k>/`**: **`explanations/<explainer>/`**, **`visualizations/<explainer>/graphs/`** (PNGs from the optional viz script). Re-running overwrites after an **`[INFO]`** line.

**Shell orchestration:** [**`scripts/mprov3/README.md`**](../scripts/mprov3/README.md) runs **`run_explanations.py`** then **`generate_visualizations.py`**.

## Flow

1. **Pick fold** – Read `mprov3_gine/results/classifications/classification_summary.json` (**test accuracy**, default) or `trainings/training_summary.json` (**train accuracy at best validation**). Requires **`evaluate.py`** and/or **`train.py`** to have written those summaries.
2. **Resolve paths** – Checkpoint `trainings/fold_<k>/best_gnn.pt`, `datasets/data.pt`; splits from `data_root/Splits/`.
3. **Load model** – `MProGNN` with defaults from **`mprov3_gine_explainer_defaults`** (must match training).
4. **Explainers** – All entries in **`AVAILABLE_EXPLAINERS`**; **PGEXPL** trains on the train loader, then explains the test set.
5. **Preprocessing** – Correct-class filter; **degenerate-mask filter** (τ = 10⁻³ on max−min per mask, default **on**; disable with **`--no_mask_spread_filter`**); min–max normalization to [0, 1].
6. **Metrics** – PyG fid+/fid− and characterization; Longa **Fsuf**, **Fcom**, **Ff1** (Nt = 100).
7. **Outputs** – Per-explainer **`explanation_report.json`** and **`masks/<pdb_id>.json`**; aggregate **`explanations/comparison_report.json`** (JSON only, no HTML).

## CLI (`scripts/run_explanations.py`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--results_root` | `../mprov3_gine/results` | GNN results (trainings, classifications, datasets). |
| `--data_root` | workspace MPro snapshot | Raw data with `Splits/` and ligand SDFs. |
| `--fold_metric` | `test_accuracy` | `test_accuracy` → best fold from **`classification_summary.json`**; `train_accuracy` → **`training_summary.json`** (`best_train_accuracy_fold_index`). |
| `--no_mask_spread_filter` | off | Skip τ = 10⁻³ mask spread discard. |

## CLI (`scripts/generate_visualizations.py`)

Writes **PNG** files only (no HTML, no HTTP server). Expects **exactly one** fold directory under **`results/folds/`** with explainer outputs; otherwise raises.

| Flag | Default | Meaning |
|------|---------|---------|
| `--results_root` | `mprov3_explainer/results` | Explainer project results. |
| `--data_root` | workspace MPro snapshot | For **`Ligand/Ligand_SDF/<pdb>_ligand.sdf`**. |

## Available explainers (registry)

| Name | Description |
|------|-------------|
| **GRADEXPINODE** | Captum Saliency on node features |
| **GRADEXPLEDGE** | Captum Saliency on edge mask |
| **GUIDEDBP** | Guided backprop on node features |
| **IGNODE** | Integrated Gradients on node features |
| **IGEDGE** | Integrated Gradients on edge mask |
| **GNNEXPL** | PyG GNNExplainer |
| **PGEXPL** | PGExplainer (train MLP on train loader, then explain) |
| **PGMEXPL** | PGMExplainer |

### Implementation notes

- **Integrated Gradients:** Uses bridge modules so Captum does not break PyG `edge_index`; on **MPS**, IG may run on CPU then move masks to float32.
- **PGExplainer:** Full training pass over the train loader each run (no CLI cap).

## Usage

```bash
uv sync
uv run python scripts/run_explanations.py
uv run python scripts/generate_visualizations.py
```

Custom paths:

```bash
uv run python scripts/run_explanations.py \
  --results_root /path/to/mprov3_gine/results \
  --data_root /path/to/MPro_snapshot

uv run python scripts/generate_visualizations.py --data_root /path/to/MPro_snapshot
```

Pick fold by training metric instead of test accuracy:

```bash
uv run python scripts/run_explanations.py --fold_metric train_accuracy
```

Plausibility (AUROC) with a ground-truth mask is not enabled in the default script.
