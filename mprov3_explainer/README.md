# mprov3-explainer

MPro-GINE Explainer – PyTorch Geometric pipeline for graph-level explanations of the trained GINE classifier. Loads the model and dataset from **mprov3_gine/results**, runs **all** registered explainers on the **single best CV fold** (chosen from summaries produced by **classify.py** or **train.py**), and follows the **common representation** from *"Explaining the Explainers in Graph Neural Networks: a Comparative Study"* (Longa et al.): **(1) explanation masks** and **(2) preprocessing** (filtering, normalization) before metrics.

Artifacts go under **`mprov3_explainer/results/folds/fold_<k>/`**: **`explanations/<explainer>/`**, **`visualizations/<explainer>/graphs/`** (PNGs from the optional viz script). Re-running overwrites after an **`[INFO]`** line.

## Flow (summary)

Fold selection and I/O are handled in [`scripts/run_explanations.py`](scripts/run_explanations.py); per-graph explain → preprocess → metrics is in [`src/mprov3_explainer/pipeline.py`](src/mprov3_explainer/pipeline.py). The detailed **explanation substeps** table below maps each phase to code.

## `scripts/run_explanations.py`: command without flags

From **`mprov3_explainer/`** (after `uv sync`):

```bash
uv run python scripts/run_explanations.py
```

This run:

- Resolves **mprov3_gine/results** and picks the **best fold** using **`test_accuracy`** from **`classifications/classification_summary.json`**.
- Uses the **test** split loader for the explanation loop (`--split` default).
- Loads **`best_gnn.pt`** for that fold and **`MProGNN`** hyperparameters from **`mprov3_gine_explainer_defaults`** (must match training).
- Loads splits and structures from the workspace MPro snapshot directory named in **`DEFAULT_MPRO_SNAPSHOT_DIR_NAME`** (same default as the GNN pipeline).
- Runs **every** name in **`AVAILABLE_EXPLAINERS`** in order.
- Enables the **mask spread filter** (τ = 10⁻³): masks with max − min below τ are marked invalid; disable only with **`--no_mask_spread_filter`**.
- Uses **Nt = 100** threshold steps for Longa-style paper metrics (`PAPER_N_THRESHOLDS` in the script).
- Writes **`explanation_report.json`** and **`masks/`** per explainer, then **`explanations/comparison_report.json`** for all explainers.

**PGEXPL:** always trains its MLP on the **train** loader first, then explains graphs from the split selected by **`--split`** (default test).

## `scripts/run_explanations.py`: flags and parameters

Paths are **not** configurable on the CLI: GNN artifacts are under **`mprov3_gine/results`**, snapshot/splits under the workspace default from **`mprov3_gine_explainer_defaults`**.

| Flag | Default | Meaning |
|------|---------|---------|
| `--split` | `test` | Which loader’s graphs are explained: `train`, `validation`, or `test`. Does **not** change where **PGEXPL** trains (always **train**). |
| `--fold_metric` | `test_accuracy` | `test_accuracy` → best fold from **`classification_summary.json`**. `train_accuracy` → **`training_summary.json`** field **`best_train_accuracy_fold_index`**. |
| `--no_mask_spread_filter` | off (filter **on**) | If set, skips degenerate-mask rejection: no check that max(mask) − min(mask) ≥ τ before normalization. |

Example combinations:

```bash
uv run python scripts/run_explanations.py --split validation
uv run python scripts/run_explanations.py --fold_metric train_accuracy
uv run python scripts/run_explanations.py --split train --no_mask_spread_filter
```

## Explanation substeps (code anchors)

Each row is one conceptual step: what the code does, and the main line or call to read first. Line numbers refer to the current tree; adjust if you edit those files.

| Step | What the code does | Anchor |
|------|-------------------|--------|
| **Configuration (dataset, classes, model)** | Builds loaders, device, checkpoint, `MProGNN`, output root, and mask-filter flag into **`ExplanationRunContext`**. | [`ExplanationRunContext`](scripts/run_explanations.py) (dataclass ~81); fill in [`build_explanation_run_context`](scripts/run_explanations.py) (~137). |
| **Explainer configuration** | Per explainer: kwargs (`num_classes`, IG steps, PGM samples) and epoch count; then `run_explanations` (~506). | [`run_one_explainer`](scripts/run_explanations.py): `run_explanations(...)` call (~245). |
| **Explain graph** | Runs the PyG explainer forward to produce raw **`edge_mask` / `node_mask`**. | [`_forward_raw_explanation`](src/mprov3_explainer/pipeline.py): `raw_explanation = explainer(x, edge_index, **call_kwargs)` (~308). |
| **Preprocessing wrapper** | Applies Longa-style pipeline on the raw explanation and copies masks onto a clone for metrics. | [`_preprocess_for_metrics`](src/mprov3_explainer/pipeline.py): `apply_preprocessing(...)` (~327). |
| **Filtering** | Marks explanation invalid if edge or node mask spread (max − min) is below τ; optional correct-class filter. | [`apply_preprocessing`](src/mprov3_explainer/preprocessing.py): `_mask_weight_spread(...) < tol` (~151–158). |
| **Normalization** | Per-mask min–max to [0, 1] on the preprocessed explanation (PyG path). | [`apply_preprocessing`](src/mprov3_explainer/preprocessing.py): `normalize_mask(edge_mask)` / `normalize_mask(node_mask)` (~168–174). |
| **Conversion (edge → node, when needed)** | For **paper** metrics only: if there is no node mask, build one by **averaging incident edge weights** per node, then normalize. Preprocessing keeps **`convert_edge_to_node=False`** so PyG fidelity still uses edge masks when present. | [`edge_mask_to_node_mask`](src/mprov3_explainer/preprocessing.py): mean via `scatter_add_` and `node_mask / degree` (~44–50); invoked from [`_paper_normalized_node_mask_from_explanation`](src/mprov3_explainer/pipeline.py) (~133–135). |
| **PyG fidelity** | GraphFramEx **fid+** / **fid−** via PyG **`fidelity`**. | [`_compute_pyg_fidelity`](src/mprov3_explainer/pipeline.py): `fidelity(explainer, _fidelity_explanation(explanation))` (~360). |
| **Sufficiency & comprehensiveness (paper)** | Threshold sweep over hard node masks: average **full_prob − subgraph_prob** (sufficiency) and **full_prob − complement_prob** (comprehensiveness). | [`_paper_sufficiency_and_comprehensiveness`](src/mprov3_explainer/pipeline.py): `suf_sum += (full_prob - exp_prob)` and `com_sum += (full_prob - comp_prob)` (~238–239). |
| **Paper F1-fidelity** | Combines Fsuf and Fcom into **Ff1** (Longa et al.). | [`_paper_f1_fidelity`](src/mprov3_explainer/pipeline.py) (~246–248). |
| **Framework metric (PyG)** | **Characterization** score from fid+ and fid−. | [`_compute_pyg_characterization`](src/mprov3_explainer/pipeline.py): `characterization_score(...)` (~384). |
| **Metrics aggregation (§5.2 style)** | After one explainer finishes, **arithmetic means** over **all** graphs in the split for fid+/− (via **`aggregate_fidelity(..., valid_only=False)`**), characterization, Fsuf, Fcom, Ff1; **`num_valid`** counts rows with **`valid`** true. Cross-explainer JSON merges per-explainer summaries. | [`run_one_explainer`](scripts/run_explanations.py): means ~276–280; [`aggregate_fidelity`](src/mprov3_explainer/pipeline.py) (~654); [`write_comparison_report`](scripts/run_explanations.py) (~370). |

**Longa PDF §5.2:** See the paper copy at [`../doc/2025_Longa_Benchmarking.pdf`](../doc/2025_Longa_Benchmarking.pdf) for the dataset-level aggregation protocol. This implementation reports **simple means over every graph in the chosen split** for the scalar summaries (invalid graphs remain in the sum with their stored scores; the **`valid`** flag documents filtering). If §5.2 requires restricting means to **valid** instances only, change the aggregations in [`run_one_explainer`](scripts/run_explanations.py) accordingly.

## CLI (`scripts/generate_visualizations.py`)

Writes **PNG** files only (no HTML, no HTTP server). Expects **exactly one** fold directory under **`mprov3_explainer/results/folds/`** with explainer outputs; otherwise raises. Reads **`mprov3_explainer/results`** and ligand SDFs from the same default MPro snapshot path as the GNN pipeline (no path flags).

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

Plausibility (AUROC) with a ground-truth mask is not enabled in the default script.
