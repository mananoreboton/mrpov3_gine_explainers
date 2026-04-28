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
| `--seed` | `42` | RNG seed for `torch` / `numpy` / `random` / `PyG`. Re-seeded before every explainer so the order of explainers does not affect their individual results. |
| `--top_k_fraction` | `0.2` | Fraction of top-ranked entries kept by the GraphFramEx top-k binarized fidelity (the headline `mean_fidelity_*` numbers). `0.2` is the GraphFramEx canonical value. |

Example combinations:

```bash
uv run python scripts/run_explanations.py --split validation
uv run python scripts/run_explanations.py --fold_metric train_accuracy
uv run python scripts/run_explanations.py --split train --no_mask_spread_filter
uv run python scripts/run_explanations.py --seed 7 --top_k_fraction 0.1
```

## Metrics: definitions, ranges, how they are computed

Every entry in `explanation_report.json` (per explainer) and in the
per-explainer block of `comparison_report.json` is documented below. The
formulas reflect the **post-fix** implementation that replaced the silent
zeros / unclamped Ff1 / soft-mask fidelity headline of the previous version.

| JSON key | Formula | Range | Source / notes |
|---------|---------|-------|---------------|
| `mean_fidelity_plus` | mean over **valid** graphs (NaN-skipped) of GraphFramEx **Fid+** computed on the **top-k binarized** mask: 1 − P_target(complement subgraph) / P_target(full graph) | `[0, 1]` (0 = trivial, 1 = best) | Amara et al., *GraphFramEx*, 2022. Top-k binarization (default k=0.2) makes the explanation/complement well-defined hard subsets. |
| `mean_fidelity_minus` | mean of GraphFramEx **Fid−** on top-k binarized mask: 1 − P_target(explanation subgraph) / P_target(full graph) | `[0, 1]` (0 = best) | same source |
| `mean_pyg_characterization` | harmonic-mean-style score over (Fid+, 1 − Fid−) with weights 0.5 / 0.5; NaN if Fid+/Fid− are NaN | `[0, 1]` | PyG `torch_geometric.explain.metric.characterization_score` |
| `mean_paper_sufficiency` | average drop in P_target across a **percentile threshold sweep**: at each kept-fraction *q*, build the explanation subgraph from the top-*q* nodes (or edges, for edge-only explainers), then sum `(P_target(full) − P_target(subgraph))` and divide by the number of sweep steps | `[-1, 1]` (lower / negative = better; values may be negative when the explanation slightly *helps* the model) | Longa et al., *Benchmarking*, 2025 §5.2 |
| `mean_paper_comprehensiveness` | same sweep, but the complement subgraph: `P_target(full) − P_target(complement)` | `[-1, 1]` (higher = better) | same source |
| `mean_paper_f1_fidelity` | per-graph **clamped** Ff1 = `2·(1 − Fsuf_c)·Fcom_c / ((1 − Fsuf_c) + Fcom_c)` with `Fsuf_c, Fcom_c = clip(Fsuf, Fcom, [0, 1])`, then NaN-skipped mean | `[0, 1]` | Longa et al. The clamp guarantees the harmonic-mean is well-defined and matches the paper's domain assumption (the pre-fix code returned negative values when Fsuf or Fcom were negative). |
| `mean_fidelity_plus_all_graphs`, `mean_fidelity_minus_all_graphs`, `mean_pyg_characterization_all_graphs`, `mean_paper_sufficiency_all_graphs`, `mean_paper_comprehensiveness_all_graphs`, `mean_paper_f1_fidelity_all_graphs` | same per-graph values, averaged over **every** explained graph (including invalid ones) | same as the matching headline | Legacy fallback so reports diff against the pre-fix values. |
| `mean_fidelity_plus_soft`, `mean_fidelity_minus_soft`, `mean_pyg_characterization_soft` | GraphFramEx fidelity / characterization computed on the **soft** mask (no top-k binarization), valid-only NaN-skipped | `[0, 1]` | Diagnostic. The soft-mask values typically collapse to `Fid+ ≈ Fid−` because `mask · x` and `(1 − mask) · x` are just two rescalings of the same input; this is why the headline switched to top-k. |
| `num_graphs` | total graphs run through the explainer | integer ≥ 0 | — |
| `num_valid` | graphs with `valid=True` (correct class, mask spread ≥ τ, no NaN metric) | integer ≤ `num_graphs` | — |
| `num_degenerate_mask` | graphs whose representative mask has spread (max − min) below τ = 1e-3 | integer | New diagnostic. Tracks how often a mask is effectively constant (PGEXPL frequently degenerates without enough training). |
| `num_misclassified` | graphs where the fold-level model prediction baseline differs from the ground-truth label | integer | Computed once before any explainer runs, so this count should be identical for all explainers in the same fold and split. |
| `num_prediction_baseline_mismatch` | graphs where an explainer-time prediction differs from the precomputed baseline | integer | Should be 0. Nonzero values indicate model state drift or explainer side effects. |
| `mean_mask_spread` | NaN-skipped mean of `max(mask) − min(mask)` across explained graphs | `[0, ∞)` | — |
| `mean_mask_entropy` | NaN-skipped mean of Shannon entropy (in nats) of the normalized mask interpreted as a probability distribution | `[0, log(N)]` | Sharper masks → lower entropy. |
| `top_k_fraction` | the *k* used by the headline top-k fidelity (CLI `--top_k_fraction`) | `(0, 1]` | Self-describing parameter; default `0.2`. |
| `seed` | the RNG seed used for this run (CLI `--seed`) | integer | Self-describing parameter; default `42`. |
| `run_status`, `run_status_note` | compact quality flag for the explainer run | text | `failed_all_degenerate_masks` means every attempted mask had spread below τ and the headline means must not be used as valid thesis evidence. |
| `wall_time_s` | wall-clock seconds spent inside `run_explanations` for this explainer | `[0, ∞)` | — |
| `per_graph[*]` | per-graph mirror of the headline keys (`fidelity_plus`, `fidelity_minus`, `pyg_characterization`, `*_soft` siblings, `paper_*`, `mask_spread`, `mask_entropy`, `valid`, `correct_class`, `has_node_mask`, `has_edge_mask`, `elapsed_s`) | per key | NaN values are serialized as JSON `null`. |

### Sweep dispatch (paper metrics)

`_paper_metrics_from_masks` dispatches by the available masks:

| Mask present | Sweep granularity | Function |
|--------------|------------------|----------|
| node mask (with or without edge mask) | top-*q* nodes ⇒ induced node subgraph | `_paper_sufficiency_and_comprehensiveness` |
| edge mask only | top-*q* edges ⇒ induced edge subgraph (node set = endpoints of kept edges) | `_paper_metrics_from_edge_mask` |
| neither | NaN triple (returned as `null`) | — |

This restores native granularity for the four edge-only explainers
(`GRADEXPLEDGE`, `IGEDGE`, `GNNEXPL`, `PGEXPL`) instead of coercing their edge
masks into node masks via incident-edge averaging.

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
| **Conversion (edge → node, when needed)** | The legacy edge → node coercion is still implemented (`edge_mask_to_node_mask`) but is **no longer used** by paper metrics: edge-only explanations are now scored with an **edge-native** percentile sweep so granularity is preserved. | [`edge_mask_to_node_mask`](src/mprov3_explainer/preprocessing.py); see also `_paper_metrics_from_edge_mask` in [`pipeline.py`](src/mprov3_explainer/pipeline.py). |
| **PyG fidelity (top-k headline)** | GraphFramEx **Fid+** / **Fid−** computed on the **top-k binarized** mask (k = `top_k_fraction`, default 0.2). | [`_compute_pyg_fidelity_top_k`](src/mprov3_explainer/pipeline.py); the soft-mask variant is preserved as `_compute_pyg_fidelity` for the `*_soft` diagnostic columns. |
| **Sufficiency & comprehensiveness (paper)** | Percentile sweep over kept-fraction *q*: average `P_target(full) − P_target(subgraph)` (sufficiency) and `P_target(full) − P_target(complement)` (comprehensiveness). Dispatched by mask type (node-native vs edge-native). | [`_paper_sufficiency_and_comprehensiveness`](src/mprov3_explainer/pipeline.py); [`_paper_metrics_from_edge_mask`](src/mprov3_explainer/pipeline.py); dispatcher [`_paper_metrics_from_masks`](src/mprov3_explainer/pipeline.py). |
| **Paper F1-fidelity (clamped)** | Combines Fsuf and Fcom into **Ff1** after clamping each to `[0, 1]` (the Longa et al. domain). NaN-propagating. | [`_paper_f1_fidelity`](src/mprov3_explainer/pipeline.py). |
| **Framework metric (PyG)** | **Characterization** score from fid+ and fid−. NaN-aware. | [`_compute_pyg_characterization`](src/mprov3_explainer/pipeline.py). |
| **Metrics aggregation** | After one explainer finishes, headline `mean_*` keys are **NaN-aware, valid-only** means. Legacy "all-graphs" siblings (`mean_*_all_graphs`) and soft-mask siblings (`mean_*_soft`) are also written for backwards comparison. The **diagnostics block** records `num_degenerate_mask`, `num_misclassified`, `mean_mask_spread`, `mean_mask_entropy`, `top_k_fraction`, `seed`. | [`run_one_explainer`](scripts/run_explanations.py); [`aggregate_fidelity`](src/mprov3_explainer/pipeline.py) (with `nan_skip=True` by default); [`nanmean`](src/mprov3_explainer/pipeline.py); [`write_comparison_report`](scripts/run_explanations.py). |

**Longa PDF §5.2:** See the paper copy at [`../doc/2025_Longa_Benchmarking.pdf`](../doc/2025_Longa_Benchmarking.pdf) for the dataset-level aggregation protocol. This implementation reports **valid-only, NaN-skipped means** for the headline `mean_*` keys (the scientifically correct headline). The pre-fix "all-graphs" arithmetic means are preserved under `mean_*_all_graphs` for backwards comparison.

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
