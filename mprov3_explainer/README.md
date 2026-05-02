# mprov3-explainer

PyTorch Geometric pipeline for graph-level explanations of the trained
GINE classifier in `mprov3_gine`. The runner loads the trained model and
dataset from `mprov3_gine/results`, runs **every registered explainer** on
the chosen CV fold(s), and follows the **common representation** of *Longa
et al.* — *"Explaining the Explainers in Graph Neural Networks: a
Comparative Study"* — i.e. (1) explainer-native mask generation followed by
(2) Conversion / Filtering / Normalization preprocessing before any metric
is computed.

Per-fold artifacts live under
`mprov3_explainer/results/folds/fold_<k>/`:

- `explanations/<EXPLAINER>/explanation_report.json` and `masks/<graph>.json`,
- `explanations/comparison_report.json` (cross-explainer summary for the fold),
- `explanations/model_prediction_baseline.json` (one fold-level baseline shared by every explainer),
- `visualizations/<EXPLAINER>/graphs/mask_<graph>.png` (RDKit drawings, written by `generate_visualizations.py`),
- `explanation_web_report/index.html` (per-fold HTML report).

A multi-fold global index is written to
`results/explanation_web_report/index.html` whenever more than one fold is
processed in a single visualization run. Alongside it, an **explainer
summary page** is written to
`results/explanation_web_report/explainer_summary.html` — this is the
primary comparative view with one row per explainer containing:

- **Mean across folds** (valid and all-graph variants),
- **Valid-graph coverage** (total valid / total graphs, per-fold breakdown),
- **Weighted statistics** (median, IQR, mean, std weighted by graph count),
- **Unweighted statistics** (each fold counts equally).

**Per-class summary pages** are also generated alongside the main summary:
`results/explanation_web_report/explainer_summary_class_<N>.html` (one per
classification class). Each page contains the same weighted/unweighted
statistics, coverage, and mean-across-folds tables, but restricted to
**correctly-classified graphs** of that class only. The all-graph ("Result
metrics") tables are omitted. Links to the per-class pages appear in a
dedicated section of the main explainer summary page.

Re-running the explanation script overwrites any existing per-explainer
output after an `[INFO] Output exists; overwriting under: …` line.

## Flow at a glance

The pipeline has three thin layers:

1. **Fold and I/O resolution** — [`scripts/run_explanations.py`](scripts/run_explanations.py)
   builds an `ExplanationRunContext` that bundles the dataset, model, split
   loaders, device, and per-fold output root.
2. **Per-graph explain → preprocess → metrics** — [`src/mprov3_explainer/pipeline.py`](src/mprov3_explainer/pipeline.py)
   yields one `ExplanationResult` per graph carrying the Longa paper metrics
   plus the PyG metrics from `torch_geometric.explain.metric`.
3. **Aggregate + persist** — the runner builds two metric tables
   (`valid_result_metrics` and `result_metrics`, see below) and writes them
   into the per-explainer JSON; [`src/mprov3_explainer/web_report.py`](src/mprov3_explainer/web_report.py)
   renders the same two tables in the HTML reports.

## `scripts/run_explanations.py` — defaults

From `mprov3_explainer/` after `uv sync`:

```bash
uv run python scripts/run_explanations.py
```

This invocation:

- Resolves `mprov3_gine/results` and picks the **best fold** using
  `test_accuracy` from `classifications/classification_summary.json`
  (`--fold_metric` default).
- Uses the **test** split loader for the explanation loop (`--split` default).
- Loads `best_gnn.pt` for that fold and the `MProGNN` hyperparameters from
  `mprov3_gine_explainer_defaults` (these must match the values used at
  training time).
- Reads splits and ligand graphs from the workspace MPro snapshot directory
  named in `DEFAULT_MPRO_SNAPSHOT_DIR_NAME`.
- Runs every name in `AVAILABLE_EXPLAINERS` in registry order, re-seeding
  RNGs before each explainer so explainer order does not affect individual
  results.
- Enables the **mask-spread filter** (τ = 10⁻³); masks with `max − min < τ`
  are marked invalid. Disable only with `--no_mask_spread_filter`.
- Uses **`Nt = 100`** percentile thresholds for the Longa paper-metric sweep
  (`PAPER_N_THRESHOLDS` in the script).
- Builds the PyG fidelity curve over the sparsity grid
  `(0.1, 0.2, …, 0.9)` (`DEFAULT_FIDELITY_CURVE_TOP_K`) so PyG's
  `fidelity_curve_auc` has something to integrate.
- Writes `explanation_report.json` plus `masks/` per explainer, then the
  cross-explainer `explanations/comparison_report.json`.

**PGExplainer:** trains its MLP on the **train** loader first, even when
`--split=test` or `--split=validation`. PGExplainer is currently disabled in
`AVAILABLE_EXPLAINERS` because every mask it produces on this dataset
collapses to a constant.

## `scripts/run_explanations.py` — flags

Path arguments are intentionally **not** configurable on the CLI; they
follow `DEFAULT_DATA_ROOT`, `DEFAULT_RESULTS_ROOT`, and
`DEFAULT_MPRO_SNAPSHOT_DIR_NAME` from `mprov3_gine_explainer_defaults`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--split` | `test` | Loader used by the explanation loop: `train`, `validation`, or `test`. PGExplainer always trains its MLP on `train` regardless of this flag. |
| `--folds` | unset | Explicit list of fold indices to run, e.g. `--folds 0 2 4`. Overrides `--fold_metric`. |
| `--fold_metric` | `test_accuracy` | Best-fold selector when `--folds` is unset. `test_accuracy` reads `classifications/classification_summary.json`; `train_accuracy` reads `training_summary.json` (`best_train_accuracy_fold_index`). |
| `--no_mask_spread_filter` | filter on | If set, skips the `max − min ≥ τ` degenerate-mask check. Use only for debugging. |
| `--seed` | `42` | RNG seed for `torch` / `numpy` / `random` / `PyG`. Re-seeded before every explainer. |

Examples:

```bash
uv run python scripts/run_explanations.py --split validation
uv run python scripts/run_explanations.py --fold_metric train_accuracy
uv run python scripts/run_explanations.py --folds 0 2 4
uv run python scripts/run_explanations.py --split train --no_mask_spread_filter
uv run python scripts/run_explanations.py --seed 7
```

## Metrics

Each `explanation_report.json` carries one `per_graph` list and **two
aggregate tables** built from the same per-graph metric set:

- **`valid_result_metrics`** — aggregated only over graphs whose
  explanation produced a complete, well-defined metric set (`valid == true`).
- **`result_metrics`** — aggregated over **every** graph in the fold (NaN-
  skipped means), plus `wall_time_s` and `num_graphs`.

A graph is `valid` iff every metric below is finite, the model prediction
matches the ground-truth class, and the explanation's representative mask
passes the spread filter (or the filter is disabled).

### Per-graph metrics

| JSON key | Definition | Range | Source |
|----------|------------|-------|--------|
| `paper_sufficiency` | Longa **Fsuf**: average `P_target(full) − P_target(subgraph)` across the percentile sweep, on the explanation subgraph induced by the top-`q` fraction of nodes (or edges, for edge-only explanations). | `[-1, 1]` (lower / negative is better) | Longa et al. (2025). Reported unclamped. |
| `paper_comprehensiveness` | Longa **Fcom**: same sweep, on the **complement** subgraph: `P_target(full) − P_target(complement)`. | `[-1, 1]` (higher is better) | Longa et al. (2025). Reported unclamped. |
| `paper_f1_fidelity` | Longa **Ff1** = `2·(1 − Fsuf_c)·Fcom_c / ((1 − Fsuf_c) + Fcom_c)` with `Fsuf_c, Fcom_c = clip([Fsuf, Fcom], 0, 1)`. | `[0, 1]` | Longa et al. (2025). Clamping is applied **only for the F-score combination**; the raw `Fsuf`/`Fcom` keys stay signed. |
| `pyg_fidelity_plus` | PyG `fidelity` Fid+, called on the preprocessed (soft) explanation as PyG ships it. Class-decision rate (model mode) or target-class rate (phenomenon mode). | `[0, 1]` (higher = stronger necessity) | `torch_geometric.explain.metric.fidelity` |
| `pyg_fidelity_minus` | PyG `fidelity` Fid−, same call. | `[0, 1]` (lower = stronger sufficiency) | Same. |
| `pyg_characterization_score` | Weighted harmonic mean of Fid+ and `1 − Fid−` (`pos_weight = neg_weight = 0.5`). | `[0, 1]` | `torch_geometric.explain.metric.characterization_score` |
| `pyg_fidelity_curve_auc` | AUC of the curve `f(x) = Fid+ / (1 − Fid−)` evaluated on the per-graph top-`k` sparsity grid `(0.1, 0.2, …, 0.9)`. NaN when any `Fid− == 1`. | `[0, ∞)` | `torch_geometric.explain.metric.fidelity_curve_auc` (sweep is required because the helper integrates a curve). |
| `pyg_unfaithfulness` | Graph-Explanation-Faithfulness (GEF) = `1 − exp(−KL(p_full ‖ p_masked))`. | `[0, 1]` (lower = more faithful) | `torch_geometric.explain.metric.unfaithfulness` |
| `valid` | True iff every metric above is finite, prediction is correct, and the mask passes the spread filter (when on). | bool | Validity flag used to populate `valid_result_metrics`. |
| `elapsed_s` | Wall-clock seconds inside the explainer's forward call for this graph. | `[0, ∞)` | — |
| `correct_class`, `pred_class`, `target_class`, `prediction_baseline_mismatch`, `has_node_mask`, `has_edge_mask` | Validity bookkeeping; do not appear in the aggregate tables. | — | — |

NaN and infinity are recursively serialised as JSON `null` (writers use
`allow_nan=False` to keep the JSON strict).

### `valid_result_metrics`

| Key | Description |
|-----|-------------|
| `num_valid_graphs` | Number of graphs with `valid == true` that contributed to the means. |
| `mean_paper_sufficiency`, `mean_paper_comprehensiveness`, `mean_paper_f1_fidelity` | NaN-skipped means of the corresponding per-graph paper metric over valid graphs. |
| `mean_pyg_fidelity_plus`, `mean_pyg_fidelity_minus`, `mean_pyg_characterization_score`, `mean_pyg_fidelity_curve_auc`, `mean_pyg_unfaithfulness` | NaN-skipped means of the corresponding per-graph PyG metric over valid graphs. |

### `result_metrics`

| Key | Description |
|-----|-------------|
| `wall_time_s` | Total wall-clock seconds the explainer spent inside `run_explanations` for this fold. |
| `num_graphs` | Total number of graphs explained on the split (valid + invalid). |
| `mean_paper_sufficiency`, `mean_paper_comprehensiveness`, `mean_paper_f1_fidelity` | NaN-skipped means over **all** graphs. |
| `mean_pyg_fidelity_plus`, `mean_pyg_fidelity_minus`, `mean_pyg_characterization_score`, `mean_pyg_fidelity_curve_auc`, `mean_pyg_unfaithfulness` | NaN-skipped means over all graphs. |

The report also carries `explainer`, `seed`, `run_status`, and
`run_status_note` at the top level. `run_status` ∈ {`ok`, `partial_invalid`,
`failed_no_valid_metrics`, `empty_run`}; the note explains the cause when
the status is not `ok` so a fold whose `valid_result_metrics` are all NaN
cannot be silently mistaken for a successful run.

### Paper-metric sweep dispatch

`_paper_metrics_from_masks` dispatches the percentile sweep by which masks
the explainer produced; this preserves native granularity for edge-only
explainers instead of coercing edges into nodes via incident-edge averaging:

| Mask present | Sweep granularity | Function |
|--------------|------------------|----------|
| node mask (with or without edge mask) | top-`q` nodes ⇒ induced node subgraph | `_paper_sufficiency_and_comprehensiveness` |
| edge mask only | top-`q` edges ⇒ induced edge subgraph (nodes = endpoints of kept edges) | `_paper_metrics_from_edge_mask` |
| neither | NaN triple | — |

## Code anchors (one row per substep)

| Step | What it does | Anchor |
|------|--------------|--------|
| **Run context** | Builds loaders, device, checkpoint path, `MProGNN`, output root, and mask-filter flag. | `ExplanationRunContext`, `build_explanation_run_context_for_fold` in [`scripts/run_explanations.py`](scripts/run_explanations.py). |
| **Prediction baseline** | Computes per-graph `(pred_class, target_class, correct_class)` once per fold, before any explainer runs. Subsequent explainers reuse this baseline to keep `valid` consistent. | `collect_prediction_baseline` in [`pipeline.py`](src/mprov3_explainer/pipeline.py); `write_prediction_baseline` in [`run_explanations.py`](scripts/run_explanations.py). |
| **Explain graph** | Runs the PyG `Explainer` forward to produce the raw `edge_mask`/`node_mask`. Times the call to populate `elapsed_s`. | `_forward_raw_explanation` in [`pipeline.py`](src/mprov3_explainer/pipeline.py). |
| **Filtering + Normalization** | Longa-style: mark invalid if mask spread `< τ` or class mismatch (when `correct_class_only=True`); min-max normalize to `[0, 1]`. | `apply_preprocessing` in [`preprocessing.py`](src/mprov3_explainer/preprocessing.py); `_preprocess_for_metrics` in [`pipeline.py`](src/mprov3_explainer/pipeline.py). |
| **PyG metrics (no extra binarization)** | `fidelity`, `characterization_score` and `unfaithfulness` are called as PyG ships them. `fidelity_curve_auc` is built from a sparsity sweep because the helper integrates a curve. | `_compute_pyg_fidelity`, `_compute_pyg_characterization`, `_compute_pyg_fidelity_curve_auc`, `_compute_pyg_unfaithfulness` in [`pipeline.py`](src/mprov3_explainer/pipeline.py). |
| **Paper metrics (Longa percentile sweep)** | Mask-type-aware sweep that returns `(Fsuf, Fcom, Ff1)`. Ff1 is clamped to `[0, 1]` only for the F-score combination. | `_paper_metrics_from_masks` (dispatcher), `_paper_sufficiency_and_comprehensiveness`, `_paper_metrics_from_edge_mask`, `_paper_f1_fidelity` in [`pipeline.py`](src/mprov3_explainer/pipeline.py). |
| **Validity flag** | Any NaN metric forces `valid = False`, even if the graph passed the class and spread filters. | End of the per-graph loop in `run_explanations` ([`pipeline.py`](src/mprov3_explainer/pipeline.py)). |
| **Aggregation + persist** | Build `valid_result_metrics` (valid-only) and `result_metrics` (all graphs, plus runtime); write `explanation_report.json` and per-graph `masks/<id>.json`; finally write the cross-explainer `comparison_report.json`. | `_build_valid_result_metrics`, `_build_result_metrics`, `run_one_explainer`, `write_comparison_report` in [`run_explanations.py`](scripts/run_explanations.py). |

`run_status` (`diagnose_explanation_run` in [`pipeline.py`](src/mprov3_explainer/pipeline.py))
makes any "no valid graphs" run loud in the JSON and the HTML report so
those rows are not silently consumed as thesis evidence.

## `scripts/generate_visualizations.py`

Draws RDKit PNGs from the saved masks and writes static HTML reports. By
default it auto-discovers every `results/folds/fold_*/explanations/` tree
that already contains explainer outputs. Use `--folds 0 2 4` to restrict
to specific folds, `--report-only` to skip RDKit drawing (regenerates HTML
only, no SDFs needed) and `--no-report` to skip the HTML output.

The per-fold HTML report renders the two metric tables side by side
(`Valid result metrics` and `Result metrics`), one row per explainer, and
embeds every per-graph mask JSON / PNG underneath. When more than one fold
is processed, a global cross-fold index is written to
`results/explanation_web_report/index.html` and the explainer summary to
`results/explanation_web_report/explainer_summary.html`.

```bash
uv run python scripts/generate_visualizations.py
uv run python scripts/generate_visualizations.py --folds 0 2 4
uv run python scripts/generate_visualizations.py --report-only
```

## Available explainers

| Name | Mask | Method |
|------|------|--------|
| `GRADEXPINODE` | node features | Captum Saliency on `x` |
| `GRADEXPLEDGE` | edge | Captum Saliency on PyG's edge-mask channel |
| `GUIDEDBP` | node features | Captum Guided Backprop on `x` (requires `nn.ReLU` modules) |
| `IGNODE` | node features | Integrated Gradients on `x` (Captum-safe bridge) |
| `IGEDGE` | edge | Integrated Gradients on the edge-mask channel |
| `GNNEXPL` | edge | PyG `GNNExplainer` (per-instance soft-mask optimisation) |
| `PGEXPL` | edge | PyG `PGExplainer` (parametric edge mask, trains an MLP first). **Currently disabled** in `AVAILABLE_EXPLAINERS` because every produced mask is degenerate on this dataset. |
| `PGMEXPL` | node features | `PGMExplainer` (perturbations + chi-square test) |

### Implementation notes

- **Integrated Gradients** uses a Captum-safe bridge so PyG's `edge_index`
  is not perturbed by Captum. On MPS the IG path may run on CPU; mask
  tensors are then downcast to `float32`.
- **PGExplainer** trains its MLP on the **train** loader for every run and
  is therefore disabled by default — re-enable in `AVAILABLE_EXPLAINERS`
  only if you change its training schedule.

## Setup

```bash
uv sync
uv run python scripts/run_explanations.py
uv run python scripts/generate_visualizations.py
```

Plausibility (AUROC against a ground-truth mask) is **not** enabled by the
default scripts; this dataset has no ground-truth masks.
