# Explainer Metrics Reference

All metrics collected per explainer in `run_explanations.py`. Each per-graph
metric appears under both aggregate tables in `explanation_report.json`:
`valid_result_metrics` (aggregated only over graphs with `valid == true`) and
`result_metrics` (aggregated over **every** explained graph).

## Per-graph metrics

| Metric (JSON key) | Description | Longa 2025 | PyG library |
|----|----|----|----|
| `paper_sufficiency` | Longa Fsuf: average drop in target-class probability when the model is evaluated on explanation subgraphs across a percentile threshold sweep (lower is better). | Yes | No |
| `paper_comprehensiveness` | Longa Fcom: average drop in target-class probability when the model is evaluated on complement subgraphs across a percentile threshold sweep (higher is better). | Yes | No |
| `paper_f1_fidelity` | Longa Ff1: harmonic mean of `(1 - Fsuf)` and `Fcom` after clamping both to `[0, 1]`. | Yes | No |
| `pyg_fidelity_plus` | PyG `fidelity` Fid+ on the (preprocessed) explanation, as PyG generates it. | No | Yes |
| `pyg_fidelity_minus` | PyG `fidelity` Fid- on the (preprocessed) explanation, as PyG generates it. | No | Yes |
| `pyg_characterization_score` | PyG `characterization_score(Fid+, Fid-)` with default `pos/neg_weight=0.5`. | No | Yes |
| `pyg_fidelity_curve_auc` | PyG `fidelity_curve_auc` over a per-graph top-k sparsity sweep `(0.1, 0.2, …, 0.9)`. | No | Yes |
| `pyg_unfaithfulness` | PyG `unfaithfulness` (graph explanation faithfulness, GEF). | No | Yes |
| `valid` | True iff the prediction is correct and the mask passes the degenerate-mask filter. Used to populate `valid_result_metrics`. | – | – |
| `elapsed_s` | Wall-clock seconds the explainer's forward call took for this graph. | – | – |

The remaining per-graph fields (`correct_class`, `pred_class`, `target_class`,
`prediction_baseline_mismatch`, `has_node_mask`, `has_edge_mask`) are
diagnostics that feed the validity check; they do not appear in the aggregate
tables.

## Aggregate tables

Both tables share the eight per-graph metrics above (mean across the
selected graphs). They differ only in the population they aggregate over and
in the leading bookkeeping columns.

### `valid_result_metrics`

| Column | Description |
|----|----|
| `num_valid_graphs` | Number of graphs with `valid == true` that contributed to the means. |
| `mean_paper_sufficiency` | Mean of `paper_sufficiency` over valid graphs (NaN-skipped). |
| `mean_paper_comprehensiveness` | Mean of `paper_comprehensiveness` over valid graphs. |
| `mean_paper_f1_fidelity` | Mean of `paper_f1_fidelity` over valid graphs. |
| `mean_pyg_fidelity_plus` | Mean PyG Fid+ over valid graphs. |
| `mean_pyg_fidelity_minus` | Mean PyG Fid- over valid graphs. |
| `mean_pyg_characterization_score` | Mean PyG characterization score over valid graphs. |
| `mean_pyg_fidelity_curve_auc` | Mean PyG fidelity-curve AUC over valid graphs. |
| `mean_pyg_unfaithfulness` | Mean PyG unfaithfulness (GEF) over valid graphs. |

### `result_metrics`

| Column | Description |
|----|----|
| `wall_time_s` | Total wall-clock seconds for the explainer's loop over the fold. |
| `num_graphs` | Total number of graphs explained on the split (valid + invalid). |
| `mean_paper_sufficiency` | NaN-skipped mean over **all** graphs. |
| `mean_paper_comprehensiveness` | NaN-skipped mean over all graphs. |
| `mean_paper_f1_fidelity` | NaN-skipped mean over all graphs. |
| `mean_pyg_fidelity_plus` | NaN-skipped mean over all graphs. |
| `mean_pyg_fidelity_minus` | NaN-skipped mean over all graphs. |
| `mean_pyg_characterization_score` | NaN-skipped mean over all graphs. |
| `mean_pyg_fidelity_curve_auc` | NaN-skipped mean over all graphs. |
| `mean_pyg_unfaithfulness` | NaN-skipped mean over all graphs. |
