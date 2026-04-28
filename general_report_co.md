# General Report: GNN Explainer Metric Implementation and Result Review

Project reviewed: **"A Comparative Study of Metrics to Evaluate the Explainers of Graph Neural Networks Applied to Compound Binding Affinity Classification."**

Review date: 2026-04-28.

## Scope

This review covers the local code and generated artifacts in:

- `mprov3_gine`: dataset construction, GINE model, training, validation, and classification results.
- `mprov3_explainer`: explainer registry, preprocessing, metric computation, result JSON, and HTML summary tables.
- `mprov3_explainer/results/folds/fold_0` through `fold_4`: per-fold explainer results.
- `mprov3_gine/results`: dataset checks, training summaries, and classification summaries.

The per-fold HTML `summary-table` columns are defined in `mprov3_explainer/src/mprov3_explainer/web_report.py:150-168`. The table contains:

`Explainer`, `Mean Fid+`, `Mean Fid-`, `Mean PyG char`, `Mean Fsuf`, `Mean Fcom`, `Mean Ff1`, `Graphs`, `Valid`, `Wall (s)`, `Mean Fid+ (soft)`, `Mean Fid- (soft)`, `Mean PyG char (soft)`, `Degen.`, `Misclass.`, `Spread`, and `Entropy`.

`Explainer` is an identifier, not a metric. The other numeric columns are reviewed below. The global cross-fold HTML index uses a smaller subset: `Mean Fid+`, `Mean Fid-`, `Mean Fsuf`, `Mean Fcom`, `Mean Ff1`, `Valid`, and `Wall (s)`.

## Project Context

The predictive model is a **3-class GINE graph classifier**, not a continuous affinity regressor. `mprov3_gine/README.md` defines the classes as potency categories derived from `pIC50`: category `-1` for `pIC50 < 5.5`, category `0` for `5.5 <= pIC50 < 6.5`, and category `1` for `pIC50 >= 6.5`, mapped internally to class indices `0`, `1`, and `2`.

The built dataset contains **378 ligand graphs** according to `mprov3_gine/results/check_format/datasets/check_output.log`. The graph representation is ligand-only: node features are 3D coordinates plus atomic number, and edge features are scalar bond types.

Five folds are present. The test accuracies in `mprov3_gine/results/classifications/classification_summary.json` are:

| Fold | Test accuracy |
|---:|---:|
| 0 | 0.618421 |
| 1 | 0.565789 |
| 2 | 0.631579 |
| 3 | 0.653333 |
| 4 | 0.626667 |

The explainer results cover eight explainers on the test split for all five folds: `GRADEXPINODE`, `GRADEXPLEDGE`, `GUIDEDBP`, `IGNODE`, `IGEDGE`, `GNNEXPL`, `PGEXPL`, and `PGMEXPL`. The current run uses seed `42`, top-k fraction `0.2`, and Longa-style threshold count `100`.

I ran the explainer test suite with:

```bash
mprov3_explainer/.venv/bin/python -m pytest mprov3_explainer/tests
```

Result: **35 passed**, with only dependency deprecation warnings.

## Result-Level Observations

The current explainer result set contains **40 explainer-fold summary rows** and **3024 per-graph rows**.

Important observations:

- `PGEXPL` is unusable in the headline summary as currently produced. Across all five folds, `num_valid = 0`, `num_degenerate_mask` is 75 or 76, `mean_mask_spread = 0`, and all valid-only headline metrics are `NaN`.
- For `PGEXPL`, the JSON files contain literal `NaN` tokens. The helper `_nan_to_none` in `run_explanations.py` is passed as `json.dumps(..., default=...)`, but Python's JSON encoder does not call `default` for float `NaN`; therefore strict JSON parsers will reject these files. The reports are readable by Python's permissive parser and by the existing project code, but they are not standards-compliant JSON.
- `PGMEXPL` has a notable anomaly in the `Misclass.` column. For the first six explainers, misclassified counts match the model's test errors by fold: 29, 33, 28, 26, and 28. For `PGMEXPL`, they are 51, 44, 56, 45, and 50. Misclassification should normally be a property of the model and fold, not of the explainer. This should be investigated before drawing conclusions from `PGMEXPL` valid-only means.
- The headline GraphFramEx metrics are **top-k binarized** values. The soft-mask metrics are kept only as diagnostics.
- `Mean Fid+` and `Mean Fid-` are decision-change rates from PyG's `fidelity`, not probability-drop ratios. This distinction is important because the README's short metric table describes a probability-ratio intuition, while the code calls PyG's class-decision-based implementation.

## Observed Ranges in the Summary Table

Observed ranges below are computed across the 40 per-fold/per-explainer summary rows. For metric means, the five `PGEXPL` rows are missing because the valid-only means are `NaN`.

| Summary column | Non-missing rows | Observed min | Observed max | Notes |
|---|---:|---:|---:|---|
| Mean Fid+ | 35 | 0.333333 | 0.734694 | Missing only for `PGEXPL`. |
| Mean Fid- | 35 | 0.354167 | 0.734694 | Missing only for `PGEXPL`. Lower is better for this metric. |
| Mean PyG char | 35 | 0.000000 | 0.448980 | Many rows are exactly zero. |
| Mean Fsuf | 35 | 0.010026 | 0.373276 | Summary means are positive, but per-graph values can be negative. |
| Mean Fcom | 35 | 0.162018 | 0.512676 | Summary means are positive, but per-graph values can be negative. |
| Mean Ff1 | 35 | 0.233787 | 0.615307 | Clamped into `[0, 1]`. |
| Graphs | 40 | 75 | 76 | Folds 0-2 have 76 test graphs; folds 3-4 have 75. |
| Valid | 40 | 0 | 49 | `PGEXPL` has 0 valid graphs in every fold. |
| Wall (s) | 40 | 59.386529 | 156.410570 | Runtime per explainer per fold. |
| Mean Fid+ (soft) | 35 | 0.150000 | 1.000000 | Missing only for `PGEXPL`. |
| Mean Fid- (soft) | 35 | 0.183673 | 0.734694 | Missing only for `PGEXPL`. |
| Mean PyG char (soft) | 35 | 0.000000 | 0.816327 | Missing only for `PGEXPL`. |
| Degen. | 40 | 0 | 76 | Nonzero only for `PGEXPL`; it is all or nearly all graphs. |
| Misclass. | 40 | 26 | 56 | `PGMEXPL` is anomalously high. |
| Spread | 40 | 0.000000 | 1.000000 | After min-max normalization, most non-degenerate masks have spread 1. |
| Entropy | 40 | 0.000000 | 3.845082 | Not normalized by graph/mask size. |

Per-graph ranges relevant to interpretation:

- `paper_sufficiency`: -0.843830 to 0.870954.
- `paper_comprehensiveness`: -0.940741 to 0.931273.
- `paper_f1_fidelity`: 0.000000 to 0.953762.
- Per-graph PyG fidelity and characterization fields are in `[0, 1]`.

## Metric-by-Metric Review

### 1. Mean Fid+

**Metric name:** `Mean Fid+`, stored as `mean_fidelity_plus`.

**Simple explanation:** This asks: "If I remove the highlighted atoms/bonds/features from the graph, does the model's class decision change?" A higher value means the highlighted part is more necessary for the model's decision.

**Implementation in this codebase:** The raw explanation mask is first preprocessed: signed edge masks are converted to absolute values, signed 1D node masks are converted to absolute values, node-feature masks are reduced to per-node scores when needed, degenerate masks can be filtered, and masks are min-max normalized (`preprocessing.py:128-204`). The headline version then binarizes the mask by keeping the top `k = 0.2` entries (`preprocessing.py:212-246`, `pipeline.py:599-624`) and calls PyG's `fidelity` (`pipeline.py:627-656`). The final table value is a valid-only, NaN-skipped mean over graphs (`run_explanations.py:370-382`).

For most explainers, `explanation_type="model"`, so PyG computes `fid+ = 1 - I(complement_prediction == original_prediction)` for each graph (`torch_geometric/explain/metric/fidelity.py:91-93`). Since this pipeline explains one graph at a time, each per-graph value is usually `0` or `1`, and the mean is a proportion. For `PGEXPL`, `explanation_type="phenomenon"`, so PyG uses correctness with respect to the target class instead (`fidelity.py:94-98`).

**Theoretical range and observed/implemented range:** The theoretical and implemented finite range is `[0, 1]`, with `NaN` when computation fails or no valid graph exists. Observed summary range is `0.333333` to `0.734694`, with five missing `PGEXPL` rows.

**Alignment assessment:** The implementation correctly aligns with PyG/GraphFramEx's class-decision fidelity definition. The top-k binarization is a strong choice because it restores hard-subgraph semantics. The main caveat is documentation/interpretation: this is not a probability-drop ratio, so it should not be described as one.

### 2. Mean Fid-

**Metric name:** `Mean Fid-`, stored as `mean_fidelity_minus`.

**Simple explanation:** This asks: "If I keep only the highlighted part, can the model still make the same decision?" Lower is better: a low `Fid-` means the explanation alone preserves the decision.

**Implementation in this codebase:** It is computed by the same top-k PyG `fidelity` call as `Mean Fid+` (`pipeline.py:627-656`). In PyG model mode, `fid- = 1 - I(explanation_prediction == original_prediction)` (`fidelity.py:91-93`). The report value is a valid-only, NaN-skipped mean (`run_explanations.py:370-382`).

**Theoretical range and observed/implemented range:** The theoretical and implemented finite range is `[0, 1]`, plus possible `NaN`. Observed summary range is `0.354167` to `0.734694`, with five missing `PGEXPL` rows.

**Alignment assessment:** The implementation aligns with PyG's definition. The interpretation must be careful: unlike many "higher is better" metrics, lower `Fid-` is better. This is why the characterization score uses `1 - Fid-`.

### 3. Mean PyG char

**Metric name:** `Mean PyG char`, stored as `mean_pyg_characterization`.

**Simple explanation:** This combines `Fid+` and `Fid-` into one score. It rewards explanations that are necessary when removed (`Fid+` high) and sufficient when kept alone (`Fid-` low).

**Implementation in this codebase:** For each graph, the code calls PyG's `characterization_score(fid_plus, fid_minus, pos_weight=0.5, neg_weight=0.5)` (`pipeline.py:659-689`). PyG computes:

```text
1 / (0.5 / Fid+ + 0.5 / (1 - Fid-))
```

(`torch_geometric/explain/metric/fidelity.py:103-132`). The table value is the valid-only mean of per-graph characterization scores (`run_explanations.py:379`), not the characterization of the already-averaged `Fid+` and `Fid-`.

**Theoretical range and observed/implemented range:** If `Fid+` and `Fid-` are in `[0, 1]`, characterization is in `[0, 1]`, with boundary cases becoming zero or `NaN`. Observed summary range is `0.000000` to `0.448980`, with five missing `PGEXPL` rows.

**Alignment assessment:** The implementation correctly uses PyG's GraphFramEx characterization. Mean-of-per-graph-characterization is defensible, but the thesis should state this aggregation choice explicitly because it is not identical to applying characterization to mean `Fid+` and mean `Fid-`.

### 4. Mean Fsuf

**Metric name:** `Mean Fsuf`, stored as `mean_paper_sufficiency`.

**Simple explanation:** This asks: "If I keep only the explanation, how much does the model's probability for the target class drop?" Lower is better. A good explanation should be sufficient, so the target probability should stay close to the full-graph probability.

**Implementation in this codebase:** The code computes target-class softmax probability on the full graph, then sweeps through top-ranked node or edge fractions. With `n_thresholds=100`, the keep fractions are `0.99, 0.98, ..., 0.01` (`pipeline.py:175-184`). For node-mask explainers, it keeps top-scoring nodes and uses the induced node subgraph (`pipeline.py:187-245`). For edge-only explainers, it keeps top-scoring edges and the endpoint node set (`pipeline.py:292-364`). At each sweep point:

```text
Fsuf contribution = P_target(full graph) - P_target(explanation subgraph)
```

The per-graph `Fsuf` is the average across sweep points. The table value is the valid-only, NaN-skipped mean (`run_explanations.py:380`).

**Theoretical range and observed/implemented range:** Since both probabilities are in `[0, 1]`, the implemented per-graph range is `[-1, 1]`. Negative values mean the explanation subgraph increases target-class probability relative to the full graph. Observed per-graph range is `-0.843830` to `0.870954`; observed summary range is `0.010026` to `0.373276`.

**Alignment assessment:** The implementation aligns with the general sufficiency idea. It is not a literal raw-threshold sweep over mask values; it uses percentile/top-k fractions, which is often more stable across different mask distributions. The edge-native path preserves edge explainer granularity, but if the thesis claims a single Longa-style "common representation" for all explainers, this choice should be explicitly justified because node and edge sweeps are not identical evaluation units.

### 5. Mean Fcom

**Metric name:** `Mean Fcom`, stored as `mean_paper_comprehensiveness`.

**Simple explanation:** This asks: "If I remove the explanation and keep the rest, how much does the target-class probability drop?" Higher is better. A high value means the explanation contains information the rest of the graph cannot replace.

**Implementation in this codebase:** It is computed in the same percentile sweep as `Fsuf`, but on the complement subgraph. For each keep fraction:

```text
Fcom contribution = P_target(full graph) - P_target(complement subgraph)
```

Node and edge dispatch are implemented in `pipeline.py:187-245` and `pipeline.py:292-364`; the dispatcher is in `pipeline.py:447-492`. The table value is the valid-only, NaN-skipped mean (`run_explanations.py:381`).

**Theoretical range and observed/implemented range:** The implemented per-graph range is `[-1, 1]` because it is a probability difference. Observed per-graph range is `-0.940741` to `0.931273`; observed summary range is `0.162018` to `0.512676`.

**Alignment assessment:** The implementation aligns with the comprehensiveness concept. The same caveats apply as for `Fsuf`: percentile thresholds and edge-native dispatch are defensible, but they should be described as implementation choices.

### 6. Mean Ff1

**Metric name:** `Mean Ff1`, stored as `mean_paper_f1_fidelity`.

**Simple explanation:** This is a single score that tries to reward both low sufficiency drop and high comprehensiveness drop. Higher is better.

**Implementation in this codebase:** The code first clamps `Fsuf` and `Fcom` into `[0, 1]`, then computes:

```text
Ff1 = 2 * (1 - Fsuf_clamped) * Fcom_clamped
      / ((1 - Fsuf_clamped) + Fcom_clamped)
```

This is implemented in `pipeline.py:421-444`. The dispatcher returns `(Fsuf, Fcom, Ff1)` in `pipeline.py:447-492`. The table value is the valid-only, NaN-skipped mean (`run_explanations.py:382`).

**Theoretical range and observed/implemented range:** The intended F-score range is `[0, 1]`. The implementation guarantees `[0, 1]` for finite inputs and propagates `NaN` when either input is `NaN`. Observed per-graph range is `0.000000` to `0.953762`; observed summary range is `0.233787` to `0.615307`.

**Alignment assessment:** The formula is aligned with the F-score-style definition when `Fsuf` and `Fcom` are already in the assumed domain. The clamp is a practical safety fix because raw probability differences can be negative in this project. However, clamping is not a neutral operation: it can hide the fact that a subgraph or complement increased target probability. The thesis should report and justify this.

### 7. Graphs

**Metric name:** `Graphs`, stored as `num_graphs`.

**Simple explanation:** Number of graphs attempted for that explainer in that fold.

**Implementation in this codebase:** It is `len(results)` in `run_explanations.py:469`, after the explainer loop has processed the selected split.

**Theoretical range and observed/implemented range:** Integer in `[0, infinity)`. Observed range is `75` to `76`.

**Alignment assessment:** This is an administrative count, not an explainer-quality metric. It is implemented correctly and is useful for checking fold sizes.

### 8. Valid

**Metric name:** `Valid`, stored as `num_valid`.

**Simple explanation:** Number of graphs that actually contribute to headline mean metrics.

**Implementation in this codebase:** `valid_results = [r for r in results if r.valid]`; `num_valid = len(valid_results)` (`run_explanations.py:370-371`). A graph can become invalid if the model prediction is not the target class (`correct_class_only=True`), if mask spread is below tolerance, or if top-k/paper metrics become `NaN` (`preprocessing.py:151-177`, `pipeline.py:998-1011`).

**Theoretical range and observed/implemented range:** Integer in `[0, num_graphs]`. Observed range is `0` to `49`.

**Alignment assessment:** The implementation is internally consistent. Methodologically, valid-only aggregation changes the research question to "How good are explanations for correct, non-degenerate cases?" That is defensible, but not equivalent to evaluating explanations over the whole test split. Both valid-only and all-graph metrics should be reported in thesis conclusions.

### 9. Wall (s)

**Metric name:** `Wall (s)`, stored as `wall_time_s`.

**Simple explanation:** How long that explainer took to run for the fold.

**Implementation in this codebase:** The script measures wall-clock time around `run_explanations` in `run_one_explainer` and writes it into `explanation_report.json` (`run_explanations.py:447-479`).

**Theoretical range and observed/implemented range:** Real number in `[0, infinity)`. Observed range is `59.386529` to `156.410570` seconds.

**Alignment assessment:** Correct as a runtime diagnostic. It is not a fidelity or explanation-quality metric.

### 10. Mean Fid+ (soft)

**Metric name:** `Mean Fid+ (soft)`, stored as `mean_fidelity_plus_soft`.

**Simple explanation:** Same intuition as `Mean Fid+`, but uses the continuous soft mask directly instead of converting it into a hard top-k subgraph.

**Implementation in this codebase:** It calls `_compute_pyg_fidelity` before top-k binarization (`pipeline.py:573-596`, `pipeline.py:943-957`). The value is averaged over valid graphs (`run_explanations.py:395`).

**Theoretical range and observed/implemented range:** PyG fidelity remains in `[0, 1]`, plus possible `NaN`. Observed summary range is `0.150000` to `1.000000`, with five missing `PGEXPL` rows.

**Alignment assessment:** It is correctly computed for the soft-mask perturbation PyG receives. It is less aligned with GraphFramEx hard-subgraph semantics because `mask * x` and `(1 - mask) * x` are continuous rescalings, not discrete graph subsets. It is best treated as a diagnostic, as the code already does.

### 11. Mean Fid- (soft)

**Metric name:** `Mean Fid- (soft)`, stored as `mean_fidelity_minus_soft`.

**Simple explanation:** Same intuition as `Mean Fid-`, but with the continuous soft mask instead of a hard top-k mask.

**Implementation in this codebase:** Same `_compute_pyg_fidelity` call as `Mean Fid+ (soft)` (`pipeline.py:573-596`, `pipeline.py:943-957`), then valid-only averaging (`run_explanations.py:396`).

**Theoretical range and observed/implemented range:** PyG fidelity remains in `[0, 1]`, plus possible `NaN`. Observed summary range is `0.183673` to `0.734694`, with five missing `PGEXPL` rows.

**Alignment assessment:** Correct as a soft-mask diagnostic. It should not be used as the main GraphFramEx result if the thesis describes explanations as hard explanatory subgraphs.

### 12. Mean PyG char (soft)

**Metric name:** `Mean PyG char (soft)`, stored as `mean_pyg_characterization_soft`.

**Simple explanation:** Single-score combination of the two soft-mask fidelity values.

**Implementation in this codebase:** The code calls `_compute_pyg_characterization` on `fid_plus_soft` and `fid_minus_soft` (`pipeline.py:951-957`) and averages valid graphs (`run_explanations.py:397`).

**Theoretical range and observed/implemented range:** `[0, 1]` for finite fidelity inputs, plus possible `NaN`. Observed summary range is `0.000000` to `0.816327`, with five missing `PGEXPL` rows.

**Alignment assessment:** Correct for PyG characterization applied to soft fidelity. It is useful for comparing against older results, but less theoretically clean than the top-k headline characterization.

### 13. Degen.

**Metric name:** `Degen.`, stored as `num_degenerate_mask`.

**Simple explanation:** Counts how many explanations produced an almost constant mask. A constant mask does not meaningfully rank atoms, features, or bonds.

**Implementation in this codebase:** The code computes a representative mask for each explanation, preferring node mask over edge mask (`pipeline.py:723-730`), then computes spread as `max(mask) - min(mask)` (`pipeline.py:692-699`). A graph is counted as degenerate if this spread is below `MASK_SPREAD_TOLERANCE = 1e-3` (`run_explanations.py:399-400`; tolerance defined consistently with `preprocessing.py:57-82`).

**Theoretical range and observed/implemented range:** Integer in `[0, num_graphs]`. Observed range is `0` to `76`. All nonzero cases are `PGEXPL`.

**Alignment assessment:** Correct as a diagnostic. It is not a standard explainer metric by itself, but it is essential here because it explains why `PGEXPL` has no valid headline values.

### 14. Misclass.

**Metric name:** `Misclass.`, stored as `num_misclassified`.

**Simple explanation:** Number of graphs in the fold where the model's predicted class does not match the ground-truth class.

**Implementation in this codebase:** The explainer loop computes `pred_class` from the model logits and `target_class` from `data.category`; preprocessing stores `correct_class = pred_class == target_class` (`pipeline.py:912-934`, `preprocessing.py:151-155`). The summary reports `sum(1 for r in results if not r.correct_class)` (`run_explanations.py:401`).

**Theoretical range and observed/implemented range:** Integer in `[0, num_graphs]`. Observed range is `26` to `56`.

**Alignment assessment:** Conceptually correct, but the results expose a problem: `PGMEXPL` reports much higher misclassification counts than the other explainers on the same folds. Because misclassification should be independent of explainer method, this is a red flag for the `PGMEXPL` execution path or model state handling.

### 15. Spread

**Metric name:** `Spread`, stored as `mean_mask_spread`.

**Simple explanation:** Average difference between the largest and smallest mask score. Higher spread means the explainer produces more separated importance scores; zero means constant or empty importance.

**Implementation in this codebase:** For each graph, the representative mask is chosen (`pipeline.py:723-730`), spread is computed as `max - min` (`pipeline.py:692-699`), and the summary table reports the mean over all graphs (`run_explanations.py:402`).

**Theoretical range and observed/implemented range:** For raw masks, the range is `[0, infinity)`. In this implementation, masks are usually min-max normalized before diagnostics, so non-degenerate masks typically have spread `1`, while degenerate masks have spread `0`. Observed summary range is `0.000000` to `1.000000`.

**Alignment assessment:** Correct as a degeneracy/sharpness diagnostic. Because min-max normalization collapses most non-degenerate rows to spread `1`, it is not very informative as a fine-grained quality metric.

### 16. Entropy

**Metric name:** `Entropy`, stored as `mean_mask_entropy`.

**Simple explanation:** Measures how diffuse the importance scores are. Low entropy means importance is concentrated on few atoms/bonds/features; high entropy means importance is spread across many entries.

**Implementation in this codebase:** The representative mask is flattened, converted to absolute values, normalized by its sum into a probability distribution, and Shannon entropy in nats is computed (`pipeline.py:702-720`). Empty or all-zero masks return entropy `0`. The table reports the mean over all graphs (`run_explanations.py:403`).

**Theoretical range and observed/implemented range:** For a mask with `M` positive entries, entropy is in `[0, log(M)]`. Because `M` differs by molecule and by mask type, the maximum differs across node and edge explainers. Observed per-graph range is `0.000000` to `4.503914`; observed summary range is `0.000000` to `3.845082`.

**Alignment assessment:** Correct as an entropy diagnostic. It is not size-normalized, so comparing raw entropy across molecules or between node-mask and edge-mask explainers can be unfair. A normalized entropy, such as `entropy / log(M)`, would be more comparable.

## Overall Assessment

The metric implementation is generally strong and better than a naive soft-mask-only evaluation. In particular:

- The top-k headline fidelity avoids the weak semantics of continuous soft-mask rescaling.
- NaN-aware aggregation avoids silently averaging failed metrics as zero.
- The code keeps diagnostic soft-mask metrics and all-graph metrics, which is useful for auditability.
- The tests cover important corner cases: top-k binarization, Ff1 clamping, percentile sweeps, NaN handling, and end-to-end metric fields.

The main concerns to address before using the results as final thesis evidence are:

1. `PGEXPL` currently has no valid headline results because all masks are degenerate.
2. `PGMEXPL` reports inconsistent misclassification counts across the same folds, unlike every other explainer.
3. `Mean Fid+` and `Mean Fid-` must be described as PyG class-decision GraphFramEx metrics, not probability-drop ratios.
4. `Fsuf` and `Fcom` can be negative. The report should explain that these are raw probability differences and that `Ff1` clamps them before combination.
5. Valid-only means should not be the only thesis conclusion; the all-graph diagnostic means should also be reported or at least discussed.
6. The result JSON files should be made strict-JSON compliant by replacing `NaN` with `null` before serialization.
7. Raw entropy should be normalized by mask size if it is used for explainer comparison.

## Practical Recommendations

- Investigate `PGEXPL` training and mask generation. Current `PGEXPL` rows should be treated as failed runs, not low-quality but valid explainer evidence.
- Investigate why `PGMEXPL` changes `Misclass.` counts. This should be stable across explainers for the same model and fold.
- Add a small strict-JSON serialization test using `json.dumps(..., allow_nan=False)` or a strict parser, and recursively replace non-finite floats before writing result files.
- Report both valid-only and all-graph means in thesis tables.
- Add sensitivity analysis for `top_k_fraction` values such as `0.1`, `0.2`, and `0.3`.
- Add normalized entropy or explicit sparsity metrics if `Spread` and `Entropy` are interpreted as explainer-quality metrics.
- Clarify the thesis wording: the implemented task is ligand potency category classification from discretized `pIC50`, not direct continuous binding affinity regression.
