# GNN Explainer Results — `summary-table` Backtrack and Audit

**Source:** `mprov3_explainer/results/folds/fold_0/explanation_web_report/index.html`
**Underlying JSON:** `mprov3_explainer/results/folds/fold_0/explanations/<EXPL>/explanation_report.json`
**Code paths inspected:**

- `mprov3_explainer/src/mprov3_explainer/pipeline.py` (metric computation)
- `mprov3_explainer/src/mprov3_explainer/preprocessing.py` (mask conversion / normalization / spread filter)
- `mprov3_explainer/src/mprov3_explainer/explainers.py` (per‑explainer mask types)
- `mprov3_explainer/src/mprov3_explainer/web_report.py` (table rendering)
- `mprov3_explainer/scripts/run_explanations.py` (aggregation, threshold sweep size)
- PyG (vendored): `.venv/.../torch_geometric/explain/metric/fidelity.py`,
  `.venv/.../torch_geometric/explain/explainer.py::get_masked_prediction`

---

## 1. The `summary-table` snapshot (fold 0, test split)

Numbers extracted verbatim from the HTML (and re‑confirmed against
`explanation_report.json` for a few rows):

| Explainer       | Mean Fid+ | Mean Fid− | Mean PyG char | Mean Fsuf  | Mean Fcom    | Mean Ff1      | Graphs | Valid | Wall (s) |
|-----------------|-----------|-----------|---------------|------------|--------------|---------------|--------|-------|----------|
| GRADEXPINODE    | 0.6316    | 0.6316    | 0.000         | 0.1423     | 0.1519       | −0.0478       | 76     | 52    | 64.5     |
| GRADEXPLEDGE    | 0.5921    | 0.6316    | 0.000         | 0.1201     | 0.1415       | −0.0976       | 76     | 52    | 58.1     |
| GUIDEDBP        | 0.6316    | 0.6316    | 0.000         | 0.0836     | 0.2000       | +0.0640       | 76     | 52    | 57.3     |
| IGNODE          | 0.6316    | 0.6316    | 0.000         | 0.1952     | 0.1013       | −0.1172       | 76     | 52    | 60.0     |
| IGEDGE          | 0.6316    | 0.5921    | 0.0395        | 0.0239     | 0.2225       | +0.0390       | 76     | 52    | 59.8     |
| GNNEXPL         | 0.6316    | 0.3816    | 0.250         | 0.0086     | 0.1754       | −0.2630       | 76     | 52    | 145.0    |
| PGEXPL          | 0.000     | 0.5132    | 0.000         | 0.5907     | −9.7 × 10⁻¹¹ | −1.9 × 10⁻¹⁰  | 76     | 0     | 107.7    |
| PGMEXPL         | 0.1842    | 0.6447    | 0.000         | 0.000      | 0.000        | 0.000         | 76     | 26    | 48.1     |

`Graphs = 76` is the size of the test split for fold 0. `Valid` is the count
that passed `correct_class && mask_spread > τ` with τ = 1e‑3.

---

## 2. What each column actually means in this code

### 2.1 Mean Fid+ — `mean_fidelity_plus`

- Defined in `pipeline.py::_compute_pyg_fidelity`, which calls
  `torch_geometric.explain.metric.fidelity()` per graph and **stores the first
  element of the returned tuple**.
- For our setup `explanation_type = ExplanationType.model` (from
  `DEFAULT_EXPLANATION_TYPE` = `model`, applied to every Captum/GNN/IG
  explainer; PGExplainer uses `phenomenon`).
- For `model` explanations PyG’s formula is
  $$\text{fid}_+ = 1 - \frac{1}{N}\sum_i \mathbb{1}\!\left(\hat{y}_i^{G_{C\setminus S}} = \hat{y}_i\right)$$
  i.e. *“fraction of times the **complement** subgraph’s prediction differs
  from the original full‑graph prediction”*. Each graph is one independent
  call, so `N = 1` and the per‑graph `fid_plus ∈ {0, 1}`. The mean across the
  76 graphs is what shows in the column.
- Implementation detail: `_fidelity_explanation` reshapes `node_mask` to
  `(N, 1)` so `node_mask * x` broadcasts correctly across feature columns;
  this only changes the layout, not the soft/binary nature of the mask.
- **Higher is better** (the explanation’s complement should disturb the
  prediction).

### 2.2 Mean Fid− — `mean_fidelity_minus`

- Second element of the same `fidelity()` tuple.
- For `model` explanations:
  $$\text{fid}_- = 1 - \frac{1}{N}\sum_i \mathbb{1}\!\left(\hat{y}_i^{G_S} = \hat{y}_i\right)$$
  i.e. *“fraction of times the **explanation subgraph** prediction differs
  from the original full‑graph prediction”*.
- **Lower is better** (keeping only the explanation should preserve the
  prediction).

### 2.3 Mean PyG char — `mean_pyg_characterization`

- `_compute_pyg_characterization` calls
  `characterization_score(fid+, fid−, pos_weight=0.5, neg_weight=0.5)`.
- Formula (PyG / GraphFramEx):
  $$\text{char} = \frac{w_+ + w_-}{\dfrac{w_+}{\text{fid}_+} + \dfrac{w_-}{1-\text{fid}_-}}$$
  i.e. weighted **harmonic mean** of $\text{fid}_+$ and $1 - \text{fid}_-$.
- The mean is the arithmetic average of per‑graph characterization scores.
- **Higher is better** (ideal = 1 when fid+ = 1 *and* fid− = 0).

### 2.4 Mean Fsuf — `mean_paper_sufficiency`

- Computed in `_paper_sufficiency_and_comprehensiveness` (Longa et al. 2025).
- Threshold sweep over the **normalized [0, 1] node mask** (edge‑only
  explainers are converted to a node mask via mean of incident edges in
  `edge_mask_to_node_mask`). For
  $t = 1/N_t, 2/N_t, \dots, (N_t-1)/N_t$ with $N_t = 100$ (so 99 thresholds):
  $$F_{\text{suf}} = \frac{1}{N_t-1}\sum_t \big( p(y\mid G) - p(y\mid G_{S_t}) \big)$$
  where $G_{S_t}$ is the induced subgraph on $\{v : m_v > t\}$ and
  $p(y\mid \cdot)$ is the softmax probability of the **target class**.
- **Lower is better** (subgraph alone should still produce the target class
  prediction with full confidence).

### 2.5 Mean Fcom — `mean_paper_comprehensiveness`

- Same sweep, complementary subgraph $G_{C_t}$ on $\{v : m_v \leq t\}$:
  $$F_{\text{com}} = \frac{1}{N_t-1}\sum_t \big( p(y\mid G) - p(y\mid G_{C_t}) \big)$$
- **Higher is better** (removing the explanation should drop confidence).

### 2.6 Mean Ff1 — `mean_paper_f1_fidelity`

- `_paper_f1_fidelity(Fsuf, Fcom)`:
  $$F_1^{\text{fid}} = \frac{2\,(1 - F_{\text{suf}})\,F_{\text{com}}}
                            {(1 - F_{\text{suf}}) + F_{\text{com}}}$$
  the harmonic mean of $(1 - F_{\text{suf}})$ and $F_{\text{com}}$.
- **Higher is better** *under the implicit assumption that both arguments are
  in $[0, 1]$* — see §5 for the bug this assumption hides.

### 2.7 Graphs / Valid

- `num_graphs` = number of graphs streamed through the loop (full split).
- `num_valid` = graphs where `valid == True`. From `apply_preprocessing`:
  `valid = correct_class AND (mask_spread ≥ 1e-3)`. So a graph fails validity
  if (a) the model misclassified it (because `correct_class_only=True`) or
  (b) its post‑processed mask is essentially constant.

### 2.8 Wall (s) — `wall_time_s`

- Total `time.perf_counter()` for the whole explanation loop of that
  explainer (training + per‑graph forward + metrics + I/O excluded except for
  the loop itself; mask serialization is outside this measurement).

---

## 3. Theoretical valid ranges for each metric

| Metric    | Mathematical range           | Sensible range in this code                         | Direction |
|-----------|------------------------------|-----------------------------------------------------|-----------|
| Fid+      | $[0, 1]$                     | $[0, 1]$ (per‑graph in $\{0,1\}$, mean in $[0,1]$)  | ↑ better  |
| Fid−      | $[0, 1]$                     | $[0, 1]$                                            | ↓ better  |
| PyG char  | $[0, 1]$                     | $[0, 1]$ — but **0 whenever fid+ = fid−** (see §5)  | ↑ better  |
| Fsuf      | Paper: $[0, 1]$              | **Code allows $[-1, 1]$** because nothing clamps    | ↓ better  |
| Fcom      | Paper: $[0, 1]$              | **Code allows $[-1, 1]$**                           | ↑ better  |
| Ff1       | Paper: $[0, 1]$              | **Code allows $\mathbb{R}$** (no clamp, no abs)     | ↑ better  |
| Graphs    | $\mathbb{N}$                 | exactly the loader size (76 for test, fold 0)       | —         |
| Valid     | $\{0, …, \text{Graphs}\}$    | $\le$ test accuracy × 76 = ~52                      | ↑ better  |
| Wall (s)  | $\mathbb{R}_{\ge 0}$         | depends on explainer (PGEXPL/GNNEXPL longer)        | ↓ better  |

---

## 4. How good are the values in `summary-table`?

In one phrase: **the explainers are mediocre to poor on this model/dataset
combination**. Per‑metric reading:

- **Fid+ ≈ 0.63 across the gradient/Captum family.** Only ~63% of complement
  subgraphs flip the prediction, and PGEXPL never flips it (Fid+ = 0).
  GraphFramEx considers Fid+ ≥ 0.7 a competent explainer; everyone here is
  below the bar.
- **Fid− is very high (≈ 0.6–0.65).** This is *bad*. It says the explanation
  subgraph alone changes the prediction in ~60% of graphs — the masks do not
  preserve the predictive signal. A good explainer should give Fid− close to
  zero.
- **PyG char ≤ 0.25 everywhere.** GNNExplainer is the best at 0.25 (mediocre);
  most others sit at 0 because Fid+ = Fid− per‑graph (see §5.1). 0.25 is well
  below the 0.6–0.8 commonly seen in GraphFramEx benchmarks for synthetic
  datasets.
- **Fsuf ≈ 0.0–0.6.** PGEXPL’s 0.59 says its “explanation” is essentially the
  graph minus a non‑predictive edge set — confidence collapses. The
  Captum/IG/GuidedBP/GNNExpl explainers all sit between 0 and 0.2, which is
  reasonable in absolute terms, but combined with the next column it is
  clearly noisy rather than informative (means dragged toward zero by sign
  cancellation, not by true sufficiency).
- **Fcom ≈ 0.10–0.22**, except PGEXPL ≈ 0 and PGMEXPL = 0. These are *low* —
  removing the explanation barely hurts the prediction. A useful explanation
  should drive Fcom much closer to 1.
- **Ff1 ranges from −0.26 (GNNEXPL) to +0.06 (GUIDEDBP).** The negatives are
  not just numerically tiny — for GNNEXPL `−0.263` is a *large* negative
  number, signalling a structural problem with the formula on this code path
  (see §5.2).
- **Valid count.** 52/76 ≈ 68 % matches the model’s test accuracy on this fold
  for most explainers (i.e. the only graphs marked invalid are the
  misclassified ones; the masks themselves are non‑degenerate for those
  explainers). PGMEXPL drops further to 26/76 (chi‑square p‑values are flat
  for many graphs, killing the spread filter), and PGEXPL collapses to 0/76
  (uniform edge masks, see §5.4).
- **Wall time.** GNNEXPL and PGEXPL dominate (per‑instance optimization /
  MLP training). The Captum family is fastest (single backward pass).

**Bottom line for the thesis:** none of the eight explainers is producing a
genuinely faithful explanation of this GINE model on this MPro dataset. The
gradient‑based methods are mathematically well‑defined here but the model
seems to be a mostly “distributed” learner — soft‑mask multiplication of node
features barely moves the prediction, so PyG fidelity reduces to two paired
indicators that almost always agree.

---

## 5. Why the table looks suspicious — bug / methodology audit

### 5.1 Why so many `0` in the *Mean PyG char* column

This is **expected given the rest of the data, but it exposes a methodological
problem**. The PyG `characterization_score` is

$$\text{char} = \frac{1}{0.5/\text{fid}_+ + 0.5/(1-\text{fid}_-)}.$$

For one graph the per‑graph fids are `(fid+, fid−) ∈ {0, 1}²`. With
`pos_weight = neg_weight = 0.5`:

- `(0, 0)` → `0 / (0.5/0 + 0.5/1)` = **0** (and PyTorch returns 0 because the
  $w_+/0$ term yields `inf`, dominating the denominator).
- `(0, 1)` → `0 / (∞ + ∞)` = **0**.
- `(1, 0)` → `1 / (0.5/1 + 0.5/1)` = **1**.
- `(1, 1)` → `1 / (0.5 + ∞)` = **0**.

So `char` is **non‑zero only on the graphs where fid+ = 1 and fid− = 0
simultaneously**. Counting those rows in the per‑graph JSON:

- GRADEXPINODE / GRADEXPLEDGE / GUIDEDBP / IGNODE: **0** such rows → mean = 0.
- IGEDGE: **3** such rows → mean = 3/76 = 0.039.
- GNNEXPL: **19** such rows → mean = 19/76 = 0.25.
- PGEXPL: 0 (Fid+ is always 0 because complement = 1 − uniform_mask still
  predicts the same class).
- PGMEXPL: 0.

So the column does reflect a real property of the data. **But it is
amplified by a real defect**: the soft `node_mask ∈ [0, 1]` is fed straight
into PyG’s fidelity, so `node_mask * x` and `(1 − node_mask) * x` are *both*
just rescaled versions of `x`. For a robust pooled GNN like GINE, both
predictions usually equal the original prediction (giving `(0, 0)` →
char = 0) or both flip (giving `(1, 1)` → char = 0). This is a documented
weakness of GraphFramEx fidelity when used with **soft** masks instead of
**top‑k binary** masks.

**Verdict:** the zeros are *not* a calculation typo, they are a consequence
of (a) PyG’s harmonic‑mean formula and (b) the *use of soft masks* in
`fidelity()`. The fix is to binarize the mask at top‑k% before calling
`fidelity()` — see the recommendations in §7.

### 5.2 Why so many negatives in *Mean Ff1*

`_paper_f1_fidelity(Fsuf, Fcom)` is

```python
num = 2.0 * (1.0 - Fsuf) * Fcom
den = (1.0 - Fsuf) + Fcom
return num / den
```

This is the harmonic mean of `(1 − Fsuf)` and `Fcom`, **but only valid when
both arguments are ≥ 0**. Three things break that assumption in this code:

1. `paper_sufficiency = full_prob − exp_prob` is a *signed* probability
   difference. It can be negative (the subgraph is *more* confident than the
   full graph) — common with normalized soft masks and pooled global readouts.
2. `paper_comprehensiveness = full_prob − comp_prob` can also be negative for
   the same reason.
3. The threshold sweep averages over 99 thresholds, so the mean carries the
   sign — and `(1 − Fsuf)` is > 1 whenever Fsuf < 0.

Concrete worked example from `GNNEXPL` graph `7GCK` (per‑graph JSON):
`Fsuf = −0.253`, `Fcom = −0.205` →
`num = 2 · 1.253 · (−0.205) = −0.514`, `den = 1.253 + (−0.205) = 1.048`,
`Ff1 = −0.490`. Several such large‑magnitude negative graphs drag the mean to
−0.263 for GNNEXPL.

**Verdict:** **this is a real bug.** The Longa et al. paper defines $F_1$
fidelity on $F_{\text{suf}}, F_{\text{com}} \in [0, 1]$. The code never
clamps, never takes absolute values, and never warns. Two acceptable fixes:

- **Best:** clip $F_{\text{suf}}, F_{\text{com}}$ to $[0, 1]$ before calling
  the F1 formula (and warn / log when clipping happens — large clips signal
  noise);
- **Acceptable:** redefine the metric as `0` when either factor is negative
  (so per‑graph $F_1^{\text{fid}} \in [0, 1]$ as in the paper).
  Returning a raw negative number, as today, makes the column scientifically
  uninterpretable.

The two near‑zero negatives in the **PGEXPL** row (~ −2e‑10) are different:
they are floating‑point noise around `Fcom ≈ 0` (uniform edge mask →
complement ≈ explanation ≈ full graph). They are not a bug in arithmetic but
they are a symptom that the explainer has nothing to explain.

### 5.3 Other latent bugs / dubious choices in the metric pipeline

- **Means are computed over *all* graphs, not just `valid` ones.** In
  `run_explanations.py::run_one_explainer`:
  ```python
  mean_fid_plus, mean_fid_minus = aggregate_fidelity(results, valid_only=False)
  mean_char = sum(r.pyg_characterization for r in results) / len(results)
  ...
  ```
  This is fine for sanity, but the *headline* numbers in the report should
  also be available `valid_only=True`. PGEXPL is the obvious victim: 0 valid
  graphs → reporting Fsuf = 0.59 over invalid masks is misleading. A clean
  report would show *both* (overall, valid‑only) means, or only the
  valid‑only mean and a separate `degenerate_count`.

- **`_compute_pyg_fidelity` swallows exceptions and silently returns
  `(0.0, 0.0)`.** That value is then averaged in. A failed call should mark
  the graph invalid, not be silently averaged as a perfect/ broken score.

- **Edge masks are not taken in absolute value before normalization.**
  `preprocessing.py` does `reduce_node_mask = .abs().mean(dim=-1)` for
  multi‑feature node masks, but for 1‑D node masks and for **edge masks**
  there is no `.abs()` call — `normalize_mask` then linearly maps the
  signed range to `[0, 1]`. For IG/GuidedBP/Saliency on edges, a strongly
  negative attribution and a near‑zero attribution end up at almost the same
  normalized score, while a small positive attribution may be ranked above a
  large‑magnitude negative one. Captum’s `Saliency` already returns absolute
  values, but `GuidedBackprop` and `IntegratedGradients` do not. This is a
  small but real correctness issue affecting GRADEXPLEDGE, IGEDGE, IGNODE,
  GUIDEDBP at the 1‑D path.

- **`normalize_mask` uses `1e-12` as the “constant‑mask” threshold but the
  spread filter uses `1e-3`.** The two thresholds are inconsistent: a mask
  with spread between 1e‑12 and 1e‑3 is *normalized* (so it stretches across
  the full $[0,1]$) but later marked **invalid**. This is fine for filtering,
  but the normalized mask is what is written to disk and is what PyG’s
  fidelity sees — so the metric is computed on a stretched representation
  even for graphs that are eventually flagged invalid. (PGEXPL is exactly
  this case: spread is order 1e‑8, normalized to a noisy `[0,1]` mask, fed to
  fidelity, mean Fsuf 0.59 reported, then flagged invalid.) The cleanest fix
  is to raise `normalize_mask`’s threshold to match the spread filter (or
  return early without computing metrics for invalid graphs).

- **The threshold sweep’s lower bound.** `for k in range(1, Nt)` skips
  `t = 0` and `t = Nt/Nt = 1`. Together with the strict `node_mask > t` and
  `node_mask` minimum exactly 0 after normalization, the *first* iteration
  (`t = 0.01`) drops only the single minimum‑importance node. That biases the
  Fsuf estimate toward 0 (subgraph is essentially the full graph) and Fcom
  toward 0 too. A finer sweep with **percentile thresholds** (e.g. drop the
  lowest 1 %, 5 %, …, 95 % of nodes) is the standard practice in
  GraphFramEx/Longa. Implementing the sweep on **k‑percentages** rather than
  **score‑thresholds** would also automatically cope with masks that are not
  evenly distributed in `[0, 1]`.

- **Edge‑only explainers folded into a node mask for the paper metrics.**
  `_paper_normalized_node_mask_from_explanation` calls
  `edge_mask_to_node_mask(... aggregation="mean")`, then re‑normalizes. This
  *throws away the explanation’s native granularity* and is mathematically
  asymmetric: a node with one important incident edge and 10 unimportant ones
  receives a small score. PyG fidelity does not have this problem because it
  applies the edge mask in message passing directly. The two metric families
  therefore see *different* explanations, which is part of why the columns
  disagree (e.g., `IGEDGE` PyG char 0.04 vs Ff1 +0.04 vs Fsuf 0.024 vs Fcom
  0.22 — they are not on the same axis at all).

- **Determinism / RNG.** No global `torch.manual_seed`. PGExplainer’s MLP
  init, IG’s baseline interpolation, PGMExplainer’s perturbation samples and
  GNNExplainer’s init are all RNG‑driven. The `wall_time_s` in the table is
  thus repeatable but the *metric values* are not — re‑running PGEXPL or
  PGMEXPL will give different `num_valid`. For a thesis, fix the seed in
  `run_explanations.py`.

### 5.4 Why PGEXPL has 0 valid and PGMEXPL only 26

- **PGEXPL.** Trained for `DEFAULT_PG_EXPLAINER_EPOCHS` epochs on the train
  split, then asked to produce edge masks for the test split. Inspection of
  one mask file (e.g. `PGEXPL/masks/7GCK.json`) shows `edge_mask` values that
  differ by less than `1e‑3` (max − min) — the spread filter rejects them
  all. Concretely, the per‑graph numbers in `PGEXPL/explanation_report.json`
  show `paper_comprehensiveness` of order $10^{-8}$, which is the signature
  of an effectively constant mask. Likely causes:
  1. The MLP did not learn (insufficient epochs, learning rate too high/low,
     or no early stopping based on validation).
  2. The phenomenon target is too easy (a global pooled binary problem),
     pushing the MLP toward a saturated edge probability close to a constant.
  3. PyG’s PGExplainer applies a temperature schedule that, late in training,
     squashes outputs — without a stopping criterion the masks collapse.

  Recommendations for PGEXPL (in order of likely impact):
  - run with fewer epochs and validate, or implement early stopping;
  - lower learning rate;
  - log the *spread* of the produced mask each epoch and stop when it
    plateaus;
  - feed the explainer un‑pooled node embeddings as edge features
    (PGExplainer is sensitive to the embedding it conditions on).

- **PGMEXPL.** PGMExplainer uses chi‑square testing on perturbed forward
  passes; `DEFAULT_PGM_NUM_SAMPLES` controls statistical power. With too few
  samples the test returns identical p‑values for all nodes →
  `normalize_mask` returns zeros (because spread is 0), `mask_spread_filter`
  marks the graph invalid. 50/76 graphs failing means PGMExplainer is
  under‑powered for this dataset. Increase `num_samples` and consider
  filtering nodes whose perturbed forward pass changes the prediction at all
  before the chi‑square step.

---

## 6. Are the metrics computed following scientific best practice?

### 6.1 What is good

- The PyG fidelity / characterization implementation **matches the
  GraphFramEx paper** definitions and is used through the official PyG
  interface — no math errors there.
- The Longa Fsuf / Fcom is a faithful threshold‑sweep implementation and the
  conversion of edge masks to node masks (mean of incident edge weights)
  follows the paper.
- A common preprocessing pipeline (Conversion → Filtering → Normalization) is
  applied uniformly *before* metrics, which is the right ordering.
- The Captum bridge (`LeafInputCaptumExplainer`) correctly fixes a real PyG
  issue with non‑leaf gradients and removes a class of warnings/incorrect
  attributions.
- The table is reproducible from on‑disk JSON; the HTML is purely a renderer.

### 6.2 What violates best practice

| Issue                                                                                                                | Severity | Where                                                  |
|----------------------------------------------------------------------------------------------------------------------|----------|--------------------------------------------------------|
| Means averaged over *all* graphs (incl. invalid)                                                                     | High     | `run_explanations.py::run_one_explainer`               |
| `Ff1` not clamped and can be ±∞ in principle, large negative in practice                                             | High     | `pipeline.py::_paper_f1_fidelity`                      |
| PyG fidelity called with **soft** masks instead of top‑k% binarized masks                                            | High     | `pipeline.py::_compute_pyg_fidelity`                   |
| Exceptions in fidelity / paper metrics **silently coerced to 0**, then averaged                                      | Med      | `_compute_pyg_fidelity`, `_compute_pyg_characterization`, `run_explanations` |
| No `.abs()` on signed 1‑D node masks / edge masks before normalization                                               | Med      | `preprocessing.py::apply_preprocessing`                |
| Threshold sweep uses raw $[0,1]$ thresholds rather than percentile thresholds                                        | Med      | `_paper_sufficiency_and_comprehensiveness`             |
| `normalize_mask` constant tolerance (1e‑12) inconsistent with spread filter (1e‑3) → invalid masks still feed metrics | Med      | `preprocessing.py`                                     |
| Edge‑mask explainers reduced to node masks for paper metrics, asymmetric vs PyG metrics                              | Med      | `_paper_normalized_node_mask_from_explanation`         |
| No RNG seed                                                                                                           | Med      | `run_explanations.py`                                  |
| No confidence intervals / bootstrapped means; single‑fold reporting only                                              | Low      | run‑level                                              |
| `valid_only` mean not reported even though it is the meaningful number                                                | Low      | `aggregate_fidelity`                                   |

---

## 7. Are the masks computed following scientific best practice?
What can be improved?

### 7.1 Per‑explainer assessment

- **GRADEXPINODE / GUIDEDBP.** Captum hooks attached to `nn.ReLU` *modules*
  only. The `mprov3_gine` model uses `F.relu` (functional) in some places —
  any path through `F.relu` is invisible to GuidedBackprop. The README/blurb
  warns about this but the warning does not stop the run. Either replace all
  `F.relu` with `nn.ReLU` modules in `model.py`, or drop GuidedBackprop from
  the comparison.
- **GRADEXPLEDGE / IGEDGE.** Edge attributions are *signed*. Currently no
  `.abs()` before normalization, so direction information leaks into the
  threshold sweep. Standard fix: take `edge_mask = edge_mask.abs()` (or
  `.clamp(min=0)`, or rank by absolute value) before any thresholding.
- **IGNODE / IGEDGE.** Integrated Gradients needs a *baseline*. PyG defaults
  to all‑zero, which for protein graphs (and one‑hot or chemical descriptor
  features) is not a meaningful neutral input. A more principled baseline is
  the mean feature vector across the dataset, or a class‑neutral noisy
  baseline. The blurb mentions step count but not baseline; this is a known
  source of poor IG attributions.
- **GNNEXPL.** PyG’s GNNExplainer optimizes per‑graph soft edge masks with
  entropy + sparsity regularization. Defaults work but `epochs=200`, `lr=1e‑2`
  are PyG defaults rather than tuned; on small graphs the sigmoid often
  saturates. Not a bug but a tuning gap.
- **PGEXPL.** Already discussed in §5.4 — the mask is essentially uniform.
  This is a *training* problem, not a metric problem.
- **PGMEXPL.** Discrete perturbation + chi‑square is conservative; with small
  `num_samples` p‑values are flat. Not a metric problem either.

### 7.2 Cross‑cutting improvements to the mask pipeline

1. **Always take absolute value before normalization** for *signed* masks
   (IG, GuidedBP, raw gradients). Make this a single helper used by both
   node and edge paths. Keep the signed version on disk for downstream
   analysis but feed the absolute value to metrics.
2. **Rank‑transform instead of min‑max normalize.** Min‑max is sensitive to a
   single very large attribution; rank‑normalization (`mask = rank(mask) / N`)
   gives metric values that are far more comparable across explainers.
3. **Binarize at top‑k% for fidelity.** Implement
   `binarize_top_k(mask, k=0.2)` and call PyG fidelity on the binary mask.
   Sweep `k ∈ {0.1, 0.2, 0.3, 0.5}` and report fidelity per `k`. This is what
   GraphFramEx actually defines.
4. **Use percentile thresholds in the Longa sweep.** Replace
   `keep = mask > t` with
   `keep = mask >= torch.quantile(mask, 1 − k)` and iterate over `k` values
   in `[0.05, 0.95]`.
5. **Compute paper metrics on the explainer’s native granularity.** For
   edge‑only explainers, define the explanation subgraph by *edges* (not by
   nodes derived from edges): `keep_edges = edge_mask > t`, then build the
   induced node set from those edges. This is symmetric and matches the
   semantics of edge attribution.
6. **Honor the `valid` flag in aggregation.** Either drop invalid samples
   from the means or compute and report both means (overall and valid‑only).
7. **Replace silent fallbacks with explicit invalid markers.** In
   `_compute_pyg_fidelity` / `_compute_pyg_characterization`, return
   `None`/`NaN` on exception and propagate that into the JSON. The aggregator
   should then average over non‑NaN entries.
8. **Add a `mask_disk_format`.** Store *both* the raw and the normalized mask,
   plus the binarization at top‑k. This makes the visualization layer (the
   PNGs in `visualizations/`) reproducible without re‑running the explainer.
9. **Seed all RNGs** at run start (`torch`, `numpy`, `random`, and PyG’s
   internal `seed_everything`).
10. **Add a degenerate‑mask diagnostic** to the report table:
    `degenerate_count`, `mean_spread`, `mean_entropy(mask)`. PGEXPL and
    PGMEXPL would have stood out immediately with these columns.

---

## 8. TL;DR for the thesis

- The numbers in `summary-table` are **not the result of a single arithmetic
  bug**; they are the visible end of a chain of small, defensible
  individually‑questionable choices that *together* make the explanations
  look worse than they are and the metrics harder to interpret than they
  should be.
- The `Mean PyG char = 0` column is the consequence of using **soft masks**
  with PyG’s **GraphFramEx fidelity formula** plus Fid+ ≡ Fid− per graph.
  Fix: binarize at top‑k% before calling `fidelity()`.
- The negative `Mean Ff1` is a **real implementation bug** in
  `_paper_f1_fidelity` — the formula should only be applied after clamping
  Fsuf and Fcom to $[0, 1]$.
- Aggregation should **exclude invalid graphs**, especially for PGEXPL (0/76
  valid) and PGMEXPL (26/76).
- Mask computation can be improved by (a) taking absolute values for signed
  attributions, (b) rank‑normalizing instead of min‑max, (c) sweeping
  *percentile* thresholds, and (d) keeping edge‑native semantics for
  edge‑only explainers.
- After those fixes, expect Fid+ and char to **rise** (because top‑k binary
  masks really do change the prediction), Ff1 to be **non‑negative and
  meaningful**, and PGEXPL’s 0/76 to become a clear *training* signal rather
  than a metric artifact.
