# Software Proposal for Comparing Graph-Level Explanations of a Protein–Ligand GNN Classifier in PyTorch Geometric

## Executive summary

This proposal designs a **reproducible, explainer-agnostic evaluation framework** to compare graph-level explanations for a single trained GNN classifier on **protein–ligand complex structural graphs** using the specified PyG explainer set: **GRADEXPINODE, GRADEXPLEDGE, GUIDEDBP, IGEDGE, IGNODE, GNNEXPL, PGEXPL, PGMEXPL**. The core architectural decisions are:

- Use **PyG’s `torch_geometric.explain.Explainer` as the execution backend** wherever possible, because it standardizes algorithm invocation, explanation mode selection (`"model"` vs `"phenomenon"`), and optional mask thresholding, and it also attaches inputs/predictions/targets into the returned `Explanation` object for traceability. citeturn14view1turn3view0turn3view2  
- Implement a **unified adapter interface** per explainer alias. The adapter encapsulates: (a) how to instantiate the PyG explainer, (b) which mask types are valid, (c) whether an explainer needs offline training (PGExplainer), per-instance optimization (GNNExplainer), or sampling (PGMExplainer), and (d) how to canonicalize outputs into **fixed tensor shapes suitable for downstream plotting**. citeturn4view0turn16view2turn17view2turn15view4  
- Treat explanations as **single-graph, graph-level** by default (one `Data` / `Batch` item at a time), to avoid concurrency hazards and to respect PyG/Captum implementation constraints (notably Captum’s “single sample” handling inside PyG’s Captum wrapper). citeturn7view1turn2view5turn4view0  
- Standardize postprocessing via three layers: **(1) canonicalization into fixed shapes**, **(2) normalization into comparable scales**, and **(3) selectable sparsification (top‑k, hard thresholding)** using either PyG’s `ThresholdConfig` semantics or a framework-controlled equivalent. citeturn5search1turn4view2turn4view0  
- Evaluate explainers using a mix of **PyG-provided metrics** (fidelity, fidelity-curve AUC, AUROC when a ground-truth mask exists) and **protocol-defined robustness / stability tests** grounded in perturbation-based methodology. citeturn11view2turn11view3turn11view0turn12view0  

Important scope note: PyG’s explainability module explicitly warns it is “in active development” and “may not be stable,” which should be reflected in version pinning and metadata logging. citeturn3view0  

## Assumptions and data/model contract for protein–ligand structural graphs

### Graph representation assumptions

This proposal assumes each protein–ligand complex can be represented as a PyG homogeneous graph `Data` (or `Batch` of size 1), with at minimum:

- `x`: node feature tensor shaped **[N, F]**  
- `edge_index`: COO edges shaped **[2, E]** (PyG standard for message passing flow) citeturn10search9turn10search15  

Optional but recommended fields for chemistry/structure use cases (the framework should pass these through as `**kwargs` to the model/explainers and preserve them in outputs where possible):

- `edge_attr`: [E, Fe] (bond type / distance / interaction class)  
- `pos`: [N, 3] (3D coordinates for visualization/export)  
- node/edge annotations (protein chain/residue IDs, ligand atom IDs, interaction categories)

These are “application-required,” not dictated by PyG’s explain API; PyG’s `Explainer.__call__` will forward arbitrary `**kwargs` to the model and (for homogeneous explanations) will also attach those `**kwargs` back onto the returned `Explanation` object, aiding reproducibility and visualization. citeturn14view1turn4view5  

### Model invocation assumptions

The evaluation framework targets a **single trained classifier** (binary or multiclass). PyG’s `Explainer` will run the model in `eval()` mode during explanation and then restore the previous training state, which is essential for deterministic dropout/batchnorm behavior during explanation runs. citeturn14view1  

Because PGExplainer in PyG enforces `"phenomenon"` explanations only, a fair “single suite” comparison typically requires running *all* explainers in **phenomenon mode** (with ground-truth labels provided), or running two suites (phenomenon-only suite including PGExplainer; model-prediction suite excluding it). Both are supported in this proposal (see orchestration). citeturn2view3turn14view1  

## Unified API design

### High-level module layout

The proposed system is organized into:

- **`ExplainerRegistry`**: maps alias → adapter factory + capability descriptor  
- **`ExplainerAdapter`** (per method): owns backend PyG explainer objects, pretraining (if needed), and output canonicalization  
- **`ExplainRunner`**: orchestrates dataset-level runs across explainers, manages devices, seeding, logging, and error handling  
- **`Postprocessor`**: converts raw Explanation outputs into canonical tensors, normalizes, thresholds, and generates sparse subgraphs  
- **`Evaluator`**: computes metrics and aggregates across dataset; provides statistical tests and runtime profiling metadata  
- **`VisualizerContract`**: defines a stable output schema for downstream plotting and 2D/3D export

### Canonical explanation output format (for plotting and metrics)

PyG’s `Explanation` object can include `node_mask` and `edge_mask`, with documented allowable shapes:

- `node_mask`: **[num_nodes, 1]**, **[1, num_features]**, or **[num_nodes, num_features]**  
- `edge_mask`: **[num_edges]** citeturn4view0turn3view2  

To make outputs “plot-ready” and comparable across explainers, define a canonical per-graph output with **fixed, explicit tensor shapes**:

- `node_mask_raw`: float tensor **[N, F]** (always expanded/broadcast to this shape)  
- `edge_mask_raw`: float tensor **[E]** (or `None` if unavailable)  
- `node_score`: float tensor **[N]** (feature-reduced scalar importance per node)  
- `edge_score`: float tensor **[E]** (normalized importance per edge; equals normalized `edge_mask_raw`)  
- `feature_score`: float tensor **[F]** (node-feature importance aggregated across nodes, when `node_mask` is feature-level)  
- `subgraph_edge_mask`: bool tensor **[E]** (after thresholding/top‑k selection)  
- `subgraph_node_mask`: bool tensor **[N]** (derived from edges and/or node scores)  
- `aux`: dictionary for method-specific outputs (e.g., `pgm_stats`, convergence deltas, training loss curves, runtime samples)  
- `metadata`: includes explainer config, seed, versions, model hash, graph id

Two PyG-provided utilities strongly motivate this design:

- `Explanation.validate_masks()` enforces dimensional constraints for masks. citeturn3view2turn4view2  
- `Explanation.get_explanation_subgraph()` and `get_complement_subgraph()` define a standard sparsification semantics (“mask out zeros”), which the proposal uses for consistent “subgraph fidelity” tests. citeturn4view0  

### Unified method invocation signatures

The proposal defines a minimal, stable adapter call surface (expressed as “signature guidelines,” not code):

- `Adapter.fit(train_iter, *, seed, device, max_steps, log_hook) -> FitArtifact | None`  
  - Required only for PGExplainer; no-op for other methods, but still present in the interface.
- `Adapter.explain(graph, *, target, explanation_mode, device, seed, return_raw) -> CanonicalExplanation`

Where:

- `graph` is a single `Data`/`Batch` item (ideally batch size 1)
- `explanation_mode ∈ {"phenomenon", "model"}` is mapped directly to PyG’s explanation types, which affect how `Explainer` derives or requires `target`. citeturn14view1turn14view2  
- `target` is required under phenomenon mode (PyG raises if missing). citeturn14view1  
- `seed` is used for repeatability for stochastic explainers (GNNExplainer, PGExplainer, PGMExplainer)  
- `return_raw` controls whether the raw PyG `Explanation` is preserved in `aux` for debugging

## Per-explainer adapter contracts and implementation notes

This section maps each alias to (a) the PyG backend implementation, (b) required parameters and preconditions, and (c) expected output keys/shapes.

### Shared backend mapping

- **Gradient/attribution explainers** (GRADEXPINODE, GRADEXPLEDGE, GUIDEDBP, IGNODE, IGEDGE) are implemented via **PyG’s `CaptumExplainer`** with different Captum methods and mask-type settings. PyG explicitly lists supported methods including `IntegratedGradients`, `Saliency`, and `GuidedBackprop`. citeturn6view2turn0search4  
- **GNNEXPL** uses **PyG’s `GNNExplainer`** instance-wise training (optimization over masks per explained graph). citeturn1view0turn9search0  
- **PGEXPL** uses **PyG’s `PGExplainer`**, which must be trained via `algorithm.train(epoch, ...)` before inference. citeturn2view1turn17view1turn9search1  
- **PGMEXPL** uses **PyG contrib’s `PGMExplainer`**, which computes node significance using perturbations + chi-square tests (and requires `pandas`/`pgmpy`). citeturn6view6turn15view0turn9search2  

### Captum-based methods: global constraints the framework must respect

PyG’s Captum integration has two notable constraints that drive orchestration design:

- The wrapped Captum forward path asserts **sample dimension 0 is 1** (single-sample processing). citeturn7view1  
- PyG **overrides `internal_batch_size` to 1** when the Captum attribution method supports it, warning if the user tries to set it differently. This ensures Integrated Gradients runs forward/backward in a sequential “one sample at a time” manner, matching the single-sample assumption. citeturn2view5turn7view1  

Operational implication: the evaluation runner should treat Captum-based explanations as **non-batchable across graphs** at the explainer-call level; parallelism should be at the process/job level, not by feeding multi-graph batches to a single explainer call.

### Adapter contract details by explainer

#### GRADEXPINODE

- Backend: `CaptumExplainer("Saliency", abs=...)` (Saliency returns gradients w.r.t. inputs; Captum default `abs=True`). citeturn8search2turn6view2  
- Required mask config: `node_mask_type="attributes"`, `edge_mask_type=None`. PyG’s `CaptumExplainer.supports()` requires `node_mask_type` be `None` or `"attributes"`. citeturn2view4  
- Outputs:
  - `node_mask` expected feature-level (canonicalized to **[N,F]**), then reduced to `node_score` **[N]** and `feature_score` **[F]**. PyG documents allowable node_mask shapes. citeturn4view0turn3view2  
- Preconditions:
  - Differentiable model w.r.t. node features; ensure `x` participates in prediction (otherwise gradients may be uninformative—this is domain/model dependent and not explicitly checked by Captum in PyG).
- Baselines: not applicable (Saliency does not use baselines). citeturn8search2  

#### GRADEXPLEDGE

- Backend: `CaptumExplainer("Saliency")` but operating on an **edge-mask input** created by PyG.
- Required mask config: `node_mask_type=None`, `edge_mask_type="object"`.
- PyG-specific mechanism:
  - PyG creates an edge-mask tensor of ones with `requires_grad=True` and shape `[E]`. citeturn6view0turn6view1  
- Outputs:
  - `edge_mask` shape **[E]** → canonical `edge_score` **[E]**. citeturn4view0turn3view2  
- Preconditions:
  - The model must actually use edges in message passing; otherwise the edge mask may not influence predictions (not automatically detectable here, unlike GNNExplainer which explicitly errors when gradients are missing).  

#### GUIDEDBP

- Backend: `CaptumExplainer("GuidedBackprop")`, node-feature attribution.
- Required mask config: typically `node_mask_type="attributes"`, `edge_mask_type=None` (edge-guided backprop is technically possible via edge-mask inputs, but not standard; treat as “unsupported by design” unless explicitly required).
- Outputs: `node_mask` → canonical **[N,F]**.
- Major precondition / failure mode:
  - Captum warns that Guided Backpropagation (and other hook-based methods) **will not work properly with functional nonlinearities** and requires module activations (e.g., `torch.nn.ReLU`) initialized in the module constructor. This should be logged as a compatibility check for the classifier architecture. citeturn8search1turn8search3  

#### IGNODE

- Backend: `CaptumExplainer("IntegratedGradients", n_steps=..., method=..., baselines=...)`.
- Required mask config: `node_mask_type="attributes"`, `edge_mask_type=None`.
- Baseline handling:
  - Captum’s Integrated Gradients uses **zero baselines by default when `baselines=None`**. The adapter must provide a “baseline provider” hook because protein–ligand feature baselines may be domain sensitive. citeturn13view0turn9search3  
- Key performance implication:
  - IG cost scales with `n_steps` (default 50) and PyG forces `internal_batch_size=1`, implying sequential evaluation and potentially high runtime per graph. citeturn13view0turn2view5  
- Outputs: `node_mask` canonicalized to **[N,F]**; optional capture of convergence delta is available in Captum, but whether PyG surfaces it is **unspecified** in PyG docs/source for this wrapper (it forwards `**kwargs` to the Captum `attribute` call, but does not document dedicated handling of convergence deltas). citeturn6view3turn13view0  

#### IGEDGE

- Backend: `CaptumExplainer("IntegratedGradients", ...)` operating on PyG’s edge-mask input.
- Required mask config: `edge_mask_type="object"`, `node_mask_type=None`.
- Baselines:
  - Same default: if baselines are not provided, IG uses a zero baseline; with PyG’s edge input being all ones, IG effectively integrates from “edge off” (0) to “edge on” (1) unless overridden. citeturn6view0turn13view0  
- Outputs: `edge_mask` → canonical **[E]**.

#### GNNEXPL

- Backend: PyG `GNNExplainer(epochs=..., lr=..., **coeffs)`.
- Category: per-instance optimization over masks; original method is formulated as an optimization task to identify a compact subgraph and feature subset. citeturn9search0turn1view0  
- Key parameters from PyG 2.7.0 source:
  - `epochs` default **100**, `lr` default **0.01**. citeturn1view0  
  - Default regularization coefficients include `edge_size`, `edge_ent`, `node_feat_ent`, etc. citeturn1view0  
  - Masks are randomly initialized using `torch.randn(...)` (seed-sensitive). citeturn16view2  
- Supported mask types:
  - Node masks support `"object"`, `"attributes"`, `"common_attributes"`; edge masks support `"object"` per initialization logic. citeturn16view2turn14view2  
- Explicit failure modes:
  - If node-mask gradients are `None`, PyG raises an error suggesting to ensure node features are used or disable node masks. citeturn16view3  
  - If edge-mask gradients are `None`, PyG raises an error suggesting to ensure message passing uses edges or disable edge masks. citeturn16view3  
- Outputs:
  - `node_mask` and `edge_mask` returned after postprocessing (sigmoid is applied in `_create_explanation`). citeturn1view0  

#### PGEXPL

- Backend: PyG `PGExplainer(epochs, lr=0.003, **coeffs)`; original paper emphasizes generalization via a parameterized explainer network. citeturn2view2turn9search1turn6view4  
- Enforced constraints (PyG `supports()`):
  - Explanation type must be **phenomenon**. citeturn2view3turn14view1  
  - Task level must be node or graph. citeturn2view3  
  - Node feature explanations are not supported: `node_mask_type` must be `None`. citeturn2view3turn2view2  
- Training requirement:
  - The adapter must run `algorithm.train(epoch, model, ...)` for `epochs` epochs; otherwise `forward(...)` raises “not yet fully trained.” citeturn2view1turn17view2  
- Architecture sensitivity:
  - PGExplainer relies on capturing embeddings; PyG’s implementation can raise if “No embeddings were captured” or if it could not generate edge masks, indicating unsupported architectures. citeturn17view1turn2view2  
  - Heterogeneous support is restricted to models containing certain hetero conv modules (HGTConv, HANConv, HeteroConv) listed in `SUPPORTED_HETERO_MODELS`. citeturn2view2  
- Outputs: edge-only `edge_mask` (**[E]**) (or per-edge-type dict for hetero).

#### PGMEXPL

- Backend: `torch_geometric.contrib.explain.PGMExplainer(...)`, derived from PGM-Explainer methodology. citeturn9search2turn6view6  
- Enforced constraints:
  - Only supports node-level or graph-level tasks; does not support regression; does not support edge masks. citeturn2view8turn6view7  
  - For node-level mode, only a **single index** is supported (multi-index not implemented). citeturn2view8turn6view7  
- Dependency constraints:
  - Imports `pandas` and `pgmpy.estimators.CITests.chi_square` inside explanation routines; the framework must treat these as optional dependencies with clear error messaging. citeturn15view0turn15view1  
- Outputs (PyG source behavior):
  - Graph-level output includes `node_mask` created as `torch.zeros(x.size(), dtype=int)` then sets selected nodes to 1, implying a hard feature-level mask shape **[N, F]**. citeturn15view4turn4view0  
  - Also returns `pgm_stats`, a tensor of p-values (length `num_nodes`). citeturn15view4turn6view7  

## Orchestration workflow: running many explainers on a validation set

### Core workflow and scheduling model

The evaluation should treat each graph explanation as an **atomic unit** with complete metadata capture (input graph id, model version, explainer config, seed, device, runtime). PyG `Explainer.__call__` already supports attaching `prediction`, `target`, `index`, `x`, `edge_index`, and additional kwargs into the returned explanation object, but the runner must still log system-level context. citeturn14view1turn3view2  

A recommended orchestration graph is:

```mermaid
flowchart TD
  A[Load trained GNN + version metadata] --> B[Build ExplainerSpecs list]
  B --> C[Instantiate adapters]
  C --> D{Adapter requires offline training?}
  D -->|Yes: PGExplainer| E[Fit explainer on train subset]
  D -->|No| F[Proceed]
  E --> F
  F --> G[Iterate validation graphs (batch_size=1)]
  G --> H[Compute base prediction + target selection]
  H --> I[For each ExplainerSpec: explain()]
  I --> J[Canonicalize masks to fixed shapes]
  J --> K[Normalize + sparsify (topk/hard curves)]
  K --> L[Metrics: fidelity, curves, robustness, runtime]
  K --> M[Visualization exports (2D/3D schemas)]
  L --> N[Dataset-level aggregation + stats tests]
  M --> N
  N --> O[Write artifacts: JSON/Parquet + plots downstream]
```

### Explanation mode strategy

Because PyG’s `Explainer` differentiates `"phenomenon"` vs `"model"` types and requires or derives `target` accordingly, the runner should support two explicit protocols:

- **Protocol P (phenomenon)**: supply ground-truth label `y` for every graph. Required if PGExplainer is included (PyG enforces phenomenon mode for PGExplainer). citeturn2view3turn14view1  
- **Protocol M (model)**: omit `target` and let PyG derive it from the model prediction (PyG warns if `target` is provided in model mode). citeturn14view1turn14view0  

Deliverables should clearly label which protocol was used; metrics like fidelity are interpretable under both (but reflect different objectives). citeturn11view2turn14view2  

### Parallelism, device management, and concurrency hazards

#### Parallelism model

Many explainers temporarily mutate model state by applying masks (e.g., `set_masks` / `clear_masks`) during explanation runs. PyG’s Captum wrapper and mask-learning explainers (GNNExplainer, PGExplainer) use this pattern internally. citeturn7view0turn1view0turn6view5  

Therefore:

- Do **not** run multiple explanation calls concurrently on the **same model instance** in the same process/thread.
- Safe parallelism strategies:
  - **Process-level parallelism** where each worker loads its own model copy (and optionally its own GPU).  
  - **Single-process sequential** (simpler, most reproducible) with optional asynchronous I/O for artifact writing.

#### Device management

- The runner should enforce consistent device placement per explainer:
  - Captum-based, GNNExplainer, PGExplainer: typically GPU-accelerated if model/data are on GPU.
  - PGMExplainer: includes CPU-bound steps (pandas DataFrames, chi-square tests) and explicitly moves samples to CPU in the graph-level path (`samples.detach().cpu()`), which can dominate runtime if the model is on GPU. citeturn15view4turn2view7  
- Runtime profiling should include:
  - wall-clock time per graph per explainer
  - GPU synchronization boundaries (design choice; not dictated by sources)
  - counts of forward/backward calls (estimated from explainer type; see below)

#### Determinism and seeds

The runner must set and log:

- `torch` seed (controls random init in GNNExplainer mask parameters and randomness inside PGExplainer’s sampling) citeturn16view2turn2view2  
- `numpy` seed (PGMExplainer uses `numpy` randomness) citeturn15view2turn2view7  
- dataset order seed (DataLoader shuffling) (framework design choice)

### Error handling and explainer health checks

The runner should implement standardized exception categories:

- **Hard incompatibility** (skip with reason):
  - PGExplainer not trained → raises not-yet-fully-trained error. citeturn17view2  
  - PGMExplainer missing dependencies `pandas`/`pgmpy`. citeturn15view0turn15view1  
  - GuidedBP with functional nonlinearities → known Captum limitation; should be detected by architecture checks and flagged as “high risk.” citeturn8search1  
- **Graph-specific failure** (continue, log graph id):
  - Gradient missing for GNNExplainer node/edge masks (PyG raises with guidance). citeturn16view3  
  - PGExplainer “No embeddings were captured” / could not generate edge masks. citeturn17view1  

All failures should emit:
- full explainer configuration
- model config / return type
- graph id and sizes (N, E, F)
- seed and device
- traceback details

## Standardized postprocessing and visualization contracts for graph-level explanations

### Canonicalization rules (raw → fixed shapes)

Given PyG’s allowed node mask shapes, the postprocessor should map to canonical `node_mask_raw [N,F]` as follows:

- If `node_mask` is **[N, F]**: use directly. citeturn4view0  
- If `node_mask` is **[N, 1]**: broadcast across features to [N, F] (design choice; consistent with “node importance independent of feature”). citeturn4view0turn3view2  
- If `node_mask` is **[1, F]**: broadcast across nodes to [N, F] (design choice; consistent with “global feature mask”). citeturn4view0turn3view2  
- If `node_mask` is `None`: set `node_mask_raw=None` and compute node scores from edges if available (design choice).

Edge mask canonicalization:

- If `edge_mask` is present: must be **[E]** per PyG. citeturn4view0turn4view2  
- If missing: `edge_mask_raw=None`.

### Normalization and comparability across explainer families

Different explainers can output masks with very different semantics:

- GNNExplainer outputs sigmoid-postprocessed masks (0–1). citeturn1view0  
- Captum attributions can be signed and unbounded (e.g., gradients); Saliency defaults to absolute gradients but can be configured. citeturn8search2turn8search6  
- PGMExplainer returns hard 0/1 `node_mask` plus p-values in `pgm_stats`. citeturn15view4turn6view7  

The proposal standardizes comparability by defining two normalized views:

- **Magnitude view** (recommended for cross-method comparison):
  - Convert masks to non-negative magnitudes (`abs` where needed), then normalize to [0,1] per-graph per-mask (e.g., min-max or rank-based). This is a design choice; PyG does not prescribe a specific normalization beyond optional thresholding. citeturn5search1turn4view2  
- **Signed view** (optional, method-specific):
  - Preserve sign for methods where it is meaningful (gradients / IG with `multiply_by_inputs=True` semantics). This is primarily useful for debugging and is not always comparable across methods. citeturn13view0  

For PGMExplainer, define a deterministic conversion to a score:

- Keep `pgm_stats` (p-values) as a primary output.
- Define a plotted node importance score as either:
  - `importance = 1 - clamp(p, 0, 1)` or
  - `importance = -log10(p + ε)` (design choice; not specified by PyG). citeturn15view4turn6view7  

### Sparsification, thresholding, and subgraph extraction

PyG supports a standardized threshold system:

- `None`: no threshold  
- `"hard"`: mask values below `value` become 0, others become 1  
- `"topk"`: keep top `value` elements, set others to 0  
- `"topk_hard"`: like topk but set kept elements to 1 citeturn5search1turn4view2  

The proposal uses sparsification in two ways:

- **Explainer-native thresholding**: configure `threshold_config` in PyG `Explainer` so the returned `Explanation` is already thresholded. citeturn4view4turn14view1  
- **Evaluation-time thresholding**: keep raw continuous masks (preferred for comparisons), then generate a family of thresholded masks for curves (e.g., varying top‑k). This aligns with fidelity curve computation and avoids conflating algorithm output with postprocessing. citeturn11view3turn4view2  

Subgraph extraction:

- Use `Explanation.get_explanation_subgraph()` to produce an induced subgraph where zero-attribution nodes/edges are removed (PyG defines this semantics explicitly). citeturn4view0  

### Visualization contract

The visualization layer should not depend on a particular explainer, only on canonical outputs. Define a stable “visual payload” schema per graph:

- Node table:
  - `node_index`, `node_score [0,1]`, `protein_or_ligand`, optional residue/atom identifiers, optional `pos [3]`
- Edge table:
  - `edge_index_src`, `edge_index_dst`, `edge_score [0,1]`, optional `edge_type`, optional `edge_attr`
- Optional subgraph selections:
  - boolean masks `subgraph_node_mask [N]`, `subgraph_edge_mask [E]`

Recommended visualization outputs:

- **2D graph view**: edge opacity/thickness mapped to `edge_score`, node size/opacity mapped to `node_score`. PyG’s `Explanation.visualize_graph` consumes `edge_index` and `edge_mask`, which can be leveraged for quick sanity checks. citeturn4view0turn3view2  
- **Feature importance bar chart**: aggregate `node_mask` across nodes, consistent with PyG’s `Explanation.visualize_feature_importance`, useful for interpreting which structural/chemical descriptors drive predictions. citeturn4view0turn3view2  
- **3D structural mapping** (protein–ligand specific; design choice):
  - export node scores into PDB “B-factor” fields or generate a PyMOL selection script referencing residue/atom IDs
  - export ligand atom scores to SDF as per-atom properties  
  (These are not specified by PyG/Captum; they are practical integration targets.)

## Quantitative comparison metrics and evaluation protocols

### Core metrics

#### Fidelity and fidelity curves

PyG provides a fidelity metric designed for explanations as subgraphs, including both “remove subgraph” and “keep only subgraph” variants (fidelity+ and fidelity−). citeturn11view2turn10search5  

Proposal usage:

- For each explainer and graph:
  - generate a sequence of thresholded masks (e.g., top‑k with k as a fraction of edges)
  - compute fidelity at each sparsity point
- Use PyG’s `fidelity_curve_auc` as the primary summary scalar across thresholds. citeturn11view3turn5search7  

Rationale: fidelity-based curves are explainer-agnostic (work for edge or node masks, as long as a subgraph can be produced) and do not require ground-truth explanations.

#### Sparsity

Sparsity is defined as the fraction of selected nodes/edges after thresholding:

- `edge_sparsity = 1 - (#selected_edges / E)`
- `node_sparsity = 1 - (#selected_nodes / N)`

This is a reporting standard (design choice), but it pairs naturally with PyG’s thresholding and subgraph semantics. citeturn5search1turn4view0  

#### Stability / robustness

Two complementary stability measures are recommended:

1) **Seed stability** (stochastic explainers):
- For GNNExplainer, PGExplainer, PGMExplainer, run multiple seeds and compute rank correlation (Spearman) of `edge_score` / `node_score` across runs per graph; aggregate across graphs. Random seed sensitivity is expected because masks are initialized randomly (GNNExplainer) or sampling is used (PGExplainer/PGMExplainer). citeturn16view2turn2view2turn15view2  

2) **Input-perturbation sensitivity** (robustness):
- Captum provides metrics such as `sensitivity_max` that quantify explanation change under small input perturbations. These are defined for attribution functions and grounded in robustness literature. citeturn12view0  
- For non-Captum explainers (GNNExplainer/PGExplainer/PGMExplainer), mimic this protocol by re-running explanations on perturbed graphs (e.g., small coordinate noise if coordinates are features, or feature noise) and computing correlations—this is a framework design, not mandated by sources.

#### Rank-correlation agreement between explainers

Compute cross-explainer agreement per graph on a shared domain, e.g.:

- edge-level Spearman correlation between `edge_score` vectors (requires both explainers output edges; if not, skip or compare node scores)
- top‑k Jaccard overlap on selected edges/nodes

This provides “consensus vs disagreement” insights without assuming any explainer is ground truth.

#### AUC-ROC for perturbation-grounded tests

PyG provides `groundtruth_metrics` which returns metrics including `"auroc"` when comparing a predicted mask and a ground-truth mask. citeturn11view0  

Protein–ligand systems often lack gold explanation masks, so the proposal supports two modes:

- **True ground-truth AUROC** (if available): e.g., known binding-site residues, experimentally validated contacts, or curated interaction subgraphs (dataset-specific; unspecified in sources).
- **Perturbation-derived pseudo-ground-truth AUROC** (design choice):
  - define a binary label per edge/node as “causal” if removing it causes a prediction change beyond a fixed threshold
  - compute AUROC between explainer scores and these labels using `groundtruth_metrics(..., metrics="auroc")` citeturn11view0turn11view2  

This should be clearly labeled as *proxy ground truth* because it partly bakes the model’s own behavior into the evaluation objective.

### Evaluation protocol and statistical testing

The evaluation protocol should report both **per-graph** distributions and **dataset-level** summaries:

- per-graph: fidelity curve AUC, runtime, sparsity at fixed k, stability scores, agreement scores
- dataset-level: mean/median with confidence intervals (bootstrap is recommended), plus pairwise statistical tests across explainers (e.g., Wilcoxon signed-rank on per-graph AUC values; design choice)

Because PGExplainer requires offline training and is phenomenon-only, protocol must document:

- training subset used for explainer training
- number of epochs completed (PyG enforces full training to `epochs - 1` or it errors) citeturn17view2turn2view1  
- whether phenomenon mode uses true labels or predicted labels (must be true labels per PyG’s phenomenon target requirement in `Explainer.__call__`) citeturn14view1  

### Runtime profiling expectations (qualitative)

Given implementation strategies:

- Captum Saliency / GuidedBP: ~single backward pass.
- Captum IG: ~`n_steps` backward passes (default 50), and PyG forces `internal_batch_size=1`, implying sequential processing. citeturn13view0turn2view5  
- GNNExplainer: ~`epochs` optimization steps (default 100), gradient-based; random initialization adds seed sensitivity. citeturn1view0turn16view2  
- PGExplainer:
  - offline training across `epochs` plus forward/backward per training example; inference is lighter but still requires embedding capture and mask generation. citeturn2view1turn17view2  
- PGMExplainer: `num_samples` perturbation runs (default 100) plus chi-square tests and pandas overhead. citeturn6view6turn15view4  

These expectations should be validated using actual profiling logs because graph sizes (protein-ligand) will strongly affect runtime.

## Logging, metadata, and dependency constraints

### Required metadata to capture per run

The framework must attach (at minimum):

- **Explainer identity**: alias, backend class, all hyperparameters (including thresholds, normalization settings)
- **Model identity**: model checkpoint hash, commit id of model code, and model_config:
  - `mode`, `task_level`, `return_type` (these govern target selection and Captum binary postprocessing). citeturn7view3turn14view1  
- **Graph identity**: dataset split, graph id, N/E/F sizes
- **Seeding**: torch seed, numpy seed, dataset order seed
- **Versions**:
  - PyG version pinned (recommend pin to 2.7.0 in this context; log exact installed version) citeturn0search2  
  - Captum version (required for CaptumExplainer to import) citeturn6view2  
  - pandas/pgmpy versions (required for PGMExplainer paths) citeturn15view0turn15view1  

Note: PyG’s explain module warns about instability and “master” requirements; in practice, you should log whether you used the exact 2.7.0 release or a source install. citeturn3view0  

### Dependency and runtime constraints

- Captum-based explainers require `captum.attr` to be importable; PyG imports Captum inside `CaptumExplainer.__init__`. citeturn6view2turn6view3  
- PGMExplainer requires `numpy` plus runtime imports of `pandas` and `pgmpy`. citeturn15view2turn15view0turn15view1  
- PGExplainer requires embedding capture and may fail for unsupported architectures; log supported/unsupported status per model. citeturn17view1turn2view2  

## Comparative mapping table: explainer → adapter behaviors and output keys

The table below is the core implementation checklist for adapter behavior and capability gating.

| Alias | PyG backend | Offline training required | Per-instance optimization | Explanation type allowed | Baseline required | Node mask support | Edge mask support | Hetero support | Expected output keys (raw) | Critical implementation notes |
|---|---|---:|---:|---|---|---|---|---|---|---|
| GRADEXPINODE | CaptumExplainer + Saliency citeturn6view2turn8search2 | No | No | model or phenomenon (framework choice) citeturn14view1 | No | Yes, **attributes only** citeturn2view4 | Optional if configured | Yes (dict path exists) citeturn6view3turn6view1 | `node_mask` | Saliency default `abs=True` in Captum; log if changed citeturn8search2 |
| GRADEXPLEDGE | CaptumExplainer + Saliency | No | No | model or phenomenon | No | No | Yes (`edge_mask` [E]) citeturn4view0turn6view0 | Yes | `edge_mask` | Edge mask input is created as ones with grad; CaptumModel assumes single sample dim citeturn6view0turn7view1 |
| GUIDEDBP | CaptumExplainer + GuidedBackprop citeturn6view2turn8search3 | No | No | model or phenomenon | No | Yes, attributes only citeturn2view4 | Optional if configured | Yes | `node_mask` | Captum: hook-based methods don’t work with functional activations; must use module ReLU citeturn8search1 |
| IGNODE | CaptumExplainer + IntegratedGradients citeturn6view2turn13view0 | No | No | model or phenomenon | Optional (default zero) citeturn13view0 | Yes, attributes only citeturn2view4 | Optional if configured | Yes | `node_mask` | Default `n_steps=50`; PyG forces `internal_batch_size=1` citeturn13view0turn2view5 |
| IGEDGE | CaptumExplainer + IntegratedGradients | No | No | model or phenomenon | Optional (default zero) citeturn13view0 | No | Yes (`edge_mask` [E]) | Yes | `edge_mask` | Edge mask is ones; IG integrates from baseline (often zero) to ones unless overridden citeturn6view0turn13view0 |
| GNNEXPL | GNNExplainer citeturn1view0turn9search0 | No | **Yes** (`epochs`, default 100) citeturn1view0 | model or phenomenon | No | Yes (`object/attributes/common_attributes`) citeturn16view2turn14view2 | Yes (`object`) citeturn16view2 | Yes | `node_mask`, `edge_mask` | Random init (`torch.randn`) → seed sensitivity; explicit errors when grads missing citeturn16view2turn16view3 |
| PGEXPL | PGExplainer citeturn6view4turn9search1 | **Yes** (`algorithm.train(...)`) citeturn2view1turn17view2 | No | **phenomenon only** citeturn2view3turn14view1 | No | **No** (node_mask_type must be None) citeturn2view3 | Yes (`edge_mask`) | Partial (restricted models) citeturn2view2 | `edge_mask` | Fails if not fully trained; may fail if embeddings not captured / unsupported architecture citeturn17view1turn17view2 |
| PGMEXPL | contrib PGMExplainer citeturn6view6turn9search2 | No | No (sampling-based) | model or phenomenon (but classification only) citeturn2view8turn14view1 | No | Yes (hard node mask) | **No** edge masks citeturn2view8 | No (homogeneous-only) | `node_mask`, `pgm_stats` | Requires `pandas` + `pgmpy`; default `num_samples=100`; returns p-values per node citeturn15view4turn6view6turn15view0 |

## Concise actionable checklist for implementation and reproducibility

- Pin and log versions: PyG (target 2.7.0), torch, captum, numpy, pandas, pgmpy; store them in every run artifact. citeturn0search2turn6view2turn15view0  
- Implement adapters exactly following PyG-enforced constraints:
  - CaptumExplainer: `node_mask_type ∈ {None, "attributes"}` citeturn2view4  
  - PGExplainer: phenomenon-only; node_mask_type must be None; must be fully trained citeturn2view3turn17view2  
  - PGMExplainer: classification-only; no edge masks; requires pandas/pgmpy citeturn2view8turn15view0  
- Run explanations per graph (batch_size=1) and avoid concurrent explanations on a shared model instance due to mask mutation behavior. citeturn7view0turn1view0turn6view5  
- Canonicalize all node masks to `[N,F]` and edge masks to `[E]` before metric computation; validate mask shapes using PyG’s constraints. citeturn4view0turn3view2  
- Standardize postprocessing:
  - produce magnitude-normalized scores in [0,1] for cross-method plots
  - generate top‑k / hard-threshold families using PyG threshold semantics for fidelity curves citeturn5search1turn4view2turn11view3  
- Evaluate with:
  - fidelity and fidelity-curve AUC (primary, no ground-truth needed) citeturn11view2turn11view3  
  - AUROC via `groundtruth_metrics` only when a true (or explicitly proxy) target mask exists citeturn11view0  
  - robustness via repeated seeds (stochastic explainers) and perturbation sensitivity protocols (Captum’s sensitivity metric can guide design) citeturn12view0turn16view2turn15view2  
- Record full configuration + exceptions per graph/explainer; never silently drop failures (PGExplainer training state, embedding capture failures, GuidedBP activation incompatibility). citeturn17view1turn8search1turn16view3