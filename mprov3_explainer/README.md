# mprov3-explainer

MPro-GINE Explainer – PyTorch Geometric pipeline for graph-level explanations of the trained GINE classifier. Loads the model and dataset from **mprov3_gine/results**, uses a pluggable **explainer registry** (default: all **available explainers**), and follows the **common representation** from *"Explaining the Explainers in Graph Neural Networks: a Comparative Study"* (Longa et al.): **(1) explanation masks generation** and **(2) preprocessing** (Conversion, Filtering, Normalization) applied to any explainer output before metrics.

Outputs are written under **`results/explanations/<explainer>/`** and **`results/visualizations/<explainer>/`** (flat layout; re-running overwrites after an **`[INFO]`** line).

**Shell orchestration:** From the repo root, [**`scripts/mprov3/README.md`**](../scripts/mprov3/README.md) documents bash wrappers that run **`run_explanations.py`** then **`generate_visualizations.py`** in order, with fold-aware args and **`-m` / `--include-misclassified`** (or **`INCLUDE_MISCLASSIFIED=1`**) to pass **`--no_correct_class_only`** into explanations.

## Flow

1. **Resolve paths** – From `results_root`, load `trainings/best_gnn.pt` and `datasets/data.pt` (flat paths). Splits (train/val/test) are read from `data_root/Splits/`.
2. **Load model** – Instantiate `MProGNN` with the same hyperparameters as training (defaults from `mprov3_gine_explainer_defaults` / CLI), load state dict from `best_gnn.pt`, set to eval.
3. **Build Explainer** – For each chosen explainer, the registry provides a builder (PyG `Explainer` wrapping GNNExplainer, Captum-based methods, PGExplainer, PGMExplainer, etc.).
4. **Optional offline training** – **PGEXPL** fits its parametric edge-mask MLP on the **training** loader before any test graphs are explained (see **PGExplainer** below).
5. **Phase 1 – Masks generation** – For each graph: call explainer → raw `Explanation` (`edge_mask` and/or `node_mask`, depending on the method).
6. **Phase 2 – Preprocessing** (optional, on by default):
   - **Conversion** – Optional edge-mask → node-mask (e.g. mean incident weights per node) for protocols that need it.
   - **Filtering** – Restrict to **correctly classified** instances (use `--no_correct_class_only` to include misclassified).
   - **Normalization** – Scale mask to [0, 1] per instance.
   - Mask tensors are **canonicalized** (e.g. stray batch dims, `(F, N)` vs `(N, F)`, degenerate square layouts) so downstream metrics and fidelity behave consistently.
7. **Phase 3 – Metrics** – Two fidelity tracks are computed on the **preprocessed** explanation:
   - **PyG/GraphFramEx**: fidelity (fid+, fid−) plus `characterization_score` (harmonic-mean style combine of fid+/fid−).
   - **Longa et al. (paper)**: **Sufficiency (Fsuf)**, **Comprehensiveness (Fcom)**, and **F1-fidelity (Ff1)** computed via a threshold sweep (Nt=100) over hard masks, by re-running the model on the explanation subgraph and its complement.
   For PyG fidelity, node masks are reshaped so `mask * node_features` matches PyTorch broadcasting (per-node `(N,)` masks are expanded to `(N, 1)` when needed).
8. **Report and masks** – Write `explanation_report.json` and per-graph `masks/<pdb_id>.json` (preprocessed masks).

## Input

- **results_root** (default: `../mprov3_gine/results` from the project root):
  - `trainings/best_gnn.pt` – trained GINE checkpoint
  - `datasets/data.pt` – PyG dataset; `pdb_order.txt` in the same folder is used for split indexing
- **data_root** (default: from mprov3_gine config) – path to the raw MPro snapshot containing `Splits/` (train/val/test PDB ID lists)

Model and split args (e.g. `--fold_index`, `--hidden`, `--num_layers`) must match the run that produced `best_gnn.pt`.

## Output

- **Stdout** – Per-graph line: `graph_id: fid+=... fid-=... [excluded]`; then mean fidelity and graph counts (total and valid).
- **results/explanations/&lt;explainer&gt;/** (inside mprov3_explainer):
  - `explanation_report.json` – includes:
    - `mean_fidelity_plus`, `mean_fidelity_minus`
    - `mean_pyg_characterization`
    - `mean_paper_sufficiency`, `mean_paper_comprehensiveness`, `mean_paper_f1_fidelity`
    - `num_graphs`, `num_valid`, `explainer`, and `per_graph` entries with the per-graph counterparts.
  - `masks/&lt;pdb_id&gt;.json` – per-graph `edge_index` and preprocessed `edge_mask` and/or `node_mask` (depending on the explainer).

The JSON field **`num_graphs`** counts graphs processed in that explainer run. It is **not** the same as **`mprov3_gine/visualize_graphs.py`** or its **`--num-graphs-by-fold`** option (per fold × train/val/test cap on ligand preview index rows).

## Available explainers (registry)

Default **`--explainers`** (or no flag) runs all of:

| Name | Description |
|------|-------------|
| **GRADEXPINODE** | Captum Saliency on **node features** |
| **GRADEXPLEDGE** | Captum Saliency on **edge mask** |
| **GUIDEDBP** | Guided backprop on node features |
| **IGNODE** | Integrated Gradients on node features (custom Captum bridge; see below) |
| **IGEDGE** | Integrated Gradients on edge mask (custom bridge) |
| **GNNEXPL** | PyG GNNExplainer (optimised edge mask) |
| **PGEXPL** | PGExplainer (phenomenon; **train** MLP on train loader, then explain) |
| **PGMEXPL** | PGMExplainer (node masks; statistical test) |

**SubgraphX (SubX, DIG)** is **not** supported in this pipeline: it is not in `AVAILABLE_EXPLAINERS` and there is no bundled adapter or test script.

### Implementation notes

- **Integrated Gradients (IGNODE / IGEDGE):** Captum’s IG repeats tensors in `additional_forward_args` along dim 0, which **breaks** PyG `edge_index` shape `[2, E]`. This project uses **bridge modules** (`integrated_gradients_node.py`, `integrated_gradients_edge.py`) so Captum only attributes the **mask inputs**; the graph is fixed inside the wrapper. On **MPS**, IG runs on **CPU** (Captum uses float64 steps; MPS does not support float64), then results are moved back as float32.
- **Node-only / edge-only masks:** Preprocessing merges preprocessed masks with `getattr(..., "edge_mask", None)` so missing optional keys on PyG `Explanation` do not raise.
- **PGExplainer:** Training can take a long time (epochs × train graphs). The script prints **per-epoch progress**. Use **`--pg_train_max_graphs`** to cap how many training graphs are stepped **per epoch**, or rely on the automatic cap when **`--max_graphs`** is set (see CLI). **`--max_graphs`** only limits **test** explanations, not PG training, unless you also set PG training caps.

## Usage

### Setup

```bash
uv sync
```

### Run explainers (default: all available, latest results, full test set)

```bash
uv run python scripts/run_explanations.py
```

### Run a specific explainer or multiple explainers

```bash
uv run python scripts/run_explanations.py --explainer GNNEXPL
uv run python scripts/run_explanations.py --explainers GNNEXPL GRADEXPINODE IGNODE
```

### Limit number of graphs (e.g. quick run)

```bash
uv run python scripts/run_explanations.py --max_graphs 10
```

`--max_graphs` limits **evaluation** on the test loader. For **PGEXPL**, if you also want a shorter **training** phase, set `--pg_train_max_graphs` (or use the automatic cap when `--max_graphs` is set).

### Paper metrics (Longa et al.: Fsuf, Fcom, Ff1)

By default the script computes **paper** sufficiency / comprehensiveness / F1-fidelity via a threshold sweep (costly: many extra forward passes per graph). PyG **fid+**, **fid−**, and **characterization** are always computed.

```bash
# Skip paper metrics (faster; reports still include paper fields as zeros)
uv run python scripts/run_explanations.py --no_paper_metrics

# Fewer threshold steps (faster, less faithful to Nt=100 in the paper)
uv run python scripts/run_explanations.py --paper_n_thresholds 25

# Re-enable explicitly after combining with other flags (default is already on)
uv run python scripts/run_explanations.py --paper_metrics --paper_n_thresholds 100
```

### Custom paths

```bash
uv run python scripts/run_explanations.py \
  --results_root /path/to/mprov3_gine/results \
  --data_root /path/to/MPro_snapshot
```

Report and masks are written to `results/explanations/<explainer>/` in the mprov3_explainer project root. Comparison files are **`results/explanations/comparison_report.json`** and **`comparison_report.html`**.

### Generate visualizations (index + images)

After running the explainer, generate an HTML index and 2D molecular images with bond coloring from the saved masks and SDF files:

```bash
uv run python scripts/generate_visualizations.py
```

If you do not pass `--explainer` / `--explainers`, it **only processes explainer folders that exist** under `results/explanations/` (with `explanation_report.json` and `masks/`), so partial runs (e.g. only PGEXPL) do not print a long list of “Skip …” lines. Output is written to **results/visualizations/&lt;explainer&gt;/**:

- `index.html` – summary (mean fidelity, num_graphs) and a grid of thumbnails linking to each graphic.
- `graphs/mask_&lt;pdb_id&gt;.png` – 2D molecule drawn from the SDF with bonds colored by explainer importance (max of edge_mask per bond).

To choose explainers explicitly:

```bash
uv run python scripts/generate_visualizations.py --explainer GNNEXPL
uv run python scripts/generate_visualizations.py --explainers GNNEXPL GRADEXPINODE
```

Cross-explainer **`comparison.html`** is written to **`results/visualizations/comparison.html`**.

**Migration:** Old outputs under `results/explanations/<timestamp>/` or `results/visualizations/<timestamp>/` are not read automatically; move files into the flat paths or re-run.

Images require SDF files at `data_root/Ligand/Ligand_SDF/&lt;pdb_id&gt;_ligand.sdf`. Pass `--data_root` if your MPro snapshot is elsewhere; `--results_root` overrides the default results directory.

### Other options (explainer and preprocessing)

- `--explainer` – single explainer; ignored if `--explainers` is set
- `--explainers` – explainers to run (default: all available; see table above)
- `--checkpoint` – checkpoint filename (default: `best_gnn.pt`)
- `--fold_index`, `--num_folds` – which fold to use (must match training)
- `--hidden`, `--num_layers`, `--dropout`, `--num_classes` – must match the trained model
- `--explainer_epochs` – GNNExplainer optimisation epochs **per graph** (default from `DEFAULT_GNN_EXPLAINER_EPOCHS`)
- `--pg_explainer_epochs` – PGExplainer MLP training epochs (default from `DEFAULT_PG_EXPLAINER_EPOCHS`)
- `--pg_train_max_graphs` – max training graphs per epoch for PGExplainer; if omitted and `--max_graphs` is set, a subsample cap is applied so PG training is not disproportionately slow
- `--ig_n_steps` – Integrated Gradients steps for **IGNODE** / **IGEDGE**
- `--pgm_num_samples` – PGMExplainer perturbation samples
- **Preprocessing (Longa et al.):** `--no_preprocessing` disable conversion/filtering/normalization; `--no_correct_class_only` include misclassified in averaging; `--fidelity_valid_only` report mean fidelity only over valid instances
- **Paper metrics:** `--paper_metrics` / `--no_paper_metrics` (default: **on**); `--paper_n_thresholds` (default: **100**, sweep uses k = 1 … Nt−1)

Plausibility (AUROC) is only computed when a ground-truth explanation mask is supplied (e.g. via a custom callback in the pipeline); the script does not provide one by default.
