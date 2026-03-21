# mprov3-explainer

MPro-GINE Explainer – PyTorch Geometric pipeline for graph-level explanations of the trained GINE classifier. Loads the model and dataset from **mprov3_gine/results**, uses a pluggable **explainer registry** (default: all **available explainers**), and follows the **common representation** from *"Explaining the Explainers in Graph Neural Networks: a Comparative Study"* (Longa et al.): **(1) explanation masks generation** and **(2) preprocessing** (Conversion, Filtering, Normalization) applied to any explainer output before metrics.

Outputs are written under **`results/<explanations|visualizations>/<timestamp>/<explainer>/`** so one run can produce results for multiple explainers under the same timestamp.

## Flow

1. **Resolve paths** – From `results_root`, resolve latest `trainings/<timestamp>/best_gnn.pt` and `datasets/<timestamp>/` (contains `data.pt`). Splits (train/val/test) are read from `data_root/Splits/`.
2. **Load model** – Instantiate `MProGNN` with the same hyperparameters as training (defaults from `mprov3_gine_explainer_defaults` / CLI), load state dict from `best_gnn.pt`, set to eval.
3. **Build Explainer** – For each chosen explainer, the registry provides a builder (PyG `Explainer` for GNNExplainer).
4. **Phase 1 – Masks generation** – For each graph: call explainer → raw `Explanation` (edge_mask, optionally node_mask).
5. **Phase 2 – Preprocessing** (optional, on by default):
   - **Conversion** – Optional edge-mask → node-mask (e.g. mean incident weights per node) for protocols that need it.
   - **Filtering** – Restrict to **correctly classified** instances; discard **nearly constant** masks (max − min &lt; `--min_mask_range`).
   - **Normalization** – Scale mask to [0, 1] per instance.
6. **Phase 3 – Metrics** – Fidelity (fid+, fid−) and plausibility (AUROC) on the **preprocessed** explanation.
7. **Report and masks** – Write `explanation_report.json` and per-graph `masks/<pdb_id>.json` (preprocessed masks).

## Input

- **results_root** (default: `../mprov3_gine/results` from the project root):
  - `trainings/<timestamp>/best_gnn.pt` – trained GINE checkpoint
  - `datasets/<timestamp>/data.pt` – PyG dataset; `pdb_order.txt` in the same folder is used for split indexing
- **data_root** (default: from mprov3_gine config) – path to the raw MPro snapshot containing `Splits/` (train/val/test PDB ID lists)

Model and split args (e.g. `--fold_index`, `--hidden`, `--num_layers`) must match the run that produced `best_gnn.pt`.

## Output

- **Stdout** – Per-graph line: `graph_id: fid+=... fid-=... [auroc=...] [excluded]`; then mean fidelity and graph counts (total and valid).
- **results/explanations/&lt;timestamp&gt;/&lt;explainer&gt;/** (inside mprov3_explainer; timestamp is execution time in UTC):
  - `explanation_report.json` – `mean_fidelity_plus`, `mean_fidelity_minus`, `num_graphs`, `num_valid`, `explainer`, and `per_graph` (graph_id, fidelity_plus, fidelity_minus, auroc, valid, correct_class).
  - `masks/&lt;pdb_id&gt;.json` – per-graph `edge_index` and preprocessed `edge_mask` for **generate_visualizations**.

**Available explainers:** **GNNExplainer**, **SubgraphX**. Use `--explainer` or `--explainers` to select.

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
uv run python scripts/run_explanations.py --explainer GNNExplainer
uv run python scripts/run_explanations.py --explainer SubgraphX
uv run python scripts/run_explanations.py --explainers GNNExplainer SubgraphX
```

### Limit number of graphs (e.g. quick run)

```bash
uv run python scripts/run_explanations.py --max_graphs 10
```

### Custom paths

```bash
uv run python scripts/run_explanations.py \
  --results_root /path/to/mprov3_gine/results \
  --data_root /path/to/MPro_snapshot
```

Report and masks are written to `results/explanations/<timestamp>/<explainer>/` in the mprov3_explainer project root.

### Generate visualizations (index + images)

After running the explainer, generate an HTML index and 2D molecular images with bond coloring from the saved masks and SDF files:

```bash
uv run python scripts/generate_visualizations.py
```

This uses the **latest** explanation run under `results/explanations/` and, by default, all **available explainers**. Output is written to **results/visualizations/&lt;new_timestamp&gt;/&lt;explainer&gt;/**:

- `index.html` – summary (mean fidelity, num_graphs) and a grid of thumbnails linking to each graphic.
- `graphs/mask_&lt;pdb_id&gt;.png` – 2D molecule drawn from the SDF with bonds colored by explainer importance (max of edge_mask per bond).

To use a specific explainer or timestamp:

```bash
uv run python scripts/generate_visualizations.py --explainer GNNExplainer
uv run python scripts/generate_visualizations.py --timestamp 2026-03-15_133711
uv run python scripts/generate_visualizations.py --explainers GNNExplainer
```

Images require SDF files at `data_root/Ligand/Ligand_SDF/&lt;pdb_id&gt;_ligand.sdf`. Pass `--data_root` if your MPro snapshot is elsewhere; `--results_root` overrides the default results directory.

### Other options (explainer and preprocessing)

- `--explainer` – single explainer; ignored if `--explainers` is set
- `--explainers` – explainers to run (default: all available)
- `--checkpoint` – checkpoint filename (default: `best_gnn.pt`)
- `--fold_index`, `--num_folds` – which fold to use (must match training)
- `--hidden`, `--num_layers`, `--dropout`, `--num_classes` – must match the trained model
- `--explainer_epochs` – explainer optimization epochs per graph (default: `mprov3_gine_explainer_defaults.DEFAULT_GNN_EXPLAINER_EPOCHS`, currently 200; GNNExplainer)
- **SubgraphX:** `--subgraphx_rollout` (default: 10), `--subgraphx_max_nodes` (default: 10), `--subgraphx_sample_num` (default: 100)
- **Preprocessing (Longa et al.):** `--no_preprocessing` disable conversion/filtering/normalization; `--no_correct_class_only` include misclassified in averaging; `--min_mask_range` (default 1e-3) min mask range to keep; `--fidelity_valid_only` report mean fidelity only over valid instances

Plausibility (AUROC) is only computed when a ground-truth explanation mask is supplied (e.g. via a custom callback in the pipeline); the script does not provide one by default.
