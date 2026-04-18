# mprov3-gine-explainer-defaults

Single source of truth for configuration shared by:

- `mprov3_gine/` — Training/fold and GINE **default hyperparameters** from this package (`train.py` / `classify.py` argparse); **`MProGNN`** is defined in `mprov3_gine/model.py`. **`SplitConfig`**, **`WORKSPACE_ROOT`**, **`DEFAULT_DATA_ROOT`**, **`DEFAULT_RESULTS_ROOT`**, **`GINE_PROJECT_DIR`**, etc. assume sibling projects next to **`mprov3_gine_explainer_defaults`** (`gine_project_paths.py`). **`visualize_graphs.py`** uses the same path/split defaults for QC drawings under `results/visualizations/` (see **`mprov3_gine/README.md`**).
- `mprov3_explainer/` — CLI defaults aligned with the above

Modules: `data_path_defaults`, `gine_architecture`, `gine_project_paths`, `split_config`, `pyg_explainer`, `pyg_mask_types`, `explainer_algorithm_defaults`, `training_defaults`, `results_path_resolution`, …

**Results paths:** GINE and explainer pipelines use a **flat** layout under each project’s `results/` (e.g. `datasets/data.pt`, `trainings/best_gnn.pt`, `explanations/<explainer>/`). Helpers `resolve_checkpoint_path`, `resolve_dataset_dir`, `explanations_run_dir`, and `visualizations_run_dir` encode this layout.

**No heavy dependencies** (stdlib only). Add as a path dependency from sibling projects:

```toml
dependencies = [ "mprov3-gine-explainer-defaults" ]
[tool.uv.sources]
mprov3-gine-explainer-defaults = { path = "../mprov3_gine_explainer_defaults", editable = true }
```
