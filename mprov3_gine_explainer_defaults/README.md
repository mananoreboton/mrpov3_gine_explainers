# mprov3-gine-explainer-defaults

Single source of truth for configuration shared across the monorepo:

- `mprov3_gine/` — training, fold and GINE **default hyperparameters**
  consumed by `train.py` and `classify.py` argparse; the model class
  `MProGNN` itself lives in `mprov3_gine/model.py`.
  `SplitConfig`, `WORKSPACE_ROOT`, `DEFAULT_DATA_ROOT`,
  `DEFAULT_RESULTS_ROOT`, `GINE_PROJECT_DIR`, etc. are defined here and
  assume sibling project layout next to `mprov3_gine_explainer_defaults`
  (see `gine_project_paths.py`). `visualize_graphs.py` reuses the same
  path / split defaults for QC drawings under `results/visualizations/`
  (see [`mprov3_gine/README.md`](../mprov3_gine/README.md)).
- `mprov3_explainer/` — CLI defaults and PyG explainer / mask-type
  constants are kept in lock-step with the GNN side via this package.

## Modules

| Module | Role |
|--------|------|
| `data_path_defaults` | `DEFAULT_DATA_ROOT`, `DEFAULT_RESULTS_ROOT`, split filenames, MPro snapshot directory name. |
| `gine_architecture` | GINE/MProGNN hyperparameter defaults (`DEFAULT_IN_CHANNELS`, `DEFAULT_HIDDEN_CHANNELS`, `DEFAULT_NUM_LAYERS`, `DEFAULT_DROPOUT`, `DEFAULT_OUT_CLASSES`, `DEFAULT_POOL`, `DEFAULT_EDGE_DIM`). |
| `gine_project_paths` | Sibling-project paths (`WORKSPACE_ROOT`, `GINE_PROJECT_DIR`, `EXPLAINER_PROJECT_DIR`, `RESULTS_DIR_NAME`, `RESULTS_TRAININGS`, `RESULTS_DATASETS`, `RESULTS_EXPLANATIONS`, `BUILT_DATASET_FOLDER_NAME`, …). |
| `split_config` | `SplitConfig` dataclass and split-file name defaults. |
| `pyg_explainer` | Shared `Explainer` constants (`DEFAULT_EXPLANATION_TYPE`, `PHENOMENON_EXPLANATION_TYPE`, `DEFAULT_MODEL_CONFIG`). |
| `pyg_mask_types` | Mask-type constants (`NODE_MASK_ATTRIBUTES`, `EDGE_MASK_OBJECT`). |
| `explainer_algorithm_defaults` | Per-explainer training/inference defaults (e.g. `DEFAULT_GNN_EXPLAINER_EPOCHS`, `DEFAULT_PG_EXPLAINER_EPOCHS`, `DEFAULT_PG_EXPLAINER_LR`, `DEFAULT_IG_N_STEPS`, `DEFAULT_PGM_NUM_SAMPLES`). |
| `training_defaults` | `DEFAULT_TRAINING_EPOCHS`, `DEFAULT_BATCH_SIZE`, `DEFAULT_TRAINING_LR`, `DEFAULT_SEED`, `DEFAULT_TRAINING_CHECKPOINT_FILENAME`. |
| `results_path_resolution` | Helpers that encode the flat `results/` layout: `resolve_checkpoint_path`, `resolve_dataset_dir`, `explanations_run_dir`, `visualizations_run_dir`. |
| `best_fold` | `resolve_best_fold_index`, `read_num_folds_for_fold`, `resolve_fold_indices` — pick / validate fold indices against `training_summary.json` and `classification_summary.json`. |
| `fold_indices` | Lightweight CLI parser for `--fold_indices` style arguments. |

**Results layout.** GINE and explainer pipelines use a **flat** layout
under each project's `results/` directory (e.g. `datasets/data.pt`,
`trainings/fold_<k>/best_gnn.pt`, `explanations/<EXPLAINER>/`). The
`results_path_resolution` helpers encode this layout so call sites do not
hard-code path segments.

**No heavy dependencies** (stdlib only). Add as a path dependency from
sibling projects:

```toml
dependencies = [ "mprov3-gine-explainer-defaults" ]

[tool.uv.sources]
mprov3-gine-explainer-defaults = { path = "../mprov3_gine_explainer_defaults", editable = true }
```
