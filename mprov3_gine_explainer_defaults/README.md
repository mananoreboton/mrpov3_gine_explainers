# mprov3-gine-explainer-defaults

Single source of truth for configuration shared by:

- `mprov3_gine/` — Training/fold and GINE **default hyperparameters** from this package (`train.py` / `evaluate.py` argparse); **`MProGNN`** is defined in `mprov3_gine/model.py` and instantiated there (not in this package). **`SplitConfig`** lives in `mprov3_gine/config.py`.
- `v2/` — PyG `Explainer` `model_config`, explainer-algorithm hyperparameters, mask-type strings
- `mprov3_explainer/` — CLI defaults aligned with the above

Modules: `gine_architecture`, `pyg_explainer`, `pyg_mask_types`, `explainer_algorithm_defaults`, `training_defaults`.

**No heavy dependencies** (stdlib only). Add as a path dependency from sibling projects:

```toml
dependencies = [ "mprov3-gine-explainer-defaults" ]
[tool.uv.sources]
mprov3-gine-explainer-defaults = { path = "../mprov3_gine_explainer_defaults", editable = true }
```
