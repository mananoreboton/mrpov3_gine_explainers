# v2 — compare explainers

Standalone tree (no imports from `mprov3_explainer/`). Implements eight “straight forward” PyG-compatible explainers from `doc/table_of_explainer_implementations.md`.

PyG `model_config`, explainer-algorithm defaults, and mask-type strings come from **`mprov3_gine_explainer_defaults`** (sibling `../mprov3_gine_explainer_defaults`, see `pyproject.toml`) so they stay aligned with `mprov3_gine` and `mprov3_explainer`.

## Setup

```bash
cd v2
uv sync
```

## Run

```bash
cd v2
uv run python compare_explainers.py
uv run python compare_explainers.py --explainers GNNEXPL UNKNOWN
```

With explicit names (comma-separated chunks allowed):

```bash
uv run python compare_explainers.py --explainers GNNEXPL PGEXPL
```

Default list (when `--explainers` is omitted): all eight canonical names (`GRADEXPINODE`, …, `PGMEXPL`).

**Note:** PyG’s `PGExplainer` requires `explanation_type="phenomenon"` (not `"model"`); `PgExplExplainer` sets that in `build_explainer`.

## Layout

- `compare_explainers.py` — entrypoint
- `cli/compare_explainers_cli.py` — argument parsing
- `explainers/` — one module per method + `shared/` configuration
