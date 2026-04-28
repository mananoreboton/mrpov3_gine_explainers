# Plan: Adopt A Rich 2D Ligand Atom-Bond Graph Representation

## Goal

Replace the current ligand graph contract in `mprov3_gine` from:

- node features: `(x, y, z, atomic_number)`
- edge features: one bond scalar

to a richer **2D topological atom-bond representation** built from `Ligand/Ligand_SDF`, while keeping the full workflow working well across:

- `mprov3_gine`
- `mprov3_explainer`
- `mprov3_ui`

This plan does **not** change the code yet. It defines the implementation order, compatibility constraints, and validation steps.

## Current-State Findings

### 1. The graph schema is hard-coded in multiple places

The current `(N, 4)` node feature layout and `(E, 1)` edge feature layout are assumed in:

- `mprov3_gine/dataset.py`
- `mprov3_gine/model.py`
- `mprov3_gine/check_PyG_data_format.py`
- `mprov3_gine/visualize_graphs.py`
- `mprov3_gine/create_classification_report.py`
- `mprov3_gine_explainer_defaults/mprov3_gine_explainer_defaults/gine_architecture.py`
- `mprov3_gine/train.py`
- `mprov3_gine/classify.py`
- `mprov3_explainer/scripts/run_explanations.py`

### 2. GNN-side visualization currently depends on coordinates being stored inside `x`

`mprov3_gine/visualize_graphs.py` and `mprov3_gine/create_classification_report.py` rebuild RDKit molecules from:

- `x[:, :3]` for coordinates
- `x[:, 3]` for atomic number
- `edge_attr` for bond type

This will break if `x` becomes a rich 2D feature matrix instead of `[x, y, z, atomic_number]`.

### 3. Explainer drawing is already more robust

`mprov3_explainer/src/mprov3_explainer/visualize.py` draws molecules directly from the raw SDF files, then overlays node/edge masks. That is good news. If the new dataset preserves:

- RDKit atom ordering
- one directed edge per bond direction

then the explainer visualizations should continue to work with little conceptual change.

### 4. Checkpoint loading is fragile with respect to schema changes

`train.py` currently saves a raw `state_dict` only. Then:

- `classify.py`
- `mprov3_explainer/scripts/run_explanations.py`

rebuild `MProGNN` from shared defaults and load the state dict. If `in_channels` or `edge_dim` changes, these scripts will fail unless they also become schema-aware.

### 5. Development in a parallel results tree is not fully supported yet

`train.py` supports `--results_root`, but:

- `classify.py` is pinned to `DEFAULT_RESULTS_ROOT`
- `run_explanations.py` is pinned to `mprov3_gine/results`

This makes the migration riskier than necessary, because trial runs cannot be cleanly isolated from the current default outputs.

## Target Representation

## Representation contract

- Graph type: ligand-only atom-bond graph
- Source: `Ligand/Ligand_SDF/*.sdf`
- Nodes: atoms
- Edges: covalent bonds only
- Directed edge duplication: keep the current PyG convention of storing both directions
- Labels: keep `category`, `pIC50`, `pdb_id` unchanged

## Required invariants

To keep explainers and reports working:

- Preserve RDKit atom order from the input SDF
- Preserve bond connectivity exactly as read from the SDF
- Preserve the current directed-edge convention: each undirected bond becomes two directed edges
- Keep `pdb_id` and `pdb_order.txt` semantics unchanged

## Recommended node features

Freeze a fixed feature schema in code, for example:

- element identity
- degree
- formal charge
- implicit or total H count
- valence
- aromatic flag
- ring flag
- hybridization
- chirality or chiral tag

Use `float32` tensors for `x`, even when the source properties are categorical, because the current training and explainer stack expects numeric tensors.

## Recommended edge features

Freeze a fixed edge feature schema in code, for example:

- bond type
- conjugation flag
- ring flag
- stereo

This yields a richer `edge_attr` than the current single scalar, while still fitting naturally into GINE.

## Strong recommendation

Do **not** make visualization or reporting depend on decoding chemistry back out of `x`. After this migration, the raw SDF should be the source of truth for drawing molecules.

## Migration Strategy

## Phase 1: Freeze the schema before writing code

1. Define the exact node feature list and edge feature list.
2. Decide the encoding for each feature:
   - scalar
   - one-hot
   - multi-hot
3. Freeze the order of the feature columns.
4. Document the final `in_channels` and `edge_dim`.

Why this phase matters:

- `mprov3_gine`
- `mprov3_explainer`
- checkpoints
- validators
- report generators

all need a consistent contract.

## Phase 2: Add explicit schema metadata to built artifacts

Add a dataset-side manifest, for example under `results/datasets/`, containing:

- `representation_name`
- `schema_version`
- `node_feature_names`
- `edge_feature_names`
- `in_channels`
- `edge_dim`
- source description such as `ligand_rich_2d_atom_bond`

Recommended output file:

- `results/datasets/graph_schema.json`

Also add training-time model metadata, either:

- as a sidecar JSON near each checkpoint, or
- by saving a structured checkpoint dict instead of only a raw `state_dict`

Required fields:

- `in_channels`
- `edge_dim`
- `hidden_channels`
- `num_layers`
- `dropout`
- `out_classes`
- `pool`
- `representation_name`
- `schema_version`

Why this phase matters:

- `classify.py` and `run_explanations.py` should not guess dimensions from global defaults after the schema changes.

## Phase 3: Rebuild the dataset pipeline around rich 2D features

Update the build path centered on:

- `mprov3_gine/dataset.py`
- `mprov3_gine/build_dataset.py`

Required changes:

1. Replace the current `sdf_to_graph` feature extraction logic.
2. Keep graph-level labels exactly as today:
   - `pIC50`
   - `category`
   - `pdb_id`
3. Keep `pdb_order.txt`.
4. Write the new schema manifest together with `data.pt`.

Also update:

- `mprov3_gine/check_raw_data_format.py`

so that its SDF parsing smoke test validates the new representation instead of the old 3D one.

## Phase 4: Make model construction schema-aware

Update the model-construction path used by:

- `mprov3_gine/train.py`
- `mprov3_gine/classify.py`
- `mprov3_explainer/scripts/run_explanations.py`

Required changes:

1. Stop relying purely on hard-coded shared defaults for `in_channels` and `edge_dim`.
2. Read them from the built dataset manifest or the checkpoint metadata.
3. Fail with a clear message when dataset schema and checkpoint schema do not match.

Also update:

- `mprov3_gine_explainer_defaults/mprov3_gine_explainer_defaults/gine_architecture.py`

Decision to make during implementation:

- either keep `DEFAULT_IN_CHANNELS` and `DEFAULT_EDGE_DIM` as the new frozen values
- or make the loading path derive them from artifact metadata and treat the constants only as fallbacks

The second option is more robust.

## Phase 5: Decouple GNN-side visualization from `x`

Refactor:

- `mprov3_gine/visualize_graphs.py`
- `mprov3_gine/create_classification_report.py`

so that they draw molecules from the raw SDFs resolved via `pdb_id`, not from:

- `x[:, :3]`
- `x[:, 3]`
- `edge_attr` as a bond decoder

Required changes:

1. Resolve the SDF path from `pdb_id`.
2. Draw the base molecule directly with RDKit from the SDF.
3. Replace the old coordinate table with something representation-appropriate, such as:
   - atom index + element + atom feature summary
   - bond list + bond feature summary
4. Keep output filenames and output directories unchanged so `mprov3_ui` can continue serving:
   - `/gine/`
   - classification report pages

This is the cleanest way to preserve report usability after moving away from 3D coordinates in `x`.

## Phase 6: Update validators and quality checks

Update:

- `mprov3_gine/check_PyG_data_format.py`

Required changes:

1. Stop hard-coding:
   - `x.shape[1] == 4`
   - `edge_attr.shape[1] == 1`
2. Validate against the schema manifest instead.
3. Continue checking:
   - tensor dtypes
   - label presence
   - `pdb_id`
   - `pdb_order.txt` alignment
   - split index bounds

Also add explicit checks that:

- graph atom count matches the raw molecule atom count for sampled entries
- edge count still corresponds to two directed edges per bond

## Phase 7: Make the migration safe to trial

Add `--results_root` and `--data_root` configurability where missing, especially in:

- `mprov3_gine/classify.py`
- `mprov3_explainer/scripts/run_explanations.py`

Reason:

- `train.py` already supports alternate results roots
- `classify.py` and the explainer runner do not

This is important so the new representation can be validated in an isolated results tree before replacing the current default outputs that `mprov3_ui` serves.

## Phase 8: Add or extend automated regression coverage

There is existing test coverage in `mprov3_explainer`, but little or none in `mprov3_gine`.

Add tests for:

- rich-2D `sdf_to_graph`
- dataset manifest writing
- model forward pass with the new `in_channels` and `edge_dim`
- `check_PyG_data_format.py` logic against manifest-based dimensions
- report generation using SDF-based drawing

Recommended test targets:

- `mprov3_gine/tests/test_dataset.py`
- `mprov3_gine/tests/test_model_schema.py`
- `mprov3_gine/tests/test_visualization_inputs.py`

Extend explainer-side tests where helpful so they do not silently assume the old `4/1` schema.

## Phase 9: Update docs and pipeline commands

Update:

- `mprov3_gine/README.md`
- `mprov3_explainer/README.md`
- `mprov3_gine/run_pipeline.sh`

Required documentation changes:

1. Describe the new rich 2D graph representation.
2. Document the new feature schema.
3. Explain that GNN-side molecule drawings now come from raw SDFs, not from `x`.
4. Document any new CLI flags for alternate `results_root` or schema-aware loading.

## Impacted Files

## `mprov3_gine`

- `dataset.py`
- `build_dataset.py`
- `check_raw_data_format.py`
- `check_PyG_data_format.py`
- `model.py`
- `train.py`
- `classify.py`
- `cli_common.py`
- `visualize_graphs.py`
- `create_classification_report.py`
- `run_pipeline.sh`
- `README.md`

## `mprov3_gine_explainer_defaults`

- `mprov3_gine_explainer_defaults/gine_architecture.py`
- possibly path or schema helper modules if a manifest loader is centralized there

## `mprov3_explainer`

- `scripts/run_explanations.py`
- selected tests under `tests/`
- README if CLI or assumptions change

## `mprov3_ui`

Likely no code change required if:

- output directories stay the same
- report entry files stay the same

But it must be smoke-tested after the migration because it serves those generated outputs directly.

## Validation Matrix

## `mprov3_gine`

Must still work:

- raw-data validation
- dataset build
- built-dataset validation
- training
- classification
- graph gallery generation
- classification report generation

Suggested validation sequence:

1. `uv run python check_raw_data_format.py --data_root ...`
2. `uv run python build_dataset.py --data_root ... --results_root ...`
3. `uv run python check_PyG_data_format.py --results_root ...`
4. `uv run python visualize_graphs.py --results_root ...`
5. `uv run python train.py --results_root ... --seed 42`
6. `uv run python classify.py ...`
7. `uv run python create_classification_report.py ...`

## `mprov3_explainer`

Must still work:

- model loading from the new checkpoint format or metadata
- explanation generation
- per-graph mask rendering
- comparison report generation

Suggested validation:

- `uv run python scripts/run_explanations.py --folds 0 --split test`
- run `pytest` in `mprov3_explainer`

## `mprov3_ui`

Must still work:

- `/gine/`
- `/explainer/`

Suggested validation:

1. Point the UI at the validated results tree used for the migration trial.
2. Open:
   - `/gine/`
   - classification report pages
   - `/explainer/`
3. Confirm that images, HTML, and relative links resolve correctly.

## Acceptance Criteria

The migration is complete only when all of the following are true:

1. The built dataset uses the rich 2D atom-bond representation.
2. `train.py` can train without manual dimension edits.
3. `classify.py` can load the trained model and classify normally.
4. `run_explanations.py` can load the same model and produce explanation outputs normally.
5. `visualize_graphs.py` and `create_classification_report.py` render valid molecule pages without depending on coordinates embedded in `x`.
6. `mprov3_ui` serves the new generated outputs at the same routes as before.
7. Schema mismatch errors are explicit and understandable.
8. Legacy artifacts either:
   - still work, or
   - fail with a deliberate compatibility message rather than a low-level tensor shape error.

## Open Design Decisions

These should be resolved before implementation starts:

1. Exact feature vocabulary and ordering for nodes and edges.
2. Whether to save:
   - raw `state_dict` + sidecar JSON
   - or a structured checkpoint dict.
3. Whether shared defaults remain literal constants or become manifest-derived at load time.
4. Whether backward compatibility with the legacy 3D graph dataset is required or whether a clean schema cutover is acceptable.

## Recommended Implementation Order

To minimize breakage, implement in this order:

1. Freeze schema and add manifest support.
2. Update dataset build and validation.
3. Update model loading to be schema-aware.
4. Refactor GNN-side visualization to use raw SDFs.
5. Add alternate `results_root` support to the remaining scripts.
6. Add tests.
7. Rebuild dataset and retrain in an isolated results tree.
8. Run explainers and UI smoke tests.
9. Switch the default results tree only after all validations pass.
