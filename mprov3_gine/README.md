# GNN training for MPro Version 3 data

Python pipeline to train a Graph Neural Network on the MPro-URV Version 3 snapshot for **3-class classification** (Category: low / medium / high potency). The codebase is split into configuration, data loading, GINE model, and separate training, validation, and evaluation logic.

**Shared defaults:** Training, fold, GINE architecture, path segment names, `**SplitConfig`**, `**DEFAULT_DATA_ROOT**`, `**DEFAULT_RESULTS_ROOT**`, and sibling project `**Path`s (`WORKSPACE_ROOT**`, `**GINE_PROJECT_DIR**`, …) come from `**mprov3_gine_explainer_defaults**` (monorepo: parent of the `mprov3_gine_explainer_defaults` folder). The model class is `**MProGNN**` in `**model.py**`.

## Overview

### Pipeline: scripts, inputs and outputs

All outputs go under `results/` in timestamped subfolders. Scripts that read from results use the **latest** timestamp by default.

```mermaid
flowchart TB
    subgraph raw [Raw input - data_root]
        SDF[Ligand/Ligand_SDF/*.sdf]
        Info[Info.csv]
        Splits[Splits/]
    end

    subgraph build [build_dataset.py]
        BuildCLI[build_dataset.py]
    end

    subgraph out_datasets [results/datasets/<ts>/]
        DataPT[data.pt]
        PdbOrder[pdb_order.txt]
        BuildLog[build.log]
    end

    subgraph train_script [train.py]
        TrainCLI[train.py]
    end

    subgraph out_trainings [results/trainings/<ts>/]
        BestCkpt[best_gnn.pt]
        TrainLog[train.log]
    end

    subgraph eval_script [evaluate.py]
        EvalCLI[evaluate.py]
    end

    subgraph out_classifications [results/classifications/<ts>/]
        EvalJSON[evaluation_results.json]
        EvalLog[evaluate.log]
    end

    subgraph report_script [create_evaluation_report.py]
        ReportCLI[create_evaluation_report.py]
    end

    subgraph out_report [same folder]
        ReportHTML[index.html, *.html]
        ReportGraphs[graphs/*.png]
        ReportLog[create_evaluation_report.log]
    end

    subgraph viz_script [visualize_graphs.py]
        VizCLI[visualize_graphs.py]
    end

    subgraph out_viz [results/visualizations/<ts>/]
        VizPNG[PNG/SVG/HTML]
        VizLog[visualize.log]
    end

    subgraph check_in [check_raw_data_format.py]
        CheckInCLI[check_raw_data_format.py]
    end

    subgraph out_check_raw [results/check_format/raw_data/<ts>/]
        CheckInLog[check_input.log]
    end

    subgraph check_out [check_PyG_data_format.py]
        CheckOutCLI[check_PyG_data_format.py]
    end

    subgraph out_check_ds [results/check_format/datasets/<ts>/]
        CheckOutLog[check_output.log]
    end

    SDF --> BuildCLI
    Info --> BuildCLI
    BuildCLI --> DataPT
    BuildCLI --> PdbOrder
    BuildCLI --> BuildLog

    DataPT --> TrainCLI
    PdbOrder --> TrainCLI
    Splits --> TrainCLI
    TrainCLI --> BestCkpt
    TrainCLI --> TrainLog

    DataPT --> EvalCLI
    PdbOrder --> EvalCLI
    Splits --> EvalCLI
    BestCkpt --> EvalCLI
    EvalCLI --> EvalJSON
    EvalCLI --> EvalLog

    EvalJSON --> ReportCLI
    DataPT --> ReportCLI
    ReportCLI --> ReportHTML
    ReportCLI --> ReportGraphs
    ReportCLI --> ReportLog

    DataPT --> VizCLI
    VizCLI --> VizPNG
    VizCLI --> VizLog

    SDF --> CheckInCLI
    Info --> CheckInCLI
    Splits --> CheckInCLI
    CheckInCLI --> CheckInLog

    DataPT --> CheckOutCLI
    PdbOrder --> CheckOutCLI
    Splits --> CheckOutCLI
    CheckOutCLI --> CheckOutLog
```



### Code layout: modules and entry points

```mermaid
flowchart TB
    subgraph cli [CLI scripts]
        BuildScript[build_dataset.py]
        TrainScript[train.py]
        EvalScript[evaluate.py]
        ReportScript[create_evaluation_report.py]
        VizScript[visualize_graphs.py]
        CheckInScript[check_raw_data_format.py]
        CheckOutScript[check_PyG_data_format.py]
    end

    subgraph shared [Shared]
        Defaults[mprov3_gine_explainer_defaults]
        Utils[utils.py]
    end

    subgraph data_layer [Data]
        Dataset[dataset.py]
        Loaders[loaders.py]
    end

    subgraph model_layer [Model]
        Model[model.py]
    end

    subgraph train_logic [Training / validation]
        TrainEpoch[train_epoch.py]
        Validation[validation.py]
    end

    subgraph eval_logic [Evaluation]
        Evaluation[evaluation.py]
    end

    Defaults --> BuildScript
    Defaults --> TrainScript
    Defaults --> EvalScript
    Defaults --> Loaders
    Utils --> BuildScript
    Utils --> TrainScript
    Utils --> EvalScript
    Utils --> ReportScript
    Utils --> VizScript
    Utils --> CheckInScript
    Utils --> CheckOutScript
    TrainScript --> Model
    EvalScript --> Model
    Dataset --> Loaders
    Dataset --> BuildScript
    Loaders --> TrainScript
    Loaders --> EvalScript
    Model --> TrainEpoch
    Model --> Validation
    Model --> Evaluation
    TrainEpoch --> TrainScript
    Validation --> TrainScript
    Validation --> Evaluation
    Evaluation --> EvalScript
    Dataset --> ReportScript
    Dataset --> VizScript
    VizScript --> ReportScript
    Dataset --> CheckOutScript
```



## Data

- **Graphs**: One graph per ligand from `Ligand/Ligand_SDF/*.sdf`. Node features: 3D coordinates (x, y, z) and atomic number. Edges: bonds; edge features: bond type (single=1, double=2, triple=3, aromatic=1.5) for GINE.
- **Labels**: `Info.csv` provides `pIC50` and `Category` (-1: pIC50<5.5, 0: 5.5≤pIC50<6.5, 1: pIC50≥6.5). Category is mapped to classes 0, 1, 2.

## Results layout

All script outputs live under `**results/`** (config: `DEFAULT_RESULTS_ROOT`). Every run writes into a **timestamped subfolder** (`<YYYY-MM-DD_HHMMSS>`). When a script *reads* from results (e.g. train reading the dataset), it uses the **most recent** timestamp folder by default.


| Path                                         | Written by                                                                                 | Log file                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| `results/datasets/<timestamp>/`              | `build_dataset.py` (`data.pt`, `pdb_order.txt`)                                            | `build.log`                                    |
| `results/trainings/<timestamp>/`             | `train.py` (`best_gnn.pt`)                                                                 | `train.log`                                    |
| `results/classifications/<timestamp>/`       | `evaluate.py` (`evaluation_results.json`), `create_evaluation_report.py` (HTML, `graphs/`) | `evaluate.log`, `create_evaluation_report.log` |
| `results/visualizations/<timestamp>/`        | `visualize_graphs.py` (PNG/SVG/HTML)                                                       | `visualize.log`                                |
| `results/check_format/datasets/<timestamp>/` | `check_PyG_data_format.py` (log only)                                                      | `check_output.log`                             |
| `results/check_format/raw_data/<timestamp>/` | `check_raw_data_format.py` (log only)                                                      | `check_input.log`                              |


Raw input (MPro snapshot with `Info.csv`, `Ligand/`, `Splits/`) stays at `--data_root` (default: `config.DEFAULT_DATA_ROOT`). Splits are always read from that raw root. Shared helpers (timestamps, latest-folder resolution, HTML, run logging) live in `**utils.py`**.

## Setup

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
cd mprov3_gine
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

Requires: PyTorch, PyTorch Geometric, RDKit, pandas, numpy, scikit-learn.

---

## Usage

### 0. Validate raw input (optional but recommended)

Before building the dataset, you can check that your raw MPro snapshot has the expected layout and that SDFs/labels/splits parse correctly. The **built** PyG dataset is validated later (after step 1).

```bash
# Default data_root = config.DEFAULT_DATA_ROOT
uv run python check_raw_data_format.py

# Custom raw data path
uv run python check_raw_data_format.py --data_root /path/to/mprov3_data
```

If the check fails, the script exits with code 1 and prints `[ERROR]` lines. Log is written to `results/check_format/raw_data/<timestamp>/check_input.log`.

### 1. Build the PyG dataset (required once)

Train/val/test loaders use a **pre-built** PyG dataset. Create it from SDFs and `Info.csv` before training. Each run writes to a new **timestamped** folder under `results/datasets/`.

```bash
# Default: data_root = config.DEFAULT_DATA_ROOT, output = results/datasets/<timestamp>/
uv run python build_dataset.py

# Custom raw data path and results root
uv run python build_dataset.py --data_root /path/to/mprov3_data --results_root results
```

Output: `results/datasets/<timestamp>/data.pt`, `pdb_order.txt`, `build.log`. If you skip this step, training will exit with an error telling you to run `build_dataset.py` first.

#### 1.1. Validate built dataset (optional)

After building, you can verify that the PyG dataset and split indices are compatible with training/evaluation. By default this uses the **latest** `results/datasets/<timestamp>/` and `--splits_root` for the raw snapshot (Splits/).

```bash
# Default: latest results/datasets/<ts>/, splits from config.DEFAULT_DATA_ROOT
uv run python check_PyG_data_format.py

# Custom splits location (raw MPro snapshot)
uv run python check_PyG_data_format.py --splits_root /path/to/mprov3_data
```

Log is written to `results/check_format/datasets/<timestamp>/check_output.log`.

### 2. Visualize ligand graphs (visualize_graphs.py)

Draws a subset of ligand graphs using **RDKit's 2D drawer** (MolDraw2D) for publication-quality figures. By default loads the **latest** `results/datasets/<timestamp>/` and writes to `results/visualizations/<timestamp>/`. **Category is shown in the original scale (-1, 0, 1)** (low / medium / high potency). Layout uses **(x, y) only** (z is dropped). Bond styles follow chemistry conventions: single = one central line; double = two shifted lines; triple = two shifted lines plus one central line; aromatic = dashed.

```bash
# Default: first 16 graphs from latest dataset, output = results/visualizations/<timestamp>/
uv run python visualize_graphs.py

# Specify how many graphs to draw
uv run python visualize_graphs.py --num_graphs 32

# Select by dataset indices or PDB IDs
uv run python visualize_graphs.py --indices 0 1 2 10 25
uv run python visualize_graphs.py --pdb_ids 5R83 6LU7

# Also write vector SVG files (for figures)
uv run python visualize_graphs.py --svg
```

Output under `**results/visualizations/<timestamp>/**`:

- `PDB_ID.png`: 2D drawing (RDKit MolDraw2D).
- `PDB_ID.svg`: vector graphic (only with `--svg`).
- `PDB_ID.html`: report with PDB ID, category (-1/0/1), pIC50, and tables for nodes (atomic number, x, y, z) and edges (bond type).
- `index.html`: index of all visualized graphs; `visualize.log`: run log.

### 3. Train (train.py)

Training loads the **latest** `results/datasets/<timestamp>/` and reads splits from **three files** in `data_root/Splits/`:

- **Default file names**: `train_index_folder.txt`, `valid_index_folder.txt`, `test_index_folder.txt`
- Each file must contain **num_folds** lists of PDB IDs (one list per fold). Default **num_folds** is **5**.

```bash
# Default: data_root = config.DEFAULT_DATA_ROOT, latest dataset, output = results/trainings/<timestamp>/
uv run python visualize_graphs.py

# Custom raw data path (for Splits/)
uv run python train.py --data_root /path/to/mprov3_data
```

#### Split files and folds

```bash
# Override split file names
uv run python train.py --train_split_file train_index_folder.txt --val_split_file valid_index_folder.txt --test_split_file test_index_folder.txt

# Number of folds (default: 5) and which fold to use (0 .. num_folds-1)
uv run python train.py --num_folds 5 --fold_index 2
```

#### Training options

```bash
# Epochs, batch size, learning rate, seed
uv run python train.py --epochs 150 --batch_size 16 --lr 5e-4 --seed 42

# Number of classes (default 3)
uv run python train.py --num_classes 3
```

#### GINE model (architecture)

```bash
# Hidden size, depth, dropout
uv run python train.py --hidden 128 --num_layers 4 --dropout 0.2
```

#### Full example

```bash
uv run python build_dataset.py --data_root /path/to/mprov3_data
uv run python train.py \
  --data_root /path/to/mprov3_data \
  --num_folds 5 --fold_index 0 \
  --epochs 100 --batch_size 32 --lr 1e-3 \
  --hidden 64 --num_layers 3 --dropout 0.2 \
  --num_classes 3 --seed 42
```

The best model (by validation accuracy) is saved as `results/trainings/<timestamp>/best_gnn.pt`. Training does not run evaluation; use `evaluate.py` for that.

### 4. Evaluate (evaluate.py) — run independently

Evaluate a saved checkpoint on the test set without running training. By default uses the **latest** `results/trainings/<timestamp>/` (for the checkpoint) and **latest** `results/datasets/<timestamp>/` (for the test set). Use the same split/fold and model architecture as when the model was trained. **Categories are reported in the original scale (-1, 0, 1)** (low / medium / high potency). Results are saved to `results/classifications/<timestamp>/evaluation_results.json` for use by the evaluation report script.

```bash
# Default: latest trainings/<ts>/ and datasets/<ts>/, output = results/classifications/<timestamp>/
uv run python evaluate.py

# Custom data root (for Splits/) and checkpoint filename (still loaded from latest trainings/<ts>/)
uv run python evaluate.py --data_root /path/to/snapshot --checkpoint best_gnn.pt

# Same fold and architecture as training
uv run python evaluate.py --data_root /path/to/snapshot --fold_index 2 --hidden 64 --num_layers 3 --num_classes 3
```

Options: `--data_root`, `--results_root`, `--checkpoint` (filename in latest `results/trainings/<ts>/`), `--train_split_file`, `--val_split_file`, `--test_split_file`, `--num_folds`, `--fold_index`, `--batch_size`, `--hidden`, `--num_layers`, `--dropout`, `--num_classes` (must match the trained model).

#### 4.1. Evaluation report (create_evaluation_report.py)

After running `evaluate.py`, generate an HTML report with graph thumbnails and per-sample real vs predicted category. By default uses the **latest** `results/classifications/<timestamp>/evaluation_results.json` and writes the report into that same folder.

```bash
# Uses latest results/classifications/<timestamp>/evaluation_results.json
uv run python create_evaluation_report.py

# Custom results file
uv run python create_evaluation_report.py --results results/classifications/2025-03-14_120000/evaluation_results.json
```

Output is written into the same `**results/classifications/<timestamp>/**` folder as the JSON:

- `**index.html**`: index page with thumbnail links; each card shows PDB ID, real category, predicted category, and correct/incorrect.
- `**graphs/<PDB_ID>.png**`: graph image per test sample.
- `**<PDB_ID>.html**`: per-sample page with image, PDB ID, real category (-1/0/1), predicted category (-1/0/1).
- `**create_evaluation_report.log**`: run log.

---

### Programmatic use

You can reuse configs, loaders, and train/val/test logic in your own scripts.

#### Configuration

- `**SplitConfig**` (from `**mprov3_gine_explainer_defaults**`): train/val/test file names (`train_file`, `val_file`, `test_file`), `num_folds`, `fold_index`, `dataset_name` (PyG dataset folder).
- **Training hyperparameters** (`epochs`, `batch_size`, `lr`, `seed`): defaults from `**mprov3_gine_explainer_defaults`**; `train.py` uses argparse (see `DEFAULT_TRAINING_EPOCHS`, `DEFAULT_BATCH_SIZE`, `DEFAULT_TRAINING_LR`, `DEFAULT_SEED`).
- `**model.MProGNN**`: GINE architecture; construct with hyperparameters (defaults align with `**mprov3_gine_explainer_defaults**` e.g. `DEFAULT_IN_CHANNELS`, `DEFAULT_HIDDEN_CHANNELS`, …).

#### Data loaders

- `**loaders.collate_batch(batch)**`: collate list of PyG graphs into a batch (includes category labels; pIC50 still in data for reference).
- `**loaders.create_data_loaders(dataset_root, data_root, split_config, batch_size=32)**`: loads the PyG dataset from `dataset_root/split_config.dataset_name` (e.g. `results/datasets/<timestamp>`) and returns `(train_loader, val_loader, test_loader)` using split files from `data_root/Splits/` and `fold_index`.

#### Training

- `**train_epoch.train_one_epoch(model, loader, optimizer, device, criterion_ce)**`: one training epoch (cross-entropy); returns mean loss.

#### Validation

- `**validation.evaluate_validation(model, loader, device)**`: returns `**ValidationMetrics**` (`accuracy`).

#### Evaluation

- `**evaluation.evaluate_test(model, loader, device)**`: returns `**TestMetrics**` (`accuracy`).
- `**evaluation.evaluate_test_with_predictions(model, loader, device)**`: returns `**(TestMetrics, list of (pdb_id, real_category, pred_category))**` with categories in original scale (-1, 0, 1).
- `**evaluation.print_test_report(metrics)**`: prints test accuracy.

#### Example script

```python
from pathlib import Path
import torch
from mprov3_gine_explainer_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EDGE_DIM,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_POOL,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LR,
)
from mprov3_gine_explainer_defaults import SplitConfig
from loaders import create_data_loaders
from model import MProGNN
from train_epoch import train_one_epoch
from validation import evaluate_validation
from evaluation import evaluate_test, print_test_report
from utils import get_latest_timestamp_dir

data_root = Path("/path/to/mprov3_data")  # raw snapshot (Splits/, Info.csv)
dataset_base = Path("results/datasets")  # run build_dataset.py first
latest_ds = get_latest_timestamp_dir(dataset_base)
dataset_name = latest_ds.name if latest_ds else "2025-03-14_120000"  # or raise if None
split_config = SplitConfig(num_folds=5, fold_index=0, dataset_name=dataset_name)
epochs = DEFAULT_TRAINING_EPOCHS
batch_size = DEFAULT_BATCH_SIZE
lr = DEFAULT_TRAINING_LR
model = MProGNN(
    in_channels=DEFAULT_IN_CHANNELS,
    hidden_channels=DEFAULT_HIDDEN_CHANNELS,
    num_layers=DEFAULT_NUM_LAYERS,
    dropout=DEFAULT_DROPOUT,
    out_classes=DEFAULT_OUT_CLASSES,
    pool=DEFAULT_POOL,
    edge_dim=DEFAULT_EDGE_DIM,
)

train_loader, val_loader, test_loader = create_data_loaders(
    dataset_base, data_root, split_config, batch_size=batch_size
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion_ce = torch.nn.CrossEntropyLoss()

# Training loop (simplified)
for epoch in range(1, epochs + 1):
    train_one_epoch(model, train_loader, optimizer, device, criterion_ce)
    val_metrics = evaluate_validation(model, val_loader, device)
    # ... save best model by val_metrics.accuracy, etc.

# Load from latest training run
trainings_base = Path("results/trainings")
latest_training = get_latest_timestamp_dir(trainings_base)
ckpt_path = latest_training / "best_gnn.pt" if latest_training else Path("results/trainings/best_gnn.pt")
model.load_state_dict(torch.load(ckpt_path))
test_metrics = evaluate_test(model, test_loader, device)
print_test_report(test_metrics)
```

---

## Layout


| File                            | Role                                                                                                                                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **model.py**                    | GINE model: `MProGNN` (hyperparameter defaults align with `mprov3_gine_explainer_defaults`).                                                                                        |
| **dataset.py**                  | Helpers: `sdf_to_graph`, `load_activity_and_category`; `load_splits` (three files); `get_train_val_test_indices`; `MProV3Dataset` (loads pre-built PyG dataset, errors if missing). |
| **utils.py**                    | `run_timestamp()`, `get_latest_timestamp_dir()`, `html_escape()`, `html_document()`, `RunLogger` (tee to file + stdout).                                                            |
| **build_dataset.py**            | Builds PyG dataset to `results/datasets/<timestamp>/` (no dataset_name); writes `build.log`.                                                                                        |
| **check_raw_data_format.py**    | CLI: validate raw dataset at `--data_root`; writes `results/check_format/raw_data/<timestamp>/check_input.log`.                                                                     |
| **check_PyG_data_format.py**    | CLI: validate built dataset (default: latest `results/datasets/<timestamp>/`); writes `results/check_format/datasets/<timestamp>/check_output.log`.                                 |
| **loaders.py**                  | `collate_batch`, `create_data_loaders(dataset_root, data_root, ...)` (dataset under `results/datasets/`, splits from raw root).                                                     |
| **train_epoch.py**              | One-epoch training step: `train_one_epoch`.                                                                                                                                         |
| **validation.py**               | Validation: `evaluate_validation`, `ValidationMetrics`.                                                                                                                             |
| **evaluation.py**               | Evaluation: `evaluate_test`, `TestMetrics`, `print_test_report`.                                                                                                                    |
| **train.py**                    | CLI: load latest `results/datasets/<timestamp>/`, train; save to `results/trainings/<timestamp>/` and `train.log`.                                                                  |
| **evaluate.py**                 | CLI: load checkpoint from latest `results/trainings/<timestamp>/`, evaluate; save to `results/classifications/<timestamp>/` and `evaluate.log`.                                     |
| **create_evaluation_report.py** | CLI: read latest `results/classifications/<timestamp>/evaluation_results.json`, write HTML report into that folder; `create_evaluation_report.log`.                                 |
| **visualize_graphs.py**         | CLI: read latest `results/datasets/<timestamp>/`, write to `results/visualizations/<timestamp>/` and `visualize.log`; uses `utils` HTML helpers.                                    |


