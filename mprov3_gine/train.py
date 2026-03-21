"""
Train GNN on MPro Version 3: classification only (Category: low / medium / high potency).
Requires a pre-built PyG dataset (run build_dataset.py first). Saves best model to results/trainings/<timestamp>/.
Usage:
  uv run python build_dataset.py --data_root /path/to/snapshot
  uv run python train.py --data_root /path/to/snapshot [--num_folds 5] [--fold_index 0] [--epochs 100]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from mprov3_gine_explainer_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EDGE_DIM,
    DEFAULT_FOLD_INDEX,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_NUM_FOLDS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_POOL,
    DEFAULT_SEED,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LR,
)

from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_VAL_SPLIT_FILE,
    PYG_DATA_FILENAME,
    RESULTS_DATASETS,
    RESULTS_TRAININGS,
    SplitConfig,
)
from loaders import create_data_loaders
from model import MProGNN
from train_epoch import train_one_epoch
from utils import RunLogger, get_latest_timestamp_dir, run_timestamp
from validation import evaluate_validation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN on MPro Version 3 (classification)")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Splits/, Info.csv); default: config.DEFAULT_DATA_ROOT",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); uses latest results/datasets/<timestamp>/ and writes to results/trainings/<timestamp>/.",
    )
    parser.add_argument(
        "--train_split_file",
        type=str,
        default=DEFAULT_TRAIN_SPLIT_FILE,
        help=f"Train split file in Splits/ (default: {DEFAULT_TRAIN_SPLIT_FILE})",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=f"Val split file in Splits/ (default: {DEFAULT_VAL_SPLIT_FILE})",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=f"Test split file in Splits/ (default: {DEFAULT_TEST_SPLIT_FILE})",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help=f"Number of folds (default: {DEFAULT_NUM_FOLDS})",
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        default=DEFAULT_FOLD_INDEX,
        help="Which fold to use (0 .. num_folds-1)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING_LR)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_OUT_CLASSES,
        help=f"Number of classes (Category, default: {DEFAULT_OUT_CLASSES})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    results_root = Path(args.results_root or DEFAULT_RESULTS_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    dataset_base = results_root / RESULTS_DATASETS
    latest_dataset = get_latest_timestamp_dir(dataset_base)
    if latest_dataset is None or not (latest_dataset / PYG_DATA_FILENAME).exists():
        raise FileNotFoundError(
            f"No dataset found under {dataset_base}. Run build_dataset.py with --results_root {results_root} first."
        )
    dataset_name = latest_dataset.name

    ts = run_timestamp()
    out_dir = results_root / RESULTS_TRAININGS / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    torch.manual_seed(args.seed)

    split_config = SplitConfig(
        train_file=args.train_split_file,
        val_file=args.val_split_file,
        test_file=args.test_split_file,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        dataset_name=dataset_name,
    )
    with RunLogger(log_path) as log:
        log.log(f"Dataset: {dataset_base / dataset_name} (latest)")
        log.log(f"Output: {out_dir}")
        train_loader, val_loader, _ = create_data_loaders(
            dataset_base, data_root, split_config, batch_size=args.batch_size
        )
        log.log(f"Dataset size (train/val): {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MProGNN(
            in_channels=DEFAULT_IN_CHANNELS,
            hidden_channels=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
            out_classes=args.num_classes,
            pool=DEFAULT_POOL,
            edge_dim=DEFAULT_EDGE_DIM,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion_ce = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                criterion_ce,
            )
            val_metrics = evaluate_validation(model, val_loader, device)
            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                torch.save(model.state_dict(), out_dir / DEFAULT_TRAINING_CHECKPOINT_FILENAME)
            if epoch % 10 == 0 or epoch == 1:
                log.log(
                    f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_acc={val_metrics.accuracy:.4f}"
                )
        log.log(f"Best checkpoint saved to {out_dir / DEFAULT_TRAINING_CHECKPOINT_FILENAME}")
        log.log(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
