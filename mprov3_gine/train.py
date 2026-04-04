"""
Train GNN on MPro Version 3: classification only (Category: low / medium / high potency).
Requires a pre-built PyG dataset (run build_dataset.py first). Saves best model per fold to
results/trainings/fold_<k>/<checkpoint_name>.

Usage:
  uv run python build_dataset.py --data_root /path/to/snapshot
  uv run python train.py --data_root /path/to/snapshot [--num_folds 5] [--epochs 100]
  uv run python train.py ... --fold_index 0
  uv run python train.py ... --fold_indices 0 2 4
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from mprov3_gine_explainer_defaults import (
    BUILT_DATASET_FOLDER_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_DROPOUT,
    DEFAULT_EDGE_DIM,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_NUM_FOLDS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_POOL,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SEED,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LR,
    DEFAULT_VAL_SPLIT_FILE,
    RESULTS_DATASETS,
    RESULTS_TRAININGS,
    SplitConfig,
    resolve_dataset_dir,
    resolve_fold_indices,
    training_checkpoint_path,
)

from loaders import create_data_loaders
from model import MProGNN
from train_epoch import train_one_epoch
from utils import RunLogger, log_overwrite_dir_if_nonempty
from validation import evaluate_validation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN on MPro Version 3 (classification)")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Splits/, Info.csv); default: DEFAULT_DATA_ROOT",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); reads results/datasets/data.pt, writes to results/trainings/.",
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
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument(
        "--fold_index",
        type=int,
        default=None,
        help="Train a single fold (0 .. num_folds-1). Default: all folds.",
    )
    fold_group.add_argument(
        "--fold_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Train these fold indices only. Default: all folds 0..num_folds-1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_TRAINING_CHECKPOINT_FILENAME,
        help=f"Checkpoint filename under each trainings/fold_<k>/ (default: {DEFAULT_TRAINING_CHECKPOINT_FILENAME}).",
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

    fold_list = resolve_fold_indices(
        args.num_folds,
        fold_index=args.fold_index,
        fold_indices=args.fold_indices,
    )

    dataset_dir = resolve_dataset_dir(results_root)
    dataset_base = results_root / RESULTS_DATASETS
    dataset_name = BUILT_DATASET_FOLDER_NAME

    out_dir = results_root / RESULTS_TRAININGS
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    torch.manual_seed(args.seed)

    with RunLogger(log_path) as log:
        log_overwrite_dir_if_nonempty(out_dir, log.log)
        log.log(f"Dataset: {dataset_dir}")
        log.log(f"Output: {out_dir}")
        log.log(
            f"Folds to train: {fold_list} (num_folds={args.num_folds}, "
            f"checkpoint name={args.checkpoint})"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k in fold_list:
            log.log("")
            log.log(
                f"========== CV fold {k + 1}/{args.num_folds} "
                f"(fold_index={k}) =========="
            )
            split_config = SplitConfig(
                train_file=args.train_split_file,
                val_file=args.val_split_file,
                test_file=args.test_split_file,
                num_folds=args.num_folds,
                fold_index=k,
                dataset_name=dataset_name,
            )
            train_loader, val_loader, _ = create_data_loaders(
                dataset_base, data_root, split_config, batch_size=args.batch_size
            )
            log.log(
                f"Dataset size (train/val): {len(train_loader.dataset)} train, "
                f"{len(val_loader.dataset)} val"
            )

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

            ckpt_path = training_checkpoint_path(results_root, k, args.checkpoint)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            best_val_acc = 0.0
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    criterion_ce,
                )
                val_metrics = evaluate_validation(model, val_loader, device)
                if val_metrics.accuracy > best_val_acc:
                    best_val_acc = val_metrics.accuracy
                    torch.save(model.state_dict(), ckpt_path)
                if epoch % 10 == 0 or epoch == 1:
                    log.log(
                        f"Epoch {epoch:3d}  fold_index={k}  "
                        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                        f"val_acc={val_metrics.accuracy:.4f}"
                    )
            log.log(
                f"Best validation score (accuracy): {best_val_acc:.4f}  "
                f"fold_index={k}  checkpoint={ckpt_path}"
            )

        log.log(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
