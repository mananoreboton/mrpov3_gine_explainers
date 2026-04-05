"""
Train GNN on MPro Version 3: classification only (Category: low / medium / high potency).
Requires a pre-built PyG dataset (run build_dataset.py first). Saves best model per fold to
results/trainings/fold_<k>/<checkpoint_name>, fold_<k>/training_metrics.json, and
results/trainings/training_summary.json (aggregate over all fold_*/training_metrics.json).

Usage:
  uv run python build_dataset.py --data_root /path/to/snapshot
  uv run python train.py --data_root /path/to/snapshot [--num_folds 5] [--epochs 100]
  uv run python train.py ... --fold_index 0
  uv run python train.py ... --fold_indices 0 2 4
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

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
from utils import RunLogger, log_overwrite_dir_if_nonempty, log_overwrite_if_exists
from validation import evaluate_validation

_TRAINING_METRICS_JSON = "training_metrics.json"
_TRAINING_SUMMARY_JSON = "training_summary.json"
_FOLD_DIR_RE = re.compile(r"^fold_(\d+)$")


def _write_fold_training_metrics(
    fold_dir: Path,
    *,
    fold_index: int,
    num_folds: int,
    use_validation: bool,
    best_validation_accuracy: Optional[float],
    train_accuracy_at_best_validation: float,
    checkpoint: str,
    log,
) -> None:
    path = fold_dir / _TRAINING_METRICS_JSON
    log_overwrite_if_exists(path, log)
    payload = {
        "fold_index": fold_index,
        "num_folds": num_folds,
        "use_validation": use_validation,
        "best_validation_accuracy": best_validation_accuracy,
        "train_accuracy_at_best_validation": train_accuracy_at_best_validation,
        "checkpoint": checkpoint,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def scan_and_write_training_summary(trainings_root: Path, log) -> None:
    """Load all fold_*/training_metrics.json and write trainings/training_summary.json."""
    fold_entries: list[dict] = []
    for child in sorted(trainings_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not _FOLD_DIR_RE.match(child.name):
            continue
        metrics_path = child / _TRAINING_METRICS_JSON
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        raw_best_val = data.get("best_validation_accuracy")
        best_val: Optional[float] = (
            None if raw_best_val is None else float(raw_best_val)
        )
        use_val = bool(data.get("use_validation", best_val is not None))
        fold_entries.append(
            {
                "fold_index": int(data["fold_index"]),
                "num_folds": int(data["num_folds"]),
                "use_validation": use_val,
                "best_validation_accuracy": best_val,
                "train_accuracy_at_best_validation": float(
                    data["train_accuracy_at_best_validation"]
                ),
                "checkpoint": str(data["checkpoint"]),
            }
        )
    fold_entries.sort(key=lambda x: x["fold_index"])
    summary_path = trainings_root / _TRAINING_SUMMARY_JSON
    if not fold_entries:
        log_overwrite_if_exists(summary_path, log)
        summary_path.write_text(json.dumps({"folds": []}, indent=2), encoding="utf-8")
        log(f"Training summary: no {_TRAINING_METRICS_JSON} under {trainings_root}; "
            f"wrote empty {summary_path.name}")
        return

    with_val = [f for f in fold_entries if f["best_validation_accuracy"] is not None]
    if with_val:
        best_val_fold = max(
            with_val,
            key=lambda f: (f["best_validation_accuracy"], -f["fold_index"]),
        )["fold_index"]
    else:
        best_val_fold = None
    best_train_fold = max(
        fold_entries,
        key=lambda f: (f["train_accuracy_at_best_validation"], -f["fold_index"]),
    )["fold_index"]
    out = {
        "folds": fold_entries,
        "best_validation_fold_index": best_val_fold,
        "best_train_accuracy_fold_index": best_train_fold,
    }
    log_overwrite_if_exists(summary_path, log)
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(
        f"Training summary: best_validation_fold_index={best_val_fold} "
        f"best_train_accuracy_fold_index={best_train_fold} → {summary_path}"
    )


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
    parser.add_argument(
        "--no_validation",
        action="store_true",
        help="Do not read the val split file; train without validation (checkpoint on best train acc). "
        "Also implied when the val split resolves to zero samples.",
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
                use_validation=not args.no_validation,
            )
            train_loader, val_loader, _ = create_data_loaders(
                dataset_base, data_root, split_config, batch_size=args.batch_size
            )
            use_val = split_config.use_validation and len(val_loader.dataset) > 0
            if not split_config.use_validation:
                log.log("Validation disabled (--no_validation): val split file not loaded.")
            elif not use_val:
                log.log(
                    "Validation split has zero samples in the built dataset; "
                    "training without validation (checkpoint on best train accuracy)."
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

            best_val_acc: Optional[float] = 0.0
            best_train_acc_at_best_val = 0.0
            best_train_for_ckpt = -1.0
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    criterion_ce,
                )
                if use_val:
                    val_metrics = evaluate_validation(model, val_loader, device)
                    if val_metrics.accuracy > best_val_acc:
                        best_val_acc = val_metrics.accuracy
                        best_train_acc_at_best_val = train_acc
                        torch.save(model.state_dict(), ckpt_path)
                    if epoch % 10 == 0 or epoch == 1:
                        log.log(
                            f"Epoch {epoch:3d}  fold_index={k}  "
                            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                            f"val_acc={val_metrics.accuracy:.4f}"
                        )
                else:
                    if train_acc > best_train_for_ckpt:
                        best_train_for_ckpt = train_acc
                        best_train_acc_at_best_val = train_acc
                        torch.save(model.state_dict(), ckpt_path)
                    if epoch % 10 == 0 or epoch == 1:
                        log.log(
                            f"Epoch {epoch:3d}  fold_index={k}  "
                            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                        )
            if use_val:
                log.log(
                    f"Best validation score (accuracy): {best_val_acc:.4f}  "
                    f"train_acc@best_val={best_train_acc_at_best_val:.4f}  "
                    f"fold_index={k}  checkpoint={ckpt_path}"
                )
            else:
                log.log(
                    f"Best train accuracy (no validation): {best_train_for_ckpt:.4f}  "
                    f"fold_index={k}  checkpoint={ckpt_path}"
                )
            _write_fold_training_metrics(
                ckpt_path.parent,
                fold_index=k,
                num_folds=args.num_folds,
                use_validation=use_val,
                best_validation_accuracy=best_val_acc if use_val else None,
                train_accuracy_at_best_validation=best_train_acc_at_best_val,
                checkpoint=args.checkpoint,
                log=log.log,
            )

        scan_and_write_training_summary(out_dir, log.log)
        log.log(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
