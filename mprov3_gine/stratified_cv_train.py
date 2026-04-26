"""
Stratified K-fold CV on the unified split pool (unique PDBs across all fold lists).

Builds the pool with dataset.get_unified_pool_indices, runs training like train.py
(checkpoint on best train accuracy), evaluates on each held-out fold, aggregates
out-of-fold predictions, and writes sklearn metrics to results/trainings/.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from mprov3_gine_explainer_defaults import (
    BUILT_DATASET_FOLDER_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_EDGE_DIM,
    DEFAULT_IN_CHANNELS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_POOL,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LR,
    RESULTS_DATASETS,
    resolve_dataset_dir,
    training_checkpoint_path,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from cli_common import (
    add_batch_size_arg,
    add_data_and_results_roots,
    add_model_loader_args,
    add_split_file_args,
)
from dataset import MProV3Dataset, get_unified_pool_indices, load_dataset_pdb_order
from loaders import create_subset_loaders
from model import MProGNN
from train import _write_fold_training_metrics, scan_and_write_training_summary
from train_epoch import train_one_epoch
from utils import RunLogger, log_overwrite_dir_if_nonempty

_STRATIFIED_CV_METRICS_JSON = "stratified_cv_metrics.json"


def _collect_predictions(
    model: MProGNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            category = batch.category.to(device).squeeze(-1)
            edge_attr = getattr(batch, "edge_attr", None)
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr)
            pred = logits.argmax(dim=1)
            y_true.extend(category.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    return y_true, y_pred


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stratified K-fold CV on unified train+val+test PDB pool "
            "(unique IDs across all split-file folds)."
        )
    )
    add_data_and_results_roots(
        parser,
        results_help="reads results/datasets/data.pt, writes to results/trainings/.",
    )
    add_split_file_args(parser)
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_EPOCHS)
    add_batch_size_arg(parser, for_classification=False)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING_LR)
    add_model_loader_args(parser, for_classification=False, include_num_classes=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Torch / CV shuffle seed. If omitted, a random seed is chosen and logged.",
    )
    parser.add_argument(
        "--no_validation",
        action="store_true",
        help="Build the unified pool from train and test split files only (omit val file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    results_root = Path(args.results_root or DEFAULT_RESULTS_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be at least 2")

    dataset_dir = resolve_dataset_dir(results_root)
    dataset_base = results_root / RESULTS_DATASETS
    dataset_name = BUILT_DATASET_FOLDER_NAME
    out_dir = results_root / RESULTS_TRAININGS
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_stratified_cv.log"

    seed = args.seed if args.seed is not None else secrets.randbits(31)
    torch.manual_seed(seed)

    ckpt_name = DEFAULT_TRAINING_CHECKPOINT_FILENAME
    labels = list(range(DEFAULT_OUT_CLASSES))

    with RunLogger(log_path) as log:
        log_overwrite_dir_if_nonempty(out_dir, log.log)
        log.log(f"Dataset: {dataset_dir}")
        log.log(f"Output: {out_dir}")
        log.log(f"Random seed: {seed}")
        log.log(
            f"Stratified CV: cv_folds={args.cv_folds}, epochs={args.epochs}, "
            f"checkpoint={ckpt_name}"
        )

        dataset_pdb_order = load_dataset_pdb_order(dataset_base, dataset_name)
        if dataset_pdb_order is None:
            raise FileNotFoundError(
                f"Missing pdb_order next to dataset under {dataset_base / dataset_name}"
            )

        dataset = MProV3Dataset(root=str(dataset_base), dataset_name=dataset_name)
        pool_idx, y = get_unified_pool_indices(
            data_root,
            args.train_split_file,
            args.val_split_file,
            args.test_split_file,
            dataset_pdb_order,
            dataset,
            use_validation=not args.no_validation,
        )
        n_pool = len(pool_idx)
        log.log(
            f"Unified pool: {n_pool} samples (unique PDBs in split files ∩ built dataset); "
            f"labels shape {y.shape}"
        )
        if n_pool < args.cv_folds:
            raise ValueError(
                f"Unified pool size {n_pool} is smaller than --cv_folds={args.cv_folds}"
            )

        class_counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
        min_class = min(class_counts.values()) if class_counts else 0
        if min_class < args.cv_folds:
            raise ValueError(
                "StratifiedKFold requires at least cv_folds samples per class. "
                f"Class counts: {class_counts}, --cv_folds={args.cv_folds}"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        skf = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=seed,
        )
        X_idx = np.arange(n_pool)

        oof_true: List[int] = []
        oof_pred: List[int] = []
        fold_rows: List[dict] = []

        for fold_k, (train_pos, test_pos) in enumerate(skf.split(X_idx, y)):
            log.log("")
            log.log(
                f"========== Stratified CV fold {fold_k + 1}/{args.cv_folds} "
                f"(fold_index={fold_k}) =========="
            )
            train_list = pool_idx[train_pos].tolist()
            test_list = pool_idx[test_pos].tolist()
            train_loader, test_loader = create_subset_loaders(
                dataset,
                train_list,
                test_list,
                batch_size=args.batch_size,
            )
            log.log(
                f"Split sizes: {len(train_loader.dataset)} train, "
                f"{len(test_loader.dataset)} held-out"
            )

            model = MProGNN(
                in_channels=DEFAULT_IN_CHANNELS,
                hidden_channels=args.hidden,
                num_layers=args.num_layers,
                dropout=args.dropout,
                out_classes=DEFAULT_OUT_CLASSES,
                pool=DEFAULT_POOL,
                edge_dim=DEFAULT_EDGE_DIM,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion_ce = nn.CrossEntropyLoss()

            ckpt_path = training_checkpoint_path(results_root, fold_k, ckpt_name)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            best_train_for_ckpt = -1.0
            best_train_acc_at_best = 0.0
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    criterion_ce,
                )
                if train_acc > best_train_for_ckpt:
                    best_train_for_ckpt = train_acc
                    best_train_acc_at_best = train_acc
                    torch.save(model.state_dict(), ckpt_path)
                if epoch % 10 == 0 or epoch == 1:
                    log.log(
                        f"Epoch {epoch:3d}  fold_index={fold_k}  "
                        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                    )

            log.log(
                f"Best train accuracy: {best_train_for_ckpt:.4f}  "
                f"fold_index={fold_k}  checkpoint={ckpt_path}"
            )
            _write_fold_training_metrics(
                ckpt_path.parent,
                fold_index=fold_k,
                num_folds=args.cv_folds,
                use_validation=False,
                best_validation_accuracy=None,
                train_accuracy_at_best_validation=best_train_acc_at_best,
                checkpoint=ckpt_name,
                log=log.log,
            )

            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            y_true, y_pred = _collect_predictions(model, test_loader, device)
            oof_true.extend(y_true)
            oof_pred.extend(y_pred)

            cm_fold = confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist()
            p_fold, r_fold, f_fold, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average=None,
                zero_division=0,
            )
            fold_rows.append(
                {
                    "fold_index": fold_k,
                    "n_train": len(train_list),
                    "n_test": len(test_list),
                    "confusion_matrix": cm_fold,
                    "precision_per_class": p_fold.tolist(),
                    "recall_per_class": r_fold.tolist(),
                    "f1_per_class": f_fold.tolist(),
                }
            )

        cm = confusion_matrix(oof_true, oof_pred, labels=labels)
        prec_per, rec_per, f1_per, support = precision_recall_fscore_support(
            oof_true,
            oof_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            oof_true,
            oof_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            oof_true,
            oof_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        metrics_path = out_dir / _STRATIFIED_CV_METRICS_JSON
        payload = {
            "cv_folds": args.cv_folds,
            "n_samples_evaluated": len(oof_true),
            "confusion_matrix": cm.tolist(),
            "precision_per_class": prec_per.tolist(),
            "recall_per_class": rec_per.tolist(),
            "f1_per_class": f1_per.tolist(),
            "support_per_class": support.tolist(),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "per_fold": fold_rows,
            "class_counts_in_pool": class_counts,
            "seed": seed,
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.log(
            f"OOF metrics → {metrics_path}  "
            f"macro P/R/F1: {prec_macro:.4f}/{rec_macro:.4f}/{f1_macro:.4f}"
        )

        scan_and_write_training_summary(out_dir, log.log)
        log.log(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
