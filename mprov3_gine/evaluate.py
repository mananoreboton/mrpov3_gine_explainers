"""
Classify the test set on the trained GNN.
Loads a saved checkpoint from results/trainings/fold_<k>/ (or legacy flat trainings/ for fold 0).
Saves per-sample results to results/classifications/fold_<fold_index>/ and writes
results/classifications/classification_summary.json (scan of all fold_*/evaluation_results.json).
Usage:
  uv run python evaluate.py [--data_root /path/to/snapshot] [--checkpoint best_gnn.pt]
  uv run python evaluate.py ... --fold_index 0
  uv run python evaluate.py ... --fold_indices 0 1 2
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
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
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_VAL_SPLIT_FILE,
    resolve_checkpoint_path,
    resolve_dataset_dir,
    resolve_fold_indices,
    RESULTS_CLASSIFICATIONS,
    RESULTS_DATASETS,
    SplitConfig,
)

from evaluation import evaluate_test_with_predictions, print_test_report
from loaders import create_data_loaders
from model import MProGNN
from utils import RunLogger, log_overwrite_if_exists

_CLASSIFICATION_SUMMARY_JSON = "classification_summary.json"


def scan_and_write_classification_summary(classifications_root: Path, log) -> None:
    """Scan fold_*/evaluation_results.json and write classification_summary.json."""
    summary_path = classifications_root / _CLASSIFICATION_SUMMARY_JSON
    fold_entries: list[dict] = []
    for json_path in sorted(classifications_root.glob("fold_*/evaluation_results.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        fold_entries.append(
            {
                "fold_index": int(data["fold_index"]),
                "test_accuracy": float(data["accuracy"]),
            }
        )
    fold_entries.sort(key=lambda x: x["fold_index"])
    log_overwrite_if_exists(summary_path, log)
    if not fold_entries:
        summary_path.write_text(json.dumps({"folds": []}, indent=2), encoding="utf-8")
        log(
            f"Classification summary: no fold_*/evaluation_results.json under "
            f"{classifications_root}; wrote empty {summary_path.name}"
        )
        return
    best_fold = max(
        fold_entries,
        key=lambda f: (f["test_accuracy"], -f["fold_index"]),
    )["fold_index"]
    out = {
        "folds": fold_entries,
        "best_classification_fold_index": best_fold,
    }
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(
        f"Classification summary: best_classification_fold_index={best_fold} → {summary_path}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN on the test set (classification, run independently of training)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Splits/); default: DEFAULT_DATA_ROOT",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); reads trainings/ and datasets/, writes to classifications/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_TRAINING_CHECKPOINT_FILENAME,
        help=f"Checkpoint filename (default: {DEFAULT_TRAINING_CHECKPOINT_FILENAME}); under trainings/fold_<k>/ per fold.",
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
        help="Evaluate a single fold (0 .. num_folds-1). Default: all folds.",
    )
    fold_group.add_argument(
        "--fold_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Evaluate these fold indices only. Default: all folds.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=DEFAULT_HIDDEN_CHANNELS,
        help="Must match trained model",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Must match trained model",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Must match trained model",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_OUT_CLASSES,
        help=f"Number of classes (default: {DEFAULT_OUT_CLASSES})",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for k in fold_list:
        checkpoint_path = resolve_checkpoint_path(
            results_root, args.checkpoint, fold_index=k
        )
        split_config = SplitConfig(
            train_file=args.train_split_file,
            val_file=args.val_split_file,
            test_file=args.test_split_file,
            num_folds=args.num_folds,
            fold_index=k,
            dataset_name=dataset_name,
        )
        fold_dir = f"fold_{k}"
        out_dir = results_root / RESULTS_CLASSIFICATIONS / fold_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "evaluate.log"
        results_path = out_dir / "evaluation_results.json"

        with RunLogger(log_path) as log:
            log.log(
                f"CV fold: {k + 1}/{split_config.num_folds} "
                f"(fold_index={k}, num_folds={split_config.num_folds})"
            )
            log.log(f"Output: {out_dir}")
            log.log(f"Checkpoint: {checkpoint_path}")
            log.log(f"Dataset: {dataset_dir}")

            _, _, test_loader = create_data_loaders(
                dataset_base, data_root, split_config, batch_size=args.batch_size
            )
            log.log(f"Test set size: {len(test_loader.dataset)}")

            model = MProGNN(
                in_channels=DEFAULT_IN_CHANNELS,
                hidden_channels=args.hidden,
                num_layers=args.num_layers,
                dropout=args.dropout,
                out_classes=args.num_classes,
                pool=DEFAULT_POOL,
                edge_dim=DEFAULT_EDGE_DIM,
            ).to(device)
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=False)
            )

            log.log(f"Running test evaluation (fold_index={k})")
            test_metrics, results = evaluate_test_with_predictions(model, test_loader, device)

            payload = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "data_root": str(data_root.resolve()),
                "results_root": str(results_root.resolve()),
                "dataset_name": dataset_name,
                "fold_index": k,
                "num_folds": split_config.num_folds,
                "accuracy": test_metrics.accuracy,
                "results": [
                    {"pdb_id": pdb_id, "real_category": real, "predicted_category": pred}
                    for pdb_id, real, pred in results
                ],
            }
            log_overwrite_if_exists(results_path, log.log)
            log.log(
                f"Test accuracy (Category): {test_metrics.accuracy:.4f} "
                f"(fold_index={k})"
            )
            print_test_report(test_metrics)
            results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.log(f"Results saved to {results_path}")
            log.log(f"Log written to {log_path}")

    scan_and_write_classification_summary(
        results_root / RESULTS_CLASSIFICATIONS,
        print,
    )


if __name__ == "__main__":
    main()
