"""
Run test-set classification with a saved training checkpoint (independent of train.py).

Writes results/classifications/fold_<k>/classification_results.json and classification_summary.json.
See README.md (Usage) and ``classify.py --help`` for flags (splits and architecture must match training).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from mprov3_gine_explainer_defaults import (
    BUILT_DATASET_FOLDER_NAME,
    CLASSIFICATION_RESULTS_JSON,
    DEFAULT_DATA_ROOT,
    DEFAULT_EDGE_DIM,
    DEFAULT_IN_CHANNELS,
    DEFAULT_POOL,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    LEGACY_EVALUATION_RESULTS_JSON,
    resolve_checkpoint_path,
    resolve_dataset_dir,
    resolve_fold_indices,
    RESULTS_CLASSIFICATIONS,
    RESULTS_DATASETS,
    SplitConfig,
)

from cli_common import (
    add_batch_size_arg,
    add_checkpoint_arg,
    add_data_and_results_roots,
    add_model_loader_args,
    add_split_and_fold_args,
)
from classification import classify_test_with_predictions, print_test_classification_report
from loaders import create_data_loaders
from model import MProGNN
from utils import FOLD_SUBDIR_NAME_RE, RunLogger, log_overwrite_if_exists

_CLASSIFICATION_SUMMARY_JSON = "classification_summary.json"


def _fold_classification_json_paths(classifications_root: Path) -> list[Path]:
    """Per-fold JSON paths; prefer classification_results.json over legacy evaluation_results.json."""
    by_fold: dict[int, Path] = {}
    for p in classifications_root.glob(f"fold_*/{CLASSIFICATION_RESULTS_JSON}"):
        m = FOLD_SUBDIR_NAME_RE.match(p.parent.name)
        if m:
            by_fold[int(m.group(1))] = p
    for p in classifications_root.glob(f"fold_*/{LEGACY_EVALUATION_RESULTS_JSON}"):
        m = FOLD_SUBDIR_NAME_RE.match(p.parent.name)
        if m:
            k = int(m.group(1))
            if k not in by_fold:
                by_fold[k] = p
    return [by_fold[k] for k in sorted(by_fold)]


def scan_and_write_classification_summary(classifications_root: Path, log) -> None:
    """Scan per-fold classification JSON and write classification_summary.json."""
    summary_path = classifications_root / _CLASSIFICATION_SUMMARY_JSON
    fold_entries: list[dict] = []
    for json_path in _fold_classification_json_paths(classifications_root):
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
            f"Classification summary: no fold_*/{CLASSIFICATION_RESULTS_JSON} or "
            f"fold_*/{LEGACY_EVALUATION_RESULTS_JSON} under "
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
        description=(
            "Classify the test set with a trained GNN checkpoint "
            "(independent of train.py)."
        )
    )
    add_data_and_results_roots(
        parser,
        results_help="reads trainings/ and datasets/, writes to classifications/.",
        data_root_help="Path to raw MPro snapshot (Splits/); default: DEFAULT_DATA_ROOT",
    )
    add_checkpoint_arg(
        parser,
        help=f"Checkpoint filename (default: {DEFAULT_TRAINING_CHECKPOINT_FILENAME}); under trainings/fold_<k>/ per fold.",
    )
    add_split_and_fold_args(
        parser,
        fold_indices_help=(
            "Classify the test set for these fold indices only (e.g. one fold: --fold_indices 2). "
            "Default: all folds."
        ),
    )
    add_batch_size_arg(parser, for_classification=True)
    add_model_loader_args(parser, for_classification=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    results_root = Path(args.results_root or DEFAULT_RESULTS_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    fold_list = resolve_fold_indices(args.num_folds, fold_indices=args.fold_indices)

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
        log_path = out_dir / "classify.log"
        results_path = out_dir / CLASSIFICATION_RESULTS_JSON

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

            log.log(f"Running test-set classification (fold_index={k})")
            test_metrics, results = classify_test_with_predictions(
                model, test_loader, device
            )

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
            print_test_classification_report(test_metrics)
            results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.log(f"Results saved to {results_path}")
            log.log(f"Log written to {log_path}")

    scan_and_write_classification_summary(
        results_root / RESULTS_CLASSIFICATIONS,
        print,
    )


if __name__ == "__main__":
    main()
