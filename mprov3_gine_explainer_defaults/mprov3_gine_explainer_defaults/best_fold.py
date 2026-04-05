"""Resolve best CV fold from GNN result summaries (evaluate.py / train.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from mprov3_gine_explainer_defaults.data_path_defaults import (
    RESULTS_CLASSIFICATIONS,
    RESULTS_TRAININGS,
)
from mprov3_gine_explainer_defaults.training_defaults import DEFAULT_NUM_FOLDS

FoldMetric = Literal["test_accuracy", "train_accuracy"]

_CLASSIFICATION_SUMMARY = "classification_summary.json"
_TRAINING_SUMMARY = "training_summary.json"


def resolve_best_fold_index(results_root: Path, metric: FoldMetric) -> int:
    """
    Pick fold index from aggregate JSON written by mprov3_gine.

    * test_accuracy: classification_summary.json (run evaluate.py first).
    * train_accuracy: training_summary.json (run train.py first).
    """
    root = Path(results_root)
    if metric == "test_accuracy":
        path = root / RESULTS_CLASSIFICATIONS / _CLASSIFICATION_SUMMARY
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run mprov3_gine/evaluate.py to write "
                f"{_CLASSIFICATION_SUMMARY} before explainers."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        folds = data.get("folds") or []
        if not folds:
            raise FileNotFoundError(
                f"{path} has no folds; run evaluate.py on at least one fold."
            )
        return int(data["best_classification_fold_index"])

    path = root / RESULTS_TRAININGS / _TRAINING_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run mprov3_gine/train.py to write "
            f"{_TRAINING_SUMMARY} before using --fold_metric train_accuracy."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    folds = data.get("folds") or []
    if not folds:
        raise FileNotFoundError(
            f"{path} has no folds; run train.py on at least one fold."
        )
    return int(data["best_train_accuracy_fold_index"])


def read_num_folds_for_fold(results_root: Path, fold_index: int) -> int:
    """num_folds for SplitConfig: prefer classification JSON, then training metrics."""
    root = Path(results_root)
    k = int(fold_index)
    eval_path = root / RESULTS_CLASSIFICATIONS / f"fold_{k}" / "evaluation_results.json"
    if eval_path.is_file():
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        if "num_folds" in data:
            return int(data["num_folds"])
    train_path = root / RESULTS_TRAININGS / f"fold_{k}" / "training_metrics.json"
    if train_path.is_file():
        data = json.loads(train_path.read_text(encoding="utf-8"))
        if "num_folds" in data:
            return int(data["num_folds"])
    return int(DEFAULT_NUM_FOLDS)
