"""
Resolve paths under a GNN `results/` tree (trainings, datasets, explanations, visualizations).

Uses segment constants from `data_path_defaults`; callers supply concrete `Path` roots
(e.g. mprov3_gine/results, mprov3_explainer/results).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from mprov3_gine_explainer_defaults.data_path_defaults import (
    BUILT_DATASET_FOLDER_NAME,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    PYG_DATA_FILENAME,
    RESULTS_DATASETS,
    RESULTS_EXPLANATIONS,
    RESULTS_TRAININGS,
    RESULTS_VISUALIZATIONS,
)


def training_checkpoint_path(
    results_root: Path,
    fold_index: int,
    checkpoint_name: str = DEFAULT_TRAINING_CHECKPOINT_FILENAME,
) -> Path:
    """Path under ``results_root/trainings/fold_<fold_index>/<checkpoint_name>`` (train.py writes here)."""
    return results_root / RESULTS_TRAININGS / f"fold_{fold_index}" / checkpoint_name


def resolve_checkpoint_path(
    results_root: Path,
    checkpoint_name: str = DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    fold_index: int | None = None,
) -> Path:
    """
    If *fold_index* is None: ``results_root/trainings/<checkpoint_name>`` (legacy flat layout).
    If set: per-fold file first; for fold 0 only, fall back to the flat path if missing.
    """
    trainings = results_root / RESULTS_TRAININGS
    if fold_index is None:
        path = trainings / checkpoint_name
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Run train.py first."
            )
        return path

    per_fold = training_checkpoint_path(results_root, fold_index, checkpoint_name)
    if per_fold.is_file():
        return per_fold
    if fold_index == 0:
        legacy = trainings / checkpoint_name
        if legacy.is_file():
            return legacy
    raise FileNotFoundError(
        f"Checkpoint not found: {per_fold}. Run train.py for this fold first."
    )


def resolve_dataset_dir(results_root: Path) -> Path:
    """
    Return ``results_root/datasets/`` if ``data.pt`` exists there (flat layout).
    """
    d = results_root / RESULTS_DATASETS
    pt = d / PYG_DATA_FILENAME
    if not pt.is_file():
        raise FileNotFoundError(
            f"{PYG_DATA_FILENAME} not found under {d}. Run build_dataset.py first."
        )
    return d


def resolve_training_checkpoint_and_dataset_name(
    results_root: Path,
) -> Tuple[Path, str]:
    """
    Checkpoint under ``results/trainings/`` and the dataset name for SplitConfig /
    MProV3Dataset (flat ``results/datasets/data.pt`` uses BUILT_DATASET_FOLDER_NAME).
    """
    checkpoint_path = resolve_checkpoint_path(results_root)
    resolve_dataset_dir(results_root)
    return checkpoint_path, BUILT_DATASET_FOLDER_NAME


def explanations_run_dir(results_root: Path, explainer_name: str) -> Path:
    """``results_root/explanations/<explainer_name>/``."""
    return results_root / RESULTS_EXPLANATIONS / explainer_name


def visualizations_run_dir(results_root: Path, explainer_name: str) -> Path:
    """``results_root/visualizations/<explainer_name>/``."""
    return results_root / RESULTS_VISUALIZATIONS / explainer_name
