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


def resolve_checkpoint_path(
    results_root: Path,
    checkpoint_name: str = DEFAULT_TRAINING_CHECKPOINT_FILENAME,
) -> Path:
    """Resolve path to ``results_root/trainings/<checkpoint_name>``."""
    path = results_root / RESULTS_TRAININGS / checkpoint_name
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run train.py first."
        )
    return path


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
