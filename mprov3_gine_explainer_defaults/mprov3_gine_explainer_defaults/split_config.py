"""Train/val/test split configuration for MPro GINE loaders."""

from __future__ import annotations

from dataclasses import dataclass

from mprov3_gine_explainer_defaults.data_path_defaults import (
    DEFAULT_PYG_DATASET_NAME,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
)
from mprov3_gine_explainer_defaults.training_defaults import DEFAULT_FOLD_INDEX, DEFAULT_NUM_FOLDS


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/val/test split: three files and folds."""

    train_file: str = DEFAULT_TRAIN_SPLIT_FILE
    val_file: str = DEFAULT_VAL_SPLIT_FILE
    test_file: str = DEFAULT_TEST_SPLIT_FILE
    num_folds: int = DEFAULT_NUM_FOLDS
    fold_index: int = DEFAULT_FOLD_INDEX
    dataset_name: str = DEFAULT_PYG_DATASET_NAME
