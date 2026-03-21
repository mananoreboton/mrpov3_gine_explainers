"""Default paths and hyperparameters for GNN training on MPro Version 3."""

from dataclasses import dataclass
from pathlib import Path

from mprov3_gine_explainer_defaults import (
    CHECK_FORMAT_DATASETS_SUBDIR,
    CHECK_FORMAT_RAW_DATA_SUBDIR,
    DEFAULT_FOLD_INDEX,
    DEFAULT_MPRO_SNAPSHOT_DIR_NAME,
    DEFAULT_NUM_FOLDS,
    DEFAULT_PYG_DATASET_NAME,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_VAL_SPLIT_FILE,
    MPRO_INFO_CSV,
    MPRO_LIGAND_DIR,
    MPRO_LIGAND_SDF_SUBDIR,
    MPRO_SPLITS_DIR,
    PYG_DATA_FILENAME,
    PYG_PDB_ORDER_FILENAME,
    RESULTS_CHECK_FORMAT,
    RESULTS_CLASSIFICATIONS,
    RESULTS_DATASETS,
    RESULTS_DIR_NAME,
    RESULTS_TRAININGS,
    RESULTS_VISUALIZATIONS,
)

_PROJECT_ROOT = Path(__file__).resolve().parent

# Default data root: MPro snapshot directory (sibling of mprov3_gine)
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT.parent / DEFAULT_MPRO_SNAPSHOT_DIR_NAME)

# Script outputs under mprov3_gine/results/
DEFAULT_RESULTS_ROOT = str(_PROJECT_ROOT / RESULTS_DIR_NAME)

__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_PYG_DATASET_NAME",
    "DEFAULT_TRAIN_SPLIT_FILE",
    "DEFAULT_VAL_SPLIT_FILE",
    "DEFAULT_TEST_SPLIT_FILE",
    "SplitConfig",
    "MPRO_INFO_CSV",
    "MPRO_SPLITS_DIR",
    "MPRO_LIGAND_DIR",
    "MPRO_LIGAND_SDF_SUBDIR",
    "PYG_DATA_FILENAME",
    "PYG_PDB_ORDER_FILENAME",
    "DEFAULT_TRAINING_CHECKPOINT_FILENAME",
    "RESULTS_TRAININGS",
    "RESULTS_DATASETS",
    "RESULTS_CLASSIFICATIONS",
    "RESULTS_VISUALIZATIONS",
    "RESULTS_CHECK_FORMAT",
    "CHECK_FORMAT_DATASETS_SUBDIR",
    "CHECK_FORMAT_RAW_DATA_SUBDIR",
]


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/val/test split: three files and folds."""

    train_file: str = DEFAULT_TRAIN_SPLIT_FILE
    val_file: str = DEFAULT_VAL_SPLIT_FILE
    test_file: str = DEFAULT_TEST_SPLIT_FILE
    num_folds: int = DEFAULT_NUM_FOLDS
    fold_index: int = DEFAULT_FOLD_INDEX
    dataset_name: str = DEFAULT_PYG_DATASET_NAME
