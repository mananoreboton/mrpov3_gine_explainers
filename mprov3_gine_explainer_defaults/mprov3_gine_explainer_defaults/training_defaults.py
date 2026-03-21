"""Default training loop and CV settings for GNN training scripts."""

from __future__ import annotations

DEFAULT_TRAINING_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_TRAINING_LR: float = 1e-3
DEFAULT_SEED: int = 42

DEFAULT_NUM_FOLDS: int = 5
DEFAULT_FOLD_INDEX: int = 0
