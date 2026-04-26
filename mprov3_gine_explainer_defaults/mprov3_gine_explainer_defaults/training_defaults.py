"""Default training loop and CV settings for GNN training scripts."""

from __future__ import annotations

DEFAULT_TRAINING_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_TRAINING_LR: float = 1e-3
DEFAULT_SEED: int = 42

DEFAULT_NUM_FOLDS: int = 5
DEFAULT_FOLD_INDEX: int = 0


def seed_everything(seed: int) -> None:
    """Pin every RNG used downstream so results are reproducible.

    Covers Python ``random``, NumPy, PyTorch (CPU + GPU), and PyG.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from torch_geometric import seed_everything as _pyg_seed

        _pyg_seed(seed)
    except Exception:
        pass
