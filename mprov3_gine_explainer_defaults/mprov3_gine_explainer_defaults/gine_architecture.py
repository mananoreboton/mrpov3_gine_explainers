"""
Default GINE architecture for MPro v3 — keep in sync with training and explainers.
"""

from __future__ import annotations

DEFAULT_IN_CHANNELS: int = 4  # x, y, z, atomic number
DEFAULT_HIDDEN_CHANNELS: int = 64
DEFAULT_NUM_LAYERS: int = 3
DEFAULT_DROPOUT: float = 0.2
DEFAULT_OUT_CLASSES: int = 3
DEFAULT_POOL: str = "mean"
DEFAULT_EDGE_DIM: int = 1
