"""Default hyperparameters for PyG / Captum explainer algorithms."""

from __future__ import annotations

DEFAULT_GNN_EXPLAINER_EPOCHS: int = 200
DEFAULT_GNN_EXPLAINER_LR: float = 0.01

DEFAULT_PG_EXPLAINER_EPOCHS: int = 30
DEFAULT_PG_EXPLAINER_LR: float = 0.003

DEFAULT_IG_N_STEPS: int = 32
DEFAULT_IG_INTERNAL_BATCH_SIZE: int | None = None

DEFAULT_PGM_NUM_SAMPLES: int = 100
