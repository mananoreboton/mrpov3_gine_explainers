"""Re-exports shared explainer constants from mprov3_gine_explainer_defaults; local code in explainer_builder."""

from explainers.shared.explainer_builder import build_pyg_explainer
from mprov3_gine_explainer_defaults import (
    DEFAULT_EXPLANATION_TYPE,
    DEFAULT_GNN_EXPLAINER_EPOCHS,
    DEFAULT_GNN_EXPLAINER_LR,
    DEFAULT_IG_INTERNAL_BATCH_SIZE,
    DEFAULT_IG_N_STEPS,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PG_EXPLAINER_EPOCHS,
    DEFAULT_PG_EXPLAINER_LR,
    DEFAULT_PGM_NUM_SAMPLES,
    EDGE_MASK_OBJECT,
    NODE_MASK_ATTRIBUTES,
)

__all__ = [
    "build_pyg_explainer",
    "DEFAULT_EXPLANATION_TYPE",
    "DEFAULT_MODEL_CONFIG",
    "NODE_MASK_ATTRIBUTES",
    "EDGE_MASK_OBJECT",
    "DEFAULT_GNN_EXPLAINER_EPOCHS",
    "DEFAULT_GNN_EXPLAINER_LR",
    "DEFAULT_PG_EXPLAINER_EPOCHS",
    "DEFAULT_PG_EXPLAINER_LR",
    "DEFAULT_IG_N_STEPS",
    "DEFAULT_IG_INTERNAL_BATCH_SIZE",
    "DEFAULT_PGM_NUM_SAMPLES",
]
