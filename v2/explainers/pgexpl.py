"""
PGEXPL — PGExplainer (PyG).
See doc/table_of_explainer_implementations.md.
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer

from explainers.shared.explainer_builder import build_pyg_explainer
from mprov3_gine_explainer_defaults import (
    DEFAULT_PG_EXPLAINER_EPOCHS,
    DEFAULT_PG_EXPLAINER_LR,
    EDGE_MASK_OBJECT,
)


class PgExplExplainer:
    """PGExplainer (edge-level; PyG requires ``explanation_type='phenomenon'``). Training runs when explaining."""

    @staticmethod
    def build_explainer(
        model: nn.Module,
        *,
        epochs: int = DEFAULT_PG_EXPLAINER_EPOCHS,
        lr: float = DEFAULT_PG_EXPLAINER_LR,
        **algorithm_kwargs,
    ) -> Explainer:
        # PGExplainer only supports phenomenon-level explanations in PyG 2.7+.
        return build_pyg_explainer(
            model,
            algorithm=PGExplainer(epochs=epochs, lr=lr, **algorithm_kwargs),
            explanation_type="phenomenon",
            node_mask_type=None,
            edge_mask_type=EDGE_MASK_OBJECT,
        )
