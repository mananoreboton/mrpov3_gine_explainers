"""
GNNEXPL — GNNExplainer (PyG).
See doc/table_of_explainer_implementations.md.
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

from explainers.shared.explainer_builder import build_pyg_explainer
from mprov3_gine_explainer_defaults import (
    DEFAULT_GNN_EXPLAINER_EPOCHS,
    DEFAULT_GNN_EXPLAINER_LR,
    EDGE_MASK_OBJECT,
)


class GnnExplExplainer:
    """GNNExplainer with edge mask."""

    @staticmethod
    def build_explainer(
        model: nn.Module,
        *,
        epochs: int = DEFAULT_GNN_EXPLAINER_EPOCHS,
        lr: float = DEFAULT_GNN_EXPLAINER_LR,
        **algorithm_kwargs,
    ) -> Explainer:
        return build_pyg_explainer(
            model,
            algorithm=GNNExplainer(epochs=epochs, lr=lr, **algorithm_kwargs),
            node_mask_type=None,
            edge_mask_type=EDGE_MASK_OBJECT,
        )
