"""
GUIDEDBP — Guided Backpropagation on node features (Captum via PyG).
See doc/table_of_explainer_implementations.md.
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer

from explainers.shared.explainer_builder import build_pyg_explainer
from mprov3_gine_explainer_defaults import NODE_MASK_ATTRIBUTES


class GuidedBpExplainer:
    """Captum GuidedBackprop with node mask type."""

    @staticmethod
    def build_explainer(model: nn.Module, **algorithm_kwargs) -> Explainer:
        return build_pyg_explainer(
            model,
            algorithm=CaptumExplainer("GuidedBackprop", **algorithm_kwargs),
            node_mask_type=NODE_MASK_ATTRIBUTES,
            edge_mask_type=None,
        )
