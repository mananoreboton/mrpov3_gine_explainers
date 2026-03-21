"""
IGNODE — Integrated Gradients on node features (Captum via PyG).
See doc/table_of_explainer_implementations.md.
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer

from explainers.shared.explainer_builder import build_pyg_explainer
from mprov3_gine_explainer_defaults import (
    DEFAULT_IG_INTERNAL_BATCH_SIZE,
    DEFAULT_IG_N_STEPS,
    NODE_MASK_ATTRIBUTES,
)


class IgNodeExplainer:
    """Integrated Gradients with node (feature) masks."""

    @staticmethod
    def build_explainer(
        model: nn.Module,
        *,
        n_steps: int = DEFAULT_IG_N_STEPS,
        internal_batch_size: int | None = DEFAULT_IG_INTERNAL_BATCH_SIZE,
        **extra_algorithm_kwargs,
    ) -> Explainer:
        algo_kw = {"n_steps": n_steps, **extra_algorithm_kwargs}
        if internal_batch_size is not None:
            algo_kw["internal_batch_size"] = internal_batch_size
        return build_pyg_explainer(
            model,
            algorithm=CaptumExplainer("IntegratedGradients", **algo_kw),
            node_mask_type=NODE_MASK_ATTRIBUTES,
            edge_mask_type=None,
        )
