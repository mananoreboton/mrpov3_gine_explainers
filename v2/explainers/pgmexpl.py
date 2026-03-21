"""
PGMEXPL — PGMExplainer (torch_geometric.contrib.explain).
See doc/table_of_explainer_implementations.md.
"""

from __future__ import annotations

import warnings

import torch.nn as nn
from torch_geometric.explain import Explainer

from explainers.shared.explainer_builder import build_pyg_explainer
from mprov3_gine_explainer_defaults import DEFAULT_PGM_NUM_SAMPLES, NODE_MASK_ATTRIBUTES

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from torch_geometric.contrib.explain import PGMExplainer


class PgmExplExplainer:
    """PGMExplainer with node-level attribution (contrib API)."""

    @staticmethod
    def build_explainer(
        model: nn.Module,
        *,
        num_samples: int = DEFAULT_PGM_NUM_SAMPLES,
        **algorithm_kwargs,
    ) -> Explainer:
        return build_pyg_explainer(
            model,
            algorithm=PGMExplainer(num_samples=num_samples, **algorithm_kwargs),
            node_mask_type=NODE_MASK_ATTRIBUTES,
            edge_mask_type=None,
        )
