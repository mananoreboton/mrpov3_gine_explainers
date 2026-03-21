"""Small helper to construct torch_geometric.explain.Explainer with shared task config."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import ExplainerAlgorithm

from mprov3_gine_explainer_defaults import DEFAULT_EXPLANATION_TYPE, DEFAULT_MODEL_CONFIG


def build_pyg_explainer(
    model: nn.Module,
    *,
    algorithm: ExplainerAlgorithm,
    node_mask_type: Optional[str] = None,
    edge_mask_type: Optional[str] = None,
    explanation_type: str = DEFAULT_EXPLANATION_TYPE,
    model_config: Optional[dict[str, Any]] = None,
    **explainer_kwargs: Any,
) -> Explainer:
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
        cfg.update(model_config)
    return Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type=explanation_type,
        model_config=cfg,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        **explainer_kwargs,
    )
