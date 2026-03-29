"""
Explainer registry: available explainers and builder callables.
Output paths use <timestamp>/<explainer_name>/ under results/explanations and results/visualizations.

Supports eight PyG-native explainer configurations (see ``AVAILABLE_EXPLAINERS``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
from mprov3_gine_explainer_defaults import (
    DEFAULT_EXPLANATION_TYPE,
    DEFAULT_GNN_EXPLAINER_EPOCHS,
    DEFAULT_GNN_EXPLAINER_LR,
    DEFAULT_IG_N_STEPS,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PG_EXPLAINER_EPOCHS,
    DEFAULT_PG_EXPLAINER_LR,
    DEFAULT_PGM_NUM_SAMPLES,
    EDGE_MASK_OBJECT,
    NODE_MASK_ATTRIBUTES,
    PHENOMENON_EXPLANATION_TYPE,
)
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.algorithm import GNNExplainer, PGExplainer

from mprov3_explainer.captum_leaf_explainer import LeafInputCaptumExplainer

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExplainerSpec: metadata for each registered explainer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplainerSpec:
    """Capability descriptor for a registered explainer."""

    name: str
    builder: Callable[..., Any]
    needs_training: bool = False
    phenomenon_only: bool = False
    produces_node_mask: bool = False
    produces_edge_mask: bool = True


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def _build_gnn_explainer(
    model: torch.nn.Module,
    *,
    device: torch.device,
    epochs: int = DEFAULT_GNN_EXPLAINER_EPOCHS,
    lr: float = DEFAULT_GNN_EXPLAINER_LR,
    **kwargs: Any,
) -> Explainer:
    """PyG GNNExplainer — per-instance mask optimisation, edge masks."""
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        edge_mask_type=EDGE_MASK_OBJECT,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


def _build_gradexp_node(
    model: torch.nn.Module,
    *,
    device: torch.device,
    **kwargs: Any,
) -> Explainer:
    """Saliency (abs gradients) over node features."""
    return Explainer(
        model=model,
        algorithm=LeafInputCaptumExplainer("Saliency"),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        node_mask_type=NODE_MASK_ATTRIBUTES,
        edge_mask_type=None,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


def _build_gradexp_edge(
    model: torch.nn.Module,
    *,
    device: torch.device,
    **kwargs: Any,
) -> Explainer:
    """Saliency (abs gradients) over edge mask."""
    return Explainer(
        model=model,
        algorithm=LeafInputCaptumExplainer("Saliency"),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        node_mask_type=None,
        edge_mask_type=EDGE_MASK_OBJECT,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


def _build_guided_bp(
    model: torch.nn.Module,
    *,
    device: torch.device,
    **kwargs: Any,
) -> Explainer:
    """GuidedBackprop over node features (Captum hook-based)."""
    _LOG.warning(
        "GuidedBackprop requires nn.ReLU modules; functional .relu() calls "
        "in the model will NOT be hooked. Results may be incomplete."
    )
    return Explainer(
        model=model,
        algorithm=LeafInputCaptumExplainer("GuidedBackprop"),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        node_mask_type=NODE_MASK_ATTRIBUTES,
        edge_mask_type=None,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


def _build_ig_node(
    model: torch.nn.Module,
    *,
    device: torch.device,
    n_steps: int = DEFAULT_IG_N_STEPS,
    **kwargs: Any,
) -> Explainer:
    """Integrated Gradients over node features (Captum-safe bridge; see integrated_gradients_node)."""
    from mprov3_explainer.integrated_gradients_node import IntegratedGradientsNodeExplainer

    return Explainer(
        model=model,
        algorithm=IntegratedGradientsNodeExplainer(n_steps=n_steps),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        node_mask_type=NODE_MASK_ATTRIBUTES,
        edge_mask_type=None,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


def _build_ig_edge(
    model: torch.nn.Module,
    *,
    device: torch.device,
    n_steps: int = DEFAULT_IG_N_STEPS,
    **kwargs: Any,
) -> Explainer:
    """Integrated Gradients over edge mask (Captum-safe bridge; see integrated_gradients_edge)."""
    from mprov3_explainer.integrated_gradients_edge import IntegratedGradientsEdgeExplainer

    return Explainer(
        model=model,
        algorithm=IntegratedGradientsEdgeExplainer(n_steps=n_steps),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        node_mask_type=None,
        edge_mask_type=EDGE_MASK_OBJECT,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


def _build_pg_explainer(
    model: torch.nn.Module,
    *,
    device: torch.device,
    epochs: int = DEFAULT_PG_EXPLAINER_EPOCHS,
    lr: float = DEFAULT_PG_EXPLAINER_LR,
    **kwargs: Any,
) -> Explainer:
    """PGExplainer — parametric edge-mask generator (phenomenon-only, requires training)."""
    return Explainer(
        model=model,
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type=PHENOMENON_EXPLANATION_TYPE,
        node_mask_type=None,
        edge_mask_type=EDGE_MASK_OBJECT,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


def _build_pgm_explainer(
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_samples: int = DEFAULT_PGM_NUM_SAMPLES,
    **kwargs: Any,
) -> Explainer:
    """PGMExplainer — perturbation + chi-square statistical test, node masks only."""
    from torch_geometric.contrib.explain import PGMExplainer

    class _DefaultBatchAndEdgeAttrWrapper(torch.nn.Module):
        """Make `batch`/`edge_attr` optional for explainers that omit them.

        PGMExplainer may call `model(x, edge_index, ...)` without supplying `batch`
        (and sometimes `edge_attr`). Our base model (`MProGNN`) requires `batch`
        and expects `edge_attr` when `edge_dim > 0`.
        """

        def __init__(self, base: torch.nn.Module):
            super().__init__()
            self.base = base

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None,
            **model_kwargs: Any,
        ) -> torch.Tensor:
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            if edge_attr is None and edge_index is not None and edge_index.numel() > 0:
                edge_dim = getattr(self.base, "edge_dim", 1)
                edge_attr = torch.zeros(
                    edge_index.size(1),
                    edge_dim,
                    dtype=x.dtype,
                    device=x.device,
                )

            return self.base(x, edge_index, batch, edge_attr=edge_attr, **model_kwargs)

    return Explainer(
        model=_DefaultBatchAndEdgeAttrWrapper(model),
        algorithm=PGMExplainer(num_samples=num_samples),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        node_mask_type=NODE_MASK_ATTRIBUTES,
        edge_mask_type=None,
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AVAILABLE_EXPLAINERS: list[str] = [
    "GRADEXPINODE",
    "GRADEXPLEDGE",
    "GUIDEDBP",
    "IGNODE",
    "IGEDGE",
    "GNNEXPL",
    "PGEXPL",
    "PGMEXPL",
]

_EXPLAINER_SPECS: Dict[str, ExplainerSpec] = {
    "GRADEXPINODE": ExplainerSpec(
        name="GRADEXPINODE",
        builder=_build_gradexp_node,
        produces_node_mask=True,
        produces_edge_mask=False,
    ),
    "GRADEXPLEDGE": ExplainerSpec(
        name="GRADEXPLEDGE",
        builder=_build_gradexp_edge,
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "GUIDEDBP": ExplainerSpec(
        name="GUIDEDBP",
        builder=_build_guided_bp,
        produces_node_mask=True,
        produces_edge_mask=False,
    ),
    "IGNODE": ExplainerSpec(
        name="IGNODE",
        builder=_build_ig_node,
        produces_node_mask=True,
        produces_edge_mask=False,
    ),
    "IGEDGE": ExplainerSpec(
        name="IGEDGE",
        builder=_build_ig_edge,
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "GNNEXPL": ExplainerSpec(
        name="GNNEXPL",
        builder=_build_gnn_explainer,
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "PGEXPL": ExplainerSpec(
        name="PGEXPL",
        builder=_build_pg_explainer,
        needs_training=True,
        phenomenon_only=True,
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "PGMEXPL": ExplainerSpec(
        name="PGMEXPL",
        builder=_build_pgm_explainer,
        produces_node_mask=True,
        produces_edge_mask=False,
    ),
}


def get_spec(explainer_name: str) -> ExplainerSpec:
    """Return the ExplainerSpec for the given name. Raises ValueError if unknown."""
    if explainer_name not in _EXPLAINER_SPECS:
        raise ValueError(
            f"Unknown explainer: {explainer_name}. Available: {AVAILABLE_EXPLAINERS}"
        )
    return _EXPLAINER_SPECS[explainer_name]


def get_builder(explainer_name: str) -> Callable[..., Any]:
    """Return the builder callable for the given explainer. Raises ValueError if unknown."""
    return get_spec(explainer_name).builder


def validate_explainer(name: str) -> str:
    """Validate and return the explainer name. Raises ValueError if unknown."""
    if name not in _EXPLAINER_SPECS:
        raise ValueError(
            f"Unknown explainer: {name}. Available: {AVAILABLE_EXPLAINERS}"
        )
    return name
