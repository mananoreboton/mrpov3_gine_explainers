"""
Explainer registry: available explainers and builder callables.
Output paths use <explainer_name>/ under results/explanations and results/visualizations.

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
    report_paragraph: str
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
    # "PGEXPL" removed — all masks degenerate, no valid results
    "PGMEXPL",
]

_EXPLAINER_SPECS: Dict[str, ExplainerSpec] = {
    "GRADEXPINODE": ExplainerSpec(
        name="GRADEXPINODE",
        builder=_build_gradexp_node,
        report_paragraph=(
            "Mask type: node-feature mask (`node_mask_type=attributes`); no edge mask. "
            "Method: Captum Saliency—gradient of the explained output with respect to node "
            "features (typically absolute gradients). Fast, one backward pass per explanation."
        ),
        produces_node_mask=True,
        produces_edge_mask=False,
    ),
    "GRADEXPLEDGE": ExplainerSpec(
        name="GRADEXPLEDGE",
        builder=_build_gradexp_edge,
        report_paragraph=(
            "Mask type: edge mask (`edge_mask_type=object`); no node-feature mask. "
            "Method: Captum Saliency on PyG’s differentiable edge-mask input, so importance "
            "is the sensitivity of the prediction to each edge when message passing respects "
            "the mask. One backward pass per explanation."
        ),
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "GUIDEDBP": ExplainerSpec(
        name="GUIDEDBP",
        builder=_build_guided_bp,
        report_paragraph=(
            "Mask type: node-feature mask (`attributes`); no edge mask. "
            "Method: Captum Guided Backpropagation—modified ReLU backward (non-negative "
            "gradients) for sharper input attributions than plain gradients. Requires "
            "`torch.nn.ReLU` modules; functional activations may not hook correctly."
        ),
        produces_node_mask=True,
        produces_edge_mask=False,
    ),
    "IGNODE": ExplainerSpec(
        name="IGNODE",
        builder=_build_ig_node,
        report_paragraph=(
            "Mask type: node-feature mask (`attributes`); no edge mask. "
            "Method: Integrated Gradients—path-integral attribution by averaging gradients "
            "along paths from a baseline to the input (cost scales with `n_steps`). "
            "Implemented here via a Captum-safe bridge over node features."
        ),
        produces_node_mask=True,
        produces_edge_mask=False,
    ),
    "IGEDGE": ExplainerSpec(
        name="IGEDGE",
        builder=_build_ig_edge,
        report_paragraph=(
            "Mask type: edge mask (`object`); no node-feature mask. "
            "Method: Integrated Gradients on the learnable edge-mask channel—same "
            "path-integral idea as IG on features, but attributions are per edge and "
            "runtime scales with the number of integration steps."
        ),
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "GNNEXPL": ExplainerSpec(
        name="GNNEXPL",
        builder=_build_gnn_explainer,
        report_paragraph=(
            "Mask type: edge mask only in this configuration (`edge_mask_type=object`); "
            "no node mask. Method: PyG GNNExplainer—per-instance optimization of a soft "
            "edge mask with sparsity and entropy regularization over many epochs. "
            "Explanation is a local optimization problem, not a single-pass gradient map."
        ),
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "PGEXPL": ExplainerSpec(
        name="PGEXPL",
        builder=_build_pg_explainer,
        report_paragraph=(
            "Mask type: edge mask (`object`); phenomenon explanations only—no node-feature "
            "mask. Method: PGExplainer trains a small MLP to predict edge masks; "
            "`algorithm.train(...)` must complete before calling the explainer. "
            "Amortized, parametric edge explanations rather than per-instance mask optimization."
        ),
        needs_training=True,
        phenomenon_only=True,
        produces_node_mask=False,
        produces_edge_mask=True,
    ),
    "PGMEXPL": ExplainerSpec(
        name="PGMEXPL",
        builder=_build_pgm_explainer,
        report_paragraph=(
            "Mask type: node-feature mask (`attributes`); no edge mask (PGMExplainer does "
            "not support edge masks). Method: perturbation plus statistical testing—sample "
            "forward passes under feature perturbations and use chi-square conditional "
            "independence (pgmpy) to mark significant nodes; classification-oriented."
        ),
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


def explainer_report_meta() -> Dict[str, Dict[str, str]]:
    """Mask-type + method blurbs for web reports (e.g. ``report_data.json``)."""
    return {
        name: {"report_paragraph": _EXPLAINER_SPECS[name].report_paragraph}
        for name in AVAILABLE_EXPLAINERS
    }
