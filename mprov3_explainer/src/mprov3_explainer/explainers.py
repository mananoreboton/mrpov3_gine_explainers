"""
Explainer registry: available explainers and builder callables.
Output paths use <timestamp>/<explainer_name>/ under results/explanations and results/visualizations.

Supports 8 PyG-native explainer configurations plus legacy SubgraphX (DIG).
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
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)

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
    from torch_geometric.explain.algorithm import CaptumExplainer

    return Explainer(
        model=model,
        algorithm=CaptumExplainer("Saliency"),
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
    from torch_geometric.explain.algorithm import CaptumExplainer

    return Explainer(
        model=model,
        algorithm=CaptumExplainer("Saliency"),
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
    from torch_geometric.explain.algorithm import CaptumExplainer

    _LOG.warning(
        "GuidedBackprop requires nn.ReLU modules; functional .relu() calls "
        "in the model will NOT be hooked. Results may be incomplete."
    )
    return Explainer(
        model=model,
        algorithm=CaptumExplainer("GuidedBackprop"),
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
    """Integrated Gradients over node features."""
    from torch_geometric.explain.algorithm import CaptumExplainer

    return Explainer(
        model=model,
        algorithm=CaptumExplainer("IntegratedGradients", n_steps=n_steps),
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
    """Integrated Gradients over edge mask."""
    from torch_geometric.explain.algorithm import CaptumExplainer

    return Explainer(
        model=model,
        algorithm=CaptumExplainer("IntegratedGradients", n_steps=n_steps),
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

    return Explainer(
        model=model,
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


# ---------------------------------------------------------------------------
# SubgraphX (DIG): legacy adapter — kept importable but NOT in AVAILABLE_EXPLAINERS
# ---------------------------------------------------------------------------


class _GINEAdapterForDIG(torch.nn.Module):
    """Wraps GINE so DIG can call model(x, edge_index) or model(data=...)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        data: Optional[Any] = None,
    ) -> torch.Tensor:
        if data is not None:
            if hasattr(data, "to"):
                data = data.to(next(self._model.parameters()).device)
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
            edge_attr = getattr(data, "edge_attr", None)
        if batch is None and x is not None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_attr is None and edge_index is not None:
            edge_dim = getattr(self._model, "edge_dim", 1)
            edge_attr = torch.zeros(
                edge_index.size(1), edge_dim, device=x.device, dtype=torch.float32
            )
        return self._model(x, edge_index, batch, edge_attr)


def _find_closest_from_dicts(results: list[dict], max_nodes: int) -> dict:
    """Pick the best result from DIG serialized dicts (coalition, P)."""
    sorted_results = sorted(results, key=lambda d: len(d["coalition"]))
    best = sorted_results[0]
    for d in sorted_results:
        if len(d["coalition"]) <= max_nodes and d["P"] > best["P"]:
            best = d
    return best


class _SubgraphXWrapper:
    """Legacy wrapper so SubgraphX (DIG) fits the pipeline."""

    def __init__(
        self,
        dig_subgraphx: Any,
        model: torch.nn.Module,
        num_classes: int,
        max_nodes: int = 10,
    ):
        self._subgraphx = dig_subgraphx
        self.model = model
        self._num_classes = num_classes
        self._max_nodes = max_nodes
        self.model_config = ModelConfig(
            mode=ModelMode.multiclass_classification,
            task_level=ModelTaskLevel.graph,
            return_type=ModelReturnType.raw,
        )
        self.explanation_type = ExplanationType.model

    @torch.no_grad()
    def get_prediction(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch = kwargs.get("batch")
        edge_attr = kwargs.get("edge_attr")
        if batch is None and x is not None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, edge_index, batch, edge_attr)

    def get_target(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction.argmax(dim=-1)

    def get_masked_prediction(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if edge_mask is not None:
            set_masks(self.model, edge_mask, edge_index, apply_sigmoid=False)
        out = self.get_prediction(x, edge_index, **kwargs)
        if edge_mask is not None:
            clear_masks(self.model)
        return out

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Explanation:
        from dig.xgraph.method.subgraphx import find_closest_node_result

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        with torch.no_grad():
            logits = self.model(x, edge_index, batch, edge_attr)
        pred_class = int(logits.argmax(dim=-1).squeeze().item())

        _, explanation_results, _ = self._subgraphx(x, edge_index, max_nodes=self._max_nodes)
        if pred_class >= len(explanation_results) or not explanation_results[pred_class]:
            edge_mask = torch.zeros(edge_index.size(1), device=x.device, dtype=torch.float32)
        else:
            results_for_class = explanation_results[pred_class]
            if results_for_class and isinstance(results_for_class[0], dict):
                result_node = _find_closest_from_dicts(results_for_class, self._max_nodes)
                coalition_set = set(result_node["coalition"])
            else:
                result_node = find_closest_node_result(
                    results_for_class, max_nodes=self._max_nodes
                )
                coalition_set = set(result_node.coalition)
            device = edge_index.device
            row_in = torch.tensor(
                [int(i) in coalition_set for i in edge_index[0].tolist()],
                device=device,
                dtype=torch.bool,
            )
            col_in = torch.tensor(
                [int(i) in coalition_set for i in edge_index[1].tolist()],
                device=device,
                dtype=torch.bool,
            )
            edge_mask = (row_in & col_in).float()

        out = Explanation(x=x, edge_index=edge_index, edge_mask=edge_mask)
        out.batch = batch
        if edge_attr is not None:
            out.edge_attr = edge_attr
        out.target = torch.tensor([pred_class], device=x.device, dtype=torch.long)
        out._model_args = ("batch", "edge_attr") if edge_attr is not None else ("batch",)
        return out


def _build_subgraphx(
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_classes: int,
    rollout: int = 10,
    min_atoms: int = 5,
    max_nodes: int = 10,
    sample_num: int = 100,
    verbose: bool = False,
    **kwargs: Any,
) -> _SubgraphXWrapper:
    """Build SubgraphX (DIG) explainer for graph-level explanations (legacy)."""
    from dig.xgraph.method import SubgraphX

    rollout = kwargs.get("subgraphx_rollout", rollout)
    max_nodes = kwargs.get("subgraphx_max_nodes", max_nodes)
    sample_num = kwargs.get("subgraphx_sample_num", sample_num)
    adapter = _GINEAdapterForDIG(model)
    adapter.to(device)
    dig_explainer = SubgraphX(
        adapter,
        num_classes=num_classes,
        device=device,
        explain_graph=True,
        rollout=rollout,
        min_atoms=min_atoms,
        sample_num=sample_num,
        verbose=verbose,
    )
    return _SubgraphXWrapper(dig_explainer, model, num_classes, max_nodes=max_nodes)
