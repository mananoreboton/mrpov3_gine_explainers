"""
Explainer registry: available explainers and builder callables.
Output paths use <timestamp>/<explainer_name>/ under results/explanations and results/visualizations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from mprov3_gine_explainer_defaults import DEFAULT_EXPLANATION_TYPE, DEFAULT_MODEL_CONFIG
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)

AVAILABLE_EXPLAINERS: list[str] = ["GNNExplainer", "SubgraphX"]


# ---------------------------------------------------------------------------
# SubgraphX (DIG): model adapter and wrapper
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


def _find_closest_from_dicts(
    results: list[dict], max_nodes: int
) -> dict:
    """Pick the best result from DIG serialized dicts (coalition, P). Same logic as find_closest_node_result."""
    sorted_results = sorted(results, key=lambda d: len(d["coalition"]))
    best = sorted_results[0]
    for d in sorted_results:
        if len(d["coalition"]) <= max_nodes and d["P"] > best["P"]:
            best = d
    return best


class _SubgraphXWrapper:
    """Wrapper so SubgraphX (DIG) fits the pipeline: callable returning Explanation with edge_mask; quacks like PyG Explainer for fidelity()."""

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
            # DIG may return list of dicts (from write_from_MCTSNode_list) instead of MCTSNode objects
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

        # Build Explanation so PyG fidelity(explainer, explanation) works
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
    """Build SubgraphX (DIG) explainer for graph-level explanations."""
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


def _build_gnn_explainer(
    model: torch.nn.Module,
    *,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.01,
    **kwargs: Any,
) -> Explainer:
    """Build PyG Explainer with GNNExplainer for graph-level explanations."""
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type=DEFAULT_EXPLANATION_TYPE,
        edge_mask_type="object",
        model_config=dict(DEFAULT_MODEL_CONFIG),
    )


_EXPLAINER_BUILDERS: Dict[str, Callable[..., Any]] = {
    "GNNExplainer": _build_gnn_explainer,
    "SubgraphX": _build_subgraphx,
}


def get_builder(explainer_name: str):
    """Return the builder callable for the given explainer. Raises ValueError if unknown."""
    if explainer_name not in _EXPLAINER_BUILDERS:
        raise ValueError(
            f"Unknown explainer: {explainer_name}. Available: {AVAILABLE_EXPLAINERS}"
        )
    return _EXPLAINER_BUILDERS[explainer_name]


def validate_explainer(name: str) -> str:
    """Validate and return the explainer name. Raises ValueError if unknown."""
    if name not in _EXPLAINER_BUILDERS:
        raise ValueError(
            f"Unknown explainer: {name}. Available: {AVAILABLE_EXPLAINERS}"
        )
    return name
