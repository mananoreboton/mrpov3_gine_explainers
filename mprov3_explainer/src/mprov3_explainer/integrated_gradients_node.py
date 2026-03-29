"""
Integrated Gradients on node features without passing graph tensors through Captum's
``additional_forward_args``.

Captum repeats/expands every tensor in ``additional_forward_args`` along dim 0 for IG
steps. PyG ``edge_index`` is ``[2, E]``, so dim 0 is *not* a batch axis; expanding it
corrupts the tensor and breaks message passing. This module wraps the GNN so Captum
only sees node features ``x``; structure is fixed inside the wrapper.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType, ModelMode, ModelReturnType


class _IntegratedGradientsNodeBridge(torch.nn.Module):
    """Forward: ``(B, N, F)`` or ``(N, F)`` node features → ``(B, C)`` or ``(C,)`` logits."""

    def __init__(
        self,
        gnn: torch.nn.Module,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
    ):
        super().__init__()
        self.gnn = gnn
        self.register_buffer("_edge_index", edge_index.detach())
        self.register_buffer("_batch", batch.detach())
        self._edge_attr = edge_attr

    def forward(self, x: Tensor) -> Tensor:
        ea = self._edge_attr
        if x.dim() == 2:
            out = self.gnn(x, self._edge_index, self._batch, ea)
            if out.dim() == 2 and out.size(0) == 1:
                out = out.squeeze(0)
            return out
        if x.dim() != 3:
            raise ValueError(
                f"Expected node features of shape (N, F) or (B, N, F); got {tuple(x.shape)}"
            )
        parts: list[Tensor] = []
        for i in range(x.size(0)):
            o = self.gnn(x[i], self._edge_index, self._batch, ea)
            if o.dim() == 2 and o.size(0) == 1:
                o = o.squeeze(0)
            parts.append(o)
        return torch.stack(parts, dim=0)


class IntegratedGradientsNodeExplainer(ExplainerAlgorithm):
    """IG attributions over node features; graph structure is held in the bridge module."""

    def __init__(self, n_steps: int = 32, **ig_kwargs: Any):
        super().__init__()
        self.n_steps = n_steps
        self.ig_kwargs = ig_kwargs

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[int | Tensor] = None,
        **kwargs: Any,
    ) -> Explanation:
        from captum.attr import IntegratedGradients

        if index is not None:
            raise NotImplementedError("index != None not supported for IG node explainer")
        batch = kwargs.get("batch")
        edge_attr = kwargs.get("edge_attr")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Captum IG uses float64 step tensors; MPS does not support float64 — run IG on CPU.
        dev = x.device
        use_cpu_ig = dev.type == "mps"
        if use_cpu_ig:
            model.to(torch.device("cpu"))
            x_loc = x.detach().cpu()
            ei_loc = edge_index.detach().cpu()
            batch_loc = batch.detach().cpu()
            ea_loc = edge_attr.detach().cpu() if edge_attr is not None else None
        else:
            x_loc = x
            ei_loc = edge_index
            batch_loc = batch
            ea_loc = edge_attr

        try:
            bridge = _IntegratedGradientsNodeBridge(model, ei_loc, batch_loc, ea_loc)
            bridge.train(model.training)
            ig = IntegratedGradients(bridge)

            x_in = x_loc.unsqueeze(0)
            baselines = torch.zeros_like(x_in)

            if self.model_config.mode == ModelMode.regression:
                t_arg = None
            else:
                if target is None:
                    raise ValueError("target is required for classification IG")
                t_arg = int(target.flatten()[0].item())

            extra = {k: v for k, v in self.ig_kwargs.items() if k != "n_steps"}
            attributions = ig.attribute(
                inputs=x_in,
                baselines=baselines,
                target=t_arg,
                additional_forward_args=None,
                n_steps=self.n_steps,
                **extra,
            )
            node_mask = attributions.squeeze(0).to(dtype=torch.float32)
            if use_cpu_ig:
                node_mask = node_mask.to(device=dev, dtype=torch.float32)
        finally:
            if use_cpu_ig:
                model.to(dev)

        return Explanation(node_mask=node_mask)

    def supports(self) -> bool:
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type not in (None, MaskType.attributes):
            return False
        return_type = self.model_config.return_type
        if (
            self.model_config.mode == ModelMode.binary_classification
            and return_type != ModelReturnType.probs
        ):
            return False
        return True
