"""
Integrated Gradients on edge masks without passing ``edge_index`` through Captum's
``additional_forward_args`` (IG repeats dim 0, which breaks ``[2, E]`` tensors).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelReturnType


class _IntegratedGradientsEdgeBridge(torch.nn.Module):
    """Forward: batched edge mask values (``B``, ``E``) → logits (``B``, ``C``)."""

    def __init__(
        self,
        gnn: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
    ):
        super().__init__()
        self.gnn = gnn
        self.register_buffer("_x", x.detach())
        self.register_buffer("_edge_index", edge_index.detach())
        self.register_buffer("_batch", batch.detach())
        self._edge_attr = edge_attr.detach() if edge_attr is not None else None

    def forward(self, edge_mask: Tensor) -> Tensor:
        ea = self._edge_attr
        ei = self._edge_index

        def _one(m: Tensor) -> Tensor:
            set_masks(self.gnn, m, ei, apply_sigmoid=False)
            o = self.gnn(self._x, ei, self._batch, ea)
            clear_masks(self.gnn)
            if o.dim() == 2 and o.size(0) == 1:
                o = o.squeeze(0)
            return o

        if edge_mask.dim() == 1:
            return _one(edge_mask)
        if edge_mask.dim() != 2:
            raise ValueError(
                f"Expected edge mask (E,) or (B, E); got {tuple(edge_mask.shape)}"
            )
        parts = [_one(edge_mask[i]) for i in range(edge_mask.size(0))]
        return torch.stack(parts, dim=0)


class IntegratedGradientsEdgeExplainer(ExplainerAlgorithm):
    """IG attributions over edges; graph tensors live in the bridge."""

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
            raise NotImplementedError("index != None not supported for IG edge explainer")
        batch = kwargs.get("batch")
        edge_attr = kwargs.get("edge_attr")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

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

        e = int(ei_loc.size(1))
        try:
            bridge = _IntegratedGradientsEdgeBridge(model, x_loc, ei_loc, batch_loc, ea_loc)
            bridge.train(model.training)
            ig = IntegratedGradients(bridge)
            edge_in = torch.ones(1, e, dtype=torch.float32, device=x_loc.device, requires_grad=True)
            baselines = torch.zeros_like(edge_in)

            if self.model_config.mode == ModelMode.regression:
                t_arg = None
            else:
                if target is None:
                    raise ValueError("target is required for classification IG")
                t_arg = int(target.flatten()[0].item())

            extra = {k: v for k, v in self.ig_kwargs.items() if k != "n_steps"}
            attributions = ig.attribute(
                inputs=edge_in,
                baselines=baselines,
                target=t_arg,
                additional_forward_args=None,
                n_steps=self.n_steps,
                **extra,
            )
            edge_mask = attributions.squeeze(0).to(dtype=torch.float32)
            if use_cpu_ig:
                edge_mask = edge_mask.to(device=dev, dtype=torch.float32)
        finally:
            if use_cpu_ig:
                model.to(dev)

        return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        edge_mask_type = self.explainer_config.edge_mask_type
        if edge_mask_type not in (None, MaskType.object):
            return False
        return_type = self.model_config.return_type
        if (
            self.model_config.mode == ModelMode.binary_classification
            and return_type != ModelReturnType.probs
        ):
            return False
        return True
