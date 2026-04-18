"""
PyG's ``to_captum_input`` builds attribution inputs with ``unsqueeze(0)`` (and
similar), which yields **non-leaf** views. Captum's gradient helpers then read
``tensor.grad`` on those inputs and PyTorch emits:

  UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is
  being accessed.

We wrap ``to_captum_input`` and pass **detached clones** with ``requires_grad=True``
so Captum sees leaf tensors with identical values.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm.captum import (
    CaptumHeteroModel,
    CaptumModel,
    convert_captum_output,
    to_captum_input as _pyg_to_captum_input,
)
from torch_geometric.explain.algorithm.captum_explainer import CaptumExplainer
from torch_geometric.explain.config import ModelMode
from torch_geometric.typing import EdgeType, NodeType


def _captum_inputs_as_leaf_tensors(
    inputs: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    return tuple(
        t.detach().clone().requires_grad_(True)
        if isinstance(t, torch.Tensor)
        else t
        for t in inputs
    )


class LeafInputCaptumExplainer(CaptumExplainer):
    """Same as PyG :class:`CaptumExplainer`, but attribution inputs are leaf tensors."""

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        mask_type = self._get_mask_type()

        inputs, add_forward_args = _pyg_to_captum_input(
            x,
            edge_index,
            mask_type,
            *kwargs.values(),
        )
        inputs = _captum_inputs_as_leaf_tensors(inputs)

        if isinstance(x, dict):
            metadata = (list(x.keys()), list(edge_index.keys()))
            captum_model = CaptumHeteroModel(
                model,
                mask_type,
                index,
                metadata,
                self.model_config,
            )
        else:
            metadata = None
            captum_model = CaptumModel(
                model,
                mask_type,
                index,
                self.model_config,
            )

        self.attribution_method_instance = self.attribution_method_class(
            captum_model,
        )

        if self.model_config.mode == ModelMode.regression:
            target = None
        elif index is not None:
            target = target[index]

        attributions = self.attribution_method_instance.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=add_forward_args,
            **self.kwargs,
        )

        node_mask, edge_mask = convert_captum_output(
            attributions,
            mask_type,
            metadata,
        )

        if not isinstance(x, dict):
            return Explanation(node_mask=node_mask, edge_mask=edge_mask)

        explanation = HeteroExplanation()
        explanation.set_value_dict("node_mask", node_mask)
        explanation.set_value_dict("edge_mask", edge_mask)
        return explanation
