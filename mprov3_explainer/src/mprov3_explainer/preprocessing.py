"""
Preprocessing phase for explanation masks (Longa et al. common representation).
Applied uniformly to any explainer output before metrics: Conversion, Filtering, Normalization.

Supports edge-mask-only, node-mask-only, and mixed explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.explain import Explanation


@dataclass
class PreprocessedExplanation:
    """Result of preprocessing: masks + validity for metric aggregation."""

    explanation: Explanation
    valid: bool
    correct_class: bool


def edge_mask_to_node_mask(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes: Optional[int] = None,
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Convert edge mask to node mask by aggregating incident edge weights per node.
    (Longa et al.: average incident-edge weights per node for comparability.)
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
    device = edge_index.device
    node_mask = torch.zeros(num_nodes, device=device, dtype=edge_mask.dtype)
    if edge_mask.dim() == 0 or edge_mask.numel() == 0:
        return node_mask
    row, col = edge_index[0], edge_index[1]
    if aggregation == "mean":
        node_mask.scatter_add_(0, row, edge_mask)
        node_mask.scatter_add_(0, col, edge_mask)
        degree = torch.zeros(num_nodes, device=device, dtype=torch.long)
        degree.scatter_add_(0, row, torch.ones_like(row, device=device))
        degree.scatter_add_(0, col, torch.ones_like(col, device=device))
        degree = degree.clamp(min=1)
        node_mask = node_mask / degree
    else:  # max
        node_mask.scatter_reduce_(0, row, edge_mask, reduce="amax")
        node_mask.scatter_reduce_(0, col, edge_mask, reduce="amax")
    return node_mask


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Scale mask to [0, 1] per instance (min-max).
    If constant (max == min), return zeros to avoid division by zero.
    """
    if mask.numel() == 0:
        return mask
    min_val = mask.min()
    max_val = mask.max()
    if (max_val - min_val).item() < 1e-12:
        return torch.zeros_like(mask, device=mask.device)
    return (mask - min_val) / (max_val - min_val)


def reduce_node_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """Reduce a [N, F] node-feature mask to [N] per-node scores (mean across features)."""
    if node_mask.dim() == 1:
        return node_mask
    return node_mask.abs().mean(dim=-1)


def _align_node_mask_to_graph(node_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Remove stray batch dimensions (e.g. Captum ``(1, N, F)``) so masks broadcast with
    ``x`` of shape ``(N, F)``. Also fix ``(F, N)`` attributions mistaken for ``(N, F)``.
    """
    m = node_mask
    N = int(x.size(0))
    F = int(x.size(1))
    while m.dim() > 2:
        m = m.squeeze(0)
    while m.dim() >= 2 and m.size(0) == 1:
        m = m.squeeze(0)
    if m.dim() == 2 and m.size(0) == F and m.size(1) == N:
        m = m.transpose(0, 1).contiguous()
    if m.dim() == 2 and m.size(0) == N and m.size(1) == N:
        m = m.abs().mean(dim=1)
    if m.dim() == 2 and m.size(0) == N:
        return m
    if m.dim() == 1 and m.numel() == N:
        return m
    if m.numel() == N:
        return m.view(N)
    if m.numel() == N * F:
        return m.view(N, F)
    return m


def _mask_range(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return (mask.max() - mask.min()).item()


def apply_preprocessing(
    explanation: Explanation,
    *,
    pred_class: int,
    target_class: int,
    correct_class_only: bool = True,
    min_mask_range: float = 1e-3,
    normalize: bool = True,
    convert_edge_to_node: bool = False,
) -> PreprocessedExplanation:
    """
    Apply Conversion (optional), Filtering, Normalization to a raw explanation.
    Supports explanations with edge_mask only, node_mask only, or both.
    """
    edge_mask = getattr(explanation, "edge_mask", None)
    node_mask = getattr(explanation, "node_mask", None)
    edge_index = explanation.edge_index

    if edge_mask is None and node_mask is None:
        return PreprocessedExplanation(
            explanation=explanation, valid=False, correct_class=False,
        )

    correct_class = pred_class == target_class
    valid = True

    if correct_class_only:
        valid = valid and correct_class

    # Detach to float
    if edge_mask is not None:
        edge_mask = edge_mask.detach().float()
    if node_mask is not None:
        node_mask = node_mask.detach().float()
        if explanation.x is not None:
            node_mask = _align_node_mask_to_graph(node_mask, explanation.x)

    # Low-information filter: check whichever mask is available
    if edge_mask is not None and edge_mask.numel() > 0:
        if _mask_range(edge_mask) < min_mask_range:
            valid = False
    elif node_mask is not None and node_mask.numel() > 0:
        flat = reduce_node_mask(node_mask) if node_mask.dim() > 1 else node_mask
        if _mask_range(flat) < min_mask_range:
            valid = False

    # Optional conversion: edge -> node
    if convert_edge_to_node and edge_mask is not None and edge_index is not None:
        num_nodes = explanation.x.size(0) if explanation.x is not None else None
        derived_node_mask = edge_mask_to_node_mask(edge_index, edge_mask, num_nodes=num_nodes)
        if normalize:
            derived_node_mask = normalize_mask(derived_node_mask)
        if node_mask is None:
            node_mask = derived_node_mask

    # Normalization
    if normalize and edge_mask is not None:
        edge_mask = normalize_mask(edge_mask)
    if normalize and node_mask is not None:
        if node_mask.dim() > 1:
            node_mask = normalize_mask(reduce_node_mask(node_mask))
        else:
            node_mask = normalize_mask(node_mask)

    new_explanation = Explanation(
        x=explanation.x,
        edge_index=explanation.edge_index,
        edge_mask=edge_mask,
        node_mask=node_mask,
    )

    return PreprocessedExplanation(
        explanation=new_explanation, valid=valid, correct_class=correct_class,
    )
