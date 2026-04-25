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


CONSTANT_MASK_TOLERANCE = 1e-3
"""Spread (max - min) below which a mask is considered constant/degenerate.

Aligned with the default ``mask_spread_tolerance`` used by :func:`apply_preprocessing`,
so a mask that the spread filter would discard is also treated as constant by
:func:`normalize_mask` (returning zeros) instead of being stretched to a noisy
``[0, 1]`` range and silently fed downstream.
"""


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Scale mask to [0, 1] per instance (min-max).

    If the spread (``max - min``) is below :data:`CONSTANT_MASK_TOLERANCE`, the
    mask is considered degenerate and an all-zeros tensor is returned instead of
    a numerically noisy normalization. Aligned with the global spread filter
    (:func:`apply_preprocessing`'s ``mask_spread_tolerance``).
    """
    if mask.numel() == 0:
        return mask
    min_val = mask.min()
    max_val = mask.max()
    if (max_val - min_val).item() < CONSTANT_MASK_TOLERANCE:
        return torch.zeros_like(mask, device=mask.device)
    return (mask - min_val) / (max_val - min_val)


def reduce_node_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """Reduce a [N, F] node-feature mask to [N] per-node scores (mean across features)."""
    if node_mask.dim() == 1:
        return node_mask
    return node_mask.abs().mean(dim=-1)


def _mask_weight_spread(mask: torch.Tensor) -> float:
    """max − min; 0 if empty (treated as degenerate by callers)."""
    if mask is None or mask.numel() == 0:
        return 0.0
    return float(mask.max().item() - mask.min().item())


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


def apply_preprocessing(
    explanation: Explanation,
    *,
    pred_class: int,
    target_class: int,
    # When True, explanations for misclassified graphs are marked invalid (metrics / aggregation only over correct).
    correct_class_only: bool = True,
    normalize: bool = True,
    convert_edge_to_node: bool = False,
    apply_mask_spread_filter: bool = True,
    mask_spread_tolerance: float = 1e-3,
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

    # Detach to float; take absolute values for signed attributions so that
    # ranking and thresholding reflect *importance magnitude* rather than sign
    # (Captum Saliency already returns abs; GuidedBP / IntegratedGradients do
    # not). 2-D node masks already get .abs().mean(...) via reduce_node_mask.
    if edge_mask is not None:
        edge_mask = edge_mask.detach().float().abs()
    if node_mask is not None:
        node_mask = node_mask.detach().float()
        if explanation.x is not None:
            node_mask = _align_node_mask_to_graph(node_mask, explanation.x)
        if node_mask.dim() == 1:
            node_mask = node_mask.abs()

    if apply_mask_spread_filter:
        tol = float(mask_spread_tolerance)
        if edge_mask is not None and _mask_weight_spread(edge_mask) < tol:
            valid = False
        if node_mask is not None:
            nm1 = reduce_node_mask(node_mask) if node_mask.dim() > 1 else node_mask
            if _mask_weight_spread(nm1) < tol:
                valid = False

    # Optional conversion: edge -> node (raw aggregation; normalization follows below)
    if convert_edge_to_node and edge_mask is not None and edge_index is not None:
        num_nodes = explanation.x.size(0) if explanation.x is not None else None
        derived_node_mask = edge_mask_to_node_mask(edge_index, edge_mask, num_nodes=num_nodes)
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


# ---------------------------------------------------------------------------
# Binarization / rank-based helpers (used by top-k fidelity, percentile sweeps)
# ---------------------------------------------------------------------------


def binarize_top_k(mask: torch.Tensor, k: float) -> torch.Tensor:
    """Return a ``{0, 1}`` mask of the same shape keeping the top ``k`` fraction by value.

    Used by GraphFramEx-style top-k fidelity (see Amara et al., *GraphFramEx*, 2022)
    where soft attribution scores are binarized before computing fidelity, so that
    the explanation/complement subgraphs are well-defined sets rather than rescalings.

    Args:
        mask: tensor of attribution scores (any shape).
        k: fraction in ``(0, 1]`` of entries to keep (rounded up). ``k == 0`` returns
            an all-zero mask; ``k >= 1`` returns an all-one mask.

    Returns:
        A tensor of the same shape and dtype as ``mask`` containing ``0`` and ``1``.
        Empty inputs are returned unchanged.
    """
    if mask.numel() == 0:
        return mask
    k = float(k)
    if k <= 0.0:
        return torch.zeros_like(mask)
    if k >= 1.0:
        return torch.ones_like(mask)

    flat = mask.reshape(-1)
    n = flat.numel()
    n_keep = max(1, int(round(k * n + 0.5)))  # ceil semantics
    n_keep = min(n_keep, n)
    threshold = torch.topk(flat, n_keep, largest=True, sorted=False).values.min()
    out = (mask >= threshold).to(mask.dtype)
    # Ensure exactly n_keep ones by tie-breaking via topk indices when ties exist.
    if int(out.reshape(-1).sum().item()) != n_keep:
        idx = torch.topk(flat, n_keep, largest=True, sorted=False).indices
        out = torch.zeros_like(flat)
        out[idx] = 1.0
        out = out.reshape(mask.shape).to(mask.dtype)
    return out


def rank_normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Map a 1-D mask to ``[0, 1]`` by rank rather than min-max.

    More robust than :func:`normalize_mask` when a single attribution dominates the
    range. Equal values receive the same rank (averaged ties).

    Args:
        mask: 1-D tensor of attribution scores.

    Returns:
        A 1-D tensor in ``[0, 1]`` whose ordering matches ``mask``.
    """
    if mask.numel() == 0:
        return mask
    flat = mask.reshape(-1).float()
    n = flat.numel()
    if n == 1:
        return torch.zeros_like(flat).reshape(mask.shape)
    sorted_vals, sorted_idx = torch.sort(flat)
    ranks = torch.empty_like(flat)
    ranks[sorted_idx] = torch.arange(n, device=flat.device, dtype=flat.dtype)
    # Average rank for ties
    unique_vals, inverse = torch.unique(flat, return_inverse=True)
    if unique_vals.numel() != n:
        avg_ranks = torch.zeros_like(unique_vals)
        counts = torch.zeros_like(unique_vals)
        avg_ranks.scatter_add_(0, inverse, ranks)
        counts.scatter_add_(0, inverse, torch.ones_like(ranks))
        avg_ranks = avg_ranks / counts.clamp(min=1)
        ranks = avg_ranks[inverse]
    return (ranks / float(n - 1)).reshape(mask.shape)
