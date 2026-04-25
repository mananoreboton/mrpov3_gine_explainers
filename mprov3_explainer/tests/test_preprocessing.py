"""Unit tests for ``mprov3_explainer.preprocessing``."""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.explain import Explanation

from mprov3_explainer.preprocessing import (
    CONSTANT_MASK_TOLERANCE,
    apply_preprocessing,
    binarize_top_k,
    normalize_mask,
    rank_normalize_mask,
)


def test_normalize_mask_returns_zeros_for_constant_mask():
    """Spread below the global tolerance must collapse to zeros, not be stretched."""
    mask = torch.full((10,), 0.5)
    out = normalize_mask(mask)
    assert torch.equal(out, torch.zeros(10))


def test_normalize_mask_returns_zeros_for_near_constant_mask():
    """Even tiny but sub-tolerance spreads should produce zeros (avoids the
    pre-fix behaviour where 1e-12 < spread < 1e-3 was stretched to noisy [0,1])."""
    eps = CONSTANT_MASK_TOLERANCE / 10.0
    mask = torch.tensor([0.5, 0.5 + eps, 0.5 - eps, 0.5])
    out = normalize_mask(mask)
    assert torch.equal(out, torch.zeros(4))


def test_normalize_mask_min_max_scaling():
    mask = torch.tensor([0.0, 0.5, 1.0, 2.0])
    out = normalize_mask(mask)
    assert math.isclose(out.min().item(), 0.0)
    assert math.isclose(out.max().item(), 1.0)


def test_apply_preprocessing_takes_abs_of_signed_edge_mask():
    """GuidedBackprop / IG produce signed attributions; .abs() guard fixes ranking."""
    x = torch.eye(4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    # Signed attribution: most-negative entry must be most-important after abs
    edge_mask = torch.tensor([-3.0, 1.0, -1.0, 0.5])
    expl = Explanation(
        x=x,
        edge_index=edge_index,
        edge_mask=edge_mask,
        prediction=torch.tensor([[0.6, 0.4]]),
        target=torch.tensor([0]),
    )
    out = apply_preprocessing(expl, pred_class=0, target_class=0)
    em = out.explanation.edge_mask
    assert em is not None
    assert (em >= 0).all()
    # The originally most-negative entry (idx 0) should be the maximum after abs+normalize
    assert em.argmax().item() == 0


def test_apply_preprocessing_takes_abs_of_signed_node_mask_1d():
    x = torch.eye(3)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_mask = torch.tensor([-2.0, 0.5, 1.0])
    expl = Explanation(
        x=x,
        edge_index=edge_index,
        node_mask=node_mask,
        prediction=torch.tensor([[0.6, 0.4]]),
        target=torch.tensor([0]),
    )
    out = apply_preprocessing(expl, pred_class=0, target_class=0)
    nm = out.explanation.node_mask
    assert nm is not None
    assert (nm >= 0).all()
    assert nm.argmax().item() == 0  # |-2| is largest


@pytest.mark.parametrize("k,n,expected", [
    (0.2, 10, 2),  # ceil(0.2 * 10) = 2
    (0.25, 8, 2),  # ceil(0.25 * 8) = 2
    (0.5, 7, 4),   # ceil(0.5 * 7 + 0.5) = 4 (ceil semantics)
    (1.0, 5, 5),
])
def test_binarize_top_k_keeps_correct_count(k: float, n: int, expected: int):
    """For distinct-valued inputs binarize_top_k keeps ceil-ish(k * N) ones."""
    mask = torch.arange(n, dtype=torch.float32)  # all distinct
    out = binarize_top_k(mask, k)
    assert out.dtype == mask.dtype
    assert out.shape == mask.shape
    assert int(out.sum().item()) == expected
    # The kept entries are the largest values
    kept_idx = torch.nonzero(out, as_tuple=False).reshape(-1)
    assert sorted(kept_idx.tolist()) == sorted(range(n - expected, n))


def test_binarize_top_k_handles_edge_cases():
    mask = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(binarize_top_k(mask, 0.0), torch.zeros(3))
    assert torch.equal(binarize_top_k(mask, 1.0), torch.ones(3))
    assert binarize_top_k(torch.empty(0), 0.5).numel() == 0


def test_binarize_top_k_all_equal_returns_exact_count():
    """When all entries are equal we tie-break and still produce exactly the right count."""
    mask = torch.full((10,), 0.5)
    out = binarize_top_k(mask, 0.3)
    assert int(out.sum().item()) == 4  # ceil(0.3 * 10 + 0.5) = 4


def test_rank_normalize_mask_is_in_unit_interval():
    mask = torch.tensor([10.0, -5.0, 3.0, 100.0, 1.0])
    out = rank_normalize_mask(mask)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    # Order preserved: largest input -> largest output
    assert out.argmax().item() == mask.argmax().item()
    assert out.argmin().item() == mask.argmin().item()


def test_rank_normalize_mask_handles_ties():
    mask = torch.tensor([1.0, 1.0, 2.0])
    out = rank_normalize_mask(mask)
    assert math.isclose(out[0].item(), out[1].item())
    assert out[2].item() > out[0].item()
