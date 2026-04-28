"""Unit tests for ``mprov3_explainer.pipeline`` correctness fixes."""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.explain import Explanation

from mprov3_explainer.pipeline import (
    DEFAULT_TOP_K_FRACTION,
    _binarize_explanation_top_k,
    _clamp_unit,
    _paper_f1_fidelity,
    _paper_metrics_from_edge_mask,
    _paper_metrics_from_masks,
    _paper_sufficiency_and_comprehensiveness,
    _percentile_keep_fractions,
    aggregate_fidelity,
    diagnose_explanation_run,
    nanmean,
)


# ---------------------------------------------------------------------------
# Ff1 clamping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fsuf,Fcom", [
    (-0.5, 0.4),    # Fsuf below 0
    (1.2, 0.5),     # Fsuf above 1
    (0.4, -0.2),    # Fcom below 0
    (0.4, 1.5),     # Fcom above 1
    (-0.5, -0.5),   # both below 0
    (2.0, 2.0),     # both above 1
])
def test_paper_f1_fidelity_in_unit_interval_for_adversarial_inputs(Fsuf, Fcom):
    out = _paper_f1_fidelity(Fsuf, Fcom)
    assert not math.isnan(out)
    assert 0.0 <= out <= 1.0


def test_paper_f1_fidelity_propagates_nan():
    assert math.isnan(_paper_f1_fidelity(float("nan"), 0.5))
    assert math.isnan(_paper_f1_fidelity(0.5, float("nan")))


def test_paper_f1_fidelity_matches_paper_formula():
    # When inputs are already in [0, 1] the result should equal
    # 2 * (1 - Fsuf) * Fcom / ((1 - Fsuf) + Fcom)
    Fsuf, Fcom = 0.3, 0.4
    expected = 2.0 * (1.0 - Fsuf) * Fcom / ((1.0 - Fsuf) + Fcom)
    assert math.isclose(_paper_f1_fidelity(Fsuf, Fcom), expected)


def test_clamp_unit_preserves_nan():
    assert math.isnan(_clamp_unit(float("nan")))
    assert _clamp_unit(-0.1) == 0.0
    assert _clamp_unit(1.5) == 1.0
    assert _clamp_unit(0.5) == 0.5


# ---------------------------------------------------------------------------
# Top-k binarization on Explanation objects
# ---------------------------------------------------------------------------


def test_binarize_explanation_top_k_produces_hard_masks():
    x = torch.eye(5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_mask = torch.tensor([0.1, 0.4, 0.9, 0.2])
    node_mask = torch.tensor([0.05, 0.5, 0.7, 0.3, 0.9])
    expl = Explanation(
        x=x, edge_index=edge_index, edge_mask=edge_mask, node_mask=node_mask,
    )
    out = _binarize_explanation_top_k(expl, k=0.5)
    assert out.edge_mask is not None and out.node_mask is not None
    assert set(torch.unique(out.edge_mask).tolist()).issubset({0.0, 1.0})
    assert set(torch.unique(out.node_mask).tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Percentile sweep symmetry
# ---------------------------------------------------------------------------


class _ConstModel(torch.nn.Module):
    """Tiny GNN-shaped model that returns the same logits for every input."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.bias = torch.nn.Parameter(torch.tensor([0.7, 0.3]), requires_grad=False)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        # One row per graph in the batch
        if batch is None:
            return self.bias.unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        return self.bias.unsqueeze(0).expand(n_graphs, -1)


def test_percentile_keep_fractions_descends_from_almost_one_to_almost_zero():
    fractions = _percentile_keep_fractions(100)
    assert len(fractions) == 99
    assert fractions[0] == pytest.approx(0.99)
    assert fractions[-1] == pytest.approx(0.01)
    assert fractions == sorted(fractions, reverse=True)


def test_node_native_sweep_is_zero_for_constant_model():
    """A model that returns a constant prediction has Fsuf=Fcom=0 because the
    explanation/complement subgraphs do not change the output.

    We use a graph with enough nodes that no keep-fraction in the percentile
    sweep collapses the complement to an empty set (which would short-circuit
    ``comp_prob`` to 0 and bias the sufficiency curve).
    """
    n = 20
    x = torch.eye(n)
    src = torch.arange(n, dtype=torch.long)
    dst = (src + 1) % n
    edge_index = torch.stack([src, dst], dim=0)
    expl = Explanation(x=x, edge_index=edge_index)
    node_mask = torch.linspace(0.0, 1.0, n)
    Fsuf, Fcom = _paper_sufficiency_and_comprehensiveness(
        _ConstModel(),
        expl,
        node_mask=node_mask,
        target_class=0,
        n_thresholds=10,
    )
    assert math.isclose(Fsuf, 0.0, abs_tol=1e-6)
    assert math.isclose(Fcom, 0.0, abs_tol=1e-6)


def test_edge_native_sweep_is_zero_for_constant_model():
    n = 20
    x = torch.eye(n)
    src = torch.arange(n, dtype=torch.long)
    dst = (src + 1) % n
    edge_index = torch.stack([src, dst], dim=0)
    expl = Explanation(x=x, edge_index=edge_index)
    edge_mask = torch.linspace(0.1, 1.0, edge_index.size(1))
    Fsuf, Fcom = _paper_metrics_from_edge_mask(
        _ConstModel(),
        expl,
        edge_mask=edge_mask,
        target_class=0,
        n_thresholds=10,
    )
    assert math.isclose(Fsuf, 0.0, abs_tol=1e-6)
    assert math.isclose(Fcom, 0.0, abs_tol=1e-6)


def test_paper_metrics_dispatches_to_edge_native_for_edge_only_explanation():
    """An explanation with only an edge mask must NOT silently coerce to a node mask."""
    x = torch.eye(4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_mask = torch.tensor([0.1, 0.3, 0.7, 0.9])
    expl = Explanation(x=x, edge_index=edge_index, edge_mask=edge_mask)
    Fsuf, Fcom, Ff1 = _paper_metrics_from_masks(
        _ConstModel(),
        expl,
        target_class=0,
        n_thresholds=20,
    )
    # All values should be finite (edge-native sweep ran successfully)
    assert not math.isnan(Fsuf)
    assert not math.isnan(Fcom)
    assert 0.0 <= Ff1 <= 1.0 or math.isnan(Ff1)


# ---------------------------------------------------------------------------
# NaN-aware aggregation
# ---------------------------------------------------------------------------


def test_nanmean_skips_nan_values():
    assert math.isclose(nanmean([1.0, 2.0, float("nan"), 3.0]), 2.0)
    assert math.isclose(nanmean([float("nan"), 0.5]), 0.5)
    assert math.isnan(nanmean([float("nan"), float("nan")]))
    assert math.isnan(nanmean([]))


def test_nanmean_skips_none():
    assert math.isclose(nanmean([1.0, None, 3.0]), 2.0)


def test_aggregate_fidelity_nan_skip_default():
    """Mock results with mixed NaN/finite fidelities."""
    from mprov3_explainer.pipeline import ExplanationResult

    def make(fp, fm, valid=True):
        return ExplanationResult(
            graph_id="g",
            explanation=Explanation(),
            fidelity_fid_plus=fp,
            fidelity_fid_minus=fm,
            valid=valid,
        )

    rs = [
        make(0.4, 0.1),
        make(float("nan"), float("nan"), valid=False),
        make(0.6, 0.3),
    ]
    mp, mm = aggregate_fidelity(rs, valid_only=False, nan_skip=True)
    assert math.isclose(mp, 0.5)
    assert math.isclose(mm, 0.2)
    # valid_only further filters out the NaN graph
    mp2, mm2 = aggregate_fidelity(rs, valid_only=True, nan_skip=True)
    assert math.isclose(mp2, 0.5)
    assert math.isclose(mm2, 0.2)


def test_aggregate_fidelity_returns_nan_for_empty_input():
    plus, minus = aggregate_fidelity([], valid_only=False)
    assert math.isnan(plus)
    assert math.isnan(minus)


def test_diagnose_explanation_run_marks_all_degenerate_failure():
    from mprov3_explainer.pipeline import ExplanationResult

    results = [
        ExplanationResult(
            graph_id=f"g{i}",
            explanation=Explanation(),
            valid=False,
            mask_spread=0.0,
        )
        for i in range(3)
    ]

    status, note = diagnose_explanation_run(results, mask_spread_tolerance=1e-3)

    assert status == "failed_all_degenerate_masks"
    assert "headline metrics" in note


# ---------------------------------------------------------------------------
# Default top-k fraction
# ---------------------------------------------------------------------------


def test_default_top_k_fraction_is_graphframex_canonical():
    """GraphFramEx (Amara et al., 2022) uses k=0.2 by default."""
    assert DEFAULT_TOP_K_FRACTION == pytest.approx(0.2)
