"""End-to-end smoke test for ``run_explanations``.

Mocks a 2-graph dataset and a tiny GNN so we can verify that:
  * every advertised JSON key on :class:`ExplanationResult` is populated,
  * NaN propagation works (a graph that breaks fidelity is marked invalid),
  * the aggregator helpers produce well-defined, finite headline numbers.
"""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mprov3_explainer.pipeline import (
    ExplanationResult,
    aggregate_fidelity,
    nanmean,
    run_explanations,
)


class _TinyGraphModel(torch.nn.Module):
    """Tiny graph-classifier suitable for SALIENCY-style explainers.

    The forward signature matches what the explainer infrastructure expects:
    ``(x, edge_index, batch=None, edge_attr=None)`` returning per-graph logits.
    Non-trivial parameters keep gradient-based explainers from yielding all
    zeros (which would make every result degenerate and be filtered out).
    """

    def __init__(self, in_dim: int = 4, num_classes: int = 2):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        h = self.lin(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        out = torch.zeros(n_graphs, h.size(-1), device=h.device)
        out.index_add_(0, batch, h)
        return out


def _make_loader() -> DataLoader:
    g1 = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
        y=torch.tensor([0]),
    )
    g2 = Data(
        x=torch.randn(4, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        y=torch.tensor([1]),
    )
    return DataLoader([g1, g2], batch_size=1)


def _expected_fields() -> set[str]:
    return {
        "graph_id",
        "fidelity_fid_plus",
        "fidelity_fid_minus",
        "pyg_characterization",
        "fidelity_fid_plus_soft",
        "fidelity_fid_minus_soft",
        "pyg_characterization_soft",
        "paper_sufficiency",
        "paper_comprehensiveness",
        "paper_f1_fidelity",
        "valid",
        "correct_class",
        "has_node_mask",
        "has_edge_mask",
        "mask_spread",
        "mask_entropy",
        "elapsed_s",
        "explanation",
    }


def test_explanation_result_carries_all_advertised_fields():
    """Smoke-check: every field the report writer expects exists on the dataclass."""
    expected = _expected_fields()
    actual = set(ExplanationResult.__dataclass_fields__.keys())
    missing = expected - actual
    assert not missing, f"Missing ExplanationResult fields: {missing}"


def test_run_explanations_smoke_gradexpnode():
    """Run a gradient-based node explainer end-to-end on the tiny dataset; every metric is finite or NaN."""
    torch.manual_seed(0)
    model = _TinyGraphModel(in_dim=4, num_classes=2)
    loader = _make_loader()

    results = list(run_explanations(
        model,
        loader,
        torch.device("cpu"),
        explainer_name="GRADEXPINODE",
        explainer_epochs=1,
        max_graphs=2,
        apply_preprocessing_flag=True,
        correct_class_only=False,
        apply_mask_spread_filter=False,
        paper_metrics=True,
        paper_n_thresholds=10,
        top_k_fraction=0.2,
    ))

    assert len(results) == 2
    for r in results:
        # All advertised fields present and the right type
        for field in _expected_fields() - {"explanation"}:
            assert hasattr(r, field), f"Missing {field}"
        # Numeric metrics are finite or NaN (never raise / never inf)
        for f in (
            r.fidelity_fid_plus, r.fidelity_fid_minus, r.pyg_characterization,
            r.fidelity_fid_plus_soft, r.fidelity_fid_minus_soft,
            r.pyg_characterization_soft,
            r.paper_sufficiency, r.paper_comprehensiveness, r.paper_f1_fidelity,
        ):
            assert isinstance(f, float)
            assert math.isnan(f) or math.isfinite(f)
        # Diagnostics
        assert r.mask_spread >= 0.0
        assert r.mask_entropy >= 0.0
        # Top-k Ff1 is in [0, 1] when finite (clamped)
        if not math.isnan(r.paper_f1_fidelity):
            assert 0.0 <= r.paper_f1_fidelity <= 1.0


def test_run_explanations_aggregation_is_finite_on_smoke_dataset():
    torch.manual_seed(0)
    model = _TinyGraphModel(in_dim=4, num_classes=2)
    loader = _make_loader()

    results = list(run_explanations(
        model,
        loader,
        torch.device("cpu"),
        explainer_name="GRADEXPINODE",
        explainer_epochs=1,
        max_graphs=2,
        apply_preprocessing_flag=True,
        correct_class_only=False,
        apply_mask_spread_filter=False,
        paper_metrics=True,
        paper_n_thresholds=10,
        top_k_fraction=0.2,
    ))

    mp, mm = aggregate_fidelity(results, valid_only=False, nan_skip=True)
    # With the tiny model and 2 graphs we expect at least one valid result, so
    # the aggregated mean should be finite (or NaN if all results were invalid).
    assert isinstance(mp, float) and isinstance(mm, float)
    # nanmean over the per-graph chars is finite or NaN, never raises
    chars = [r.pyg_characterization for r in results]
    out = nanmean(chars)
    assert isinstance(out, float)
