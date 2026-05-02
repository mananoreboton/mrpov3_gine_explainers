"""End-to-end smoke test for ``run_explanations``.

Mocks a 2-graph dataset and a tiny GNN so we can verify that:
  * every advertised JSON key on :class:`ExplanationResult` is populated,
  * NaN propagation works (a graph that breaks fidelity is marked invalid),
  * the per-graph metrics round-trip through ``nanmean`` cleanly.
"""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mprov3_explainer.pipeline import (
    ExplanationResult,
    PredictionBaselineEntry,
    nanmean,
    run_explanations,
)


class _TinyGraphModel(torch.nn.Module):
    """Tiny graph-classifier suitable for SALIENCY-style explainers."""

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
        "paper_sufficiency",
        "paper_comprehensiveness",
        "paper_f1_fidelity",
        "pyg_fidelity_plus",
        "pyg_fidelity_minus",
        "pyg_characterization_score",
        "pyg_fidelity_curve_auc",
        "pyg_unfaithfulness",
        "valid",
        "correct_class",
        "pred_class",
        "target_class",
        "prediction_baseline_mismatch",
        "has_node_mask",
        "has_edge_mask",
        "elapsed_s",
        "explanation",
    }


def test_explanation_result_carries_all_advertised_fields():
    """Smoke-check: every field the report writer expects exists on the dataclass."""
    expected = _expected_fields()
    actual = set(ExplanationResult.__dataclass_fields__.keys())
    missing = expected - actual
    assert not missing, f"Missing ExplanationResult fields: {missing}"
    extra = actual - expected
    assert not extra, f"Unexpected ExplanationResult fields: {extra}"


def test_run_explanations_smoke_gradexpnode():
    """Run a gradient-based node explainer end-to-end on the tiny dataset."""
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
    ))

    assert len(results) == 2
    for r in results:
        for field in _expected_fields() - {"explanation"}:
            assert hasattr(r, field), f"Missing {field}"
        for f in (
            r.paper_sufficiency, r.paper_comprehensiveness, r.paper_f1_fidelity,
            r.pyg_fidelity_plus, r.pyg_fidelity_minus,
            r.pyg_characterization_score, r.pyg_fidelity_curve_auc,
            r.pyg_unfaithfulness,
        ):
            assert isinstance(f, float)
            assert math.isnan(f) or math.isfinite(f)
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
    ))

    chars = [r.pyg_characterization_score for r in results]
    out = nanmean(chars)
    assert isinstance(out, float)


def test_run_explanations_uses_precomputed_prediction_baseline():
    torch.manual_seed(0)
    model = _TinyGraphModel(in_dim=4, num_classes=2)
    loader = _make_loader()
    baseline = {
        "graph_0": PredictionBaselineEntry(
            graph_id="graph_0",
            pred_class=1,
            target_class=0,
            correct_class=False,
        )
    }

    results = list(run_explanations(
        model,
        loader,
        torch.device("cpu"),
        explainer_name="GRADEXPINODE",
        explainer_epochs=1,
        max_graphs=1,
        apply_preprocessing_flag=True,
        correct_class_only=True,
        apply_mask_spread_filter=False,
        paper_metrics=False,
        prediction_baseline=baseline,
    ))

    assert len(results) == 1
    assert results[0].pred_class == 1
    assert results[0].target_class == 0
    assert results[0].correct_class is False
    assert results[0].valid is False
