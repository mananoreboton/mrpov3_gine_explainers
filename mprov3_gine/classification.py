"""
Test-set classification: run the model on a loader (e.g. test) and compute metrics.

Provides classify_test(), classify_test_with_predictions(), TestMetrics, and
print_test_classification_report. Categories are reported in original scale (-1, 0, 1);
the model uses class indices (0, 1, 2).
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import ORIGINAL_CATEGORY_FROM_CLASS
from model import MProGNN
from validation import evaluate_validation


@dataclass(frozen=True)
class TestMetrics:
    """Test set metrics (classification accuracy)."""

    accuracy: float


def classify_test(
    model: MProGNN,
    loader: DataLoader,
    device: torch.device,
) -> TestMetrics:
    """Compute test-set classification accuracy."""
    metrics = evaluate_validation(model, loader, device)
    return TestMetrics(accuracy=metrics.accuracy)


def classify_test_with_predictions(
    model: MProGNN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[TestMetrics, List[Tuple[str, int, int]]]:
    """
    Run the model on the loader and return metrics plus per-sample results.

    Returns:
        (TestMetrics, list of (pdb_id, real_category, predicted_category))
        with real_category and predicted_category in original scale (-1, 0, 1).
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            category = batch.category.to(device).squeeze(-1)
            edge_attr = getattr(batch, "edge_attr", None)
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(category.cpu().tolist())

    total = len(all_labels)
    correct = sum(1 for p, r in zip(all_preds, all_labels) if p == r)
    accuracy = correct / total if total else 0.0
    metrics = TestMetrics(accuracy=accuracy)

    # Get pdb_id for each sample in loader order (same as all_preds / all_labels).
    subset = loader.dataset
    pdb_ids: List[str] = []
    for i in range(len(subset)):
        idx = subset.indices[i]
        data = subset.dataset[idx]
        pdb_id = getattr(data, "pdb_id", f"idx_{idx}")
        pdb_ids.append(str(pdb_id))

    # Convert class indices (0, 1, 2) to original categories (-1, 0, 1).
    results: List[Tuple[str, int, int]] = []
    for pdb_id, pred_idx, real_idx in zip(pdb_ids, all_preds, all_labels):
        real_orig = ORIGINAL_CATEGORY_FROM_CLASS.get(real_idx, real_idx)
        pred_orig = ORIGINAL_CATEGORY_FROM_CLASS.get(pred_idx, pred_idx)
        results.append((pdb_id, real_orig, pred_orig))

    return metrics, results


def print_test_classification_report(metrics: TestMetrics) -> None:
    """Print test-set classification accuracy to stdout."""
    print(f"Test accuracy (Category): {metrics.accuracy:.4f}")
