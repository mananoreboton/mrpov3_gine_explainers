"""
Validation logic: evaluate the model on the validation set (classification only).
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from model import MProGNN


@dataclass(frozen=True)
class ValidationMetrics:
    """Validation set metrics (classification accuracy)."""

    accuracy: float


def evaluate_validation(
    model: MProGNN,
    loader: DataLoader,
    device: torch.device,
) -> ValidationMetrics:
    """Compute validation accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            category = batch.category.to(device).squeeze(-1)
            edge_attr = getattr(batch, "edge_attr", None)
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr)
            pred = logits.argmax(dim=1)
            correct += (pred == category).sum().item()
            total += category.size(0)
    accuracy = correct / total if total else 0.0
    return ValidationMetrics(accuracy=accuracy)
