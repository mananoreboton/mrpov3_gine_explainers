"""
One-epoch training step: run a single training epoch (classification only) and return mean loss.
Used by train.py (training CLI).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple

from model import MProGNN


def train_one_epoch(
    model: MProGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion_ce: nn.Module,
) -> Tuple[float, float]:
    """Run one training epoch (cross-entropy on category); return (mean loss, train accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        category = batch.category.to(device).squeeze(-1)
        optimizer.zero_grad()
        edge_attr = getattr(batch, "edge_attr", None)
        logits = model(batch.x, batch.edge_index, batch.batch, edge_attr)
        loss = criterion_ce(logits, category)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == category).sum().item()
        total += category.size(0)
    mean_loss = total_loss / len(loader) if len(loader) else 0.0
    train_acc = correct / total if total else 0.0
    return mean_loss, train_acc
