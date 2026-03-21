"""
One-epoch training step: run a single training epoch (classification only) and return mean loss.
Used by train.py (training CLI).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import MProGNN


def train_one_epoch(
    model: MProGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion_ce: nn.Module,
) -> float:
    """Run one training epoch (cross-entropy on category); return mean loss."""
    model.train()
    total_loss = 0.0
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
    return total_loss / len(loader) if len(loader) else 0.0
