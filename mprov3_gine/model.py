"""
GNN for MPro ligand potency: graph-level classification (Category: low / medium / high).
Node features: (x, y, z, atomic_number). Uses GINE (Graph Isomorphism Network with Edge features).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool


# Edge feature dim for GINE (bond type scalar)
EDGE_DIM = 1


class MProGNN(nn.Module):
    """
    GINE-based graph neural network for MPro Version 3 data.
    - in_channels: 4 (x, y, z, atomic number)
    - hidden_channels: hidden dimension
    - num_layers: number of GINE layers
    - edge_attr: (E, edge_dim) bond type for each edge
    - out_classes: 3 (Category 0, 1, 2)
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        out_classes: int = 3,
        pool: str = "mean",
        edge_dim: int = EDGE_DIM,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_classes = out_classes
        self.pool = pool
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINEConv(nn1, eps=0.0, train_eps=True, edge_dim=edge_dim))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            nn_k = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(nn_k, eps=0.0, train_eps=True, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (N, in_channels), edge_index: (2, E), batch: (N,), edge_attr: (E, edge_dim) optional.
        Returns logits (N_graphs, out_classes).
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = x.relu()
            x = self.dropout(x)

        if self.pool == "mean":
            h = global_mean_pool(x, batch)
        else:
            h = global_add_pool(x, batch)

        return self.cls_head(h)
