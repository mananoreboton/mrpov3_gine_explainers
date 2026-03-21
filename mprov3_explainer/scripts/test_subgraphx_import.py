#!/usr/bin/env python3
"""Minimal script to verify SubgraphX (DIG) dependencies: import, model, and one explain call."""
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from dig.xgraph.method import SubgraphX


class TinyGCN(torch.nn.Module):
    """Minimal graph-level GNN: (x, edge_index) or (data=...) -> logits [num_classes]."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x=None, edge_index=None, batch=None, data=None):
        if data is not None:
            x, edge_index = data.x, data.edge_index
            batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x  # [1, num_classes] or [num_classes]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes, num_classes = 8, 2
    dim_node = 4

    # Tiny random graph
    x = torch.randn(num_nodes, dim_node, device=device)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 2], [1, 2, 3, 4, 5, 6, 0, 1]], device=device
    ).long()

    model = TinyGCN(dim_node, 8, num_classes).to(device).eval()

    explainer = SubgraphX(
        model,
        num_classes=num_classes,
        device=device,
        explain_graph=True,
        rollout=3,
        min_atoms=2,
        verbose=False,
    )
    _, explanation_results, related_preds = explainer(x, edge_index, max_nodes=4)

    print("SubgraphX run OK.")
    print(f"  explanation_results: {len(explanation_results)} class(es)")
    print(f"  related_preds: {len(related_preds)} class(es)")


if __name__ == "__main__":
    main()
