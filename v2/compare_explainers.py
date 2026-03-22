#!/usr/bin/env python3
"""
Run v2 explainers on the latest GINE training checkpoint and matching PyG dataset; print fidelity to stdout.

Layout: ``mprov3_gine_explainer_defaults`` and sibling projects share one workspace root
(see ``WORKSPACE_ROOT`` / ``GINE_PROJECT_DIR`` in that package).

Default explainer list is ``PGMEXPL`` only. Override with ``--explainers NAME ...``.

Run from v2/:  uv run python compare_explainers.py [--explainers GNNEXPL ...]
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ExplanationType
from torch_geometric.explain.metric import fidelity

from cli.compare_explainers_cli import explainer_names_from_args, parse_args
from explainers import (
    build_explainers_for_model,
    is_registered,
    resolve_explainer_class,
)
from mprov3_gine_explainer_defaults import (
    DEFAULT_DROPOUT,
    DEFAULT_EDGE_DIM,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_MPRO_SNAPSHOT_DIR_NAME,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_POOL,
    GINE_PROJECT_DIR,
    RESULTS_DATASETS,
    RESULTS_DIR_NAME,
    SplitConfig,
    WORKSPACE_ROOT,
    resolve_training_checkpoint_and_dataset_name,
    validate_explainer_names,
)

# Default when ``--explainers`` is omitted (this script; not the full explainers registry).
DEFAULT_COMPARE_EXPLAINER_NAMES: Tuple[str, ...] = ("PGMEXPL",)

MPRO_SNAPSHOT = WORKSPACE_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
GINE_CODE = GINE_PROJECT_DIR
GINE_RESULTS_ROOT = GINE_PROJECT_DIR / RESULTS_DIR_NAME
GINE_DATASET_BASE = GINE_RESULTS_ROOT / RESULTS_DATASETS

sys.path.insert(0, str(GINE_CODE))

from loaders import create_data_loaders  # noqa: E402
from model import MProGNN  # noqa: E402


class _DefaultBatchAndEdgeAttrWrapper(nn.Module):
    """
    PyG's ``PGMExplainer`` calls ``model(x, edge_index)`` internally without
    ``batch`` / ``edge_attr``; ``MProGNN`` requires ``batch`` and GINE expects
    ``edge_attr`` when ``edge_dim > 0``.
    """

    def __init__(self, base: MProGNN) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_attr is None and edge_index.numel() > 0:
            e = edge_index.size(1)
            edge_attr = torch.zeros(
                e, self.base.edge_dim, dtype=x.dtype, device=x.device
            )
        return self.base(x, edge_index, batch, edge_attr=edge_attr)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        d = torch.device("mps")
    elif torch.cuda.is_available():
        d = torch.device("cuda")
    else:
        d = torch.device("cpu")
    print(f"Using device: {d}", flush=True)
    return d


def _phenomenon_target_from_data(data: Any, device: torch.device) -> torch.Tensor:
    cat = getattr(data, "category", None)
    if cat is None:
        raise ValueError(
            "Phenomenon explainers need a graph label; missing ``data.category``."
        )
    return cat.view(-1).long().to(device)


def _train_pg_explainer_for_graph(
    explainer: Any,
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target: torch.Tensor,
    **model_kw: Any,
) -> None:
    """PGExplainer must be trained on the same graph before ``forward`` (PyG API)."""
    alg = explainer.algorithm
    if not isinstance(alg, PGExplainer):
        return
    dev = x.device
    alg.to(dev)
    alg.reset_parameters()
    was_training = model.training
    model.train()
    graph_index = 0
    for epoch in range(alg.epochs):
        alg.train(
            epoch,
            model,
            x,
            edge_index,
            target=target,
            index=graph_index,
            **model_kw,
        )
    model.train(was_training)
    model.eval()


def _explanation_to_device(
    explanation: Explanation, device: torch.device
) -> Explanation:
    """Move explanation tensors to ``device``; MPS does not support float64."""
    if device.type == "cpu":
        return explanation

    def _float64_to_float32(obj: Any) -> Any:
        if torch.is_tensor(obj) and obj.dtype == torch.float64:
            return obj.float()
        return obj

    out = explanation
    if device.type == "mps":
        out = explanation.apply(_float64_to_float32)
    return out.to(device)


def _ensure_edge_mask(explanation: Explanation) -> Explanation:
    # PyG Data/Explanation: absent mask keys raise AttributeError, not None.
    if getattr(explanation, "edge_mask", None) is not None:
        return explanation
    nm = getattr(explanation, "node_mask", None)
    ei = getattr(explanation, "edge_index", None)
    if nm is None or ei is None:
        return explanation
    out = explanation.clone()
    if nm.dim() > 1:
        nm = nm.float().mean(dim=-1)
    else:
        nm = nm.float()
    row, col = ei[0], ei[1]
    out.edge_mask = (nm[row] + nm[col]) * 0.5
    return out


def _load_model_and_loader() -> Tuple[nn.Module, Any, torch.device]:
    checkpoint_path, dataset_name = resolve_training_checkpoint_and_dataset_name(
        GINE_RESULTS_ROOT
    )
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Dataset: {GINE_DATASET_BASE / dataset_name}", flush=True)
    _, _, test_loader = create_data_loaders(
        GINE_DATASET_BASE,
        MPRO_SNAPSHOT,
        SplitConfig(dataset_name=dataset_name),
        batch_size=1,
    )
    device = _device()
    model = MProGNN(
        in_channels=DEFAULT_IN_CHANNELS,
        hidden_channels=DEFAULT_HIDDEN_CHANNELS,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT,
        out_classes=DEFAULT_OUT_CLASSES,
        pool=DEFAULT_POOL,
        edge_dim=DEFAULT_EDGE_DIM,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )
    model.eval()
    return _DefaultBatchAndEdgeAttrWrapper(model), test_loader, device


def run_compare_explainers(
    explainers: Sequence[Tuple[str, Any]],
    model: nn.Module,
    test_loader: Any,
    device: torch.device,
) -> int:
    """Run each ``(canonical_name, explainer)`` pair and print fidelity stats."""
    for key, explainer in explainers:
        print(f"\n--- {key} ---")
        model.eval()
        fp_sum = fm_sum = 0.0
        n = 0
        n_mask = 0
        gi = 0
        for batch in test_loader:
            data_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            graphs = (
                data_batch.to_data_list()
                if hasattr(data_batch, "to_data_list")
                else [data_batch]
            )
            for data in graphs:
                if hasattr(data, "to"):
                    data = data.to(device)
                x, edge_index = data.x, data.edge_index
                b = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
                edge_attr = getattr(data, "edge_attr", None)
                gid = getattr(data, "pdb_id", f"graph_{gi}")

                call_kw: dict[str, Any] = {"batch": b, "edge_attr": edge_attr}
                if explainer.explanation_type == ExplanationType.phenomenon:
                    call_kw["target"] = _phenomenon_target_from_data(data, x.device)
                    _train_pg_explainer_for_graph(
                        explainer,
                        model,
                        x,
                        edge_index,
                        call_kw["target"],
                        batch=b,
                        edge_attr=edge_attr,
                    )

                raw_exp = explainer(x, edge_index, **call_kw)
                if hasattr(raw_exp, "to") and device.type != "cpu":
                    raw_exp = _explanation_to_device(raw_exp, device)
                exp = _ensure_edge_mask(raw_exp)

                fr = fidelity(explainer, exp)
                if isinstance(fr, (list, tuple)):
                    fp = float(fr[0]) if fr else 0.0
                    fm = float(fr[1]) if len(fr) > 1 else 0.0
                else:
                    fp, fm = float(fr), 0.0

                ok = getattr(exp, "edge_mask", None) is not None
                if ok:
                    n_mask += 1
                fp_sum += fp
                fm_sum += fm
                n += 1
                print(
                    f"  {gid}: fid+={fp:.4f} fid-={fm:.4f}"
                    + ("" if ok else " [no edge_mask]")
                )
                gi += 1

        if n:
            print(f"Mean fidelity (fid+): {fp_sum / n:.4f}")
            print(f"Mean fidelity (fid-): {fm_sum / n:.4f}")
        else:
            print("No graphs in test loader.")
        print(f"Graphs: {n} ({n_mask} with edge_mask).")

    return 0


def main_with_names(explainer_names: Optional[List[str]]) -> int:
    names = list(explainer_names or DEFAULT_COMPARE_EXPLAINER_NAMES)
    if validate_explainer_names(
        names,
        is_registered=is_registered,
        resolve_explainer_class=resolve_explainer_class,
    ):
        return 1

    model, test_loader, device = _load_model_and_loader()
    selected = build_explainers_for_model(model, names)
    return run_compare_explainers(selected, model, test_loader, device)


def main() -> None:
    args = parse_args()
    raise SystemExit(main_with_names(explainer_names_from_args(args)))


if __name__ == "__main__":
    main()
