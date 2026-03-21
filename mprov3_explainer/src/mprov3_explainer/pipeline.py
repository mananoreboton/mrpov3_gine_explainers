"""
Explainer-agnostic pipeline: generate graph-level explanations and compute metrics.
Follows Longa et al. common representation: (1) masks generation, (2) preprocessing
(Conversion, Filtering, Normalization), (3) metrics on preprocessed masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
from torch_geometric.explain import Explanation
from torch_geometric.explain.metric import fidelity, groundtruth_metrics

from mprov3_explainer.explainers import get_builder
from mprov3_explainer.preprocessing import PreprocessedExplanation, apply_preprocessing


@dataclass
class ExplanationResult:
    """Result of explaining one graph."""

    graph_id: str
    explanation: Explanation  # Preprocessed explanation (used for metrics and saved masks)
    fidelity_fid_plus: float
    fidelity_fid_minus: float
    auroc: Optional[float] = None  # When ground-truth mask is provided
    valid: bool = True  # False if excluded by preprocessing filters (correct-class, low-info)
    correct_class: bool = True  # True if pred_class == target_class


def _single_graph_inputs(data: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extract x, edge_index, batch (single graph), edge_attr for one Data; move to device."""
    if hasattr(data, "to"):
        data = data.to(device)
    x = data.x
    edge_index = data.edge_index
    # Single graph: batch index all zeros
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
    edge_attr = getattr(data, "edge_attr", None)
    return x, edge_index, batch, edge_attr


def _get_target_class(data: Any) -> Optional[int]:
    """Extract target class from data if present (e.g. data.category)."""
    c = getattr(data, "category", None)
    if c is None:
        return None
    if hasattr(c, "squeeze"):
        c = c.squeeze()
    if hasattr(c, "item"):
        return int(c.item())
    return int(c)


def run_explanations(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    explainer_name: str,
    explainer_epochs: int = 200,
    max_graphs: Optional[int] = None,
    get_target_mask: Optional[Callable[..., Optional[torch.Tensor]]] = None,
    get_graph_id: Optional[Callable[..., str]] = None,
    apply_preprocessing_flag: bool = True,
    correct_class_only: bool = True,
    min_mask_range: float = 1e-3,
    **explainer_kwargs: Any,
) -> Iterator[ExplanationResult]:
    """
    Generate graph-level explanations, optionally preprocess (Conversion, Filtering, Normalization),
    then compute fidelity and plausibility on preprocessed masks.
    Loader yields (data_batch, pIC50, category); we iterate each graph in the batch.
    get_graph_id(data, index) -> str; get_target_mask(data, index) -> Optional[Tensor] for AUROC.
    explainer_kwargs (incl. device, num_classes) are passed to the builder.
    """
    model.eval()
    builder = get_builder(explainer_name)
    explainer = builder(
        model,
        device=device,
        epochs=explainer_epochs,
        lr=explainer_kwargs.get("lr", 0.01),
        **explainer_kwargs,
    )

    graph_index = 0
    for batch in loader:
        if max_graphs is not None and graph_index >= max_graphs:
            break
        data_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
        if hasattr(data_batch, "to_data_list"):
            graph_list = data_batch.to_data_list()
        else:
            graph_list = [data_batch]

        for data in graph_list:
            if max_graphs is not None and graph_index >= max_graphs:
                break
            x, edge_index, batch_tensor, edge_attr = _single_graph_inputs(data, device)
            graph_id = get_graph_id(data, graph_index) if get_graph_id else f"graph_{graph_index}"

            # Prediction for preprocessing (correct-class filter)
            with torch.no_grad():
                logits = model(x, edge_index, batch_tensor, edge_attr)
            pred_class = int(logits.argmax(dim=-1).squeeze().item())
            target_class = _get_target_class(data)
            if target_class is None:
                target_class = pred_class  # No label: treat as correct for filtering

            # Phase 1: Masks generation (raw explanation)
            raw_explanation = explainer(
                x,
                edge_index,
                batch=batch_tensor,
                edge_attr=edge_attr,
            )
            if hasattr(raw_explanation, "to") and device.type != "cpu":
                raw_explanation = raw_explanation.to(device)

            # Phase 2: Preprocessing (Conversion, Filtering, Normalization)
            if apply_preprocessing_flag:
                preproc = apply_preprocessing(
                    raw_explanation,
                    pred_class=pred_class,
                    target_class=target_class,
                    correct_class_only=correct_class_only,
                    min_mask_range=min_mask_range,
                    normalize=True,
                    convert_edge_to_node=False,
                )
                # Keep raw explanation container so PyG fidelity gets _model_args; only overwrite masks
                explanation = raw_explanation.clone()
                explanation.edge_mask = preproc.explanation.edge_mask
                if getattr(preproc.explanation, "node_mask", None) is not None:
                    explanation.node_mask = preproc.explanation.node_mask
                valid = preproc.valid
                correct_class = preproc.correct_class
            else:
                explanation = raw_explanation
                valid = True
                correct_class = target_class == pred_class if target_class is not None else True

            # Phase 3: Metrics on preprocessed explanation
            fid_result = fidelity(explainer, explanation)
            if isinstance(fid_result, (list, tuple)):
                fid_plus = float(fid_result[0]) if len(fid_result) > 0 else 0.0
                fid_minus = float(fid_result[1]) if len(fid_result) > 1 else 0.0
            else:
                fid_plus = float(fid_result)
                fid_minus = 0.0

            auroc_val: Optional[float] = None
            if get_target_mask is not None and explanation.edge_mask is not None:
                target_mask = get_target_mask(data, graph_index)
                if target_mask is not None:
                    try:
                        if hasattr(target_mask, "to"):
                            target_mask = target_mask.to(device)
                        auroc_val = groundtruth_metrics(
                            explanation.edge_mask,
                            target_mask,
                            metrics="auroc",
                        )
                        if hasattr(auroc_val, "item"):
                            auroc_val = float(auroc_val.item())
                        else:
                            auroc_val = float(auroc_val)
                    except Exception:
                        pass

            yield ExplanationResult(
                graph_id=graph_id,
                explanation=explanation,
                fidelity_fid_plus=fid_plus,
                fidelity_fid_minus=fid_minus,
                auroc=auroc_val,
                valid=valid,
                correct_class=correct_class,
            )
            graph_index += 1


def aggregate_fidelity(
    results: list[ExplanationResult],
    valid_only: bool = False,
) -> tuple[float, float]:
    """Return (mean fid+, mean fid-) over results. If valid_only=True, average only over valid instances."""
    if valid_only:
        results = [r for r in results if r.valid]
    if not results:
        return 0.0, 0.0
    n = len(results)
    mean_plus = sum(r.fidelity_fid_plus for r in results) / n
    mean_minus = sum(r.fidelity_fid_minus for r in results) / n
    return mean_plus, mean_minus
