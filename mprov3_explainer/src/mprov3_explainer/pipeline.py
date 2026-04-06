"""
Explainer-agnostic pipeline: generate graph-level explanations and compute metrics.
Follows Longa et al. common representation: (1) masks generation, (2) preprocessing
(Conversion, Filtering, Normalization), (3) metrics on preprocessed masks.

Supports edge-mask, node-mask, and mixed explainers.  PGExplainer offline training
is handled via ``train_explainer()``.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
from torch_geometric.explain import Explanation
from torch_geometric.explain.metric import characterization_score, fidelity
from torch_geometric.utils import subgraph

from mprov3_explainer.explainers import ExplainerSpec, get_spec
from mprov3_explainer.preprocessing import (
    _align_node_mask_to_graph,
    apply_preprocessing,
    edge_mask_to_node_mask,
    normalize_mask,
    reduce_node_mask,
)

_LOG = logging.getLogger(__name__)


def _fidelity_explanation(explanation: Explanation) -> Explanation:
    """Clone with a node mask shaped for ``node_mask * x`` (PyG fidelity).

    A 1D ``(N,)`` mask does not broadcast with ``(N, F)`` node features in PyTorch; use ``(N, 1)``.
    """
    out = explanation.clone()
    x = getattr(out, "x", None)
    nm = getattr(out, "node_mask", None)
    if x is None or nm is None:
        return out
    nm = _align_node_mask_to_graph(nm.detach().float(), x)
    if nm.dim() > 1 and nm.shape != x.shape:
        nm = reduce_node_mask(nm)
    if nm.dim() == 1:
        nm = nm.unsqueeze(1)
    out.node_mask = nm.to(dtype=x.dtype, device=x.device)
    return out


@dataclass
class ExplanationResult:
    """Result of explaining one graph."""

    graph_id: str
    explanation: Explanation
    fidelity_fid_plus: float
    fidelity_fid_minus: float
    paper_sufficiency: float = 0.0
    paper_comprehensiveness: float = 0.0
    paper_f1_fidelity: float = 0.0
    pyg_characterization: float = 0.0
    valid: bool = True
    correct_class: bool = True
    has_node_mask: bool = False
    has_edge_mask: bool = False
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_graph_inputs(
    data: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extract x, edge_index, batch (single graph), edge_attr for one Data; move to device."""
    if hasattr(data, "to"):
        data = data.to(device)
    x = data.x
    edge_index = data.edge_index
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


def _predict_target_proba(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    target_class: int,
) -> float:
    with torch.no_grad():
        logits = model(x, edge_index, batch, edge_attr)
        probs = torch.softmax(logits, dim=-1)
        return float(probs.squeeze(0)[target_class].item())


def _paper_normalized_node_mask_from_explanation(
    explanation: Explanation,
) -> Optional[torch.Tensor]:
    """
    Build a single [N] node mask in [0, 1] for Longa-style threshold metrics.
    Converts edge_mask → node (mean over incident edges) when node_mask is absent.
    """
    x = explanation.x
    edge_index = explanation.edge_index
    if x is None or edge_index is None:
        return None

    node_mask = explanation.get("node_mask")
    edge_mask = explanation.get("edge_mask")

    if node_mask is None and edge_mask is not None:
        node_mask = edge_mask_to_node_mask(edge_index, edge_mask, num_nodes=int(x.size(0)))
        node_mask = normalize_mask(node_mask)
    elif node_mask is not None:
        node_mask = _align_node_mask_to_graph(node_mask.detach().float(), x)
        node_mask = reduce_node_mask(node_mask) if node_mask.dim() > 1 else node_mask
        node_mask = normalize_mask(node_mask)
    else:
        return None
    return node_mask


def _paper_sufficiency_and_comprehensiveness(
    model: torch.nn.Module,
    explanation: Explanation,
    *,
    node_mask: torch.Tensor,
    target_class: int,
    n_thresholds: int,
) -> tuple[float, float]:
    """
    Threshold sweep: sufficiency from full_prob − subgraph(explanation-only) prob;
    comprehensiveness from full_prob − complement-subgraph prob.
    """
    x = explanation.x
    edge_index = explanation.edge_index
    if x is None or edge_index is None:
        return 0.0, 0.0

    batch = explanation.get("batch")
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    edge_attr = explanation.get("edge_attr")

    full_prob = _predict_target_proba(
        model,
        x=x,
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        target_class=target_class,
    )

    Nt = int(n_thresholds)
    if Nt <= 1:
        return 0.0, 0.0

    suf_sum = 0.0
    com_sum = 0.0
    denom = float(Nt - 1)

    N = int(x.size(0))
    all_nodes = torch.arange(N, device=x.device)

    for k in range(1, Nt):
        t = float(k) / float(Nt)
        keep = node_mask > t

        if bool(keep.any()):
            keep_nodes = all_nodes[keep]
            sub_ei, sub_edge_attr, _ = subgraph(
                keep_nodes,
                edge_index,
                edge_attr,
                relabel_nodes=True,
                num_nodes=N,
                return_edge_mask=True,
            )
            sub_x = x[keep_nodes]
            sub_batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=x.device)
            exp_prob = _predict_target_proba(
                model,
                x=sub_x,
                edge_index=sub_ei,
                batch=sub_batch,
                edge_attr=sub_edge_attr,
                target_class=target_class,
            )
        else:
            exp_prob = 0.0

        comp_keep = ~keep
        if bool(comp_keep.any()):
            comp_nodes = all_nodes[comp_keep]
            comp_ei, comp_edge_attr, _ = subgraph(
                comp_nodes,
                edge_index,
                edge_attr,
                relabel_nodes=True,
                num_nodes=N,
                return_edge_mask=True,
            )
            comp_x = x[comp_nodes]
            comp_batch = torch.zeros(comp_x.size(0), dtype=torch.long, device=x.device)
            comp_prob = _predict_target_proba(
                model,
                x=comp_x,
                edge_index=comp_ei,
                batch=comp_batch,
                edge_attr=comp_edge_attr,
                target_class=target_class,
            )
        else:
            comp_prob = 0.0

        suf_sum += (full_prob - exp_prob)
        com_sum += (full_prob - comp_prob)

    return float(suf_sum / denom), float(com_sum / denom)


def _paper_f1_fidelity(Fsuf: float, Fcom: float) -> float:
    """Paper F1-fidelity from sufficiency and comprehensiveness (Longa et al.)."""
    num = 2.0 * (1.0 - Fsuf) * Fcom
    den = (1.0 - Fsuf) + Fcom
    return (num / den) if abs(den) > 1e-12 else 0.0


def _paper_metrics_from_masks(
    model: torch.nn.Module,
    explanation: Explanation,
    *,
    target_class: int,
    n_thresholds: int = 100,
) -> tuple[float, float, float]:
    """
    Longa et al. (2025) graph classification fidelity metrics:
    Fsuf, Fcom, and F1-fidelity computed via threshold sweeping over hard masks.
    """
    node_mask = _paper_normalized_node_mask_from_explanation(explanation)
    if node_mask is None:
        return 0.0, 0.0, 0.0
    Fsuf, Fcom = _paper_sufficiency_and_comprehensiveness(
        model,
        explanation,
        node_mask=node_mask,
        target_class=target_class,
        n_thresholds=n_thresholds,
    )
    Ff1 = _paper_f1_fidelity(Fsuf, Fcom)
    return Fsuf, Fcom, Ff1


def _coerce_explanation_device_dtype(raw_explanation: Any, device: torch.device) -> Any:
    """Move explanation tensors to *device*; on MPS, downcast float64 masks to float32."""
    if not hasattr(raw_explanation, "to") or device.type == "cpu":
        return raw_explanation
    if device.type == "mps" and hasattr(raw_explanation, "apply"):
        def _float64_to_float32(obj: Any) -> Any:
            if torch.is_tensor(obj) and obj.dtype == torch.float64:
                return obj.float()
            return obj

        raw_explanation = raw_explanation.apply(_float64_to_float32)
    return raw_explanation.to(device)


def _forward_raw_explanation(
    explainer: Any,
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch_tensor: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    device: torch.device,
    spec: ExplainerSpec,
    target_class: int,
) -> tuple[Any, float]:
    """Run the explainer forward; return (raw explanation, wall seconds)."""
    call_kwargs: dict[str, Any] = dict(batch=batch_tensor, edge_attr=edge_attr)
    if spec.phenomenon_only:
        call_kwargs["target"] = torch.tensor([target_class], device=device)

    t0 = time.perf_counter()
    raw_explanation = explainer(x, edge_index, **call_kwargs)
    elapsed = time.perf_counter() - t0
    model.to(device)
    raw_explanation = _coerce_explanation_device_dtype(raw_explanation, device)
    return raw_explanation, elapsed


def _preprocess_for_metrics(
    raw_explanation: Explanation,
    *,
    pred_class: int,
    target_class: int,
    apply_preprocessing_flag: bool,
    correct_class_only: bool,
    apply_mask_spread_filter: bool,
    mask_spread_tolerance: float,
) -> tuple[Explanation, bool, bool]:
    """Longa-style preprocessing on masks; return (explanation clone with masks, valid, correct_class)."""
    if apply_preprocessing_flag:
        preproc = apply_preprocessing(
            raw_explanation,
            pred_class=pred_class,
            target_class=target_class,
            correct_class_only=correct_class_only,
            normalize=True,
            convert_edge_to_node=False,
            apply_mask_spread_filter=apply_mask_spread_filter,
            mask_spread_tolerance=mask_spread_tolerance,
        )
        explanation = raw_explanation.clone()
        _pre_edge_mask = getattr(preproc.explanation, "edge_mask", None)
        if _pre_edge_mask is not None:
            explanation.edge_mask = _pre_edge_mask
        if getattr(preproc.explanation, "node_mask", None) is not None:
            explanation.node_mask = preproc.explanation.node_mask
        return explanation, preproc.valid, preproc.correct_class

    explanation = raw_explanation
    valid = True
    correct_class = target_class == pred_class if target_class is not None else True
    return explanation, valid, correct_class


def _compute_pyg_fidelity(
    explainer: Any,
    explanation: Explanation,
    *,
    explainer_name: str,
    graph_id: str,
) -> tuple[float, float]:
    """PyG GraphFramEx fidelity+ / fidelity− on the preprocessed explanation."""
    try:
        fid_result = fidelity(explainer, _fidelity_explanation(explanation))
        if isinstance(fid_result, (list, tuple)):
            fid_plus = float(fid_result[0]) if len(fid_result) > 0 else 0.0
            fid_minus = float(fid_result[1]) if len(fid_result) > 1 else 0.0
        else:
            fid_plus = float(fid_result)
            fid_minus = 0.0
        return fid_plus, fid_minus
    except Exception as exc:
        _LOG.warning("Fidelity failed for %s / %s: %s", explainer_name, graph_id, exc)
        return 0.0, 0.0


def _compute_pyg_characterization(
    fid_plus: float,
    fid_minus: float,
    *,
    device: torch.device,
    explainer_name: str,
    graph_id: str,
) -> float:
    """PyG framework metric: characterization_score from fid+ and fid−."""
    try:
        return float(
            characterization_score(
                torch.tensor(fid_plus, device=device),
                torch.tensor(fid_minus, device=device),
                pos_weight=0.5,
                neg_weight=0.5,
            ).item()
        )
    except Exception as exc:
        _LOG.warning(
            "characterization_score failed for %s / %s: %s",
            explainer_name,
            graph_id,
            exc,
        )
        return 0.0


# ---------------------------------------------------------------------------
# PGExplainer offline training
# ---------------------------------------------------------------------------


def train_explainer(
    explainer: Any,
    model: torch.nn.Module,
    train_loader: Any,
    device: torch.device,
    *,
    epochs: int,
    max_graphs_per_epoch: Optional[int] = None,
) -> None:
    """Train a PGExplainer's internal MLP on the training set.

    Must be called before ``explainer(...)`` for PGExplainer; no-op check is
    done by PyG (it raises if not trained).

    PyG's PGExplainer returns ``float(loss)`` on a tensor that still has
    ``requires_grad``; we filter that warning. Training can take a long time
    because each epoch walks the full ``train_loader`` unless
    *max_graphs_per_epoch* caps how many graphs are stepped per epoch
    (``--pg_train_max_graphs`` in the CLI).
    """
    model.eval()
    model.to(device)
    algo = explainer.algorithm
    if hasattr(algo, "mlp"):
        algo.mlp.to(device)
    ds = getattr(train_loader, "dataset", None)
    n_train_hint: Optional[int] = None
    if ds is not None and hasattr(ds, "__len__"):
        try:
            n_train_hint = len(ds)
        except TypeError:
            n_train_hint = None
    cap_msg = (
        f", capping at {max_graphs_per_epoch} graph(s)/epoch"
        if max_graphs_per_epoch is not None
        else ""
    )
    if n_train_hint is not None:
        _LOG.info(
            "PGExplainer train: ~%d graph(s) in loader × %d epoch(s)%s.",
            n_train_hint,
            epochs,
            cap_msg,
        )
        print(
            f"  PGExplainer: training {epochs} epoch(s) on the train loader "
            f"(~{n_train_hint} graph(s)){cap_msg or ' — this can take a long time'}.",
            flush=True,
        )
    else:
        print(
            f"  PGExplainer: training {epochs} epoch(s){cap_msg or ' — this can take a long time'}.",
            flush=True,
        )
    _pg_warn_re = r".*Converting a tensor with requires_grad=True to a scalar.*"
    for epoch in range(epochs):
        graphs_this_epoch = 0
        stop_epoch = False
        for batch in train_loader:
            if stop_epoch:
                break
            data_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            if hasattr(data_batch, "to_data_list"):
                graph_list = data_batch.to_data_list()
            else:
                graph_list = [data_batch]
            for data in graph_list:
                if max_graphs_per_epoch is not None and graphs_this_epoch >= max_graphs_per_epoch:
                    stop_epoch = True
                    break
                x, edge_index, batch_t, edge_attr = _single_graph_inputs(data, device)
                target_class = _get_target_class(data)
                if target_class is None:
                    with torch.no_grad():
                        logits = model(x, edge_index, batch_t, edge_attr)
                    target_class = int(logits.argmax(dim=-1).squeeze().item())
                target_t = torch.tensor([target_class], device=device)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=_pg_warn_re, category=UserWarning)
                    explainer.algorithm.train(
                        epoch,
                        model,
                        x,
                        edge_index,
                        target=target_t,
                        batch=batch_t,
                        edge_attr=edge_attr,
                    )
                graphs_this_epoch += 1
        print(
            f"  PGExplainer: epoch {epoch + 1}/{epochs} done ({graphs_this_epoch} gradient step(s)).",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main explanation loop
# ---------------------------------------------------------------------------


def run_explanations(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    explainer_name: str,
    explainer_epochs: int = 200,
    max_graphs: Optional[int] = None,
    get_graph_id: Optional[Callable[..., str]] = None,
    apply_preprocessing_flag: bool = True,
    correct_class_only: bool = True,
    apply_mask_spread_filter: bool = True,
    mask_spread_tolerance: float = 1e-3,
    train_loader: Optional[Any] = None,
    pg_train_max_graphs: Optional[int] = None,
    paper_metrics: bool = True,
    paper_n_thresholds: int = 100,
    **explainer_kwargs: Any,
) -> Iterator[ExplanationResult]:
    """Generate graph-level explanations, preprocess, then compute fidelity.

    When *explainer_name* refers to PGExplainer (``needs_training=True``),
    *train_loader* must be supplied so the explainer MLP can be fitted first.
    *pg_train_max_graphs* caps training steps per epoch (full loader if ``None``).
    """
    model.eval()
    model.to(device)

    spec = get_spec(explainer_name)
    builder = spec.builder
    explainer = builder(model, device=device, epochs=explainer_epochs, lr=explainer_kwargs.get("lr", 0.01), **explainer_kwargs)

    if spec.needs_training:
        if train_loader is None:
            raise ValueError(
                f"{explainer_name} requires offline training but no train_loader was provided."
            )
        _LOG.info("Training %s explainer MLP (%d epochs)…", explainer_name, explainer_epochs)
        train_explainer(
            explainer,
            model,
            train_loader,
            device,
            epochs=explainer_epochs,
            max_graphs_per_epoch=pg_train_max_graphs,
        )
        _LOG.info("Training complete for %s.", explainer_name)

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

            with torch.no_grad():
                logits = model(x, edge_index, batch_tensor, edge_attr)
            pred_class = int(logits.argmax(dim=-1).squeeze().item())
            target_class = _get_target_class(data)
            if target_class is None:
                target_class = pred_class

            raw_explanation, elapsed = _forward_raw_explanation(
                explainer,
                model,
                x=x,
                edge_index=edge_index,
                batch_tensor=batch_tensor,
                edge_attr=edge_attr,
                device=device,
                spec=spec,
                target_class=int(target_class),
            )

            has_edge_mask = getattr(raw_explanation, "edge_mask", None) is not None
            has_node_mask = getattr(raw_explanation, "node_mask", None) is not None

            explanation, valid, correct_class = _preprocess_for_metrics(
                raw_explanation,
                pred_class=pred_class,
                target_class=target_class,
                apply_preprocessing_flag=apply_preprocessing_flag,
                correct_class_only=correct_class_only,
                apply_mask_spread_filter=apply_mask_spread_filter,
                mask_spread_tolerance=mask_spread_tolerance,
            )

            fid_plus, fid_minus = _compute_pyg_fidelity(
                explainer,
                explanation,
                explainer_name=explainer_name,
                graph_id=graph_id,
            )
            pyg_char = _compute_pyg_characterization(
                fid_plus,
                fid_minus,
                device=device,
                explainer_name=explainer_name,
                graph_id=graph_id,
            )

            # Paper metrics (Longa et al.): threshold-sweep probability drops
            if paper_metrics:
                try:
                    paper_suf, paper_com, paper_f1 = _paper_metrics_from_masks(
                        model,
                        explanation,
                        target_class=int(target_class),
                        n_thresholds=paper_n_thresholds,
                    )
                except Exception as exc:
                    _LOG.warning(
                        "Paper metrics failed for %s / %s: %s",
                        explainer_name,
                        graph_id,
                        exc,
                    )
                    paper_suf, paper_com, paper_f1 = 0.0, 0.0, 0.0
            else:
                paper_suf, paper_com, paper_f1 = 0.0, 0.0, 0.0

            yield ExplanationResult(
                graph_id=graph_id,
                explanation=explanation,
                fidelity_fid_plus=fid_plus,
                fidelity_fid_minus=fid_minus,
                paper_sufficiency=paper_suf,
                paper_comprehensiveness=paper_com,
                paper_f1_fidelity=paper_f1,
                pyg_characterization=pyg_char,
                valid=valid,
                correct_class=correct_class,
                has_node_mask=has_node_mask,
                has_edge_mask=has_edge_mask,
                elapsed_s=elapsed,
            )
            graph_index += 1


def aggregate_fidelity(
    results: list[ExplanationResult],
    valid_only: bool = False,
) -> tuple[float, float]:
    """Return (mean fid+, mean fid-) over results."""
    if valid_only:
        results = [r for r in results if r.valid]
    if not results:
        return 0.0, 0.0
    n = len(results)
    mean_plus = sum(r.fidelity_fid_plus for r in results) / n
    mean_minus = sum(r.fidelity_fid_minus for r in results) / n
    return mean_plus, mean_minus
