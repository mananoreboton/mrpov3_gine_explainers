"""
Explainer-agnostic pipeline: generate graph-level explanations and compute metrics.
Follows Longa et al. common representation: (1) masks generation, (2) preprocessing
(Conversion, Filtering, Normalization), (3) metrics on preprocessed masks.

Supports edge-mask, node-mask, and mixed explainers.  PGExplainer offline training
is handled via ``train_explainer()``.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional

import torch
from torch_geometric.explain import Explanation
from torch_geometric.explain.metric import characterization_score, fidelity
from torch_geometric.utils import subgraph

from mprov3_explainer.explainers import ExplainerSpec, get_spec
from mprov3_explainer.preprocessing import (
    _align_node_mask_to_graph,
    apply_preprocessing,
    binarize_top_k,
    edge_mask_to_node_mask,
    normalize_mask,
    reduce_node_mask,
)

_LOG = logging.getLogger(__name__)

#: Default fraction of top-ranked entries kept by the binarized fidelity helpers.
#: 0.2 is the GraphFramEx canonical value (Amara et al., 2022).
DEFAULT_TOP_K_FRACTION = 0.2

#: Default number of percentile thresholds in the Longa et al. sweep.
DEFAULT_PAPER_N_THRESHOLDS = 100

_NAN = float("nan")


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
    """Result of explaining one graph.

    The ``fidelity_*`` and ``pyg_characterization`` fields hold the **top-k
    binarized** GraphFramEx fidelity (the scientifically correct headline);
    the ``*_soft`` fields preserve the legacy soft-mask values for
    backwards-compatible reporting and downstream comparison. Every metric
    field uses :data:`math.nan` to signal "could not be computed" so the
    aggregator can skip such entries cleanly.
    """

    graph_id: str
    explanation: Explanation
    fidelity_fid_plus: float = _NAN
    fidelity_fid_minus: float = _NAN
    pyg_characterization: float = _NAN
    fidelity_fid_plus_soft: float = _NAN
    fidelity_fid_minus_soft: float = _NAN
    pyg_characterization_soft: float = _NAN
    paper_sufficiency: float = _NAN
    paper_comprehensiveness: float = _NAN
    paper_f1_fidelity: float = _NAN
    valid: bool = True
    correct_class: bool = True
    pred_class: int = -1
    target_class: int = -1
    prediction_baseline_mismatch: bool = False
    has_node_mask: bool = False
    has_edge_mask: bool = False
    mask_spread: float = 0.0
    mask_entropy: float = 0.0
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class PredictionBaselineEntry:
    """Model prediction for one graph before any explainer is run."""

    graph_id: str
    pred_class: int
    target_class: int
    correct_class: bool


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


def collect_prediction_baseline(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    get_graph_id: Optional[Callable[..., str]] = None,
    max_graphs: Optional[int] = None,
) -> dict[str, PredictionBaselineEntry]:
    """Compute graph predictions once, before running explainer-specific code.

    Misclassification is a property of the trained model, fold, and split. It
    should not change across explainers, so the runner can collect this
    baseline once and pass it back into :func:`run_explanations`.
    """
    was_training = model.training
    model.eval()
    model.to(device)
    baseline: dict[str, PredictionBaselineEntry] = {}
    graph_index = 0
    try:
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
                baseline[graph_id] = PredictionBaselineEntry(
                    graph_id=graph_id,
                    pred_class=pred_class,
                    target_class=int(target_class),
                    correct_class=pred_class == int(target_class),
                )
                graph_index += 1
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    return baseline


def _coerce_prediction_baseline_entry(
    entry: PredictionBaselineEntry | Mapping[str, Any],
) -> PredictionBaselineEntry:
    """Accept dataclass or dict baselines for test and integration flexibility."""
    if isinstance(entry, PredictionBaselineEntry):
        return entry
    graph_id = str(entry.get("graph_id", ""))
    pred_class = int(entry["pred_class"])
    target_class = int(entry["target_class"])
    correct_class = bool(entry.get("correct_class", pred_class == target_class))
    return PredictionBaselineEntry(
        graph_id=graph_id,
        pred_class=pred_class,
        target_class=target_class,
        correct_class=correct_class,
    )


def _predict_target_proba(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    target_class: int,
) -> float:
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, batch, edge_attr)
            probs = torch.softmax(logits, dim=-1)
            return float(probs.squeeze(0)[target_class].item())
    finally:
        if was_training:
            model.train()


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


def _percentile_keep_fractions(n_thresholds: int) -> list[float]:
    """Return the sequence of "keep" fractions used by the Longa percentile sweep.

    For ``n_thresholds = 100`` this returns ``[0.99, 0.98, …, 0.01]`` — i.e. the
    fraction of top-ranked nodes/edges retained at each step. Iterating from
    "keep almost everything" to "keep almost nothing" mirrors the GraphFramEx
    sufficiency curve.
    """
    Nt = max(2, int(n_thresholds))
    return [1.0 - float(k) / float(Nt) for k in range(1, Nt)]


def _paper_sufficiency_and_comprehensiveness(
    model: torch.nn.Module,
    explanation: Explanation,
    *,
    node_mask: torch.Tensor,
    target_class: int,
    n_thresholds: int,
) -> tuple[float, float]:
    """Node-native Longa sweep using *percentile* thresholds.

    For each ``keep_fraction k ∈ {1/Nt, …, (Nt-1)/Nt}``, build the explanation
    subgraph from the top-``k`` fraction of nodes by mask value, and the
    complement subgraph from the rest. Sufficiency / comprehensiveness are the
    average drops in target-class probability across the sweep.

    Compared to the previous raw ``mask > t`` thresholding, this is robust to
    masks that are not uniformly distributed in ``[0, 1]`` (e.g. a single very
    large saliency value would bias raw thresholds toward "keep almost
    nothing").
    """
    x = explanation.x
    edge_index = explanation.edge_index
    if x is None or edge_index is None:
        return _NAN, _NAN

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

    keep_fractions = _percentile_keep_fractions(n_thresholds)
    if not keep_fractions:
        return _NAN, _NAN

    suf_sum = 0.0
    com_sum = 0.0

    N = int(x.size(0))
    all_nodes = torch.arange(N, device=x.device)

    for keep_frac in keep_fractions:
        keep_mask = binarize_top_k(node_mask, keep_frac).bool()
        if bool(keep_mask.any()):
            keep_nodes = all_nodes[keep_mask]
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

        comp_mask = ~keep_mask
        if bool(comp_mask.any()):
            comp_nodes = all_nodes[comp_mask]
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

    denom = float(len(keep_fractions))
    return float(suf_sum / denom), float(com_sum / denom)


def _paper_metrics_from_edge_mask(
    model: torch.nn.Module,
    explanation: Explanation,
    *,
    edge_mask: torch.Tensor,
    target_class: int,
    n_thresholds: int,
) -> tuple[float, float]:
    """Edge-native Longa sweep over a normalized edge mask.

    Mirrors :func:`_paper_sufficiency_and_comprehensiveness` but operates on
    edges directly: the explanation subgraph keeps the top-``k`` fraction of
    edges, and the induced node set is everything they touch. The complement
    subgraph keeps the remaining edges. This avoids the lossy "average incident
    edge weight" coercion to a node mask that was previously applied to
    edge-only explainers (GRADEXPLEDGE, IGEDGE, GNNEXPL, PGEXPL) and keeps the
    paper metric symmetric with how PyG fidelity uses the edge mask.
    """
    x = explanation.x
    edge_index = explanation.edge_index
    if x is None or edge_index is None or edge_mask is None or edge_mask.numel() == 0:
        return _NAN, _NAN

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

    keep_fractions = _percentile_keep_fractions(n_thresholds)
    if not keep_fractions:
        return _NAN, _NAN

    N = int(x.size(0))
    suf_sum = 0.0
    com_sum = 0.0

    for keep_frac in keep_fractions:
        keep_edges = binarize_top_k(edge_mask, keep_frac).bool()
        comp_edges = ~keep_edges

        exp_prob = _predict_with_edge_subset(
            model,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_subset=keep_edges,
            target_class=target_class,
            num_nodes=N,
        )
        comp_prob = _predict_with_edge_subset(
            model,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_subset=comp_edges,
            target_class=target_class,
            num_nodes=N,
        )

        suf_sum += (full_prob - exp_prob)
        com_sum += (full_prob - comp_prob)

    denom = float(len(keep_fractions))
    return float(suf_sum / denom), float(com_sum / denom)


def _predict_with_edge_subset(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    edge_subset: torch.Tensor,
    target_class: int,
    num_nodes: int,
) -> float:
    """Predict the target-class probability on the subgraph induced by ``edge_subset``.

    The induced node set is the set of endpoints of the kept edges; isolated
    nodes are dropped so the pooled readout is computed on the connected
    substructure that the edge-level explanation actually selects.
    """
    if not bool(edge_subset.any()):
        return 0.0

    sub_ei_orig = edge_index[:, edge_subset]
    sub_edge_attr = edge_attr[edge_subset] if edge_attr is not None else None

    touched = torch.unique(sub_ei_orig.reshape(-1))
    if touched.numel() == 0:
        return 0.0

    # Build old -> new index mapping for the touched nodes, then relabel edges.
    node_remap = torch.full(
        (num_nodes,), fill_value=-1, dtype=torch.long, device=x.device,
    )
    node_remap[touched] = torch.arange(
        touched.numel(), dtype=torch.long, device=x.device,
    )
    relabeled_ei = node_remap[sub_ei_orig]

    sub_x = x[touched]
    sub_batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=x.device)
    return _predict_target_proba(
        model,
        x=sub_x,
        edge_index=relabeled_ei,
        batch=sub_batch,
        edge_attr=sub_edge_attr,
        target_class=target_class,
    )


def _clamp_unit(value: float) -> float:
    """Clamp a real number into ``[0, 1]``; pass NaN through unchanged."""
    if value != value:  # NaN
        return value
    return max(0.0, min(1.0, value))


def _paper_f1_fidelity(Fsuf: float, Fcom: float) -> float:
    """Paper F1-fidelity from sufficiency and comprehensiveness (Longa et al.).

    The Longa et al. (2025) definition assumes ``Fsuf, Fcom ∈ [0, 1]``. The
    implementation in earlier versions of this file fed raw signed probability
    differences into the harmonic-mean formula, which can return values outside
    ``[0, 1]`` (and even negative numbers) when one of the operands is negative
    or ``Fsuf > 1``. Here we clamp both arguments to ``[0, 1]`` first, so the
    return value is always a well-defined harmonic-mean F-score in ``[0, 1]``.
    A DEBUG log entry is emitted when clamping changes either input by more
    than 0.05 so noisy graphs stay traceable.
    """
    if Fsuf != Fsuf or Fcom != Fcom:  # NaN propagation
        return _NAN
    Fsuf_c = _clamp_unit(Fsuf)
    Fcom_c = _clamp_unit(Fcom)
    if abs(Fsuf - Fsuf_c) > 0.05 or abs(Fcom - Fcom_c) > 0.05:
        _LOG.debug(
            "Ff1 clamp: Fsuf %.4f -> %.4f, Fcom %.4f -> %.4f",
            Fsuf, Fsuf_c, Fcom, Fcom_c,
        )
    num = 2.0 * (1.0 - Fsuf_c) * Fcom_c
    den = (1.0 - Fsuf_c) + Fcom_c
    return (num / den) if den > 1e-12 else 0.0


def _paper_metrics_from_masks(
    model: torch.nn.Module,
    explanation: Explanation,
    *,
    target_class: int,
    n_thresholds: int = DEFAULT_PAPER_N_THRESHOLDS,
) -> tuple[float, float, float]:
    """Longa et al. (2025) graph-classification fidelity metrics with mask-type dispatch.

    - **Node-only or mixed explanation** -> node-native percentile sweep on the
      preprocessed node mask (induced node subgraphs).
    - **Edge-only explanation** -> edge-native percentile sweep on the
      preprocessed edge mask (induced edge subgraphs), preserving the explainer's
      native granularity instead of coercing edges into node scores.
    - **No mask at all** -> NaN triple, signalling "could not be computed".

    The returned ``Ff1`` is the clamped harmonic mean (see :func:`_paper_f1_fidelity`)
    so it is always a well-defined F-score in ``[0, 1]`` (or NaN).
    """
    node_mask_raw = explanation.get("node_mask")
    edge_mask_raw = explanation.get("edge_mask")

    if node_mask_raw is not None:
        node_mask = _paper_normalized_node_mask_from_explanation(explanation)
        if node_mask is None:
            return _NAN, _NAN, _NAN
        Fsuf, Fcom = _paper_sufficiency_and_comprehensiveness(
            model,
            explanation,
            node_mask=node_mask,
            target_class=target_class,
            n_thresholds=n_thresholds,
        )
    elif edge_mask_raw is not None:
        Fsuf, Fcom = _paper_metrics_from_edge_mask(
            model,
            explanation,
            edge_mask=edge_mask_raw.detach().float(),
            target_class=target_class,
            n_thresholds=n_thresholds,
        )
    else:
        return _NAN, _NAN, _NAN

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
    # Some explainers (PGMExplainer, GNNExplainer) internally toggle model.train();
    # restore eval mode so BatchNorm/Dropout behave correctly during metric sweeps.
    model.eval()
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
    """PyG GraphFramEx fidelity+ / fidelity− on a (soft-mask) explanation.

    PyG's :func:`fidelity` returns class-decision fidelity rates. In model mode
    it checks whether the masked/complemented graph preserves the full-graph
    predicted class; in phenomenon mode it checks target-class correctness.
    These values are not probability-drop ratios.

    Returns ``(NaN, NaN)`` on exception so the aggregator can drop the graph
    cleanly instead of silently averaging in zeros.
    """
    try:
        fid_result = fidelity(explainer, _fidelity_explanation(explanation))
        if isinstance(fid_result, (list, tuple)):
            fid_plus = float(fid_result[0]) if len(fid_result) > 0 else _NAN
            fid_minus = float(fid_result[1]) if len(fid_result) > 1 else _NAN
        else:
            fid_plus = float(fid_result)
            fid_minus = _NAN
        return fid_plus, fid_minus
    except Exception as exc:
        _LOG.warning("Fidelity failed for %s / %s: %s", explainer_name, graph_id, exc)
        return _NAN, _NAN


def _binarize_explanation_top_k(
    explanation: Explanation,
    *,
    k: float,
) -> Explanation:
    """Clone *explanation* and replace its masks with their top-``k`` binarizations.

    Soft attribution scores (continuous in ``[0, 1]`` after preprocessing) are not
    well-defined "subgraphs"; ``node_mask * x`` and ``(1 - node_mask) * x`` are
    just two rescalings of the same input and almost always elicit the same
    prediction from a robust GNN, which is why the soft-mask Fid+/Fid- pair so
    often coincides on this dataset (see ``result_description.md`` §5.1). Top-k
    binarization restores the GraphFramEx semantics of "explanation = a hard
    subset, complement = its set complement".
    """
    out = explanation.clone()
    nm = explanation.get("node_mask")
    em = explanation.get("edge_mask")
    if nm is not None:
        nm_bin = binarize_top_k(nm.detach().float(), k)
        # _fidelity_explanation will reshape (N,) -> (N, 1) downstream.
        out.node_mask = nm_bin
    if em is not None:
        em_bin = binarize_top_k(em.detach().float(), k)
        out.edge_mask = em_bin
    return out


def _compute_pyg_fidelity_top_k(
    explainer: Any,
    explanation: Explanation,
    *,
    k: float,
    explainer_name: str,
    graph_id: str,
) -> tuple[float, float]:
    """Top-k binarized GraphFramEx fidelity+ / fidelity−.

    This is the GraphFramEx-canonical formulation (Amara et al., 2022): mask is
    binarized at the top-``k`` fraction of entries before being fed to PyG's
    class-decision :func:`fidelity`. ``k = 0.2`` is the paper's default.
    """
    try:
        binarized = _binarize_explanation_top_k(explanation, k=k)
        fid_result = fidelity(explainer, _fidelity_explanation(binarized))
        if isinstance(fid_result, (list, tuple)):
            fid_plus = float(fid_result[0]) if len(fid_result) > 0 else _NAN
            fid_minus = float(fid_result[1]) if len(fid_result) > 1 else _NAN
        else:
            fid_plus = float(fid_result)
            fid_minus = _NAN
        return fid_plus, fid_minus
    except Exception as exc:
        _LOG.warning(
            "Top-k fidelity (k=%.2f) failed for %s / %s: %s",
            k, explainer_name, graph_id, exc,
        )
        return _NAN, _NAN


def _compute_pyg_characterization(
    fid_plus: float,
    fid_minus: float,
    *,
    device: torch.device,
    explainer_name: str,
    graph_id: str,
) -> float:
    """PyG framework metric: characterization_score from fid+ and fid−.

    Returns NaN if either input is NaN or if the underlying call raises.
    """
    if fid_plus != fid_plus or fid_minus != fid_minus:
        return _NAN
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
        return _NAN


def _mask_spread(mask: Optional[torch.Tensor]) -> float:
    """``max - min`` of a mask, or 0 if absent/empty/NaN."""
    if mask is None or mask.numel() == 0:
        return 0.0
    try:
        return float(mask.max().item() - mask.min().item())
    except Exception:
        return 0.0


def _mask_entropy(mask: Optional[torch.Tensor]) -> float:
    """Shannon entropy (nats) of a normalized 1-D mask interpreted as a distribution.

    Useful as a diagnostic of mask sharpness: a uniform mask has entropy
    ``log(N)``, a one-hot mask has entropy 0. Returns 0 for absent/empty masks.
    """
    if mask is None or mask.numel() == 0:
        return 0.0
    try:
        flat = mask.detach().float().reshape(-1).abs()
        s = float(flat.sum().item())
        if s <= 1e-12:
            return 0.0
        p = flat / s
        # log(0) -> 0 contribution by convention
        nz = p > 0
        return float(-(p[nz] * p[nz].log()).sum().item())
    except Exception:
        return 0.0


def _representative_mask(explanation: Explanation) -> Optional[torch.Tensor]:
    """Return whichever mask is present, preferring node mask, for spread/entropy."""
    nm = explanation.get("node_mask")
    if nm is not None:
        if nm.dim() > 1:
            nm = reduce_node_mask(nm)
        return nm
    return explanation.get("edge_mask")


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
    paper_n_thresholds: int = DEFAULT_PAPER_N_THRESHOLDS,
    top_k_fraction: float = DEFAULT_TOP_K_FRACTION,
    prediction_baseline: Optional[
        Mapping[str, PredictionBaselineEntry | Mapping[str, Any]]
    ] = None,
    **explainer_kwargs: Any,
) -> Iterator[ExplanationResult]:
    """Generate graph-level explanations, preprocess, then compute fidelity.

    When *explainer_name* refers to PGExplainer (``needs_training=True``),
    *train_loader* must be supplied so the explainer MLP can be fitted first.
    *pg_train_max_graphs* caps training steps per epoch (full loader if ``None``).

    The headline ``fidelity_fid_plus`` / ``fidelity_fid_minus`` /
    ``pyg_characterization`` fields on each :class:`ExplanationResult` are now
    the **top-k binarized** GraphFramEx values (``k = top_k_fraction``); the
    legacy soft-mask values are retained as ``*_soft`` siblings for
    comparability. A graph whose preprocessing failed or whose metrics raised
    an exception is yielded with ``valid=False`` and NaN metric values, so the
    aggregator can drop it without polluting the means.
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

            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, batch_tensor, edge_attr)
            observed_pred_class = int(logits.argmax(dim=-1).squeeze().item())
            target_class = _get_target_class(data)
            if target_class is None:
                target_class = observed_pred_class
            pred_class = observed_pred_class
            prediction_baseline_mismatch = False
            if prediction_baseline is not None and graph_id in prediction_baseline:
                baseline_entry = _coerce_prediction_baseline_entry(
                    prediction_baseline[graph_id]
                )
                prediction_baseline_mismatch = (
                    baseline_entry.pred_class != observed_pred_class
                    or baseline_entry.target_class != int(target_class)
                )
                if prediction_baseline_mismatch:
                    _LOG.warning(
                        "Prediction baseline mismatch for %s / %s: baseline=(pred=%s,target=%s), observed=(pred=%s,target=%s)",
                        explainer_name,
                        graph_id,
                        baseline_entry.pred_class,
                        baseline_entry.target_class,
                        observed_pred_class,
                        int(target_class),
                    )
                pred_class = baseline_entry.pred_class
                target_class = baseline_entry.target_class

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

            # Soft-mask fidelity (legacy GraphFramEx behaviour, kept for
            # backwards-compatible reporting under *_soft keys).
            fid_plus_soft, fid_minus_soft = _compute_pyg_fidelity(
                explainer,
                explanation,
                explainer_name=explainer_name,
                graph_id=graph_id,
            )
            pyg_char_soft = _compute_pyg_characterization(
                fid_plus_soft,
                fid_minus_soft,
                device=device,
                explainer_name=explainer_name,
                graph_id=graph_id,
            )

            # Top-k binarized fidelity (scientifically correct headline; the
            # GraphFramEx paper's actual definition with k=0.2 by default).
            fid_plus, fid_minus = _compute_pyg_fidelity_top_k(
                explainer,
                explanation,
                k=top_k_fraction,
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

            # Paper metrics (Longa et al.): percentile-sweep probability drops,
            # dispatched by mask type. NaN return values propagate as "could
            # not be computed" and mark the graph invalid.
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
                    paper_suf, paper_com, paper_f1 = _NAN, _NAN, _NAN
            else:
                paper_suf, paper_com, paper_f1 = _NAN, _NAN, _NAN

            # If a metric we expected to compute came back NaN (because of an
            # exception in PyG fidelity, characterization_score, or the paper
            # sweep), mark the graph invalid so the aggregator skips it. We
            # only invalidate on missing top-k fidelity / paper metrics — the
            # soft-mask fields are diagnostic only.
            metric_failed = any(
                math.isnan(v) for v in (fid_plus, fid_minus, pyg_char)
            )
            if paper_metrics:
                metric_failed = metric_failed or any(
                    math.isnan(v) for v in (paper_suf, paper_com, paper_f1)
                )
            if metric_failed:
                valid = False

            rep_mask = _representative_mask(explanation)
            spread_val = _mask_spread(rep_mask)
            entropy_val = _mask_entropy(rep_mask)

            yield ExplanationResult(
                graph_id=graph_id,
                explanation=explanation,
                fidelity_fid_plus=fid_plus,
                fidelity_fid_minus=fid_minus,
                pyg_characterization=pyg_char,
                fidelity_fid_plus_soft=fid_plus_soft,
                fidelity_fid_minus_soft=fid_minus_soft,
                pyg_characterization_soft=pyg_char_soft,
                paper_sufficiency=paper_suf,
                paper_comprehensiveness=paper_com,
                paper_f1_fidelity=paper_f1,
                valid=valid,
                correct_class=correct_class,
                pred_class=int(pred_class),
                target_class=int(target_class),
                prediction_baseline_mismatch=prediction_baseline_mismatch,
                has_node_mask=has_node_mask,
                has_edge_mask=has_edge_mask,
                mask_spread=spread_val,
                mask_entropy=entropy_val,
                elapsed_s=elapsed,
            )
            graph_index += 1


def nanmean(xs: list[float]) -> float:
    """Mean over non-None, non-NaN entries; returns NaN if no valid entry exists."""
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return (sum(vals) / len(vals)) if vals else _NAN


def aggregate_fidelity(
    results: list[ExplanationResult],
    valid_only: bool = False,
    nan_skip: bool = True,
) -> tuple[float, float]:
    """Return ``(mean fid+, mean fid-)`` over ``results``.

    Args:
        results: per-graph explanation results.
        valid_only: if True, only ``r.valid`` results contribute.
        nan_skip: if True (default), NaN per-graph values are skipped instead
            of polluting the mean. A NaN signals "the metric could not be
            computed for this graph" and should not be averaged in as 0.
    """
    if valid_only:
        results = [r for r in results if r.valid]
    if not results:
        return _NAN, _NAN
    plus = [r.fidelity_fid_plus for r in results]
    minus = [r.fidelity_fid_minus for r in results]
    if nan_skip:
        return nanmean(plus), nanmean(minus)
    n = len(results)
    return sum(plus) / n, sum(minus) / n


def diagnose_explanation_run(
    results: list[ExplanationResult],
    *,
    mask_spread_tolerance: float = 1e-3,
) -> tuple[str, str]:
    """Return a compact status and note for an explainer run.

    Headline metrics are intentionally valid-only, so a method with zero valid
    explanations can otherwise look like a normal row whose means happen to be
    missing. This helper makes failed or partially degenerate runs explicit in
    the machine-readable JSON and the HTML report.
    """
    if not results:
        return "empty_run", "No graphs were explained; headline metrics are unavailable."

    n_graphs = len(results)
    n_valid = sum(1 for r in results if r.valid)
    n_degenerate = sum(
        1 for r in results if r.mask_spread < mask_spread_tolerance
    )

    if n_valid == 0 and n_degenerate == n_graphs:
        return (
            "failed_all_degenerate_masks",
            "No headline metrics are valid because every produced mask is degenerate.",
        )
    if n_valid == 0 and n_degenerate > 0:
        return (
            "failed_degenerate_or_invalid_masks",
            "No headline metrics are valid; at least one mask is degenerate and the remaining graphs are invalid.",
        )
    if n_valid == 0:
        return (
            "failed_no_valid_metrics",
            "No headline metrics are valid after correctness and metric-failure filtering.",
        )
    if n_degenerate > 0:
        return (
            "partial_degenerate_masks",
            f"{n_degenerate} of {n_graphs} masks are degenerate and excluded from valid-only headline metrics.",
        )
    return "ok", "Run produced at least one valid non-degenerate explanation."
