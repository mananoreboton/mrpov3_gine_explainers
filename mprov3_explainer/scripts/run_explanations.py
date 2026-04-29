#!/usr/bin/env python3
"""
Run all explainers on trained GINE fold(s) and write metric reports.

For every (fold, explainer) the script writes
``results/folds/fold_<k>/explanations/<EXPLAINER>/explanation_report.json``
containing two aggregate tables computed from the same per-graph metric set:

* ``valid_result_metrics``: aggregated **only over graphs that produced a
  complete metric set** (``valid == True``).
* ``result_metrics``: aggregated over **every explained graph** in the fold,
  with NaN-aware means so failed graphs do not bias the aggregate.

Both tables expose the Longa et al. (2025) paper metrics (``Fsuf``,
``Fcom``, ``Ff1``) and the PyG metrics from
:mod:`torch_geometric.explain.metric` (``fidelity``, ``characterization_score``,
``fidelity_curve_auc``, ``unfaithfulness``) as PyG generates them. The
``result_metrics`` table also carries ``wall_time_s`` and the total
``num_graphs``.

A cross-explainer ``comparison_report.json`` is written under
``results/folds/fold_<k>/explanations/`` for each fold.

Usage:
  uv run python scripts/run_explanations.py
  uv run python scripts/run_explanations.py --split validation
  uv run python scripts/run_explanations.py --fold_metric train_accuracy
  uv run python scripts/run_explanations.py --folds 0 2 4
  uv run python scripts/run_explanations.py --no_mask_spread_filter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_MPROV3_EXPLAINER_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _MPROV3_EXPLAINER_ROOT.parent
_GNN_PROJECT_ROOT = _REPO_ROOT / "mprov3_gine"
if _GNN_PROJECT_ROOT.exists() and str(_GNN_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_PROJECT_ROOT))

import torch
from mprov3_gine_explainer_defaults import (
    BUILT_DATASET_FOLDER_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EDGE_DIM,
    DEFAULT_GNN_EXPLAINER_EPOCHS,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_IG_N_STEPS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_MPRO_SNAPSHOT_DIR_NAME,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_PG_EXPLAINER_EPOCHS,
    DEFAULT_PGM_NUM_SAMPLES,
    DEFAULT_POOL,
    DEFAULT_SEED,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_VAL_SPLIT_FILE,
    RESULTS_DATASETS,
    RESULTS_DIR_NAME,
    RESULTS_EXPLANATIONS,
    RESULTS_TRAININGS,
    SplitConfig,
    read_num_folds_for_fold,
    resolve_best_fold_index,
    resolve_checkpoint_path,
    resolve_fold_indices,
    seed_everything,
)

from mprov3_explainer import (
    AVAILABLE_EXPLAINERS,
    DEFAULT_PAPER_N_THRESHOLDS,
    ExplanationResult,
    PredictionBaselineEntry,
    collect_prediction_baseline,
    diagnose_explanation_run,
    dumps_strict_json,
    explanations_run_dir,
    get_device,
    nanmean,
    resolve_dataset_dir,
    run_explanations,
    validate_explainer,
)
from mprov3_explainer.explainers import get_spec

PAPER_N_THRESHOLDS = DEFAULT_PAPER_N_THRESHOLDS
MASK_SPREAD_TOLERANCE = 1e-6

# ----------------------------------------------------------------------------
# Aggregate metric layout
# ----------------------------------------------------------------------------

#: Per-graph fields (and their report keys) used to build both aggregate tables.
#: Order is preserved end-to-end: per-graph entries, in-progress logging, the
#: ``valid_result_metrics`` / ``result_metrics`` blocks of the JSON report and
#: the HTML report tables all iterate over this same sequence.
_METRIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("paper_sufficiency", "Fsuf"),
    ("paper_comprehensiveness", "Fcom"),
    ("paper_f1_fidelity", "Ff1"),
    ("pyg_fidelity_plus", "Fid+"),
    ("pyg_fidelity_minus", "Fid-"),
    ("pyg_characterization_score", "char"),
    ("pyg_fidelity_curve_auc", "AUC"),
    ("pyg_unfaithfulness", "GEF"),
)


@dataclass
class ExplanationRunContext:
    """Resolved (dataset, model, splits, outputs) for one benchmark run."""

    fold_index: int
    num_folds: int
    split_name: str
    fold_metric: str
    data_root: Path
    dataset_base: Path
    dataset_name: str
    train_loader: Any
    val_loader: Any
    test_loader: Any
    explain_loader: Any
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    explainer_results_root: Path
    num_classes: int
    apply_mask_spread_filter: bool


# ----------------------------------------------------------------------------
# CLI / context
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all explainers on the best fold (test accuracy from classify.py "
            "by default). Uses mprov3_gine/results and the default MPro snapshot "
            "path; PGExplainer still trains on the train split, then explains "
            "the split from --split."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "validation", "test"),
        default="test",
        help="Data split whose loader is used for the explanation loop (default: test).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Explicit fold indices to run (e.g. --folds 0 2 4). Overrides --fold_metric. "
             "Error if any index is out of range for the CV split.",
    )
    parser.add_argument(
        "--fold_metric",
        type=str,
        choices=("test_accuracy", "train_accuracy"),
        default="test_accuracy",
        help="Pick fold from classification_summary (test) or training_summary (train). "
             "Ignored when --folds is given.",
    )
    parser.add_argument(
        "--no_mask_spread_filter",
        action="store_true",
        help="Disable τ=1e⁻³ degenerate-mask discard (default: filter on).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"RNG seed for torch / numpy / random / PyG (default: {DEFAULT_SEED}). "
            "Pin to make PGExplainer / PGMExplainer / IG runs reproducible."
        ),
    )
    return parser.parse_args()


def _get_graph_id(data: Any, index: int) -> str:
    return getattr(data, "pdb_id", f"graph_{index}")


def _read_num_folds_from_summary(results_root: Path) -> int:
    """Read ``num_folds`` from the training summary without picking a best fold."""
    summary_path = results_root / RESULTS_TRAININGS / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {summary_path}; run mprov3_gine/train.py first."
        )
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = data.get("folds") or []
    if not folds:
        raise FileNotFoundError(
            f"{summary_path} has no folds; run train.py on at least one fold."
        )
    return int(folds[0]["num_folds"])


def build_explanation_run_context_for_fold(
    args: argparse.Namespace,
    fold_index: int,
    num_folds: int,
) -> ExplanationRunContext:
    """Build an ExplanationRunContext for an explicit fold index."""
    results_root = _GNN_PROJECT_ROOT / RESULTS_DIR_NAME
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    k = fold_index
    print(
        f"Using fold_index={k} (num_folds={num_folds}, split={args.split})",
        flush=True,
    )

    data_root = _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    resolve_dataset_dir(results_root)
    dataset_base = results_root / RESULTS_DATASETS
    dataset_name = BUILT_DATASET_FOLDER_NAME

    from loaders import create_data_loaders
    from model import MProGNN

    device = get_device()
    checkpoint_path = resolve_checkpoint_path(
        results_root, DEFAULT_TRAINING_CHECKPOINT_FILENAME, fold_index=k
    )

    split_config = SplitConfig(
        train_file=DEFAULT_TRAIN_SPLIT_FILE,
        val_file=DEFAULT_VAL_SPLIT_FILE,
        test_file=DEFAULT_TEST_SPLIT_FILE,
        num_folds=num_folds,
        fold_index=k,
        dataset_name=dataset_name,
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_base, data_root, split_config, batch_size=DEFAULT_BATCH_SIZE,
    )
    split_loaders = {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }
    explain_loader = split_loaders[args.split]

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

    fold_seg = f"fold_{k}"
    explainer_results_root = (
        _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME / "folds" / fold_seg
    )

    return ExplanationRunContext(
        fold_index=k,
        num_folds=num_folds,
        split_name=args.split,
        fold_metric=args.fold_metric,
        data_root=data_root,
        dataset_base=dataset_base,
        dataset_name=dataset_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        explain_loader=explain_loader,
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        explainer_results_root=explainer_results_root,
        num_classes=DEFAULT_OUT_CLASSES,
        apply_mask_spread_filter=not args.no_mask_spread_filter,
    )


def build_explanation_run_context(args: argparse.Namespace) -> ExplanationRunContext:
    """Resolve paths, loaders, best fold and trained GNN."""
    results_root = _GNN_PROJECT_ROOT / RESULTS_DIR_NAME
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    k = resolve_best_fold_index(results_root, args.fold_metric)
    num_folds = read_num_folds_for_fold(results_root, k)
    print(
        f"Best fold: fold_index={k} (fold_metric={args.fold_metric})",
        flush=True,
    )
    return build_explanation_run_context_for_fold(args, k, num_folds)


# ----------------------------------------------------------------------------
# Per-graph -> dict + aggregation helpers
# ----------------------------------------------------------------------------


def _fmt(x: float) -> str:
    """Pretty-print a float that may be NaN."""
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"


def _per_graph_entry(r: ExplanationResult) -> dict:
    """Convert one :class:`ExplanationResult` into the JSON-serialisable shape."""
    entry: dict[str, Any] = {"graph_id": r.graph_id, "valid": bool(r.valid)}
    for field, _ in _METRIC_FIELDS:
        entry[field] = float(getattr(r, field))
    entry.update({
        "correct_class": bool(r.correct_class),
        "pred_class": int(r.pred_class),
        "target_class": int(r.target_class),
        "prediction_baseline_mismatch": bool(r.prediction_baseline_mismatch),
        "has_node_mask": bool(r.has_node_mask),
        "has_edge_mask": bool(r.has_edge_mask),
        "elapsed_s": float(r.elapsed_s),
    })
    return entry


def _aggregate_metrics(results: list[ExplanationResult]) -> dict[str, float]:
    """NaN-aware mean of every metric in :data:`_METRIC_FIELDS` over *results*."""
    return {
        f"mean_{field}": nanmean([float(getattr(r, field)) for r in results])
        for field, _ in _METRIC_FIELDS
    }


def _build_valid_result_metrics(results: list[ExplanationResult]) -> dict[str, Any]:
    """Aggregate metrics over the subset of *results* with ``valid == True``."""
    valid = [r for r in results if r.valid]
    return {
        "num_valid_graphs": len(valid),
        **_aggregate_metrics(valid),
    }


def _build_result_metrics(
    results: list[ExplanationResult],
    *,
    wall_time_s: float,
) -> dict[str, Any]:
    """Aggregate metrics over **every** graph in the fold, plus runtime / count."""
    return {
        "wall_time_s": float(wall_time_s),
        "num_graphs": len(results),
        **_aggregate_metrics(results),
    }


def _print_metric_block(label: str, block: dict[str, Any]) -> None:
    """Compact one-line summary of a metric table."""
    parts: list[str] = []
    for field, short in _METRIC_FIELDS:
        parts.append(f"{short}={_fmt(block[f'mean_{field}'])}")
    extra_keys = [k for k in ("wall_time_s", "num_graphs", "num_valid_graphs") if k in block]
    extras = " ".join(
        f"{k}={block[k]:.1f}" if isinstance(block[k], float) else f"{k}={block[k]}"
        for k in extra_keys
    )
    print(f"  {label}: {extras} | " + " ".join(parts), flush=True)


# ----------------------------------------------------------------------------
# Core explainer driver
# ----------------------------------------------------------------------------


def _explainer_kwargs(explainer_name: str, num_classes: int) -> tuple[dict, int]:
    """Return (forward kwargs, builder epochs) tuned per explainer."""
    kwargs: dict[str, Any] = {"num_classes": num_classes}
    if explainer_name in ("IGNODE", "IGEDGE"):
        kwargs["n_steps"] = DEFAULT_IG_N_STEPS
    if explainer_name == "PGMEXPL":
        kwargs["num_samples"] = DEFAULT_PGM_NUM_SAMPLES
    epochs = (
        DEFAULT_PG_EXPLAINER_EPOCHS
        if explainer_name == "PGEXPL"
        else DEFAULT_GNN_EXPLAINER_EPOCHS
    )
    return kwargs, epochs


def _explain_all_graphs(
    ctx: ExplanationRunContext,
    explainer_name: str,
    *,
    prediction_baseline: dict[str, PredictionBaselineEntry] | None,
) -> tuple[list[ExplanationResult], float]:
    """Run *explainer_name* over the explain loader; return (results, wall_time_s)."""
    spec = get_spec(explainer_name)
    explainer_kwargs, epochs = _explainer_kwargs(explainer_name, ctx.num_classes)

    print(f"\n--- {explainer_name} ---", flush=True)
    t0 = time.perf_counter()
    results: list[ExplanationResult] = []
    for r in run_explanations(
        ctx.model,
        ctx.explain_loader,
        ctx.device,
        explainer_name=explainer_name,
        explainer_epochs=epochs,
        max_graphs=None,
        get_graph_id=_get_graph_id,
        apply_preprocessing_flag=True,
        correct_class_only=True,
        apply_mask_spread_filter=ctx.apply_mask_spread_filter,
        mask_spread_tolerance=MASK_SPREAD_TOLERANCE,
        train_loader=ctx.train_loader if spec.needs_training else None,
        pg_train_max_graphs=None,
        paper_metrics=True,
        paper_n_thresholds=PAPER_N_THRESHOLDS,
        prediction_baseline=prediction_baseline,
        **explainer_kwargs,
    ):
        results.append(r)
        per_graph_summary = " ".join(
            f"{short}={_fmt(getattr(r, field))}"
            for field, short in _METRIC_FIELDS
        )
        suffix = "" if r.valid else " [excluded]"
        elapsed = f" ({r.elapsed_s:.2f}s)" if r.elapsed_s > 0 else ""
        print(f"  {r.graph_id}: {per_graph_summary}{suffix}{elapsed}", flush=True)
    return results, time.perf_counter() - t0


def _build_report(
    *,
    explainer_name: str,
    seed: int,
    results: list[ExplanationResult],
    wall_time_s: float,
) -> dict[str, Any]:
    """Build the per-explainer ``explanation_report.json`` payload."""
    valid_block = _build_valid_result_metrics(results)
    all_block = _build_result_metrics(results, wall_time_s=wall_time_s)
    run_status, run_status_note = diagnose_explanation_run(results)
    return {
        "explainer": explainer_name,
        "seed": int(seed),
        "run_status": run_status,
        "run_status_note": run_status_note,
        "valid_result_metrics": valid_block,
        "result_metrics": all_block,
        "per_graph": [_per_graph_entry(r) for r in results],
    }


def _save_explainer_outputs(
    *,
    out_path: Path,
    report: dict[str, Any],
    results: list[ExplanationResult],
) -> None:
    """Write ``explanation_report.json`` and per-graph ``masks/<id>.json``."""
    if out_path.exists() and any(out_path.iterdir()):
        print(f"[INFO] Output exists; overwriting under: {out_path}", flush=True)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "explanation_report.json").write_text(
        dumps_strict_json(report, indent=2), encoding="utf-8",
    )

    masks_dir = out_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        mask_data: dict[str, Any] = {}

        ei = r.explanation.edge_index
        if hasattr(ei, "cpu"):
            ei = ei.cpu()
        mask_data["edge_index"] = ei.tolist() if hasattr(ei, "tolist") else ei

        em = getattr(r.explanation, "edge_mask", None)
        if em is not None:
            if hasattr(em, "cpu"):
                em = em.cpu()
            mask_data["edge_mask"] = em.tolist() if hasattr(em, "tolist") else em

        nm = getattr(r.explanation, "node_mask", None)
        if nm is not None:
            if hasattr(nm, "cpu"):
                nm = nm.cpu()
            mask_data["node_mask"] = nm.tolist() if hasattr(nm, "tolist") else nm

        (masks_dir / f"{r.graph_id}.json").write_text(
            dumps_strict_json(mask_data, indent=2), encoding="utf-8",
        )


def run_one_explainer(
    ctx: ExplanationRunContext,
    explainer_name: str,
    *,
    seed: int = DEFAULT_SEED,
    prediction_baseline: dict[str, PredictionBaselineEntry] | None = None,
) -> dict[str, Any]:
    """Run *explainer_name* on the fold and persist its report. Return the report dict."""
    results, wall_time_s = _explain_all_graphs(
        ctx, explainer_name, prediction_baseline=prediction_baseline,
    )
    report = _build_report(
        explainer_name=explainer_name,
        seed=seed,
        results=results,
        wall_time_s=wall_time_s,
    )

    _print_metric_block("valid_result_metrics", report["valid_result_metrics"])
    _print_metric_block("result_metrics", report["result_metrics"])

    n_baseline_mismatch = sum(1 for r in results if r.prediction_baseline_mismatch)
    if n_baseline_mismatch:
        print(
            f"[WARN] {n_baseline_mismatch} graph(s) differed from the precomputed "
            "model prediction baseline.",
            flush=True,
        )
    if report["run_status"] != "ok":
        print(
            f"[WARN] Run status: {report['run_status']} - {report['run_status_note']}",
            flush=True,
        )

    out_path = explanations_run_dir(ctx.explainer_results_root, explainer_name)
    _save_explainer_outputs(out_path=out_path, report=report, results=results)
    print(f"Report and masks saved to {out_path}", flush=True)
    return report


# ----------------------------------------------------------------------------
# Cross-explainer report + baseline I/O
# ----------------------------------------------------------------------------


def write_comparison_report(
    *,
    explainer_results_root: Path,
    fold_index: int,
    fold_metric: str,
    split_name: str,
    explainer_names: list[str],
    all_reports: dict[str, dict],
    seed: int,
) -> Path:
    """Write a cross-explainer comparison JSON for one fold."""
    comparison_path = explainer_results_root / RESULTS_EXPLANATIONS
    comparison_path.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comparison = {
        "generated_at": generated_at,
        "fold_index": fold_index,
        "fold_metric": fold_metric,
        "split": split_name,
        "seed": int(seed),
        "explainers": explainer_names,
        "per_explainer": {
            name: {
                "valid_result_metrics": all_reports[name]["valid_result_metrics"],
                "result_metrics": all_reports[name]["result_metrics"],
                "run_status": all_reports[name]["run_status"],
                "run_status_note": all_reports[name]["run_status_note"],
            }
            for name in explainer_names
        },
        "per_graph_per_explainer": {
            name: all_reports[name]["per_graph"] for name in explainer_names
        },
    }
    json_path = comparison_path / "comparison_report.json"
    json_path.write_text(
        dumps_strict_json(comparison, indent=2), encoding="utf-8",
    )
    return json_path


def write_prediction_baseline(
    *,
    explainer_results_root: Path,
    prediction_baseline: dict[str, PredictionBaselineEntry],
) -> Path:
    """Write the model/fold/split prediction baseline used by all explainers."""
    explanations_path = explainer_results_root / RESULTS_EXPLANATIONS
    explanations_path.mkdir(parents=True, exist_ok=True)
    baseline_path = explanations_path / "model_prediction_baseline.json"
    payload = {
        "entries": [asdict(entry) for entry in prediction_baseline.values()],
        "num_graphs": len(prediction_baseline),
        "num_misclassified": sum(
            1 for entry in prediction_baseline.values() if not entry.correct_class
        ),
    }
    baseline_path.write_text(dumps_strict_json(payload, indent=2), encoding="utf-8")
    return baseline_path


# ----------------------------------------------------------------------------
# Per-fold orchestration
# ----------------------------------------------------------------------------


def _run_fold(args: argparse.Namespace, ctx: ExplanationRunContext) -> None:
    """Run all explainers on one fold and write the comparison report."""
    explainer_names = list(AVAILABLE_EXPLAINERS)

    prediction_baseline = collect_prediction_baseline(
        ctx.model,
        ctx.explain_loader,
        ctx.device,
        get_graph_id=_get_graph_id,
    )
    baseline_path = write_prediction_baseline(
        explainer_results_root=ctx.explainer_results_root,
        prediction_baseline=prediction_baseline,
    )
    print(f"Model prediction baseline: {baseline_path}", flush=True)

    all_reports: dict[str, dict] = {}
    for explainer_name in explainer_names:
        seed_everything(args.seed)
        all_reports[explainer_name] = run_one_explainer(
            ctx,
            explainer_name,
            seed=args.seed,
            prediction_baseline=prediction_baseline,
        )

    json_path = write_comparison_report(
        explainer_results_root=ctx.explainer_results_root,
        fold_index=ctx.fold_index,
        fold_metric=ctx.fold_metric,
        split_name=ctx.split_name,
        explainer_names=explainer_names,
        all_reports=all_reports,
        seed=args.seed,
    )
    print(f"\nComparison report: {json_path}", flush=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    print(
        f"[INFO] Seeded RNGs (torch / numpy / random / PyG) with seed={args.seed}",
        flush=True,
    )

    for name in AVAILABLE_EXPLAINERS:
        validate_explainer(name)

    if args.folds is not None:
        results_root = _GNN_PROJECT_ROOT / RESULTS_DIR_NAME
        num_folds = _read_num_folds_from_summary(results_root)
        fold_indices = resolve_fold_indices(num_folds, fold_indices=args.folds)
        print(
            f"[INFO] Running explanations for {len(fold_indices)} fold(s): "
            f"{fold_indices} (num_folds={num_folds})",
            flush=True,
        )
        for k in fold_indices:
            print(f"\n{'='*60}\n  FOLD {k}\n{'='*60}", flush=True)
            ctx = build_explanation_run_context_for_fold(args, k, num_folds)
            _run_fold(args, ctx)
    else:
        ctx = build_explanation_run_context(args)
        _run_fold(args, ctx)


if __name__ == "__main__":
    main()
