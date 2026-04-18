#!/usr/bin/env python3
"""
Run all explainers on the trained GINE for the single best CV fold (from classify.py or train.py
summaries). Writes per-explainer reports under results/folds/fold_<k>/explanations/ and
comparison_report.json (no HTML).

Usage:
  uv run python scripts/run_explanations.py
  uv run python scripts/run_explanations.py --split validation
  uv run python scripts/run_explanations.py --fold_metric train_accuracy
  uv run python scripts/run_explanations.py --no_mask_spread_filter

GNN results are read from ``mprov3_gine/results`` and the MPro snapshot from the workspace
default directory (see ``DEFAULT_MPRO_SNAPSHOT_DIR_NAME`` in mprov3_gine_explainer_defaults).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_VAL_SPLIT_FILE,
    RESULTS_DATASETS,
    RESULTS_DIR_NAME,
    RESULTS_EXPLANATIONS,
    SplitConfig,
    read_num_folds_for_fold,
    resolve_best_fold_index,
    resolve_checkpoint_path,
)

from mprov3_explainer import (
    AVAILABLE_EXPLAINERS,
    ExplanationResult,
    aggregate_fidelity,
    explanations_run_dir,
    get_device,
    resolve_dataset_dir,
    run_explanations,
    validate_explainer,
)
from mprov3_explainer.explainers import get_spec

PAPER_N_THRESHOLDS = 100
MASK_SPREAD_TOLERANCE = 1e-3


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all explainers on the best fold (test accuracy from classify.py by default). "
            "Uses mprov3_gine/results and the default MPro snapshot path; "
            "PGExplainer still trains on the train split, then explains the split from --split."
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
        "--fold_metric",
        type=str,
        choices=("test_accuracy", "train_accuracy"),
        default="test_accuracy",
        help="Pick fold from classification_summary (test) or training_summary (train).",
    )
    parser.add_argument(
        "--no_mask_spread_filter",
        action="store_true",
        help="Disable τ=1e⁻³ degenerate-mask discard (default: filter on).",
    )
    return parser.parse_args()


def _get_graph_id(data: Any, index: int) -> str:
    return getattr(data, "pdb_id", f"graph_{index}")


def build_explanation_run_context(args: argparse.Namespace) -> ExplanationRunContext:
    """Resolve paths, loaders, best fold, and trained GNN (configuration for all explainers)."""
    results_root = _GNN_PROJECT_ROOT / RESULTS_DIR_NAME
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    k = resolve_best_fold_index(results_root, args.fold_metric)
    num_folds = read_num_folds_for_fold(results_root, k)
    print(
        f"Using fold_index={k} (num_folds={num_folds}, fold_metric={args.fold_metric}, "
        f"split={args.split})",
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


def run_one_explainer(
    ctx: ExplanationRunContext,
    explainer_name: str,
) -> tuple[dict, list[dict]]:
    """Drive pipeline for one explainer: explain → metrics → per-graph JSON + report dict."""
    spec = get_spec(explainer_name)

    explainer_kwargs: dict = {"num_classes": ctx.num_classes}
    if explainer_name in ("IGNODE", "IGEDGE"):
        explainer_kwargs["n_steps"] = DEFAULT_IG_N_STEPS
    if explainer_name == "PGMEXPL":
        explainer_kwargs["num_samples"] = DEFAULT_PGM_NUM_SAMPLES

    epochs_for_builder = DEFAULT_GNN_EXPLAINER_EPOCHS
    if explainer_name == "PGEXPL":
        epochs_for_builder = DEFAULT_PG_EXPLAINER_EPOCHS

    print(f"\n--- {explainer_name} ---", flush=True)
    t0 = time.perf_counter()
    results: list[ExplanationResult] = []
    for result in run_explanations(
        ctx.model,
        ctx.explain_loader,
        ctx.device,
        explainer_name=explainer_name,
        explainer_epochs=epochs_for_builder,
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
        **explainer_kwargs,
    ):
        results.append(result)
        print(
            f"  {result.graph_id}: fid+={result.fidelity_fid_plus:.4f} "
            f"fid-={result.fidelity_fid_minus:.4f}"
            f" char={result.pyg_characterization:.4f}"
            f" Fsuf={result.paper_sufficiency:.4f}"
            f" Fcom={result.paper_comprehensiveness:.4f}"
            f" Ff1={result.paper_f1_fidelity:.4f}"
            + ("" if result.valid else " [excluded]")
            + (f" ({result.elapsed_s:.2f}s)" if result.elapsed_s > 0 else "")
        )
    wall_time = time.perf_counter() - t0

    mean_fid_plus, mean_fid_minus = aggregate_fidelity(results, valid_only=False)
    mean_char = sum(r.pyg_characterization for r in results) / len(results) if results else 0.0
    mean_fsuf = sum(r.paper_sufficiency for r in results) / len(results) if results else 0.0
    mean_fcom = sum(r.paper_comprehensiveness for r in results) / len(results) if results else 0.0
    mean_ff1 = sum(r.paper_f1_fidelity for r in results) / len(results) if results else 0.0
    n_valid = sum(1 for r in results if r.valid)
    print(f"Mean fidelity (fid+): {mean_fid_plus:.4f}")
    print(f"Mean fidelity (fid-): {mean_fid_minus:.4f}")
    print(f"Mean characterization (PyG): {mean_char:.4f}")
    print(f"Mean paper sufficiency (Fsuf): {mean_fsuf:.4f}")
    print(f"Mean paper comprehensiveness (Fcom): {mean_fcom:.4f}")
    print(f"Mean paper F1-fidelity (Ff1): {mean_ff1:.4f}")
    print(f"Explained {len(results)} graphs ({n_valid} valid) in {wall_time:.1f}s.")

    out_path = explanations_run_dir(ctx.explainer_results_root, explainer_name)
    if out_path.exists() and any(out_path.iterdir()):
        print(f"[INFO] Output exists; overwriting under: {out_path}", flush=True)
    out_path.mkdir(parents=True, exist_ok=True)

    per_graph_entries = []
    for r in results:
        per_graph_entries.append({
            "graph_id": r.graph_id,
            "fidelity_plus": r.fidelity_fid_plus,
            "fidelity_minus": r.fidelity_fid_minus,
            "pyg_characterization": r.pyg_characterization,
            "paper_sufficiency": r.paper_sufficiency,
            "paper_comprehensiveness": r.paper_comprehensiveness,
            "paper_f1_fidelity": r.paper_f1_fidelity,
            "valid": r.valid,
            "correct_class": r.correct_class,
            "has_node_mask": r.has_node_mask,
            "has_edge_mask": r.has_edge_mask,
            "elapsed_s": r.elapsed_s,
        })

    report = {
        "mean_fidelity_plus": mean_fid_plus,
        "mean_fidelity_minus": mean_fid_minus,
        "mean_pyg_characterization": mean_char,
        "mean_paper_sufficiency": mean_fsuf,
        "mean_paper_comprehensiveness": mean_fcom,
        "mean_paper_f1_fidelity": mean_ff1,
        "num_graphs": len(results),
        "num_valid": n_valid,
        "explainer": explainer_name,
        "wall_time_s": wall_time,
        "per_graph": per_graph_entries,
    }
    (out_path / "explanation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
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
            json.dumps(mask_data, indent=2), encoding="utf-8",
        )
    print(f"Report and masks saved to {out_path}")

    summary = {
        "mean_fid_plus": mean_fid_plus,
        "mean_fid_minus": mean_fid_minus,
        "mean_pyg_characterization": mean_char,
        "mean_paper_sufficiency": mean_fsuf,
        "mean_paper_comprehensiveness": mean_fcom,
        "mean_paper_f1_fidelity": mean_ff1,
        "num_graphs": len(results),
        "num_valid": n_valid,
        "wall_time_s": wall_time,
    }
    return summary, per_graph_entries


def write_comparison_report(
    *,
    explainer_results_root: Path,
    fold_index: int,
    fold_metric: str,
    split_name: str,
    explainer_names: list[str],
    all_summaries: dict[str, dict],
    all_per_graph: dict[str, list[dict]],
) -> Path:
    """Write cross-explainer comparison JSON (Longa-style aggregation over the split)."""
    comparison_path = explainer_results_root / RESULTS_EXPLANATIONS
    comparison_path.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comparison = {
        "generated_at": generated_at,
        "fold_index": fold_index,
        "fold_metric": fold_metric,
        "split": split_name,
        "explainers": explainer_names,
        "per_explainer": all_summaries,
        "per_graph_per_explainer": all_per_graph,
    }
    json_path = comparison_path / "comparison_report.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return json_path


def main() -> None:
    args = parse_args()

    for name in AVAILABLE_EXPLAINERS:
        validate_explainer(name)

    ctx = build_explanation_run_context(args)
    explainer_names = list(AVAILABLE_EXPLAINERS)

    all_summaries: dict[str, dict] = {}
    all_per_graph: dict[str, list[dict]] = {}

    for explainer_name in explainer_names:
        summary, per_graph = run_one_explainer(ctx, explainer_name)
        all_summaries[explainer_name] = summary
        all_per_graph[explainer_name] = per_graph

    json_path = write_comparison_report(
        explainer_results_root=ctx.explainer_results_root,
        fold_index=ctx.fold_index,
        fold_metric=ctx.fold_metric,
        split_name=ctx.split_name,
        explainer_names=explainer_names,
        all_summaries=all_summaries,
        all_per_graph=all_per_graph,
    )
    print(f"\nComparison report: {json_path}")


if __name__ == "__main__":
    main()
