#!/usr/bin/env python3
"""
Run all explainers on the trained GINE for the single best CV fold (from classify.py or train.py
summaries). Writes per-explainer reports under results/folds/fold_<k>/explanations/ and
comparison_report.json (no HTML).

Usage:
  uv run python scripts/run_explanations.py
  uv run python scripts/run_explanations.py --fold_metric train_accuracy
  uv run python scripts/run_explanations.py --no_mask_spread_filter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    aggregate_fidelity,
    explanations_run_dir,
    get_device,
    resolve_dataset_dir,
    run_explanations,
    validate_explainer,
)
from mprov3_explainer.explainers import get_spec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all explainers on the best fold (test accuracy from classify.py by default)."
        ),
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="mprov3_gine results (trainings/, classifications/, datasets/).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Raw MPro snapshot (Splits/).",
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


def _get_graph_id(data, index: int) -> str:
    return getattr(data, "pdb_id", f"graph_{index}")


def main() -> None:
    args = _parse_args()

    for name in AVAILABLE_EXPLAINERS:
        validate_explainer(name)

    results_root = Path(args.results_root) if args.results_root else _GNN_PROJECT_ROOT / RESULTS_DIR_NAME
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    k = resolve_best_fold_index(results_root, args.fold_metric)
    num_folds = read_num_folds_for_fold(results_root, k)
    print(
        f"Using fold_index={k} (num_folds={num_folds}, fold_metric={args.fold_metric})",
        flush=True,
    )

    data_root = Path(args.data_root) if args.data_root else _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
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
    train_loader, _, test_loader = create_data_loaders(
        dataset_base, data_root, split_config, batch_size=DEFAULT_BATCH_SIZE,
    )

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

    apply_spread = not args.no_mask_spread_filter
    explainer_names = list(AVAILABLE_EXPLAINERS)

    all_summaries: dict[str, dict] = {}
    all_per_graph: dict[str, list[dict]] = {}

    for explainer_name in explainer_names:
        spec = get_spec(explainer_name)

        explainer_kwargs: dict = {"num_classes": DEFAULT_OUT_CLASSES}
        if explainer_name in ("IGNODE", "IGEDGE"):
            explainer_kwargs["n_steps"] = DEFAULT_IG_N_STEPS
        if explainer_name == "PGMEXPL":
            explainer_kwargs["num_samples"] = DEFAULT_PGM_NUM_SAMPLES

        epochs_for_builder = DEFAULT_GNN_EXPLAINER_EPOCHS
        if explainer_name == "PGEXPL":
            epochs_for_builder = DEFAULT_PG_EXPLAINER_EPOCHS

        print(f"\n--- {explainer_name} ---", flush=True)
        t0 = time.perf_counter()
        results: list = []
        for result in run_explanations(
            model,
            test_loader,
            device,
            explainer_name=explainer_name,
            explainer_epochs=epochs_for_builder,
            max_graphs=None,
            get_graph_id=_get_graph_id,
            apply_preprocessing_flag=True,
            correct_class_only=True,
            apply_mask_spread_filter=apply_spread,
            mask_spread_tolerance=1e-3,
            train_loader=train_loader if spec.needs_training else None,
            pg_train_max_graphs=None,
            paper_metrics=True,
            paper_n_thresholds=100,
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

        out_path = explanations_run_dir(explainer_results_root, explainer_name)
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

        all_summaries[explainer_name] = {
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
        all_per_graph[explainer_name] = per_graph_entries

    comparison_path = explainer_results_root / RESULTS_EXPLANATIONS
    comparison_path.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comparison = {
        "generated_at": generated_at,
        "fold_index": k,
        "fold_metric": args.fold_metric,
        "explainers": explainer_names,
        "per_explainer": all_summaries,
        "per_graph_per_explainer": all_per_graph,
    }
    json_path = comparison_path / "comparison_report.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nComparison report: {json_path}")


if __name__ == "__main__":
    main()
