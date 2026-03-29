#!/usr/bin/env python3
"""
Run explainer(s) on the trained GINE: load model and dataset from mprov3_gine/results,
generate graph-level explanations, compute fidelity.
Outputs go to results/explanations/<timestamp>/<explainer>/.

After all explainers run, writes a comparison_report.json at
results/explanations/<timestamp>/.

Usage:
  uv run python scripts/run_explanations.py
  uv run python scripts/run_explanations.py --explainers GNNEXPL GRADEXPINODE
  uv run python scripts/run_explanations.py --max_graphs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MPROV3_EXPLAINER_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _MPROV3_EXPLAINER_ROOT.parent
_GNN_PROJECT_ROOT = _REPO_ROOT / "mprov3_gine"
if _GNN_PROJECT_ROOT.exists() and str(_GNN_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_PROJECT_ROOT))

import torch
from mprov3_gine_explainer_defaults import (
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
    SplitConfig,
)

from mprov3_explainer import (
    AVAILABLE_EXPLAINERS,
    aggregate_fidelity,
    explanations_run_dir,
    get_device,
    resolve_checkpoint_path,
    resolve_dataset_dir,
    run_explanations,
    run_timestamp,
    validate_explainer,
)
from mprov3_explainer.explainers import get_spec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explainer(s) on trained GINE; produce comparison report.",
    )
    parser.add_argument(
        "--explainer",
        type=str,
        default=None,
        help="Single explainer to run. Ignored if --explainers is set.",
    )
    parser.add_argument(
        "--explainers",
        type=str,
        nargs="*",
        default=None,
        help=f"Explainers to run (default: all: {AVAILABLE_EXPLAINERS}).",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Path to mprov3_gine/results (default: ../mprov3_gine/results).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Splits/).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_TRAINING_CHECKPOINT_FILENAME,
        help=f"Checkpoint filename (default: {DEFAULT_TRAINING_CHECKPOINT_FILENAME}).",
    )
    parser.add_argument(
        "--train_split_file", type=str, default=DEFAULT_TRAIN_SPLIT_FILE,
    )
    parser.add_argument(
        "--val_split_file", type=str, default=DEFAULT_VAL_SPLIT_FILE,
    )
    parser.add_argument(
        "--test_split_file", type=str, default=DEFAULT_TEST_SPLIT_FILE,
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_OUT_CLASSES)

    # Per-explainer hyperparameters
    parser.add_argument(
        "--explainer_epochs", type=int, default=DEFAULT_GNN_EXPLAINER_EPOCHS,
        help="Optimisation epochs for GNNExplainer (per graph).",
    )
    parser.add_argument(
        "--pg_explainer_epochs", type=int, default=DEFAULT_PG_EXPLAINER_EPOCHS,
        help="Training epochs for PGExplainer MLP.",
    )
    parser.add_argument(
        "--pg_train_max_graphs",
        type=int,
        default=None,
        help=(
            "Max training graphs per epoch for PGExplainer (default: full train loader). "
            "When --max_graphs is set and this is omitted, a subsample cap is applied so "
            "PG training is not orders of magnitude slower than test evaluation."
        ),
    )
    parser.add_argument(
        "--ig_n_steps", type=int, default=DEFAULT_IG_N_STEPS,
        help="Integrated Gradients interpolation steps.",
    )
    parser.add_argument(
        "--pgm_num_samples", type=int, default=DEFAULT_PGM_NUM_SAMPLES,
        help="PGMExplainer perturbation samples.",
    )

    parser.add_argument("--max_graphs", type=int, default=None)

    # Preprocessing
    parser.add_argument("--no_preprocessing", action="store_true")
    parser.add_argument(
        "--no_correct_class_only",
        action="store_true",
        help="Include misclassified graphs in preprocessing validity (default: only correctly classified).",
    )
    parser.add_argument("--fidelity_valid_only", action="store_true")

    # Paper metrics (Longa et al.): threshold-sweep Fsuf / Fcom / Ff1
    parser.set_defaults(paper_metrics=True)
    parser.add_argument(
        "--paper_metrics",
        action="store_true",
        help="Compute paper sufficiency / comprehensiveness / F1-fidelity (default: on).",
    )
    parser.add_argument(
        "--no_paper_metrics",
        action="store_false",
        dest="paper_metrics",
        help="Skip paper metrics (faster; PyG fid+ / fid- / characterization still computed).",
    )
    parser.add_argument(
        "--paper_n_thresholds",
        type=int,
        default=100,
        help="Number of threshold levels Nt for paper metrics sweep (default: 100, paper uses k=1..Nt-1).",
    )

    return parser.parse_args()


def _get_graph_id(data, index: int) -> str:
    return getattr(data, "pdb_id", f"graph_{index}")


def main() -> None:
    args = _parse_args()

    explainer_names: list[str] = (
        args.explainers
        if args.explainers
        else ([args.explainer] if args.explainer is not None else list(AVAILABLE_EXPLAINERS))
    )
    for name in explainer_names:
        validate_explainer(name)

    # Resolve paths
    results_root = Path(args.results_root) if args.results_root else _GNN_PROJECT_ROOT / RESULTS_DIR_NAME
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    data_root = Path(args.data_root) if args.data_root else _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    checkpoint_path = resolve_checkpoint_path(results_root, args.checkpoint)
    dataset_dir = resolve_dataset_dir(results_root)
    dataset_name = dataset_dir.name
    dataset_base = results_root / RESULTS_DATASETS

    from loaders import create_data_loaders
    from model import MProGNN

    split_config = SplitConfig(
        train_file=args.train_split_file,
        val_file=args.val_split_file,
        test_file=args.test_split_file,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        dataset_name=dataset_name,
    )
    train_loader, _, test_loader = create_data_loaders(
        dataset_base, data_root, split_config, batch_size=args.batch_size,
    )

    device = get_device()
    model = MProGNN(
        in_channels=DEFAULT_IN_CHANNELS,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_classes=args.num_classes,
        pool=DEFAULT_POOL,
        edge_dim=DEFAULT_EDGE_DIM,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )
    model.eval()

    explainer_results_root = _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME
    ts = run_timestamp()

    # Collect per-explainer summaries for the comparison report
    all_summaries: dict[str, dict] = {}
    all_per_graph: dict[str, list[dict]] = {}

    for explainer_name in explainer_names:
        spec = get_spec(explainer_name)

        # Resolve per-explainer kwargs
        explainer_kwargs: dict = {"num_classes": args.num_classes}
        if explainer_name in ("IGNODE", "IGEDGE"):
            explainer_kwargs["n_steps"] = args.ig_n_steps
        if explainer_name == "PGMEXPL":
            explainer_kwargs["num_samples"] = args.pgm_num_samples

        epochs_for_builder = args.explainer_epochs
        if explainer_name == "PGEXPL":
            epochs_for_builder = args.pg_explainer_epochs

        pg_train_cap: int | None = args.pg_train_max_graphs
        if (
            explainer_name == "PGEXPL"
            and pg_train_cap is None
            and args.max_graphs is not None
        ):
            pg_train_cap = min(2048, max(256, args.max_graphs * 128))

        print(f"\n--- {explainer_name} ---")
        t0 = time.perf_counter()
        results: list = []
        for result in run_explanations(
            model,
            test_loader,
            device,
            explainer_name=explainer_name,
            explainer_epochs=epochs_for_builder,
            max_graphs=args.max_graphs,
            get_graph_id=_get_graph_id,
            apply_preprocessing_flag=not args.no_preprocessing,
            correct_class_only=not args.no_correct_class_only,
            train_loader=train_loader if spec.needs_training else None,
            pg_train_max_graphs=pg_train_cap if explainer_name == "PGEXPL" else None,
            paper_metrics=args.paper_metrics,
            paper_n_thresholds=args.paper_n_thresholds,
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

        mean_fid_plus, mean_fid_minus = aggregate_fidelity(
            results, valid_only=args.fidelity_valid_only,
        )
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

        # Save per-explainer report + masks
        out_path = explanations_run_dir(explainer_results_root, ts, explainer_name)
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

    # Write comparison report
    comparison_path = explainer_results_root / "explanations" / ts
    comparison_path.mkdir(parents=True, exist_ok=True)
    comparison = {
        "timestamp": ts,
        "explainers": explainer_names,
        "per_explainer": all_summaries,
        "per_graph_per_explainer": all_per_graph,
    }
    (comparison_path / "comparison_report.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8",
    )
    print(f"\nComparison report: {comparison_path / 'comparison_report.json'}")
    print(f"Run timestamp: {ts}")


if __name__ == "__main__":
    main()
