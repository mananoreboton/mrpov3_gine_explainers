#!/usr/bin/env python3
"""
Run explainer(s) on the trained GINE: load model and dataset from mprov3_gine/results,
generate graph-level explanations, compute fidelity (and AUROC when ground truth is available).
Outputs go to results/explanations/<timestamp>/<explainer>/.
Usage:
  uv run python scripts/run_explanations.py
  uv run python scripts/run_explanations.py --explainer GNNExplainer
  uv run python scripts/run_explanations.py --explainers GNNExplainer SubgraphX
  uv run python scripts/run_explanations.py [--results_root ...] [--data_root ...] [--max_graphs 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add mprov3_gine to path so we can import model, loaders, config (before other local imports that need it)
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
    DEFAULT_IN_CHANNELS,
    DEFAULT_MPRO_SNAPSHOT_DIR_NAME,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explainer(s) on trained GINE; use data from mprov3_gine/results.",
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
        help="Path to mprov3_gine/results (default: ../mprov3_gine/results from project root).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Splits/); default from gnn config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_TRAINING_CHECKPOINT_FILENAME,
        help=f"Checkpoint filename (default: {DEFAULT_TRAINING_CHECKPOINT_FILENAME}).",
    )
    parser.add_argument(
        "--train_split_file",
        type=str,
        default=DEFAULT_TRAIN_SPLIT_FILE,
        help=f"Train split file in Splits/ (default: {DEFAULT_TRAIN_SPLIT_FILE}).",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=f"Val split file in Splits/ (default: {DEFAULT_VAL_SPLIT_FILE}).",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=f"Test split file in Splits/ (default: {DEFAULT_TEST_SPLIT_FILE}).",
    )
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--fold_index", type=int, default=0, help="Fold to use (0 .. num_folds-1).")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Loader batch size (1 recommended for explanation).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=DEFAULT_HIDDEN_CHANNELS,
        help="Must match trained model.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Must match trained model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Must match trained model.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_OUT_CLASSES,
        help="Number of classes.",
    )
    parser.add_argument(
        "--explainer_epochs",
        type=int,
        default=DEFAULT_GNN_EXPLAINER_EPOCHS,
        help="Explainer optimization epochs per graph (e.g. GNNExplainer).",
    )
    parser.add_argument(
        "--max_graphs",
        type=int,
        default=None,
        help="Limit number of graphs to explain (default: all test set).",
    )
    # Preprocessing (Longa et al. protocol)
    parser.add_argument(
        "--no_preprocessing",
        action="store_true",
        help="Disable preprocessing (conversion, filtering, normalization) before metrics.",
    )
    parser.add_argument(
        "--no_correct_class_only",
        action="store_true",
        help="Include misclassified instances in metric averaging (default: only correct-class).",
    )
    parser.add_argument(
        "--min_mask_range",
        type=float,
        default=1e-3,
        help="Min edge_mask range to keep instance (discard nearly constant masks).",
    )
    parser.add_argument(
        "--fidelity_valid_only",
        action="store_true",
        help="Report mean fidelity only over valid instances (after preprocessing filters).",
    )
    # SubgraphX (DIG) options
    parser.add_argument(
        "--subgraphx_rollout",
        type=int,
        default=10,
        help="SubgraphX: MCTS rollouts per graph (default: 10).",
    )
    parser.add_argument(
        "--subgraphx_max_nodes",
        type=int,
        default=10,
        help="SubgraphX: max nodes in explanation subgraph (default: 10).",
    )
    parser.add_argument(
        "--subgraphx_sample_num",
        type=int,
        default=100,
        help="SubgraphX: Monte Carlo samples for Shapley (default: 100). Lower is faster, noisier.",
    )
    return parser.parse_args()


def _get_graph_id(data, index: int) -> str:
    return getattr(data, "pdb_id", f"graph_{index}")


def main() -> None:
    args = _parse_args()

    explainer_names = (
        args.explainers
        if args.explainers
        else ([args.explainer] if args.explainer is not None else AVAILABLE_EXPLAINERS)
    )
    for name in explainer_names:
        validate_explainer(name)

    # Resolve results_root (GNN project)
    if args.results_root is not None:
        results_root = Path(args.results_root)
    else:
        results_root = _GNN_PROJECT_ROOT / RESULTS_DIR_NAME
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    # Data root (for Splits)
    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        data_root = _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
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
    _, _, test_loader = create_data_loaders(
        dataset_base,
        data_root,
        split_config,
        batch_size=args.batch_size,
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

    explainer_kwargs = {
        "num_classes": args.num_classes,
        "subgraphx_rollout": args.subgraphx_rollout,
        "subgraphx_max_nodes": args.subgraphx_max_nodes,
        "subgraphx_sample_num": args.subgraphx_sample_num,
    }

    for explainer_name in explainer_names:
        print(f"\n--- {explainer_name} ---")
        results: list = []
        for result in run_explanations(
            model,
            test_loader,
            device,
            explainer_name=explainer_name,
            explainer_epochs=args.explainer_epochs,
            max_graphs=args.max_graphs,
            get_graph_id=_get_graph_id,
            apply_preprocessing_flag=not args.no_preprocessing,
            correct_class_only=not args.no_correct_class_only,
            min_mask_range=args.min_mask_range,
            **explainer_kwargs,
        ):
            results.append(result)
            print(
                f"  {result.graph_id}: fid+={result.fidelity_fid_plus:.4f} fid-={result.fidelity_fid_minus:.4f}"
                + (f" auroc={result.auroc:.4f}" if result.auroc is not None else "")
                + ("" if result.valid else " [excluded]")
            )

        mean_fid_plus, mean_fid_minus = aggregate_fidelity(
            results, valid_only=args.fidelity_valid_only
        )
        n_valid = sum(1 for r in results if r.valid)
        print(f"Mean fidelity (fid+): {mean_fid_plus:.4f}")
        print(f"Mean fidelity (fid-): {mean_fid_minus:.4f}")
        print(f"Explained {len(results)} graphs ({n_valid} valid for protocol).")

        out_path = explanations_run_dir(explainer_results_root, ts, explainer_name)
        out_path.mkdir(parents=True, exist_ok=True)
        report = {
            "mean_fidelity_plus": mean_fid_plus,
            "mean_fidelity_minus": mean_fid_minus,
            "num_graphs": len(results),
            "num_valid": n_valid,
            "explainer": explainer_name,
            "per_graph": [
                {
                    "graph_id": r.graph_id,
                    "fidelity_plus": r.fidelity_fid_plus,
                    "fidelity_minus": r.fidelity_fid_minus,
                    "auroc": r.auroc,
                    "valid": r.valid,
                    "correct_class": r.correct_class,
                }
                for r in results
            ],
        }
        (out_path / "explanation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        masks_dir = out_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            ei = r.explanation.edge_index
            em = r.explanation.edge_mask
            if hasattr(ei, "cpu"):
                ei = ei.cpu()
            if hasattr(em, "cpu"):
                em = em.cpu()
            mask_data = {
                "edge_index": ei.tolist() if hasattr(ei, "tolist") else ei,
                "edge_mask": em.tolist() if hasattr(em, "tolist") else em,
            }
            (masks_dir / f"{r.graph_id}.json").write_text(
                json.dumps(mask_data, indent=2), encoding="utf-8"
            )
        print(f"Report and masks saved to {out_path}")

    print(f"\nRun timestamp: {ts}")


if __name__ == "__main__":
    main()
