#!/usr/bin/env python3
"""
Generate visualization report from a previous explanation run: read explanation_report.json
and masks from results/explanations/<timestamp>/<explainer>/, draw 2D molecules with bond coloring,
write index.html and graphs under results/visualizations/<new_timestamp>/<explainer>/.

If --timestamp is not passed, uses the latest explanation run folder.
Usage:
  uv run python scripts/generate_visualizations.py
  uv run python scripts/generate_visualizations.py --explainer GNNExplainer
  uv run python scripts/generate_visualizations.py --explainers GNNExplainer SubgraphX [--timestamp 2026-03-15_133711]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MPROV3_EXPLAINER_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _MPROV3_EXPLAINER_ROOT.parent
_GNN_PROJECT_ROOT = _REPO_ROOT / "mprov3_gine"
if _GNN_PROJECT_ROOT.exists() and str(_GNN_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_PROJECT_ROOT))

from mprov3_gine_explainer_defaults import (
    DEFAULT_MPRO_SNAPSHOT_DIR_NAME,
    MPRO_LIGAND_DIR,
    MPRO_LIGAND_SDF_SUBDIR,
    RESULTS_DIR_NAME,
    RESULTS_EXPLANATIONS,
)

from mprov3_explainer import (
    AVAILABLE_EXPLAINERS,
    explanations_run_dir,
    run_timestamp,
    validate_explainer,
    visualizations_run_dir,
)
from mprov3_explainer.paths import get_latest_timestamp_dir
from mprov3_explainer.visualize import draw_molecule_with_mask, write_explanation_index_html


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate index and graphic explanations from a previous explanation run.",
    )
    parser.add_argument(
        "--explainer",
        type=str,
        default=None,
        help="Single explainer to visualize. Ignored if --explainers is set.",
    )
    parser.add_argument(
        "--explainers",
        type=str,
        nargs="*",
        default=None,
        help=f"Explainers to visualize (default: all: {AVAILABLE_EXPLAINERS}).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Explanation run folder name under results/explanations/ (default: latest).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Ligand/Ligand_SDF/); default from gnn config.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Root for results (default: mprov3_explainer/results).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    explainer_names = (
        args.explainers
        if args.explainers
        else ([args.explainer] if args.explainer is not None else AVAILABLE_EXPLAINERS)
    )
    for name in explainer_names:
        validate_explainer(name)

    results_root = Path(args.results_root or _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME)
    explanations_base = results_root / RESULTS_EXPLANATIONS
    if not explanations_base.exists():
        raise FileNotFoundError(
            f"Explanations folder not found: {explanations_base}. Run run_explanations.py first."
        )

    if args.timestamp:
        timestamp_dir = explanations_base / args.timestamp
        if not timestamp_dir.is_dir():
            raise FileNotFoundError(f"Explanation run not found: {timestamp_dir}")
        ts = args.timestamp
    else:
        timestamp_dir = get_latest_timestamp_dir(explanations_base)
        if timestamp_dir is None:
            raise FileNotFoundError(
                f"No timestamped run found under {explanations_base}. Run run_explanations.py first."
            )
        ts = timestamp_dir.name

    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        data_root = _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")

    new_ts = run_timestamp()

    for explainer_name in explainer_names:
        explanation_dir = explanations_run_dir(results_root, ts, explainer_name)
        if not explanation_dir.is_dir():
            print(f"  Skip {explainer_name}: no folder at {explanation_dir}")
            continue

        report_path = explanation_dir / "explanation_report.json"
        if not report_path.exists():
            print(f"  Skip {explainer_name}: report not found at {report_path}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))

        vis_out = visualizations_run_dir(results_root, new_ts, explainer_name)
        graphs_dir = vis_out / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        masks_dir = explanation_dir / "masks"
        if not masks_dir.exists():
            print(f"  Skip {explainer_name}: masks folder not found at {masks_dir}")
            continue

        report["source_explanation_timestamp"] = ts
        report["explainer"] = explainer_name
        drawn = 0
        for e in report.get("per_graph", []):
            graph_id = e.get("graph_id", "")
            if not graph_id:
                continue
            mask_path = masks_dir / f"{graph_id}.json"
            if not mask_path.exists():
                print(f"    Skip {graph_id}: mask file not found")
                continue
            mask_data = json.loads(mask_path.read_text(encoding="utf-8"))
            edge_index = mask_data.get("edge_index")
            edge_mask = mask_data.get("edge_mask")
            if edge_index is None or edge_mask is None:
                print(f"    Skip {graph_id}: missing edge_index or edge_mask")
                continue
            sdf_path = sdf_dir / f"{graph_id}_ligand.sdf"
            out_png = graphs_dir / f"mask_{graph_id}.png"
            if draw_molecule_with_mask(sdf_path, edge_index, edge_mask, out_png):
                drawn += 1
                print(f"  {explainer_name} / {graph_id} -> {out_png.name}")

        write_explanation_index_html(vis_out, report)
        print(f"\n{explainer_name}: visualizations written to {vis_out}")
        print(f"  Index: {vis_out / 'index.html'}")
        print(f"  Images: {drawn} in {graphs_dir}")

    print(f"\nRun timestamp: {new_ts}")


if __name__ == "__main__":
    main()
