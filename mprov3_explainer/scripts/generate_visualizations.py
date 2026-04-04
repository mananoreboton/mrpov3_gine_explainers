#!/usr/bin/env python3
"""
Generate visualization report from a previous explanation run.
Reads explanation_report.json and masks from results/explanations/<explainer>/,
draws 2D molecules with bond/atom coloring, writes index.html per explainer and a
cross-explainer comparison.html under results/visualizations/.

Usage:
  uv run python scripts/generate_visualizations.py
  uv run python scripts/generate_visualizations.py --explainers GNNEXPL GRADEXPINODE
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
    RESULTS_VISUALIZATIONS,
)

from mprov3_explainer import (
    AVAILABLE_EXPLAINERS,
    explanations_run_dir,
    validate_explainer,
    visualizations_run_dir,
)
from mprov3_explainer.visualize import (
    draw_molecule_with_mask,
    write_comparison_index_html,
    write_explanation_index_html,
)


def _explainers_present(explanations_base: Path) -> list[str]:
    """
    Registered explainer names that have a completed run (report + masks/).
    Used when --explainers is omitted so partial runs do not spam skips.
    """
    present: list[str] = []
    if not explanations_base.is_dir():
        return present
    for sub in sorted(explanations_base.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name not in AVAILABLE_EXPLAINERS:
            continue
        if not (sub / "explanation_report.json").is_file():
            continue
        if not (sub / "masks").is_dir():
            continue
        present.append(name)
    order = {n: i for i, n in enumerate(AVAILABLE_EXPLAINERS)}
    present.sort(key=lambda n: (order.get(n, 999), n))
    return present


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualizations from a previous explanation run.",
    )
    parser.add_argument(
        "--explainer", type=str, default=None,
        help="Single explainer to visualize. Ignored if --explainers is set.",
    )
    parser.add_argument(
        "--explainers", type=str, nargs="*", default=None,
        help=(
            "Explainers to visualize. If omitted, discover explainer folders "
            "under results/explanations/ (recommended for partial runs)."
        ),
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    results_root = Path(args.results_root or _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME)
    explanations_base = results_root / RESULTS_EXPLANATIONS
    if not explanations_base.exists():
        raise FileNotFoundError(
            f"Explanations folder not found: {explanations_base}. Run run_explanations.py first."
        )

    if args.explainers:
        explainer_names = list(args.explainers)
    elif args.explainer is not None:
        explainer_names = [args.explainer]
    else:
        explainer_names = _explainers_present(explanations_base)
        if not explainer_names:
            raise FileNotFoundError(
                f"No explainer outputs found under {explanations_base} "
                f"(need explanation_report.json and masks/ per explainer). "
                f"Run run_explanations.py first, or pass --explainers explicitly."
            )
        print(
            f"Using explainers present: {', '.join(explainer_names)}",
            flush=True,
        )

    for name in explainer_names:
        validate_explainer(name)

    data_root = Path(args.data_root) if args.data_root else _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")

    # For cross-explainer comparison grid
    comparison_data: dict = {
        "explainers": [],
        "graph_ids": [],
        "per_explainer": {},
        "grid": {},
    }
    all_graph_ids: set[str] = set()

    for explainer_name in explainer_names:
        explanation_dir = explanations_run_dir(results_root, explainer_name)
        if not explanation_dir.is_dir():
            print(f"  Skip {explainer_name}: no folder at {explanation_dir}")
            continue

        report_path = explanation_dir / "explanation_report.json"
        if not report_path.exists():
            print(f"  Skip {explainer_name}: report not found at {report_path}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))

        vis_out = visualizations_run_dir(results_root, explainer_name)
        if vis_out.exists() and any(vis_out.iterdir()):
            print(f"[INFO] Output exists; overwriting under: {vis_out}", flush=True)
        graphs_dir = vis_out / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        masks_dir = explanation_dir / "masks"
        if not masks_dir.exists():
            print(f"  Skip {explainer_name}: masks folder not found at {masks_dir}")
            continue

        report["explainer"] = explainer_name
        drawn = 0

        comparison_data["explainers"].append(explainer_name)
        comparison_data["per_explainer"][explainer_name] = {
            "mean_fid_plus": report.get("mean_fidelity_plus", 0.0),
            "mean_fid_minus": report.get("mean_fidelity_minus", 0.0),
            "mean_pyg_characterization": report.get("mean_pyg_characterization", 0.0),
            "mean_paper_sufficiency": report.get("mean_paper_sufficiency", 0.0),
            "mean_paper_comprehensiveness": report.get("mean_paper_comprehensiveness", 0.0),
            "mean_paper_f1_fidelity": report.get("mean_paper_f1_fidelity", 0.0),
        }

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
            node_mask = mask_data.get("node_mask")

            if edge_index is None and node_mask is None:
                print(f"    Skip {graph_id}: no mask data")
                continue

            sdf_path = sdf_dir / f"{graph_id}_ligand.sdf"
            out_png = graphs_dir / f"mask_{graph_id}.png"
            if draw_molecule_with_mask(
                sdf_path,
                edge_index=edge_index,
                edge_mask=edge_mask,
                out_path_png=out_png,
                node_mask=node_mask,
            ):
                drawn += 1
                print(f"  {explainer_name} / {graph_id} -> {out_png.name}")

            all_graph_ids.add(graph_id)
            if graph_id not in comparison_data["grid"]:
                comparison_data["grid"][graph_id] = {}
            comparison_data["grid"][graph_id][explainer_name] = {
                "img": f"{explainer_name}/graphs/mask_{graph_id}.png",
                "fid_plus": e.get("fidelity_plus", 0.0),
                "fid_minus": e.get("fidelity_minus", 0.0),
                "paper_f1_fidelity": e.get("paper_f1_fidelity", 0.0),
            }

        write_explanation_index_html(vis_out, report)
        print(f"\n{explainer_name}: visualizations written to {vis_out}")
        print(f"  Index: {vis_out / 'index.html'}")
        print(f"  Images: {drawn} in {graphs_dir}")

    # Write cross-explainer comparison page
    comparison_data["graph_ids"] = sorted(all_graph_ids)
    vis_root = results_root / RESULTS_VISUALIZATIONS
    vis_root.mkdir(parents=True, exist_ok=True)
    cmp_html = vis_root / "comparison.html"
    if cmp_html.exists():
        print(f"[INFO] Output exists; overwriting: {cmp_html}", flush=True)
    write_comparison_index_html(vis_root, comparison_data)
    print(f"\nComparison page: {cmp_html}")


if __name__ == "__main__":
    main()
