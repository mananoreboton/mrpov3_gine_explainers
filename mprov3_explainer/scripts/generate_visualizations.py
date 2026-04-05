#!/usr/bin/env python3
"""
Draw RDKit PNGs from saved explanation masks (single fold under results/folds/fold_*/).
No HTML, no HTTP server, no multi-fold report.

Usage:
  uv run python scripts/generate_visualizations.py
  uv run python scripts/generate_visualizations.py --results_root /path/to/mprov3_explainer/results
"""

from __future__ import annotations

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

from mprov3_explainer import AVAILABLE_EXPLAINERS, validate_explainer, visualizations_run_dir
from mprov3_explainer.visualize import draw_molecule_with_mask


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Write mask PNGs from one fold of explainer outputs (RDKit only).",
    )
    p.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="mprov3_explainer/results (expects folds/fold_*/explanations/).",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Raw MPro snapshot (ligand SDFs).",
    )
    return p.parse_args()


def _discover_single_fold(results_root: Path) -> tuple[int, Path]:
    """Return (fold_index, fold_root) for the unique fold that has explanation outputs."""
    folds_parent = results_root / "folds"
    if not folds_parent.is_dir():
        raise FileNotFoundError(
            f"No folds/ under {results_root}. Run scripts/run_explanations.py first.",
        )
    found: list[tuple[int, Path]] = []
    for sub in sorted(folds_parent.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("fold_"):
            continue
        try:
            fi = int(sub.name.removeprefix("fold_"))
        except ValueError:
            continue
        exp_base = sub / RESULTS_EXPLANATIONS
        if not exp_base.is_dir():
            continue
        has_any = False
        for name in AVAILABLE_EXPLAINERS:
            rep = exp_base / name / "explanation_report.json"
            masks = exp_base / name / "masks"
            if rep.is_file() and masks.is_dir():
                has_any = True
                break
        if has_any:
            found.append((fi, sub))
    if not found:
        raise FileNotFoundError(
            f"No explainer outputs under {folds_parent}/*/explanations/. "
            "Run run_explanations.py first.",
        )
    if len(found) > 1:
        ids = [f[0] for f in found]
        raise FileNotFoundError(
            f"Multiple folds with explanations: {ids}. "
            "Keep a single fold under results/folds/ or remove stale outputs.",
        )
    return found[0]


def _explainers_in_fold(explanations_base: Path) -> list[str]:
    present: list[str] = []
    for name in AVAILABLE_EXPLAINERS:
        d = explanations_base / name
        if (
            d.is_dir()
            and (d / "explanation_report.json").is_file()
            and (d / "masks").is_dir()
        ):
            present.append(name)
    return present


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_root or _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    fold_index, fold_root = _discover_single_fold(results_root)
    explanations_base = fold_root / RESULTS_EXPLANATIONS
    explainer_names = _explainers_in_fold(explanations_base)
    if not explainer_names:
        raise FileNotFoundError(f"No explainer dirs with masks under {explanations_base}")
    for n in explainer_names:
        validate_explainer(n)

    data_root = Path(args.data_root) if args.data_root else _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")

    print(f"Fold {fold_index}: writing PNGs for {', '.join(explainer_names)}", flush=True)

    for explainer_name in explainer_names:
        explanation_dir = explanations_base / explainer_name
        report = json.loads(
            (explanation_dir / "explanation_report.json").read_text(encoding="utf-8"),
        )
        masks_dir = explanation_dir / "masks"
        vis_out = visualizations_run_dir(fold_root, explainer_name)
        graphs_dir = vis_out / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        drawn = 0
        for e in report.get("per_graph", []):
            graph_id = e.get("graph_id", "")
            if not graph_id:
                continue
            mask_path = masks_dir / f"{graph_id}.json"
            if not mask_path.is_file():
                continue
            mask_data = json.loads(mask_path.read_text(encoding="utf-8"))
            edge_index = mask_data.get("edge_index")
            edge_mask = mask_data.get("edge_mask")
            node_mask = mask_data.get("node_mask")
            if edge_index is None and node_mask is None:
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
                print(f"  {explainer_name}: {out_png.relative_to(fold_root)}", flush=True)
        print(f"{explainer_name}: {drawn} image(s) under {vis_out}", flush=True)


if __name__ == "__main__":
    main()
