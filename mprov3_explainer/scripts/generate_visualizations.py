#!/usr/bin/env python3
"""
Draw RDKit PNGs from saved explanation masks and write static HTML reports.

Supports a single fold (auto-discovered or via ``--folds``) or multiple folds at once.
When multiple folds are processed, a global index page linking all per-fold reports is
written to ``results/explanation_web_report/index.html``.

Usage:
  uv run python scripts/generate_visualizations.py
  uv run python scripts/generate_visualizations.py --folds 0 2 4
  uv run python scripts/generate_visualizations.py --report-only
  uv run python scripts/generate_visualizations.py --no-report

Reads ``mprov3_explainer/results`` and ligand SDFs from the workspace default MPro snapshot
(unless ``--report-only``). Per-fold reports are written to
``results/folds/fold_*/explanation_web_report/index.html`` with relative links to
``visualizations/`` and embedded mask JSON; open the file in a browser (no HTTP server required).
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
from mprov3_explainer.web_report import (
    write_explainer_summary_page,
    write_fold_explanation_web_report,
    write_global_explanation_index,
    write_per_class_summary_pages,
)


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Write mask PNGs from explainer outputs (RDKit) and/or emit static "
            "HTML reports. Supports one or many folds. Uses mprov3_explainer/results."
        ),
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Explicit fold indices to visualize (e.g. --folds 0 2 4). "
             "If omitted, auto-discovers all folds with explanation outputs.",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing explanation_web_report/index.html (per-fold and global).",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Only regenerate HTML reports from existing explanations/ and visualizations/ "
        "(no RDKit drawing; does not require MPro SDFs).",
    )
    return p.parse_args()


def _discover_folds(results_root: Path) -> list[tuple[int, Path]]:
    """Return all (fold_index, fold_root) pairs that have explanation outputs."""
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
        for name in AVAILABLE_EXPLAINERS:
            rep = exp_base / name / "explanation_report.json"
            masks = exp_base / name / "masks"
            if rep.is_file() and masks.is_dir():
                found.append((fi, sub))
                break
    if not found:
        raise FileNotFoundError(
            f"No explainer outputs under {folds_parent}/*/explanations/. "
            "Run run_explanations.py first.",
        )
    return found


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


def _process_fold(
    fold_index: int,
    fold_root: Path,
    *,
    report_only: bool,
    no_report: bool,
    sdf_dir: Path | None,
) -> dict | None:
    """Process a single fold: draw PNGs and/or write per-fold HTML report.

    Returns a fold entry dict for the global index, or None if --no-report.
    """
    explanations_base = fold_root / RESULTS_EXPLANATIONS
    explainer_names = _explainers_in_fold(explanations_base)
    if not explainer_names:
        print(f"Fold {fold_index}: no explainer dirs with masks, skipping.", flush=True)
        return None
    for n in explainer_names:
        validate_explainer(n)

    if report_only:
        print(f"Fold {fold_index}: --report-only (skipping RDKit PNG generation)", flush=True)

    if not report_only and sdf_dir is not None:
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

    if no_report:
        return None

    out_html = write_fold_explanation_web_report(fold_root, fold_index, explainer_names)
    print(f"Web report written: {out_html}", flush=True)

    per_explainer_summary: dict[str, dict] = {}
    per_graph_per_explainer: dict[str, list[dict]] = {}
    comparison_path = explanations_base / "comparison_report.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        per_explainer_summary = comparison.get("per_explainer", {})
        per_graph_per_explainer = comparison.get("per_graph_per_explainer", {})

    return {
        "fold_index": fold_index,
        "explainer_names": explainer_names,
        "per_explainer_summary": per_explainer_summary,
        "per_graph_per_explainer": per_graph_per_explainer,
    }


def main() -> None:
    args = _parse_args()
    results_root = _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    all_folds = _discover_folds(results_root)

    if args.folds is not None:
        available_indices = {fi for fi, _ in all_folds}
        for k in args.folds:
            if k not in available_indices:
                raise ValueError(
                    f"Fold {k} has no explanation outputs under "
                    f"{results_root / 'folds' / f'fold_{k}' / 'explanations'}. "
                    f"Available folds with explanations: {sorted(available_indices)}"
                )
        seen: set[int] = set()
        target_folds = []
        for k in args.folds:
            if k not in seen:
                seen.add(k)
                fold_root = next(p for fi, p in all_folds if fi == k)
                target_folds.append((k, fold_root))
    else:
        target_folds = all_folds

    sdf_dir: Path | None = None
    if not args.report_only:
        data_root = _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")
        sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
        if not sdf_dir.is_dir():
            raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")

    fold_entries: list[dict] = []
    for fold_index, fold_root in target_folds:
        entry = _process_fold(
            fold_index,
            fold_root,
            report_only=args.report_only,
            no_report=args.no_report,
            sdf_dir=sdf_dir,
        )
        if entry is not None:
            fold_entries.append(entry)

    if fold_entries and len(target_folds) > 1:
        global_html = write_global_explanation_index(results_root, fold_entries)
        print(f"Global index written: {global_html}", flush=True)
        summary_html = write_explainer_summary_page(results_root, fold_entries)
        print(f"Explainer summary written: {summary_html}", flush=True)
        class_pages = write_per_class_summary_pages(results_root, fold_entries)
        for cp in class_pages:
            print(f"Per-class summary written: {cp}", flush=True)


if __name__ == "__main__":
    main()
