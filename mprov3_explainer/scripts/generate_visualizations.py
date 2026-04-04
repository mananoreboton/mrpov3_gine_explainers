#!/usr/bin/env python3
"""
Build the consolidated exploration report: RDKit images, report_data.json, static web UI,
and optionally legacy per-fold HTML under results/visualizations/.

Scans results/folds/fold_*/explanations/ (or legacy results/explanations/), merges
classification JSON from mprov3_gine results, writes mprov3_explainer/results/exploration_report/.

Usage:
  uv run python scripts/generate_visualizations.py
  uv run python scripts/generate_visualizations.py --explainers GNNEXPL GRADEXPINODE
  uv run python scripts/generate_visualizations.py --no-serve
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MPROV3_EXPLAINER_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _MPROV3_EXPLAINER_ROOT.parent
_GNN_PROJECT_ROOT = _REPO_ROOT / "mprov3_gine"
_WEB_SOURCE_DIR = _MPROV3_EXPLAINER_ROOT / "web"

if _GNN_PROJECT_ROOT.exists() and str(_GNN_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_PROJECT_ROOT))

from mprov3_gine_explainer_defaults import (
    DEFAULT_DROPOUT,
    DEFAULT_EDGE_DIM,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_MPRO_SNAPSHOT_DIR_NAME,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_POOL,
    MPRO_LIGAND_DIR,
    MPRO_LIGAND_SDF_SUBDIR,
    RESULTS_CLASSIFICATIONS,
    RESULTS_DIR_NAME,
    RESULTS_EXPLANATIONS,
    RESULTS_VISUALIZATIONS,
)

from mprov3_explainer import AVAILABLE_EXPLAINERS, validate_explainer, visualizations_run_dir
from mprov3_explainer.visualize import (
    draw_molecule_base,
    draw_molecule_with_mask,
    write_comparison_index_html,
    write_explanation_index_html,
)

EXPLORATION_REPORT_DIR = "exploration_report"


def _explainers_present(explanations_base: Path) -> list[str]:
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


def discover_fold_explanations(results_root: Path) -> list[tuple[int, Path, Path]]:
    """
    Return (fold_index, fold_root, explanations_base) where explanations_base is
    the directory containing explainer subfolders (…/explanations/).
    """
    folds_parent = results_root / "folds"
    out: list[tuple[int, Path, Path]] = []
    if folds_parent.is_dir():
        for sub in sorted(folds_parent.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("fold_"):
                continue
            try:
                fi = int(sub.name.removeprefix("fold_"))
            except ValueError:
                continue
            exp_base = sub / RESULTS_EXPLANATIONS
            if exp_base.is_dir() and any(exp_base.iterdir()):
                out.append((fi, sub, exp_base))
        if out:
            return sorted(out, key=lambda x: x[0])
    exp_base = results_root / RESULTS_EXPLANATIONS
    if exp_base.is_dir():
        return [(0, results_root, exp_base)]
    return []


def _load_classification(gnn_results_root: Path, fold_index: int) -> dict | None:
    fold_path = (
        gnn_results_root
        / RESULTS_CLASSIFICATIONS
        / f"fold_{fold_index}"
        / "evaluation_results.json"
    )
    if fold_path.is_file():
        return json.loads(fold_path.read_text(encoding="utf-8"))
    if fold_index == 0:
        legacy = gnn_results_root / RESULTS_CLASSIFICATIONS / "evaluation_results.json"
        if legacy.is_file():
            return json.loads(legacy.read_text(encoding="utf-8"))
    return None


def _summary_row_from_report(report: dict) -> dict:
    return {
        "mean_fidelity_plus": report.get("mean_fidelity_plus", 0.0),
        "mean_fidelity_minus": report.get("mean_fidelity_minus", 0.0),
        "mean_pyg_characterization": report.get("mean_pyg_characterization", 0.0),
        "mean_paper_sufficiency": report.get("mean_paper_sufficiency", 0.0),
        "mean_paper_comprehensiveness": report.get("mean_paper_comprehensiveness", 0.0),
        "mean_paper_f1_fidelity": report.get("mean_paper_f1_fidelity", 0.0),
        "num_graphs": report.get("num_graphs", 0),
        "num_valid": report.get("num_valid", 0),
    }


def _copy_web_assets(out_dir: Path) -> None:
    for name in ("index.html", "app.js", "styles.css"):
        src = _WEB_SOURCE_DIR / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing web asset: {src}")
        shutil.copy2(src, out_dir / name)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build exploration report (JSON, RDKit images, web UI) from explanation runs.",
    )
    p.add_argument("--explainer", type=str, default=None)
    p.add_argument(
        "--explainers",
        type=str,
        nargs="*",
        default=None,
        help="Explainers to include; default: union of explainer dirs across folds.",
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="mprov3_explainer/results (explanations + folds/).",
    )
    p.add_argument(
        "--gnn_results_root",
        type=str,
        default=None,
        help="mprov3_gine/results (classifications/fold_*/). Default: sibling mprov3_gine/results.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=f"Report output directory (default: …/results/{EXPLORATION_REPORT_DIR}).",
    )
    p.add_argument(
        "--no-legacy-html",
        action="store_true",
        help="Skip per-explainer index.html and comparison.html under results/visualizations/.",
    )
    p.add_argument("--no-serve", action="store_true", help="Do not start HTTP server or open browser.")
    p.add_argument("--port", type=int, default=8765, help="Port for local HTTP server (default: 8765).")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    p.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--num_classes", type=int, default=DEFAULT_OUT_CLASSES)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    results_root = Path(args.results_root or _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME)
    gnn_results_root = Path(
        args.gnn_results_root or _GNN_PROJECT_ROOT / RESULTS_DIR_NAME,
    )
    out_dir = Path(
        args.out_dir or _MPROV3_EXPLAINER_ROOT / RESULTS_DIR_NAME / EXPLORATION_REPORT_DIR,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fold_entries = discover_fold_explanations(results_root)
    if not fold_entries:
        raise FileNotFoundError(
            f"No explanations found under {results_root / 'folds'} or "
            f"{results_root / RESULTS_EXPLANATIONS}. Run run_explanations.py first.",
        )

    if args.explainers:
        explainer_names = list(args.explainers)
    elif args.explainer is not None:
        explainer_names = [args.explainer]
    else:
        discovered: set[str] = set()
        for _, _, exp_base in fold_entries:
            discovered.update(_explainers_present(exp_base))
        explainer_names = sorted(discovered, key=lambda n: (AVAILABLE_EXPLAINERS.index(n) if n in AVAILABLE_EXPLAINERS else 999, n))
        if not explainer_names:
            raise FileNotFoundError("No explainer outputs found under discovered fold explanation dirs.")
        print(f"Using explainers present: {', '.join(explainer_names)}", flush=True)

    for name in explainer_names:
        validate_explainer(name)

    data_root = Path(args.data_root) if args.data_root else _REPO_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")

    folds_data: dict[str, dict] = {}
    summary_by_fold: dict[str, list[dict]] = {}
    summary_by_explainer: dict[str, list[dict]] = defaultdict(list)
    max_fold_index = max(fi for fi, _, _ in fold_entries)
    num_folds_hint = None

    for fold_index, fold_root, explanations_base in fold_entries:
        fold_str = str(fold_index)
        class_payload = _load_classification(gnn_results_root, fold_index)
        if class_payload is None:
            print(
                f"[WARN] No classification JSON for fold {fold_index} under {gnn_results_root}",
                flush=True,
            )
        if class_payload is not None and num_folds_hint is None:
            num_folds_hint = int(class_payload.get("num_folds", max_fold_index + 1))

        classification_block: dict = {
            "accuracy": None,
            "evaluation_timestamp": None,
            "graphs": [],
        }
        if class_payload:
            classification_block["accuracy"] = float(class_payload.get("accuracy", 0.0))
            classification_block["evaluation_timestamp"] = class_payload.get("timestamp")
            num_folds_hint = int(class_payload.get("num_folds", num_folds_hint or max_fold_index + 1))

        explainers_block: dict[str, dict] = {}
        fold_summary_rows: list[dict] = []

        comparison_data: dict = {
            "explainers": [],
            "graph_ids": [],
            "per_explainer": {},
            "grid": {},
        }
        all_graph_ids: set[str] = set()

        for explainer_name in explainer_names:
            explanation_dir = explanations_base / explainer_name
            if not explanation_dir.is_dir():
                continue
            report_path = explanation_dir / "explanation_report.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            masks_dir = explanation_dir / "masks"
            if not masks_dir.is_dir():
                continue

            graphs_out: dict[str, dict] = {}

            if not args.no_legacy_html:
                vis_out = visualizations_run_dir(fold_root, explainer_name)
                if vis_out.exists() and any(vis_out.iterdir()):
                    print(f"[INFO] Overwriting legacy viz: {vis_out}", flush=True)
                graphs_dir = vis_out / "graphs"
                graphs_dir.mkdir(parents=True, exist_ok=True)
            else:
                graphs_dir = None

            comparison_data["explainers"].append(explainer_name)
            comparison_data["per_explainer"][explainer_name] = {
                "mean_fid_plus": report.get("mean_fidelity_plus", 0.0),
                "mean_fid_minus": report.get("mean_fidelity_minus", 0.0),
                "mean_pyg_characterization": report.get("mean_pyg_characterization", 0.0),
                "mean_paper_sufficiency": report.get("mean_paper_sufficiency", 0.0),
                "mean_paper_comprehensiveness": report.get("mean_paper_comprehensiveness", 0.0),
                "mean_paper_f1_fidelity": report.get("mean_paper_f1_fidelity", 0.0),
            }

            row = {"explainer": explainer_name, **_summary_row_from_report(report)}
            fold_summary_rows.append(row)
            summary_by_explainer[explainer_name].append(
                {"fold": fold_str, **_summary_row_from_report(report)},
            )

            report_for_html = dict(report)
            report_for_html["explainer"] = explainer_name
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
                base_name = f"fold{fold_index}_{graph_id}_base.png"
                base_rel = f"images/{base_name}"
                base_abs = out_dir / base_rel
                if not base_abs.is_file():
                    if draw_molecule_base(sdf_path, base_abs):
                        print(f"  base image: {base_rel}", flush=True)

                mask_name = f"fold{fold_index}_{explainer_name}_{graph_id}_mask.png"
                mask_rel = f"images/{mask_name}"
                mask_abs = out_dir / mask_rel
                if draw_molecule_with_mask(
                    sdf_path,
                    edge_index=edge_index,
                    edge_mask=edge_mask,
                    out_path_png=mask_abs,
                    node_mask=node_mask,
                ):
                    print(f"  mask image: {mask_rel}", flush=True)

                if graphs_dir is not None:
                    leg_png = graphs_dir / f"mask_{graph_id}.png"
                    if draw_molecule_with_mask(
                        sdf_path,
                        edge_index=edge_index,
                        edge_mask=edge_mask,
                        out_path_png=leg_png,
                        node_mask=node_mask,
                    ):
                        drawn += 1

                all_graph_ids.add(graph_id)
                if graph_id not in comparison_data["grid"]:
                    comparison_data["grid"][graph_id] = {}
                comparison_data["grid"][graph_id][explainer_name] = {
                    "img": f"{explainer_name}/graphs/mask_{graph_id}.png",
                    "fid_plus": e.get("fidelity_plus", 0.0),
                    "fid_minus": e.get("fidelity_minus", 0.0),
                    "paper_f1_fidelity": e.get("paper_f1_fidelity", 0.0),
                }

                graphs_out[graph_id] = {
                    "fidelity_plus": e.get("fidelity_plus", 0.0),
                    "fidelity_minus": e.get("fidelity_minus", 0.0),
                    "pyg_characterization": e.get("pyg_characterization", 0.0),
                    "paper_sufficiency": e.get("paper_sufficiency", 0.0),
                    "paper_comprehensiveness": e.get("paper_comprehensiveness", 0.0),
                    "paper_f1_fidelity": e.get("paper_f1_fidelity", 0.0),
                    "valid": e.get("valid", False),
                    "correct_class": e.get("correct_class", False),
                    "has_node_mask": e.get("has_node_mask", False),
                    "has_edge_mask": e.get("has_edge_mask", False),
                    "mask_image": mask_rel,
                    "mask_raw": mask_data,
                }

            explainers_block[explainer_name] = {
                "summary": _summary_row_from_report(report),
                "graphs": graphs_out,
            }

            if not args.no_legacy_html and graphs_dir is not None:
                write_explanation_index_html(vis_out, report_for_html)
                print(f"Legacy index: {vis_out / 'index.html'} ({drawn} graphs)", flush=True)

        if class_payload:
            for rec in class_payload.get("results", []):
                pdb_id = rec["pdb_id"]
                real_c = int(rec["real_category"])
                pred_c = int(rec["predicted_category"])
                base_rel = f"images/fold{fold_index}_{pdb_id}_base.png"
                base_abs = out_dir / base_rel
                sdf_path = sdf_dir / f"{pdb_id}_ligand.sdf"
                if not base_abs.is_file():
                    draw_molecule_base(sdf_path, base_abs)
                classification_block["graphs"].append({
                    "pdb_id": pdb_id,
                    "real_category": real_c,
                    "predicted_category": pred_c,
                    "correct": real_c == pred_c,
                    "base_image": base_rel,
                })

        comparison_data["graph_ids"] = sorted(all_graph_ids)
        if not args.no_legacy_html and comparison_data["explainers"]:
            vis_root = fold_root / RESULTS_VISUALIZATIONS
            vis_root.mkdir(parents=True, exist_ok=True)
            write_comparison_index_html(vis_root, comparison_data)
            print(f"Legacy comparison: {vis_root / 'comparison.html'}", flush=True)

        folds_data[fold_str] = {
            "fold_index": fold_index,
            "num_folds": num_folds_hint or (max_fold_index + 1),
            "classification": classification_block,
            "explainers": explainers_block,
        }
        summary_by_fold[fold_str] = fold_summary_rows

    report_payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "num_folds": num_folds_hint or (max_fold_index + 1),
            "fold_indices": [str(fi) for fi, _, _ in fold_entries],
            "explainers": explainer_names,
            "gnn": {
                "type": "MProGNN (GINE)",
                "pool": DEFAULT_POOL,
                "in_channels": DEFAULT_IN_CHANNELS,
                "hidden_channels": args.hidden,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "out_classes": args.num_classes,
                "edge_dim": DEFAULT_EDGE_DIM,
            },
        },
        "folds": folds_data,
        "summary_by_fold": summary_by_fold,
        "summary_by_explainer": dict(summary_by_explainer),
    }

    (out_dir / "report_data.json").write_text(
        json.dumps(report_payload, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir / 'report_data.json'}", flush=True)

    _copy_web_assets(out_dir)
    print(f"Copied web assets to {out_dir}", flush=True)

    if not args.no_serve:
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(out_dir.resolve()))
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=False)
        thread.start()
        url = f"http://127.0.0.1:{args.port}/index.html"
        print(f"Serving report at {url}", flush=True)
        webbrowser.open(url)
        try:
            if sys.stdin.isatty():
                input("Press Enter to stop the server...\n")
            else:
                print(
                    "(stdin is not a TTY; block until SIGINT or use --no-serve in CI.)",
                    flush=True,
                )
                threading.Event().wait()
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    main()
