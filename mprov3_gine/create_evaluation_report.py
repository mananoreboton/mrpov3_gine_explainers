"""
Create an HTML report for a previous run of evaluate.py.

Scans results/classifications/ for fold_*/evaluation_results.json (or legacy flat
evaluation_results.json), writes index.html (one tab per fold), graphs/, and
per-sample HTML into that classifications folder. A path to a single
evaluation_results.json is also accepted (report written next to that file).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List, NamedTuple, Sequence

import torch

from mprov3_gine_explainer_defaults import (
    BUILT_DATASET_FOLDER_NAME,
    DEFAULT_RESULTS_ROOT,
    PYG_DATA_FILENAME,
    PYG_PDB_ORDER_FILENAME,
    RESULTS_CLASSIFICATIONS,
    RESULTS_DATASETS,
)
from dataset import MProV3Dataset, load_dataset_pdb_order
from utils import RunLogger, log_overwrite_dir_if_nonempty, html_document, html_escape
from visualize_graphs import draw_graph

_FOLD_DIR = re.compile(r"^fold_(\d+)$")


class LoadedFold(NamedTuple):
    fold_index: int
    path: Path
    payload: dict[str, Any]


def discover_evaluation_json_paths(classifications_dir: Path) -> List[Path]:
    """Paths to evaluation_results.json under fold_*; else legacy flat file."""
    fold_paths: List[tuple[int, Path]] = []
    for p in classifications_dir.glob("fold_*/evaluation_results.json"):
        m = _FOLD_DIR.match(p.parent.name)
        if m:
            fold_paths.append((int(m.group(1)), p))
    fold_paths.sort(key=lambda x: x[0])
    paths = [p for _, p in fold_paths]
    if not paths:
        legacy = classifications_dir / "evaluation_results.json"
        if legacy.is_file():
            paths.append(legacy)
    return paths


def _fold_index_for_path(path: Path, payload: dict[str, Any]) -> int:
    m = _FOLD_DIR.match(path.parent.name)
    if m:
        return int(m.group(1))
    return int(payload.get("fold_index", 0))


def _dataset_identity(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("dataset_name"),
        payload.get("results_root"),
        payload.get("data_root"),
    )


def _load_folds_for_targets(
    paths: Sequence[Path],
) -> List[LoadedFold]:
    out: List[LoadedFold] = []
    for path in paths:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        k = _fold_index_for_path(path, data)
        out.append(LoadedFold(k, path, data))
    out.sort(key=lambda lf: lf.fold_index)
    return out


def _apply_fold_filter(
    folds: List[LoadedFold], fold_filter: List[int] | None
) -> List[LoadedFold]:
    if fold_filter is None:
        return folds
    wanted = set(fold_filter)
    filtered = [f for f in folds if f.fold_index in wanted]
    have = {f.fold_index for f in filtered}
    missing = sorted(wanted - have)
    if missing:
        raise FileNotFoundError(
            f"No evaluation_results.json for fold(s) {missing} "
            f"(available: {sorted({f.fold_index for f in folds})})"
        )
    return filtered


def _write_sample_page(
    out_dir: Path,
    pdb_id: str,
    real_cat: int,
    pred_cat: int,
    correct: bool,
) -> None:
    """Write one HTML page for a single evaluation sample."""
    img_rel = f"graphs/{html_escape(pdb_id)}.png"
    status_class = "correct" if correct else "wrong"
    status_text = "Correct" if correct else "Incorrect"
    body = [
        f"<h1>PDB ID: {html_escape(pdb_id)}</h1>",
        f"<p><strong>Real category</strong>: {real_cat} &nbsp; "
        f"<strong>Predicted category</strong>: {pred_cat} &nbsp; "
        f"<span class='{status_class}'>{status_text}</span></p>",
        f"<p><img src='{img_rel}' alt='Graph {html_escape(pdb_id)}' /></p>",
        "<p><a href='index.html'>← Back to index</a></p>",
    ]
    html = html_document(
        f"Evaluation: {html_escape(pdb_id)}",
        body,
        style="body { font-family: sans-serif; } .correct { color: green; } .wrong { color: red; }",
    )
    (out_dir / f"{pdb_id}.html").write_text(html, encoding="utf-8")


def _grid_cards_for_entries(entries: List[dict]) -> List[str]:
    lines: List[str] = ["<div class='grid'>"]
    for e in entries:
        pdb_id = e["pdb_id"]
        real = e["real_category"]
        pred = e["predicted_category"]
        correct = real == pred
        img_src = html_escape(f"graphs/{pdb_id}.png")
        report_href = html_escape(f"{pdb_id}.html")
        status_class = "correct" if correct else "wrong"
        status_text = "✓" if correct else "✗"
        lines.append("<div class='card'>")
        lines.append(f"  <a href='{report_href}'>")
        lines.append(f"    <img src='{img_src}' alt='{html_escape(pdb_id)}' loading='lazy' />")
        lines.append(f"    <span class='label'>{html_escape(pdb_id)}</span>")
        lines.append(
            f"    <span class='meta'>Real: {real} → Pred: {pred} "
            f"<span class='{status_class}'>{status_text}</span></span>"
        )
        lines.append("  </a>")
        lines.append("</div>")
    lines.append("</div>")
    return lines


def _write_index_html_folds(
    out_dir: Path,
    fold_rows: List[tuple[int, str, float, List[dict]]],
) -> None:
    """
    fold_rows: (fold_index, eval_timestamp, accuracy, written entries per fold).
    """
    style = (
        "body { font-family: sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; } "
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1em; } "
        ".card { border: 1px solid #ccc; border-radius: 6px; overflow: hidden; text-align: center; } "
        ".card a { text-decoration: none; color: inherit; display: block; } "
        ".card img { width: 100%; height: auto; display: block; } "
        ".card .label { padding: 0.25em; font-size: 14px; font-weight: bold; } "
        ".card .meta { font-size: 12px; color: #666; padding: 0 0.25em 0.5em; } "
        ".correct { color: green; } .wrong { color: red; } "
        "h1 { margin-bottom: 0.25em; } .timestamp { color: #666; font-size: 14px; margin-bottom: 0.5em; } "
        ".accuracy { font-size: 16px; margin-bottom: 1em; } "
        "h2.fold-section { margin-top: 0.5em; margin-bottom: 0.75em; font-size: 1.15rem; } "
        ".fold-tablist { display: flex; flex-wrap: wrap; gap: 0.35em; margin: 1em 0 0.5em 0; "
        "padding: 0; list-style: none; border-bottom: 2px solid #ddd; } "
        ".fold-tablist button { font: inherit; padding: 0.5em 1em; margin-bottom: -2px; "
        "border: 2px solid transparent; border-radius: 6px 6px 0 0; background: #f5f5f5; "
        "cursor: pointer; color: #333; } "
        ".fold-tablist button:hover { background: #eee; } "
        ".fold-tablist button[aria-selected='true'] { background: #fff; border-color: #ddd; "
        "border-bottom-color: #fff; font-weight: 600; } "
        ".fold-tabpanel { padding-top: 0.5em; } "
        ".fold-tabpanel[hidden] { display: none !important; } "
    )
    body: List[str] = ["<h1>Evaluation report</h1>"]
    use_tabs = len(fold_rows) > 0

    if use_tabs:
        body.append('<div class="fold-tabs" id="eval-fold-tabs-root">')
        body.append('<div role="tablist" class="fold-tablist" aria-label="CV fold">')
        for i, (fold_k, _ts, acc, entries) in enumerate(fold_rows):
            panel_id = f"eval-fold-{fold_k}"
            n = len(entries)
            sel = "true" if i == 0 else "false"
            body.append(
                f'<button type="button" role="tab" id="tab-{html_escape(panel_id)}" '
                f'aria-controls="{html_escape(panel_id)}" aria-selected="{sel}" '
                f'data-eval-panel="{html_escape(panel_id)}">'
                f"Fold {fold_k} (n={n}, acc={acc:.4f})</button>"
            )
        body.append("</div>")

        for i, (fold_k, eval_timestamp, accuracy, entries) in enumerate(fold_rows):
            panel_id = f"eval-fold-{fold_k}"
            hidden = "" if i == 0 else " hidden"
            labelled_by = f"tab-{html_escape(panel_id)}"
            body.append(
                f'<div class="fold-tabpanel" role="tabpanel" id="{html_escape(panel_id)}" '
                f'aria-labelledby="{labelled_by}"{hidden}>'
            )
            body.append(f"<h2 class='fold-section'>Fold {fold_k}</h2>")
            body.append(
                f"<p class='timestamp'>Evaluation run: {html_escape(eval_timestamp)}</p>"
            )
            body.append(
                f"<p class='accuracy'><strong>Test accuracy</strong>: {accuracy:.4f}</p>"
            )
            body.extend(_grid_cards_for_entries(entries))
            body.append("</div>")

        body.append("</div>")
        body.append(
            "<script>"
            "(function(){"
            "var root=document.getElementById('eval-fold-tabs-root');"
            "if(!root)return;"
            "var tabs=root.querySelectorAll('[role=tab][data-eval-panel]');"
            "var panels=root.querySelectorAll('[role=tabpanel]');"
            "function show(id){"
            "panels.forEach(function(p){var on=(p.id===id);p.hidden=!on;});"
            "tabs.forEach(function(t){var on=(t.getAttribute('data-eval-panel')===id);"
            "t.setAttribute('aria-selected',on?'true':'false');});"
            "}"
            "tabs.forEach(function(t){"
            "t.addEventListener('click',function(){show(t.getAttribute('data-eval-panel'));});"
            "});"
            "}());"
            "</script>"
        )

    html = html_document("Evaluation report — MPro test set", body, style=style)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create HTML report from evaluate.py outputs (evaluation_results.json per fold)."
    )
    parser.add_argument(
        "--classifications-dir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Directory with fold_*/evaluation_results.json (default: results/classifications "
            "under DEFAULT_RESULTS_ROOT), or a path to a single evaluation_results.json."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Include only these fold indices (default: all discovered folds).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    default_root = Path(DEFAULT_RESULTS_ROOT) / RESULTS_CLASSIFICATIONS

    if args.classifications_dir:
        target = Path(args.classifications_dir)
        if not target.is_absolute():
            target = project_root / target
    else:
        target = default_root

    target = target.resolve()

    if target.is_file() and target.name == "evaluation_results.json":
        report_dir = target.parent
        folds = _load_folds_for_targets([target])
        folds = _apply_fold_filter(folds, args.folds)
    elif target.is_dir():
        report_dir = target
        paths = discover_evaluation_json_paths(target)
        if not paths:
            raise FileNotFoundError(
                f"No evaluation_results.json under {target}. "
                "Expected fold_*/evaluation_results.json or evaluation_results.json. "
                "Run evaluate.py first."
            )
        folds = _load_folds_for_targets(paths)
        folds = _apply_fold_filter(folds, args.folds)
    else:
        raise FileNotFoundError(
            f"Not a directory or evaluation_results.json file: {target}"
        )

    if not folds:
        raise ValueError("No folds to report after filtering.")

    first = folds[0].payload
    ident0 = _dataset_identity(first)
    for lf in folds[1:]:
        if _dataset_identity(lf.payload) != ident0:
            raise ValueError(
                "Inconsistent dataset_name / data_root / results_root across fold JSON files."
            )

    dataset_name = first["dataset_name"]
    results_root_str = first.get("results_root")
    if results_root_str:
        dataset_root = Path(results_root_str) / RESULTS_DATASETS
    else:
        dataset_root = Path(first["data_root"])

    flat_pt = dataset_root / PYG_DATA_FILENAME
    nested_pt = dataset_root / dataset_name / PYG_DATA_FILENAME
    if flat_pt.is_file():
        load_root = dataset_root
        load_name = BUILT_DATASET_FOLDER_NAME
    elif nested_pt.is_file():
        load_root = dataset_root
        load_name = dataset_name
    else:
        raise FileNotFoundError(
            f"Dataset not found: expected {flat_pt} or legacy {nested_pt}"
        )

    ds = MProV3Dataset(root=str(load_root), dataset_name=load_name)
    pdb_order = load_dataset_pdb_order(load_root, load_name)
    if pdb_order is None:
        raise ValueError(f"{PYG_PDB_ORDER_FILENAME} missing; cannot resolve PDB IDs to graphs.")
    pdb_to_idx = {p: i for i, p in enumerate(pdb_order)}

    graphs_dir = report_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "create_evaluation_report.log"

    pdb_union: set[str] = set()
    for lf in folds:
        for rec in lf.payload["results"]:
            pdb_union.add(rec["pdb_id"])

    fold_rows: List[tuple[int, str, float, List[dict]]] = []

    with RunLogger(log_path) as log:
        log_overwrite_dir_if_nonempty(graphs_dir, log.log)
        log.log(f"Report directory: {report_dir}")
        log.log(f"Folds: {[lf.fold_index for lf in folds]}")
        for lf in folds:
            log.log(f"  Results: {lf.path}")
        log.log(f"Dataset: {load_root / load_name}")
        log.log(f"Unique PDBs (draw graphs once): {len(pdb_union)}")

        for pdb_id in sorted(pdb_union):
            idx = pdb_to_idx.get(pdb_id)
            if idx is None:
                log.log(f"  Skip graph {pdb_id}: not in dataset")
                continue
            g = ds[idx]
            x = g.x
            pos3d = x[:, :3]
            atomic_numbers = x[:, 3].round().to(torch.long)
            img_path = graphs_dir / f"{pdb_id}.png"
            draw_graph(
                pos3d=pos3d,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                atomic_numbers=atomic_numbers,
                out_path_png=img_path,
                out_path_svg=None,
            )

        for lf in folds:
            results_list: List[dict] = lf.payload["results"]
            accuracy = float(lf.payload["accuracy"])
            eval_timestamp = lf.payload.get("timestamp", "unknown")
            written: List[dict] = []

            for rec in results_list:
                pdb_id = rec["pdb_id"]
                real_cat = int(rec["real_category"])
                pred_cat = int(rec["predicted_category"])
                idx = pdb_to_idx.get(pdb_id)
                if idx is None:
                    log.log(f"  Fold {lf.fold_index} skip {pdb_id}: not in dataset")
                    continue
                _write_sample_page(
                    report_dir,
                    pdb_id,
                    real_cat,
                    pred_cat,
                    correct=(real_cat == pred_cat),
                )
                written.append(rec)

            log.log(f"Fold {lf.fold_index}: {len(written)} sample pages")
            fold_rows.append((lf.fold_index, eval_timestamp, accuracy, written))

        _write_index_html_folds(report_dir, fold_rows)
        log.log(f"Index: {report_dir / 'index.html'}")
        log.log("Done.")


if __name__ == "__main__":
    main()
