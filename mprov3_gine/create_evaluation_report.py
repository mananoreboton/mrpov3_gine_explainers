"""
Create an HTML report for a previous run of evaluate.py.

By default uses results/classifications/evaluation_results.json and writes
the report (index.html, graphs/, per-sample HTML) into that same folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

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


def _write_index_html(
    out_dir: Path,
    entries: List[dict],
    eval_timestamp: str,
    accuracy: float,
) -> None:
    """Write index.html with thumbnails and real/pred/correct for each sample."""
    style = (
        "body { font-family: sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; } "
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1em; } "
        ".card { border: 1px solid #ccc; border-radius: 6px; overflow: hidden; text-align: center; } "
        ".card a { text-decoration: none; color: inherit; display: block; } "
        ".card img { width: 100%; height: auto; display: block; } "
        ".card .label { padding: 0.25em; font-size: 14px; font-weight: bold; } "
        ".card .meta { font-size: 12px; color: #666; padding: 0 0.25em 0.5em; } "
        ".correct { color: green; } .wrong { color: red; } "
        "h1 { margin-bottom: 0.25em; } .timestamp { color: #666; font-size: 14px; margin-bottom: 1em; } "
        ".accuracy { font-size: 16px; margin-bottom: 1em; }"
    )
    body = [
        "<h1>Evaluation report</h1>",
        f"<p class='timestamp'>Evaluation run: {html_escape(eval_timestamp)}</p>",
        f"<p class='accuracy'><strong>Test accuracy</strong>: {accuracy:.4f}</p>",
        "<div class='grid'>",
    ]
    for e in entries:
        pdb_id = e["pdb_id"]
        real = e["real_category"]
        pred = e["predicted_category"]
        correct = real == pred
        img_src = html_escape(f"graphs/{pdb_id}.png")
        report_href = html_escape(f"{pdb_id}.html")
        status_class = "correct" if correct else "wrong"
        status_text = "✓" if correct else "✗"
        body.append("<div class='card'>")
        body.append(f"  <a href='{report_href}'>")
        body.append(f"    <img src='{img_src}' alt='{html_escape(pdb_id)}' loading='lazy' />")
        body.append(f"    <span class='label'>{html_escape(pdb_id)}</span>")
        body.append(
            f"    <span class='meta'>Real: {real} → Pred: {pred} <span class='{status_class}'>{status_text}</span></span>"
        )
        body.append("  </a>")
        body.append("</div>")
    body.append("</div>")
    html = html_document("Evaluation report — MPro test set", body, style=style)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create HTML report from a previous evaluate.py run (evaluation_results.json)."
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Path to evaluation_results.json (default: fold_0 under classifications/, "
            "then legacy flat evaluation_results.json)"
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    classifications_base = Path(DEFAULT_RESULTS_ROOT) / RESULTS_CLASSIFICATIONS

    if args.results:
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = project_root / results_path
        report_dir = results_path.parent
    else:
        fold0 = classifications_base / "fold_0" / "evaluation_results.json"
        legacy = classifications_base / "evaluation_results.json"
        if fold0.is_file():
            results_path = fold0
            report_dir = fold0.parent
        elif legacy.is_file():
            results_path = legacy
            report_dir = classifications_base
        else:
            raise FileNotFoundError(
                f"evaluation_results.json not found at {fold0} or {legacy}. Run evaluate.py first."
            )

    payload: Any = json.loads(results_path.read_text(encoding="utf-8"))
    dataset_name = payload["dataset_name"]
    results_list: List[dict] = payload["results"]
    accuracy = float(payload["accuracy"])
    eval_timestamp = payload.get("timestamp", "unknown")

    results_root_str = payload.get("results_root")
    if results_root_str:
        dataset_root = Path(results_root_str) / RESULTS_DATASETS
    else:
        dataset_root = Path(payload["data_root"])

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

    written_entries: List[dict] = []

    with RunLogger(log_path) as log:
        log_overwrite_dir_if_nonempty(graphs_dir, log.log)
        log.log(f"Results: {results_path}")
        log.log(f"Report directory: {report_dir}")
        log.log(f"Dataset: {load_root / load_name}")
        log.log(f"Writing report for {len(results_list)} samples")

        for rec in results_list:
            pdb_id = rec["pdb_id"]
            real_cat = int(rec["real_category"])
            pred_cat = int(rec["predicted_category"])
            idx = pdb_to_idx.get(pdb_id)
            if idx is None:
                log.log(f"  Skip {pdb_id}: not in dataset")
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
            _write_sample_page(
                report_dir,
                pdb_id,
                real_cat,
                pred_cat,
                correct=(real_cat == pred_cat),
            )
            written_entries.append(rec)

        _write_index_html(report_dir, written_entries, eval_timestamp, accuracy)
        log.log(f"Index: {report_dir / 'index.html'}")
        log.log("Done.")


if __name__ == "__main__":
    main()
