#!/usr/bin/env python3
"""
Rebuild ``results/labeled_explanation_sample.csv`` from saved fold outputs.

Scans ``results/folds/fold_<k>/explanations/<EXPLAINER>/explanation_report.json``
(``per_graph`` blocks). Split name is taken from
``results/folds/fold_<k>/explanations/comparison_report.json`` when present.

Usage (from ``mprov3_explainer``):

  uv run python scripts/export_labeled_explanation_sample.py
  uv run python scripts/export_labeled_explanation_sample.py --folds 0 1
  uv run python scripts/export_labeled_explanation_sample.py \\
      --folds-root /path/to/results/folds --output /path/to/labeled_explanation_sample.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_MPROV3_EXPLAINER_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR_NAME = "results"

# Same metric keys as ``run_explanations.py`` / ``explanation_report.json`` per_graph.
_METRIC_FIELD_NAMES: tuple[str, ...] = (
    "paper_sufficiency",
    "paper_comprehensiveness",
    "paper_f1_fidelity",
    "pyg_fidelity_plus",
    "pyg_fidelity_minus",
    "pyg_characterization_score",
    "pyg_fidelity_curve_auc",
    "pyg_unfaithfulness",
)


def _csv_columns() -> tuple[str, ...]:
    identity = ("fold", "split", "explainer", "graph_id")
    labels = ("target_class", "pred_class", "correct_class")
    flags = (
        "valid",
        "prediction_baseline_mismatch",
        "has_node_mask",
        "has_edge_mask",
    )
    timing = ("elapsed_s",)
    return identity + labels + flags + timing + _METRIC_FIELD_NAMES


def _iter_fold_dirs(folds_root: Path) -> Iterator[tuple[int, Path]]:
    if not folds_root.is_dir():
        raise FileNotFoundError(f"Folds root not found: {folds_root}")
    for sub in sorted(folds_root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("fold_"):
            continue
        try:
            k = int(sub.name.removeprefix("fold_"))
        except ValueError:
            continue
        yield k, sub


def _read_split_name(fold_dir: Path) -> str:
    comp = fold_dir / "explanations" / "comparison_report.json"
    if not comp.is_file():
        return "unknown"
    data = json.loads(comp.read_text(encoding="utf-8"))
    return str(data.get("split", "unknown"))


def _iter_explainer_reports(fold_dir: Path) -> Iterator[tuple[str, Path]]:
    root = fold_dir / "explanations"
    if not root.is_dir():
        return
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        report = sub / "explanation_report.json"
        if report.is_file():
            yield sub.name, report


def _validate_per_graph_entry(entry: dict[str, Any], *, explainer: str, path: Path) -> None:
    cols = _csv_columns()
    need = set(cols[3:])
    missing = need - entry.keys()
    if missing:
        raise KeyError(
            f"{path}: per_graph entry for graph_id={entry.get('graph_id')!r} "
            f"(explainer={explainer}) missing keys: {sorted(missing)}",
        )


def collect_rows(
    folds_root: Path,
    *,
    fold_filter: set[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cols = _csv_columns()
    for fold_index, fold_dir in _iter_fold_dirs(folds_root):
        if fold_filter is not None and fold_index not in fold_filter:
            continue
        split = _read_split_name(fold_dir)
        n_reports = 0
        for folder_explainer, report_path in _iter_explainer_reports(fold_dir):
            n_reports += 1
            data = json.loads(report_path.read_text(encoding="utf-8"))
            explainer = str(data.get("explainer", folder_explainer))
            per_graph = data.get("per_graph")
            if not isinstance(per_graph, list):
                raise TypeError(f"{report_path}: expected list per_graph")
            for entry in per_graph:
                if not isinstance(entry, dict):
                    raise TypeError(f"{report_path}: expected dict per_graph rows")
                _validate_per_graph_entry(entry, explainer=explainer, path=report_path)
                row: dict[str, Any] = {
                    "fold": int(fold_index),
                    "split": split,
                    "explainer": explainer,
                }
                for key in cols[3:]:
                    row[key] = entry[key]
                rows.append(row)
        if n_reports == 0:
            print(
                f"[WARN] No explanation_report.json under {fold_dir / 'explanations'}",
                file=sys.stderr,
            )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build labeled_explanation_sample.csv from existing "
            "results/folds/*/explanations/*/explanation_report.json data."
        ),
    )
    p.add_argument(
        "--folds-root",
        type=Path,
        default=_MPROV3_EXPLAINER_ROOT / _RESULTS_DIR_NAME / "folds",
        help="Directory containing fold_* folders (default: mprov3_explainer/results/folds).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_MPROV3_EXPLAINER_ROOT / _RESULTS_DIR_NAME / "labeled_explanation_sample.csv",
        help="Output CSV path (default: mprov3_explainer/results/labeled_explanation_sample.csv).",
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Only include these fold indices (default: all discovered folds).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fold_filter = set(args.folds) if args.folds is not None else None
    rows = collect_rows(args.folds_root, fold_filter=fold_filter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(_csv_columns()))
    if not df.empty:
        df = df.sort_values(["fold", "explainer", "graph_id"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
