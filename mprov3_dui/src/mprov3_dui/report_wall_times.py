"""``wall_time_s`` from fold JSON (same source as ``explanation_web_report`` HTML)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import pandas as pd


def iter_fold_dirs(folds_root: Path) -> Iterator[tuple[int, Path]]:
    if not folds_root.is_dir():
        return
    for sub in sorted(folds_root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("fold_"):
            continue
        try:
            k = int(sub.name.removeprefix("fold_"))
        except ValueError:
            continue
        yield k, sub


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


def _wall_map_from_comparison(fold_dir: Path) -> dict[str, float] | None:
    comp = fold_dir / "explanations" / "comparison_report.json"
    if not comp.is_file():
        return None
    data = json.loads(comp.read_text(encoding="utf-8"))
    per = data.get("per_explainer")
    if not isinstance(per, dict):
        return {}
    out: dict[str, float] = {}
    for name, block in per.items():
        if not isinstance(block, dict):
            continue
        rm = block.get("result_metrics")
        if not isinstance(rm, dict):
            continue
        wt = rm.get("wall_time_s")
        if wt is None:
            continue
        try:
            w = float(wt)
        except (TypeError, ValueError):
            continue
        if math.isfinite(w):
            out[str(name)] = w
    return out


def _wall_map_from_explanation_reports(fold_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for folder_explainer, report_path in _iter_explainer_reports(fold_dir):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        explainer = str(data.get("explainer", folder_explainer))
        rm = data.get("result_metrics")
        if not isinstance(rm, dict):
            continue
        wt = rm.get("wall_time_s")
        if wt is None:
            continue
        try:
            w = float(wt)
        except (TypeError, ValueError):
            continue
        if math.isfinite(w):
            out[explainer] = w
    return out


def load_wall_time_s_map(folds_root: Path) -> dict[int, dict[str, float]]:
    """
    ``fold_index`` → ``explainer`` → ``result_metrics[\"wall_time_s\"]``.

    Prefers ``explanations/comparison_report.json`` (same aggregation the HTML builder
    receives from ``generate_visualizations``); falls back to per-explainer
    ``explanation_report.json`` files.
    """
    wall_map: dict[int, dict[str, float]] = {}
    for fold_index, fold_dir in iter_fold_dirs(folds_root):
        cmp_map = _wall_map_from_comparison(fold_dir)
        if cmp_map is not None:
            wall_map[fold_index] = cmp_map
        else:
            scanned = _wall_map_from_explanation_reports(fold_dir)
            if scanned:
                wall_map[fold_index] = scanned
    return wall_map


def _nanmean_like_web_report(values: list[float]) -> float:
    nums = [v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def runtime_explainer_wall_totals_html_parity(
    wall_map: dict[int, dict[str, float]],
    folds_scope: list[int],
) -> pd.DataFrame:
    """
    **explainer_summary** mean-across-folds → **Result metrics** ``Wall (s) total`` column:
    for each explainer, sum ``wall_time_s`` over folds (skip missing folds), matching
    ``_build_mean_across_folds_table`` / ``_collect_per_explainer_vectors``.
    """
    folds_sorted = sorted(f for f in folds_scope if f in wall_map)
    if not folds_sorted:
        return pd.DataFrame(columns=["explainer", "Wall (s) total"])

    explainers = set()
    for f in folds_sorted:
        explainers.update(wall_map[f].keys())
    explainer_list = sorted(explainers)

    rows: list[dict[str, object]] = []
    for ex in explainer_list:
        total = sum(wall_map[f][ex] for f in folds_sorted if ex in wall_map[f])
        rows.append({"explainer": ex, "Wall (s) total": total})
    return pd.DataFrame(rows)


def runtime_fold_mean_wall_html_parity(
    wall_map: dict[int, dict[str, float]],
    folds_scope: list[int],
) -> pd.DataFrame:
    """
    **index** summary-by-fold → **Result metrics** ``Wall (s)`` column: per fold,
    mean of ``wall_time_s`` over ``per_explainer`` summaries (same as
    ``_per_fold_avg_table``).
    """
    folds_sorted = sorted(f for f in folds_scope if f in wall_map)
    rows: list[dict[str, object]] = []
    for f in folds_sorted:
        vals = list(wall_map[f].values())
        rows.append({"fold": f, "Wall (s)": _nanmean_like_web_report(vals)})
    return pd.DataFrame(rows)
