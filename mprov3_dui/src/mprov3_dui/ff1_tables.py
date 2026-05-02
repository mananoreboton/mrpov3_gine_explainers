"""Mean / std Ff1 tables from labeled sample rows (``paper_f1_fidelity``, valid-only)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from mprov3_dui.ranking import RANKING_METRIC

FF1_COLUMN = RANKING_METRIC


def _nanmean_safe(values: Sequence[float | None]) -> float:
    nums = [float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return float("nan")
    tw = sum(weights)
    if tw == 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights, strict=False)) / tw


def _weighted_std(values: list[float], weights: list[float]) -> float:
    mu = _weighted_mean(values, weights)
    if not math.isfinite(mu) or len(values) < 2:
        return float("nan")
    tw = sum(weights)
    if tw == 0:
        return float("nan")
    var = sum(w * (v - mu) ** 2 for v, w in zip(values, weights, strict=False)) / tw
    return math.sqrt(var)


def _population_std(series: pd.Series | np.ndarray | Sequence[float]) -> float:
    if isinstance(series, pd.Series):
        nums = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    else:
        nums = np.asarray(series, dtype=float)
        nums = nums[np.isfinite(nums)]
    n = nums.size
    if n < 2:
        return float("nan")
    mean = float(nums.mean())
    return math.sqrt(float(np.square(nums - mean).mean()))


def pooled_combined_std(sigmas: Sequence[float], counts: Sequence[int]) -> float:
    """√(Σ n_k σ_k² / Σ n_k) over rows with finite σ and positive n."""
    num = 0.0
    den = 0.0
    for s, n in zip(sigmas, counts, strict=False):
        if n is None or int(n) < 1 or s is None or not isinstance(s, (int, float)) or not math.isfinite(float(s)):
            continue
        nf = float(n)
        den += nf
        num += nf * float(s) ** 2
    if den == 0:
        return float("nan")
    return math.sqrt(num / den)


def explainer_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["explainer"].unique().tolist())


def ff1_one_fold_explainer_table(df_valid: pd.DataFrame, fold: int, all_explainers: list[str]) -> pd.DataFrame:
    """
    One fold: rows = all explainers (stable order); cols = counts, Mean Ff1, Std Ff1 from samples.

    Mirrors ``explainer_summary`` Valid **Per-fold explainer metrics** for a single fold, reduced to Ff1.
    Explainers absent in this fold keep ``num_valid`` 0 and NaN metric cells.
    """
    dff = df_valid.loc[df_valid["fold"] == fold]
    rows: list[dict[str, object]] = []
    for ex in all_explainers:
        ser = dff.loc[dff["explainer"] == ex, FF1_COLUMN]
        n = int(len(ser))
        if n == 0:
            rows.append({"explainer": ex, "num_valid": 0, "mean_ff1": float("nan"), "std_ff1": float("nan")})
        else:
            rows.append(
                {
                    "explainer": ex,
                    "num_valid": n,
                    "mean_ff1": float(ser.mean()),
                    "std_ff1": _population_std(ser),
                }
            )
    body = pd.DataFrame(rows)
    if body.empty:
        return body
    foot = {
        "explainer": "Aggregate",
        "num_valid": int(body["num_valid"].sum()),
        "mean_ff1": _nanmean_safe(body["mean_ff1"].tolist()),
        "std_ff1": pooled_combined_std(body["std_ff1"].tolist(), body["num_valid"].tolist()),
    }
    return pd.concat([body, pd.DataFrame([foot])], ignore_index=True)


def ff1_fold_explainer_stats(df_valid: pd.DataFrame) -> pd.DataFrame:
    """Per (fold, explainer): mean Ff1, population std of Ff1, valid graph count."""
    if df_valid.empty:
        return pd.DataFrame(columns=["fold", "explainer", "mean_ff1", "std_ff1", "num_valid"])

    def std_col(s: pd.Series) -> float:
        return _population_std(s)

    out = (
        df_valid.groupby(["fold", "explainer"], sort=True, dropna=False)[FF1_COLUMN]
        .agg(mean_ff1="mean", std_ff1=std_col, num_valid="count")
        .reset_index()
    )
    out["explainer"] = out["explainer"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out["num_valid"] = out["num_valid"].astype(int)
    return out.sort_values(["explainer", "fold"]).reset_index(drop=True)


def ff1_per_explainer_fold_body(stats: pd.DataFrame, explainer: str) -> pd.DataFrame:
    sub = stats.loc[stats["explainer"] == explainer, ["fold", "num_valid", "mean_ff1", "std_ff1"]].copy()
    return sub.sort_values("fold").reset_index(drop=True)


def ff1_per_explainer_with_footer(body: pd.DataFrame) -> pd.DataFrame:
    if body.empty:
        return body.copy()
    m_tail = _nanmean_safe(body["mean_ff1"].tolist())
    s_tail = pooled_combined_std(body["std_ff1"].tolist(), body["num_valid"].tolist())
    foot = {
        "fold": "Aggregate",
        "num_valid": int(body["num_valid"].sum()),
        "mean_ff1": m_tail,
        "std_ff1": s_tail,
    }
    return pd.concat([body, pd.DataFrame([foot])], ignore_index=True)


def ff1_fold_summary_from_samples(df_valid: pd.DataFrame) -> pd.DataFrame:
    """
    One row per fold: mean Ff1 = nanmean across explainers of within-fold explainer means;
    std Ff1 = pooled population std of all valid Ff1 samples in the fold (all explainers).
    Computed only from raw rows (not from ff1_fold_explainer_stats).
    """
    folds = sorted(df_valid["fold"].unique().tolist())
    rows: list[dict[str, object]] = []
    for fold in folds:
        dff = df_valid.loc[df_valid["fold"] == fold]
        by_ex = dff.groupby("explainer", sort=False)[FF1_COLUMN].mean()
        mean_ff1 = _nanmean_safe(by_ex.tolist())
        std_ff1 = _population_std(dff[FF1_COLUMN])
        rows.append(
            {
                "fold": fold,
                "num_valid": int(len(dff)),
                "mean_ff1": mean_ff1,
                "std_ff1": std_ff1,
            }
        )
    body = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    if body.empty:
        return body
    foot = {
        "fold": "Aggregate",
        "num_valid": int(body["num_valid"].sum()),
        "mean_ff1": _nanmean_safe(body["mean_ff1"].tolist()),
        "std_ff1": pooled_combined_std(body["std_ff1"].tolist(), body["num_valid"].tolist()),
    }
    return pd.concat([body, pd.DataFrame([foot])], ignore_index=True)


def ff1_weighted_across_folds_from_samples(df_valid: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted Mean / Std of **fold-level** sample means (weights = per-fold ``num_valid``),
    computed **directly from sample rows** (not from aggregated per-fold explainer tables).

    Matches ``explainer_summary`` Valid **Weighted statistics across folds**, reduced to Mean Ff1 / Std columns.
    """
    folds_sorted = sorted(df_valid["fold"].unique().tolist())
    order = explainer_order(df_valid)
    body_rows: list[dict[str, object]] = []
    for ex in order:
        vals: list[float] = []
        wts: list[float] = []
        for f in folds_sorted:
            ser = df_valid.loc[(df_valid["explainer"] == ex) & (df_valid["fold"] == f), FF1_COLUMN]
            if len(ser) == 0:
                continue
            vals.append(float(ser.mean()))
            wts.append(float(len(ser)))
        wm = _weighted_mean(vals, wts) if len(vals) else float("nan")
        ws = _weighted_std(vals, wts) if len(vals) else float("nan")
        ntot = int(len(df_valid.loc[df_valid["explainer"] == ex]))
        body_rows.append(
            {
                "explainer": ex,
                "num_valid": ntot,
                "mean_ff1": wm,
                "std_ff1": ws,
            }
        )
    body = pd.DataFrame(body_rows)
    if body.empty:
        return body
    foot = {
        "explainer": "Aggregate",
        "num_valid": int(body["num_valid"].sum()),
        "mean_ff1": _nanmean_safe(body["mean_ff1"].tolist()),
        "std_ff1": pooled_combined_std(body["std_ff1"].tolist(), body["num_valid"].tolist()),
    }
    return pd.concat([body, pd.DataFrame([foot])], ignore_index=True)
