"""Streamlit: hierarchical explainer ranking from ``labeled_explanation_sample.csv``."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from mprov3_dui.ff1_tables import (
    explainer_order,
    ff1_fold_explainer_stats,
    ff1_fold_summary_from_samples,
    ff1_one_fold_explainer_table,
    ff1_per_explainer_fold_body,
    ff1_per_explainer_with_footer,
    ff1_weighted_across_folds_from_samples,
)
from mprov3_dui.latex_export import dataframe_to_booktabs_latex
from mprov3_dui.paths import default_labeled_sample_csv, folds_root_from_labeled_csv
from mprov3_dui.report_wall_times import (
    load_wall_time_s_map,
    runtime_explainer_wall_totals_html_parity,
    runtime_fold_mean_wall_html_parity,
)
from mprov3_dui.ranking import (
    METRIC_COLUMNS,
    RANKING_METRIC,
    best_class_per_fold_explainer,
    class_level_scores,
    fold_level_explainer_ranks,
    global_rank_aggregate,
)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    def _truthy(v: object) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v != 0
        s = str(v).strip().lower()
        return s in ("true", "1", "1.0", "yes", "t")

    return series.map(_truthy)


def _section_anchor(html_id: str) -> None:
    st.markdown(f'<span id="{html_id}"></span>', unsafe_allow_html=True)


def _file_slug_part(label: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", label, flags=re.ASCII)[:96]


def _show_table_download(
    *,
    section_id: str,
    title: str,
    description: str,
    df: pd.DataFrame,
    file_slug: str,
) -> None:
    _section_anchor(section_id)
    st.subheader(title)
    st.caption(description)
    st.markdown(f"**Rows:** {len(df):,}")
    if df.empty:
        st.caption("_No rows._")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    tex = dataframe_to_booktabs_latex(df, caption=None)
    st.download_button(
        "Download as LaTeX",
        data=tex,
        file_name=f"{file_slug}.tex",
        mime="text/plain",
        key=f"dl_{file_slug}",
    )


st.set_page_config(page_title="mprov3 — explainer ranking", layout="wide")
st.title("Hierarchical explainer ranking")

default_path = default_labeled_sample_csv()
csv_path = st.sidebar.text_input(
    "CSV path",
    value=str(default_path),
    help="Default: mprov3_explainer/results/labeled_explanation_sample.csv",
)

_infer_csv_for_folds = Path(csv_path.strip()) if csv_path.strip() else default_path
_default_folds = folds_root_from_labeled_csv(_infer_csv_for_folds)
folds_root_str = st.sidebar.text_input(
    "Folds root (for wall_time_s)",
    value=str(_default_folds),
    help="Directory with fold_*/explanations (default: results/folds next to the CSV). "
    "Runtime tables read comparison_report.json / explanation_report.json here—same source as HTML.",
)

if not csv_path.strip():
    st.warning("Set a CSV path.")
    st.stop()

path = Path(csv_path)

if not path.is_file():
    st.error(
        f"File not found: `{path}`. Run explanations first, e.g. "
        "`uv run python scripts/run_explanations.py` from `mprov3_explainer`."
    )
    st.stop()

df = pd.read_csv(path)

required = {
    "fold",
    "split",
    "explainer",
    "graph_id",
    "target_class",
    "pred_class",
    "correct_class",
    "valid",
} | set(METRIC_COLUMNS)
missing = required - set(df.columns)
if missing:
    st.error(f"CSV missing columns: {sorted(missing)}")
    st.stop()

if df.empty:
    st.warning("The CSV has no rows.")
    st.stop()

df["fold"] = df["fold"].astype(int)
df["target_class"] = df["target_class"].astype(int)
df["valid"] = _coerce_bool_series(df["valid"])

n_total = len(df)
df = df.loc[df["valid"]].copy()
if df.empty:
    st.warning("No rows with `valid` true; nothing to rank.")
    st.stop()

st.caption(
    f"`{path.name}`: **{len(df):,}** valid sample rows "
    f"({n_total:,} total rows in file). "
    f"Ranking metric: **`{RANKING_METRIC}`**. "
    "Class-level scores: **mean** per (fold, explainer, target_class); "
    "global summary: **mean** of within-fold ranks."
)

ff1_stats = ff1_fold_explainer_stats(df)

tab_rank, tab_xexpl, tab_xfold, tab_runtime = st.tabs(
    [
        "Ranking & exports",
        "Cross-explainer Ff1",
        "Cross-fold Ff1",
        "Runtime results",
    ],
)

with tab_rank:
    st.markdown("### Tables")
    st.markdown(
        "Jump to a section: "
        "[1. Raw explanation samples](#sec-01) · "
        "[2. Class-level metric aggregates](#sec-02) · "
        "[3. Best target class score](#sec-03) · "
        "[4. Within-fold explainer ranking](#sec-04) · "
        "[5. Global rank summary](#sec-05). "
        "_Links scroll the page in the browser view._"
    )

    samples_sorted = df.sort_values(["fold", "explainer", "graph_id"]).reset_index(drop=True)
    _show_table_download(
        section_id="sec-01",
        title="1. Raw explanation samples",
        description=(
            "Selection: rows with **`valid` true** only (same for all tables). "
            "Calculation: none; values are as stored for each explained graph. "
            "Sorted by fold, explainer, and graph id."
        ),
        df=samples_sorted,
        file_slug="01_per_sample",
    )

    class_scores = class_level_scores(df, RANKING_METRIC)
    _show_table_download(
        section_id="sec-02",
        title="2. Class-level metric aggregates",
        description=(
            "Selection: same valid samples as table 1. For each (fold, explainer, target_class), "
            f"**mean** of **{RANKING_METRIC}** across graphs (NaNs ignored). "
            "One aggregate per class bucket before taking the best class per explainer."
        ),
        df=class_scores,
        file_slug="02_class_scores",
    )

    best_cls = best_class_per_fold_explainer(class_scores)
    _show_table_download(
        section_id="sec-03",
        title="3. Best target class score per fold and explainer",
        description=(
            "Selection: rows from the class-level table (section 2). Calculation: for each pair "
            "(fold, explainer), **maximum** of the class scores—the best target class under "
            f"**{RANKING_METRIC}** for that fold and explainer. "
            "Ties keep one row (first after sorting by score descending)."
        ),
        df=best_cls,
        file_slug="03_best_class_per_fold_explainer",
    )

    fold_ranks = fold_level_explainer_ranks(best_cls)
    _show_table_download(
        section_id="sec-04",
        title="4. Within-fold explainer ranking",
        description=(
            "Selection: best-per-class scores (section 3). Calculation: within each **fold**, "
            "explainers are ranked by that best score (**rank 1** is best). Tied scores receive "
            "the **average rank** (pandas `method='average'`)."
        ),
        df=fold_ranks.sort_values(["fold", "explainer_rank"]),
        file_slug="04_fold_explainer_ranks",
    )

    global_ranks = global_rank_aggregate(fold_ranks)
    _show_table_download(
        section_id="sec-05",
        title="5. Global rank summary",
        description=(
            "Selection: within-fold ranks (section 4). Calculation: for each explainer, **mean** "
            "of **explainer_rank** across folds. "
            "Lower **rank_agg** indicates better overall ranking under mean-of-ranks."
        ),
        df=global_ranks,
        file_slug="05_global_rank_aggregate",
    )

with tab_xexpl:
    st.markdown("### Cross-explainer Ff1 (valid samples)")
    st.caption(
        "Metric: **`paper_f1_fidelity` (Ff1)**. Within each (fold, explainer), **Mean Ff1** is the "
        "sample mean and **Std Ff1** is the **population** standard deviation (divide by n); "
        "undefined when fewer than two graphs. **Aggregate** row: **Mean** column = mean of body "
        "means; **Std** column = pooled √(Σ n_k σ_k² / Σ n_k) over body rows with defined σ."
    )

    for i, ex in enumerate(explainer_order(df)):
        body = ff1_per_explainer_fold_body(ff1_stats, ex)
        full = ff1_per_explainer_with_footer(body)
        slug_ex = _file_slug_part(ex)
        _show_table_download(
            section_id=f"xe-per-{i}",
            title=f"Per-explainer across folds — {ex}",
            description=(
                "One row per fold for this explainer: valid graph count, mean and population std "
                "of Ff1 over valid explained graphs in that fold. Last row aggregates across folds."
            ),
            df=full,
            file_slug=f"xe_ff1_folds_{i:02d}_{slug_ex}",
        )

    fold_summary = ff1_fold_summary_from_samples(df)
    _show_table_download(
        section_id="xe-fold-summary",
        title="Summary by fold",
        description=(
            "Computed **only from raw sample rows** (not from the per-explainer tables above). "
            "Per fold: **Mean Ff1** = nanmean across explainers of each explainer’s within-fold "
            "mean Ff1 (each explainer counts equally). **Std Ff1** = population std of Ff1 over "
            "**all** valid graphs in that fold (all explainers pooled). **Aggregate** uses the "
            "same mean-of-means and pooled-std rule on the fold rows (weights = fold graph counts)."
        ),
        df=fold_summary,
        file_slug="xe_ff1_summary_by_fold",
    )

with tab_xfold:
    st.markdown("### Cross-fold Ff1 (valid samples)")
    xf_expl = explainer_order(df)
    xf_folds = sorted(df["fold"].unique().tolist())
    st.caption(
        "Like `explainer_summary` **Per-fold explainer metrics** (valid): per fold, all explainers in "
        "sort order—**Mean Ff1**, **Std Ff1**, and valid graph counts from raw rows. Footer: mean "
        "of means and pooled combined std √(Σ n·σ²)/Σ n. **Weighted statistics across folds**: same "
        "Mean/Std Ff1 as the HTML weighted block, but recomputed directly from samples (never from "
        "the per-fold tables above)."
    )

    for i, fold in enumerate(xf_folds):
        xf_tbl = ff1_one_fold_explainer_table(df, int(fold), xf_expl)
        slug_f = _file_slug_part(str(fold))
        _show_table_download(
            section_id=f"xf-fold-{i}",
            title=f"Per-fold explainer metrics — Fold {fold}",
            description=(
                "Valid **Mean Ff1** / **Std Ff1** only: one row per explainer (explainers absent in "
                "this fold show 0 valid graphs). **Aggregate** row: mean of Mean Ff1 and pooled Std."
            ),
            df=xf_tbl,
            file_slug=f"xf_ff1_fold_{slug_f}_{i:02d}",
        )

    weighted_xf = ff1_weighted_across_folds_from_samples(df)
    _show_table_download(
        section_id="xf-weighted",
        title="Weighted statistics across folds",
        description=(
            "Computed **only from raw sample rows** (not derived from the per-fold tables above). "
            "Per explainer: **Mean Ff1** = Σ_f n_f μ_f / Σ_f n_f and **Std Ff1** = weighted RMS of fold "
            "means μ_f about that mean (weights n_f = valid graphs per fold), matching "
            "`explanation_web_report` weighted Mean/Std. **Aggregate**: mean of means and pooled std."
        ),
        df=weighted_xf,
        file_slug="xf_ff1_weighted_across_folds",
    )

with tab_runtime:
    st.markdown("### Runtime (HTML-parity **`wall_time_s`**)")

    folds_root_ui = Path(folds_root_str.strip()) if folds_root_str.strip() else _default_folds
    wall_map = load_wall_time_s_map(folds_root_ui)
    folds_scope = sorted(int(f) for f in df["fold"].unique())

    doc = (
        "Values match **`result_metrics.wall_time_s`** in **`comparison_report.json`** per fold "
        "(same payload the static HTML builder uses via `generate_visualizations`). "
        "If that file is missing, each **`explanation_report.json`** under the fold is read instead.\n\n"
        f"**Fold scope:** rows use folds **`{folds_scope}`** from your CSV union with folds found under "
        f"`{folds_root_ui.resolve()}` (both must overlap for totals to match cross-fold HTML)."
    )
    st.caption(doc)

    if not wall_map:
        st.warning(
            f"No **`wall_time_s`** data found under `{folds_root_ui.resolve()}`. "
            "Check **Folds root** points at `results/folds` containing `fold_*` runs."
        )
    elif not folds_scope:
        st.warning("No folds in CSV.")
    else:
        folds_missing = sorted(f for f in folds_scope if f not in wall_map)
        if folds_missing:
            st.warning(
                f"No JSON timings for folds {folds_missing} under this folds root; "
                "those folds are omitted (HTML may include more folds if you rerun with a fuller tree)."
            )

        folds_used = sorted(f for f in folds_scope if f in wall_map)
        if not folds_used:
            st.error("CSV folds do not overlap any fold dirs with timing JSON.")
        else:
            by_expl_total = runtime_explainer_wall_totals_html_parity(wall_map, folds_used)
            _show_table_download(
                section_id="rt-expl-mean-fold",
                title="Mean across folds — Result runtime (narrow)",
                description=(
                    "Per explainer: **Wall (s) total** = sum of `wall_time_s` over folds in scope "
                    "(same aggregation as explainer_summary → Mean across folds → Result metrics)."
                ),
                df=by_expl_total,
                file_slug="rt_wall_total_by_explainer",
            )

            by_fold = runtime_fold_mean_wall_html_parity(wall_map, folds_used)
            _show_table_download(
                section_id="rt-fold-summary",
                title="Summary by fold — Result runtime (narrow)",
                description=(
                    "Per fold: **Wall (s)** = mean across explainers' `wall_time_s` "
                    "(same as index → Summary by fold → Result metrics)."
                ),
                df=by_fold,
                file_slug="rt_wall_mean_by_fold",
            )
