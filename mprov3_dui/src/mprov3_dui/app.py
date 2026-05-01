"""Streamlit: hierarchical explainer ranking from ``labeled_explanation_sample.csv``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mprov3_dui.latex_export import dataframe_to_booktabs_latex
from mprov3_dui.paths import default_labeled_sample_csv
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
