from __future__ import annotations

import pandas as pd

METRIC_COLUMNS: tuple[str, ...] = (
    "paper_sufficiency",
    "paper_comprehensiveness",
    "paper_f1_fidelity",
    "pyg_fidelity_plus",
    "pyg_fidelity_minus",
    "pyg_characterization_score",
    "pyg_fidelity_curve_auc",
    "pyg_unfaithfulness",
)

# Fixed metric for hierarchical ranking in the Streamlit UI.
RANKING_METRIC = "paper_f1_fidelity"


def class_level_scores(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Mean of *metric_col* per (fold, explainer, target_class)."""
    g = df.groupby(["fold", "explainer", "target_class"], dropna=False)[metric_col]
    out = g.mean().reset_index(name="class_score")
    return out.sort_values(["fold", "explainer", "target_class"]).reset_index(drop=True)


def best_class_per_fold_explainer(class_scores: pd.DataFrame) -> pd.DataFrame:
    """Maximum *class_score* per (fold, explainer)."""
    cs = class_scores.copy()
    return (
        cs.sort_values(
            ["fold", "explainer", "class_score"],
            ascending=[True, True, False],
            na_position="last",
        )
        .groupby(["fold", "explainer"], as_index=False)
        .first()
        .sort_values(["fold", "explainer"])
        .reset_index(drop=True)
    )


def fold_level_explainer_ranks(best_df: pd.DataFrame) -> pd.DataFrame:
    """Rank explainers within each fold by *class_score* (1 = best)."""
    out = best_df.copy()
    out["explainer_rank"] = out.groupby("fold")["class_score"].rank(
        ascending=False,
        method="average",
    )
    return out.sort_values(["fold", "explainer_rank", "explainer"]).reset_index(drop=True)


def global_rank_aggregate(fold_ranks: pd.DataFrame) -> pd.DataFrame:
    """Mean of fold ranks per explainer."""
    rank_agg = fold_ranks.groupby("explainer")["explainer_rank"].mean()
    out = (
        pd.DataFrame({"rank_agg": rank_agg})
        .reset_index()
        .sort_values(["rank_agg", "explainer"])
        .reset_index(drop=True)
    )
    return out
