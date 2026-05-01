# mprov3_dui — Explainer ranking and LaTeX export

Interactive **Streamlit** UI for hierarchical ranking of explainers from the consolidated CSV produced by `mprov3_explainer/scripts/run_explanations.py`.

## Input data

The app reads:

`mprov3_explainer/results/labeled_explanation_sample.csv`

Generate or refresh it by running explanations from the explainer project (example):

```bash
cd mprov3_explainer
uv run python scripts/run_explanations.py
# or multiple folds, e.g.:
# uv run python scripts/run_explanations.py --folds 0 1 2
```

## Install and run (uv)

From this directory:

```bash
cd mprov3_dui
uv sync
uv run mprov3-dui
```

This starts Streamlit with the ranking UI. Alternatively:

```bash
uv run streamlit run src/mprov3_dui/app.py
```

You can change the CSV path in the sidebar if your workspace layout differs.

## Hierarchical ranking (summary)

Ranking uses the fixed metric **`paper_f1_fidelity`** (`Ff1`).

1. **Raw samples (table 1):** Rows with **`valid` true** only (invalid graphs are dropped after load).
2. **Class-level (table 2):** For each `(fold, explainer, target_class)`, **mean** of `paper_f1_fidelity` over those valid rows.
3. **Best class (table 3):** For each `(fold, explainer)`, **maximum** of those class means.
4. **Within-fold ranks (table 4):** Rank explainers inside each fold by that best score (rank 1 = best).
5. **Global summary (table 5):** For each explainer, **mean** of its within-fold ranks across folds.

Each table shows a **row count** above its grid.

A table-of-contents block at the top links to each results table.

## LaTeX export

Each table has a **Download as LaTeX** button. The file is a `tabular` using **booktabs** (`\toprule`, `\midrule`, `\bottomrule`). Add to your preamble:

```latex
\usepackage{booktabs}
```

Floating-point cells are printed with **six digits** after the decimal point.
