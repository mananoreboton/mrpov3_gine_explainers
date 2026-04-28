# Concern: `all-graph-diagnostics`

## Diagnostic

`general_report_co.md` warns that valid-only means should not be the only thesis
conclusion. The code already writes all-graph diagnostic fields under
`mean_*_all_graphs`, but the main HTML summary tables emphasized only the
valid-only headline means. That creates a reporting risk: a method can look
strong on valid graphs while silently losing many graphs to correctness,
degenerate-mask, or metric-failure filtering.

Relevant code inspected:

- `mprov3_explainer/scripts/run_explanations.py`
- `mprov3_explainer/src/mprov3_explainer/web_report.py`
- `mprov3_explainer/README.md`

## Strategy

Keep valid-only means as headline metrics, but surface all-graph diagnostic
means in the generated fold and cross-fold HTML reports. Update the README so
the thesis workflow explicitly compares both views.

## Changes Applied

- Added all-graph metric columns to the per-fold HTML summary table.
- Added all-graph diagnostic columns to the cross-fold global index.
- Updated global index explanatory text to distinguish headline valid-only
  metrics from `All` diagnostics.
- Reworded the README so `mean_*_all_graphs` are treated as required
  diagnostics rather than legacy fallback values.

## Thesis Handling

Use valid-only means as the clean headline comparison, but discuss all-graph
diagnostics wherever filtering rates differ materially across explainers or
folds.
