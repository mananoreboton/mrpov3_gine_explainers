# Concern: `pgmexpl-misclass`

## Diagnostic

`general_report_co.md` reports that `PGMEXPL` has fold-level misclassification
counts that differ from every other explainer. Misclassification should be a
property of the trained GINE model, the fold, and the split. It should not vary
with the explainer, even for perturbation-based methods.

Relevant code inspected:

- `mprov3_explainer/src/mprov3_explainer/pipeline.py`
- `mprov3_explainer/scripts/run_explanations.py`
- `mprov3_explainer/src/mprov3_explainer/explainers.py`
- `mprov3_explainer/src/mprov3_explainer/web_report.py`

## Strategy

Decouple model correctness from explainer execution. The fold runner should
compute model predictions once, before any explainer is built or invoked, then
reuse that baseline for `correct_class` and `num_misclassified` in every
explainer report. If an explainer path changes the observed prediction later,
that should be reported as prediction drift rather than silently changing the
misclassification count.

## Changes Applied

- Added `PredictionBaselineEntry` and `collect_prediction_baseline(...)`.
- Passed the precomputed baseline into every explainer run for a fold.
- Added `pred_class`, `target_class`, and `prediction_baseline_mismatch` to
  per-graph result records.
- Added `num_prediction_baseline_mismatch` to per-explainer JSON summaries.
- Wrote `model_prediction_baseline.json` alongside fold explanation outputs.
- Added HTML report columns for prediction drift and per-graph baseline fields.
- Added an end-to-end regression test proving `run_explanations(...)` honors a
  supplied prediction baseline.

## Thesis Handling

For any final result set regenerated with this code, `Misclass.` should be
constant across explainers within the same fold and split. Nonzero
`Pred. drift` counts should be treated as a run-quality warning.
