# Concern: `pgexpl-degenerate`

## Diagnostic

`general_report_co.md` reports that every `PGEXPL` fold has `num_valid = 0`,
`mean_mask_spread = 0`, and all headline means serialized as missing or `NaN`.
The code already filters degenerate masks through `MASK_SPREAD_TOLERANCE`, but
the result JSON and HTML report did not give the row an explicit failure state.
That made it too easy to treat `PGEXPL` as a normal explainer with merely low or
missing scores.

Relevant code inspected:

- `mprov3_explainer/src/mprov3_explainer/pipeline.py`
- `mprov3_explainer/scripts/run_explanations.py`
- `mprov3_explainer/src/mprov3_explainer/web_report.py`
- `mprov3_explainer/README.md`

## Strategy

Make all-degenerate runs machine-readable as failed runs. The thesis evidence
can then exclude or footnote `PGEXPL` headline metrics without relying on a
manual interpretation of `num_valid`, `Degen.`, and missing means.

## Changes Applied

- Added `diagnose_explanation_run(...)` to classify empty, all-degenerate,
  partially degenerate, and otherwise valid explainer runs.
- Added `run_status` and `run_status_note` to per-explainer JSON reports and
  cross-explainer summaries.
- Added an HTML `Status` column and per-explainer warning text for non-`ok`
  statuses.
- Documented `failed_all_degenerate_masks` in the explainer README.
- Added a regression test covering the all-degenerate failure state.

## Thesis Handling

`PGEXPL` rows with `run_status = "failed_all_degenerate_masks"` should be
reported as failed explainer runs, not as valid comparative metric evidence.
