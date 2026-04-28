# Concern: `normalized-entropy`

## Diagnostic

`general_report_co.md` notes that raw mask entropy is bounded by `log(M)`, where
`M` is the number of mask entries. Because molecule sizes differ and node masks
and edge masks have different lengths, raw entropy is not comparable across
graphs or explainers.

Relevant code inspected:

- `mprov3_explainer/src/mprov3_explainer/pipeline.py`
- `mprov3_explainer/scripts/run_explanations.py`
- `mprov3_explainer/src/mprov3_explainer/web_report.py`
- `mprov3_explainer/README.md`
- `mprov3_explainer/tests`

## Strategy

Keep raw entropy as a diagnostic, but add a normalized entropy field computed
as `entropy / log(mask_size)`. Use the normalized value in comparison-facing
report columns.

## Changes Applied

- Added `mask_entropy_normalized` to per-graph `ExplanationResult` records.
- Added `mean_mask_entropy_normalized` to per-explainer and comparison JSON.
- Added normalized entropy columns to fold and cross-fold HTML reports.
- Updated README guidance so raw entropy is not used alone for explainer
  comparison.
- Added tests proving uniform masks normalize to 1 regardless of mask size and
  point-mass masks normalize to 0.

## Thesis Handling

Use normalized entropy when comparing how diffuse explainer masks are. Raw
entropy can still be mentioned as an audit diagnostic, but not as the primary
cross-explainer comparison.
