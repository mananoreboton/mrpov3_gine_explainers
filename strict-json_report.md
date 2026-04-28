# Concern: `strict-json`

## Diagnostic

`general_report_co.md` explains that `json.dumps(..., default=_nan_to_none)` is
not enough to make strict JSON. Python's JSON encoder handles `float("nan")`
itself, so the `default` callback is never called for NaN or infinity. As a
result, result files can contain bare `NaN`, which strict JSON parsers reject.

Relevant code inspected:

- `mprov3_explainer/scripts/run_explanations.py`
- `mprov3_explainer/src/mprov3_explainer/json_utils.py`
- `mprov3_explainer/tests`

## Strategy

Sanitize artifacts recursively before encoding and call `json.dumps` with
`allow_nan=False`. That converts non-finite floats to `null` and makes future
regressions fail during serialization instead of leaking invalid JSON.

## Changes Applied

- Added `to_strict_jsonable(...)` and `dumps_strict_json(...)`.
- Switched per-explainer reports, comparison reports, prediction baselines, and
  mask JSON files to strict serialization.
- Documented that NaN and infinity serialize to `null`.
- Added unit tests proving nested NaN and infinity values become JSON `null`.

## Thesis Handling

Regenerated result JSON files can be parsed by strict JSON tooling. Existing
older artifacts with bare `NaN` should be regenerated or normalized before
being archived as thesis evidence.
