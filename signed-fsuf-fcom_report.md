# Concern: `signed-fsuf-fcom`

## Diagnostic

`general_report_co.md` notes that per-graph `Fsuf` and `Fcom` can be negative.
The implementation computes both as raw signed probability differences:
`P_target(full) - P_target(subgraph)` and
`P_target(full) - P_target(complement)`. Negative values are possible whenever
the explanation subgraph or complement raises the target-class probability
relative to the full graph. The code already clamps both operands before
combining them into `Ff1`, but the report labels did not make that distinction
visible enough.

Relevant code inspected:

- `mprov3_explainer/src/mprov3_explainer/pipeline.py`
- `mprov3_explainer/README.md`
- `mprov3_explainer/src/mprov3_explainer/web_report.py`
- `mprov3_explainer/src/mprov3_explainer/visualize.py`

## Strategy

Keep raw `Fsuf` and `Fcom` unchanged for transparency. Clarify everywhere that
they are signed probability differences, and reserve "clamped" language for
`Ff1`, where clamping is actually applied.

## Changes Applied

- Rewrote README metric definitions for `mean_paper_sufficiency`,
  `mean_paper_comprehensiveness`, and `mean_paper_f1_fidelity`.
- Updated pipeline docstrings to state that `Fsuf` and `Fcom` can be negative.
- Updated HTML and legacy visualization labels to `Fsuf (raw)`,
  `Fcom (raw)`, and `Ff1 (clamped)`.

## Thesis Handling

Interpret negative `Fsuf` or `Fcom` as a model behavior diagnostic, not a JSON
or arithmetic error. `Ff1` is bounded because it clamps those raw values before
the harmonic-style combination.
