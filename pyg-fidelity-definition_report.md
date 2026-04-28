# Concern: `pyg-fidelity-definition`

## Diagnostic

`general_report_co.md` identifies a terminology error: the code calls PyG's
`torch_geometric.explain.metric.fidelity`, but the README described
`Mean Fid+` and `Mean Fid-` as probability-drop ratios. PyG's implementation is
decision based. In model-explanation mode it compares predicted classes under
full, explanation-only, and complement perturbations. In phenomenon mode it
uses target-class correctness.

Relevant code inspected:

- `mprov3_explainer/src/mprov3_explainer/pipeline.py`
- `mprov3_explainer/README.md`
- `mprov3_explainer/src/mprov3_explainer/web_report.py`

## Strategy

Keep the implementation unchanged because it already calls PyG's fidelity
metric correctly. Fix the documentation and HTML wording so thesis readers do
not interpret `Fid+` and `Fid-` as probability ratios.

## Changes Applied

- Rewrote the `mean_fidelity_plus` and `mean_fidelity_minus` README rows as
  PyG/GraphFramEx class-decision metrics.
- Added an explicit note that they are not probability-drop ratios.
- Updated the pipeline fidelity docstrings to state the model-mode and
  phenomenon-mode semantics.
- Added an HTML report note above the summary table.

## Thesis Handling

Use `Mean Fid+` and `Mean Fid-` as PyG class-decision GraphFramEx rates. Use
`Fsuf` and `Fcom` for probability-difference interpretations.
