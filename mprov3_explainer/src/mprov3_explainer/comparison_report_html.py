"""Write a self-contained HTML comparison table for explainer metric summaries."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping


def _metric_cell(row: Mapping[str, Any], key: str) -> str:
    v = row.get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return format(float(v), ".4f")
    return "—"


def write_comparison_report_html(
    output_path: Path,
    comparison: Mapping[str, Any],
    *,
    paper_metrics_computed: bool,
    fidelity_valid_only: bool,
) -> None:
    """Write ``comparison_report.html`` next to ``comparison_report.json``.

    *comparison* must include ``per_explainer`` (dict of summary dicts) and may
    include ``generated_at`` or legacy ``timestamp``, and ``explainers`` (row order).
    """
    per_explainer = comparison.get("per_explainer") or {}
    if not isinstance(per_explainer, dict):
        raise TypeError("comparison['per_explainer'] must be a dict")

    ts = comparison.get("generated_at") or comparison.get("timestamp")
    order = comparison.get("explainers")
    if isinstance(order, list) and order:
        names = [str(x) for x in order if str(x) in per_explainer]
        for k in per_explainer:
            if k not in names:
                names.append(k)
    else:
        names = sorted(per_explainer.keys(), key=str)

    ts_esc = html.escape(str(ts), quote=True) if ts is not None else ""
    title = f"Explainer metrics — {ts_esc}" if ts_esc else "Explainer metrics"

    rows_html: list[str] = []
    for name in names:
        row = per_explainer.get(name)
        if not isinstance(row, dict):
            continue
        ne = html.escape(str(name), quote=True)

        rows_html.append(
            "<tr>"
            f"<th scope='row'>{ne}</th>"
            f"<td class='num'>{_metric_cell(row, 'mean_fid_plus')}</td>"
            f"<td class='num'>{_metric_cell(row, 'mean_fid_minus')}</td>"
            f"<td class='num'>{_metric_cell(row, 'mean_pyg_characterization')}</td>"
            f"<td class='num'>{_metric_cell(row, 'mean_paper_sufficiency')}</td>"
            f"<td class='num'>{_metric_cell(row, 'mean_paper_comprehensiveness')}</td>"
            f"<td class='num'>{_metric_cell(row, 'mean_paper_f1_fidelity')}</td>"
            f"<td class='num'>{int(row.get('num_graphs', 0))}</td>"
            f"<td class='num'>{int(row.get('num_valid', 0))}</td>"
            f"<td class='num'>{format(float(row.get('wall_time_s', 0.0)), '.2f')}</td>"
            "</tr>"
        )

    fid_note = (
        "Mean fid+ and fid− are averaged over <strong>valid</strong> graphs only."
        if fidelity_valid_only
        else "Mean fid+ and fid− are averaged over <strong>all</strong> graphs in the run."
    )
    other_note = (
        "Characterization and paper columns (Fsuf, Fcom, Ff1) are simple means over "
        "<strong>all</strong> graphs (same as in <code>explanation_report.json</code>), "
        "not restricted to valid-only."
    )
    paper_note = (
        "Paper metrics were computed (threshold sweep)."
        if paper_metrics_computed
        else (
            "Paper metrics were <strong>not</strong> computed; Fsuf, Fcom, and Ff1 are "
            "zeros in the table."
        )
    )

    legend = (
        "<section class='legend' aria-labelledby='legend-heading'>\n"
        "<h2 id='legend-heading'>How to read this table</h2>\n"
        "<ul>\n"
        "<li><strong>fid+ / fid−</strong> — PyTorch Geometric graph-level fidelity: how much the "
        "predicted probability for the explained class changes when the explanation mask is applied "
        "versus its complement. Interpretation follows PyG&rsquo;s <code>fidelity()</code> "
        f"(sign convention as in PyG). {fid_note}</li>\n"
        "<li><strong>characterization</strong> — PyG <code>characterization_score</code> combining "
        f"fid+ and fid− with equal weights (0.5 / 0.5). {other_note}</li>\n"
        "<li><strong>Fsuf / Fcom / Ff1</strong> — Longa et al.&ndash;style metrics from a threshold "
        "sweep over the node mask (edge masks are converted to a node mask for this step). "
        "<strong>Fsuf</strong> averages the drop from the full-graph class probability when keeping "
        f"only nodes above each threshold; <strong>Fcom</strong> averages the drop when using the "
        f"complement; <strong>Ff1</strong> combines the two. {paper_note}</li>\n"
        "<li><strong>graphs / valid</strong> — Total graphs explained and count marked valid after "
        "preprocessing (e.g. correct prediction when that filter is on).</li>\n"
        "<li><strong>wall (s)</strong> — Wall-clock time for that explainer over the whole run.</li>\n"
        "</ul>\n"
        "</section>\n"
    )

    doc = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>{html.escape(title, quote=True)}</title>
<style>
body {{ font-family: sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; }}
h1 {{ margin-bottom: 0.25em; }}
.timestamp {{ color: #666; font-size: 14px; margin-bottom: 1em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 14px; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.5em; text-align: left; }}
th {{ background: #f5f5f5; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tbody tr:nth-child(even) {{ background: #fafafa; }}
.legend ul {{ margin: 0.5em 0; padding-left: 1.25em; }}
.legend li {{ margin: 0.35em 0; }}
code {{ font-size: 0.95em; }}
</style>
</head>
<body>
<h1>Explainer metrics comparison</h1>
<p class='timestamp'>Run: {ts_esc or '—'}</p>
{legend}
<table>
<thead>
<tr>
<th scope='col'>Explainer</th>
<th scope='col' class='num'>fid+</th>
<th scope='col' class='num'>fid−</th>
<th scope='col' class='num'>characterization</th>
<th scope='col' class='num'>Fsuf</th>
<th scope='col' class='num'>Fcom</th>
<th scope='col' class='num'>Ff1</th>
<th scope='col' class='num'>graphs</th>
<th scope='col' class='num'>valid</th>
<th scope='col' class='num'>wall (s)</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
