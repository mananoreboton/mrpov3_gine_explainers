"""Static HTML reports: per-fold explainer metrics and a global cross-fold index.

The fold report renders the two metric tables produced by ``run_explanations``
side by side:

* **Valid Result Metrics**: aggregated only over graphs with ``valid == True``.
* **Result Metrics**: aggregated over every explained graph in the fold,
  plus the wall-clock runtime for the explainer.

The global index aggregates the same two tables across folds.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mprov3_gine_explainer_defaults import RESULTS_EXPLANATIONS

from mprov3_explainer.explainers import get_spec
from mprov3_explainer.paths import visualizations_run_dir

EXPLANATION_WEB_REPORT_DIR = "explanation_web_report"


# ---------------------------------------------------------------------------
# Metric layout shared by the fold report and the global index
# ---------------------------------------------------------------------------


_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("paper_sufficiency", "Mean Fsuf"),
    ("paper_comprehensiveness", "Mean Fcom"),
    ("paper_f1_fidelity", "Mean Ff1"),
    ("pyg_fidelity_plus", "Mean Fid+"),
    ("pyg_fidelity_minus", "Mean Fid\u2212"),
    ("pyg_characterization_score", "Mean char"),
    ("pyg_fidelity_curve_auc", "Mean AUC"),
    ("pyg_unfaithfulness", "Mean GEF"),
)

_VALID_TABLE_COUNT_COL = ("num_valid_graphs", "Valid graphs", "num")
_RESULT_TABLE_LEADING_COLS: tuple[tuple[str, str, str], ...] = (
    ("wall_time_s", "Wall (s)", "num"),
    ("num_graphs", "Graphs", "num"),
)


def _fmt_num(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, int):
        return str(x)
    return html.escape(str(x))


def _sort_value(val: Any, tp: str) -> str:
    if tp != "num":
        return ""
    if val is None:
        return ""
    try:
        return str(float(val))
    except (TypeError, ValueError):
        return ""


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-fold report
# ---------------------------------------------------------------------------


def _valid_table_columns() -> list[tuple[str, str, str]]:
    cols: list[tuple[str, str, str]] = [
        ("explainer", "Explainer", "text"),
        ("run_status", "Status", "text"),
        _VALID_TABLE_COUNT_COL,
    ]
    cols.extend((f"mean_{key}", label, "num") for key, label in _METRIC_COLUMNS)
    return cols


def _result_table_columns() -> list[tuple[str, str, str]]:
    cols: list[tuple[str, str, str]] = [
        ("explainer", "Explainer", "text"),
        ("run_status", "Status", "text"),
    ]
    cols.extend(_RESULT_TABLE_LEADING_COLS)
    cols.extend((f"mean_{key}", label, "num") for key, label in _METRIC_COLUMNS)
    return cols


def _render_table(
    *,
    table_id: str,
    cols: list[tuple[str, str, str]],
    rows: list[dict[str, Any]],
) -> str:
    """Render one summary table (header row + body rows)."""
    thead = "".join(
        f'<th data-sort="{html.escape(key)}" data-type="{tp}">{html.escape(label)}</th>'
        for key, label, tp in cols
    )
    tbody_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for key, _, tp in cols:
            val = row.get(key)
            if key == "explainer":
                cells.append(f'<td data-col="explainer">{html.escape(str(val))}</td>')
            else:
                cells.append(
                    f'<td data-col="{html.escape(key)}" data-sort-value="{_sort_value(val, tp)}">'
                    f"{_fmt_num(val)}</td>"
                )
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = (
        "\n".join(tbody_rows)
        if tbody_rows
        else f'<tr><td colspan="{len(cols)}">No data</td></tr>'
    )
    return (
        f'<table class="summary" id="{html.escape(table_id, quote=True)}">'
        f"<thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"
    )


def _summary_row(
    *,
    explainer_name: str,
    block: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a metric block into a row indexed by column key."""
    row: dict[str, Any] = {
        "explainer": explainer_name,
        **(extra or {}),
    }
    for key, value in block.items():
        row[key] = value
    return row


def _per_graph_card(
    *,
    explainer_name: str,
    entry: dict[str, Any],
    masks_dir: Path,
    graphs_dir: Path,
) -> str:
    graph_id = entry.get("graph_id") or ""
    if not graph_id:
        return ""

    mask_path = masks_dir / f"{graph_id}.json"
    png_path = graphs_dir / f"mask_{graph_id}.png"
    rel_png = f"../visualizations/{explainer_name}/graphs/mask_{graph_id}.png"
    rel_json = f"../explanations/{explainer_name}/masks/{graph_id}.json"

    if mask_path.is_file():
        raw_json = html.escape(mask_path.read_text(encoding="utf-8"))
        mask_note = ""
    else:
        raw_json = "(mask JSON file missing)"
        mask_note = ' <span class="warn">No mask file</span>'

    if png_path.is_file():
        img_block = (
            f'<img src="{html.escape(rel_png, quote=True)}" '
            f'alt="Mask visualization {html.escape(graph_id)}" loading="lazy">'
        )
    else:
        img_block = '<p class="muted">PNG not rendered (missing or draw failed).</p>'

    per_keys: list[tuple[str, str]] = [
        ("paper_sufficiency", "Fsuf"),
        ("paper_comprehensiveness", "Fcom"),
        ("paper_f1_fidelity", "Ff1"),
        ("pyg_fidelity_plus", "Fid+"),
        ("pyg_fidelity_minus", "Fid\u2212"),
        ("pyg_characterization_score", "char"),
        ("pyg_fidelity_curve_auc", "AUC"),
        ("pyg_unfaithfulness", "GEF"),
        ("valid", "valid"),
        ("correct_class", "correct class"),
        ("pred_class", "pred class"),
        ("target_class", "target class"),
        ("prediction_baseline_mismatch", "baseline mismatch"),
        ("has_node_mask", "node mask"),
        ("has_edge_mask", "edge mask"),
        ("elapsed_s", "time (s)"),
    ]
    tcells = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{_fmt_num(entry.get(key))}</td></tr>"
        for key, label in per_keys
    )

    return (
        f'<article class="graph-card" data-graph-id="{html.escape(graph_id, quote=True)}">'
        f'<h4 id="g-{html.escape(explainer_name, quote=True)}-{html.escape(graph_id, quote=True)}">'
        f"{html.escape(graph_id)}</h4>"
        f"{mask_note}"
        f'<table class="mini"><tbody>{tcells}</tbody></table>'
        f'<div class="img-wrap">{img_block}</div>'
        f"<details><summary>Raw mask JSON</summary>"
        f'<p><a href="{html.escape(rel_json, quote=True)}">Open mask file path</a> (local)</p>'
        f'<pre class="json">{raw_json}</pre></details>'
        f"</article>"
    )


def write_fold_explanation_web_report(
    fold_root: Path,
    fold_index: int,
    explainer_names: list[str],
) -> Path:
    """
    Write ``fold_root/explanation_web_report/index.html`` from existing
    ``explanations/`` and ``visualizations/`` artifacts.
    """
    explanations_base = fold_root / RESULTS_EXPLANATIONS
    comparison = _load_json(explanations_base / "comparison_report.json")

    valid_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    sections_html: list[str] = []
    nav_links: list[str] = []

    for explainer_name in explainer_names:
        report_path = explanations_base / explainer_name / "explanation_report.json"
        report = _load_json(report_path)
        if not report:
            continue

        run_status = report.get("run_status", "ok")
        run_status_note = report.get("run_status_note", "")

        valid_block = report.get("valid_result_metrics", {}) or {}
        result_block = report.get("result_metrics", {}) or {}

        valid_rows.append(_summary_row(
            explainer_name=explainer_name,
            block=valid_block,
            extra={"run_status": run_status},
        ))
        result_rows.append(_summary_row(
            explainer_name=explainer_name,
            block=result_block,
            extra={"run_status": run_status},
        ))

        nav_links.append(
            f'<a href="#explainer-{html.escape(explainer_name, quote=True)}">'
            f"{html.escape(explainer_name)}</a>"
        )

        try:
            blurb = html.escape(get_spec(explainer_name).report_paragraph)
        except ValueError:
            blurb = ""
        if run_status != "ok":
            blurb = (
                f"{blurb} "
                f"Run status: {html.escape(str(run_status))}. "
                f"{html.escape(str(run_status_note))}"
            ).strip()

        masks_dir = explanations_base / explainer_name / "masks"
        graphs_dir = visualizations_run_dir(fold_root, explainer_name) / "graphs"
        cards = [
            card for card in (
                _per_graph_card(
                    explainer_name=explainer_name,
                    entry=entry,
                    masks_dir=masks_dir,
                    graphs_dir=graphs_dir,
                )
                for entry in report.get("per_graph", [])
            )
            if card
        ]
        cards_joined = "\n".join(cards) if cards else "<p class=\"muted\">No per-graph entries.</p>"
        sections_html.append(
            f'<section class="explainer-section" id="explainer-{html.escape(explainer_name, quote=True)}">'
            f"<h2>{html.escape(explainer_name)}</h2>"
            f'<p class="blurb">{blurb}</p>'
            f'<div class="graph-grid">{cards_joined}</div>'
            f"</section>"
        )

    valid_table_html = _render_table(
        table_id="valid-result-table",
        cols=_valid_table_columns(),
        rows=valid_rows,
    )
    result_table_html = _render_table(
        table_id="result-table",
        cols=_result_table_columns(),
        rows=result_rows,
    )

    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comp_block = ""
    if comparison:
        seed_val = comparison.get("seed")
        seed_dt = (
            f"<dt>seed</dt><dd>{html.escape(str(seed_val))}</dd>"
            if seed_val is not None
            else ""
        )
        comp_block = (
            "<h3>Comparison report (on disk)</h3>"
            "<dl class=\"meta-dl\">"
            f"<dt>generated_at</dt><dd>{html.escape(str(comparison.get('generated_at', '')))}</dd>"
            f"<dt>fold_index</dt><dd>{html.escape(str(comparison.get('fold_index', '')))}</dd>"
            f"<dt>fold_metric</dt><dd>{html.escape(str(comparison.get('fold_metric', '')))}</dd>"
            f"<dt>split</dt><dd>{html.escape(str(comparison.get('split', '')))}</dd>"
            f"{seed_dt}"
            "</dl>"
        )

    nav_html = " · ".join(nav_links) if nav_links else ""

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Explainer report — fold {html.escape(str(fold_index))}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --border: #2d3a4d;
      --text: #e6edf3;
      --muted: #8b9cb3;
      --accent: #58a6ff;
      --warn: #d29922;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      margin: 0 auto;
      padding: 1rem 1.25rem 3rem;
      max-width: 1200px;
    }}
    a {{ color: var(--accent); }}
    header {{ margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.5rem; }}
    h2 {{ font-size: 1.15rem; margin-top: 2rem; border-top: 1px solid var(--border); padding-top: 1.25rem; }}
    h3 {{ font-size: 1rem; color: var(--muted); }}
    h4 {{ font-size: 0.95rem; margin: 0 0 0.5rem; }}
    .muted {{ color: var(--muted); }}
    .warn {{ color: var(--warn); }}
    .blurb {{ color: var(--muted); font-size: 0.9rem; max-width: 70ch; }}
    nav {{ font-size: 0.9rem; margin: 0.75rem 0; line-height: 1.6; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; margin: 1rem 0; }}
    .controls label {{ font-size: 0.85rem; color: var(--muted); }}
    input[type="search"] {{
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.35rem 0.6rem;
      border-radius: 6px;
      min-width: 200px;
    }}
    table.summary {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin: 1rem 0;
    }}
    table.summary th, table.summary td {{
      border: 1px solid var(--border);
      padding: 0.4rem 0.5rem;
      text-align: left;
    }}
    table.summary th {{
      background: var(--surface);
      cursor: pointer;
      user-select: none;
    }}
    table.summary th:hover {{ color: var(--accent); }}
    table.mini {{
      font-size: 0.8rem;
      border-collapse: collapse;
      width: 100%;
      margin: 0.5rem 0;
    }}
    table.mini th, table.mini td {{
      border: 1px solid var(--border);
      padding: 0.2rem 0.4rem;
    }}
    table.mini th {{ text-align: left; color: var(--muted); font-weight: 500; width: 42%; }}
    .meta-dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.25rem 1rem;
      font-size: 0.9rem;
    }}
    .meta-dl dt {{ color: var(--muted); }}
    .graph-grid {{
      display: grid;
      gap: 1.25rem;
      margin-top: 1rem;
    }}
    @media (min-width: 700px) {{
      .graph-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .graph-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
    }}
    .graph-card.hidden {{ display: none; }}
    .img-wrap {{
      margin: 0.75rem 0;
      background: #0a0e14;
      border-radius: 6px;
      padding: 0.5rem;
      text-align: center;
    }}
    .img-wrap img {{
      max-width: 100%;
      height: auto;
      vertical-align: middle;
    }}
    details {{ margin-top: 0.5rem; }}
    summary {{ cursor: pointer; color: var(--accent); font-size: 0.9rem; }}
    pre.json {{
      overflow: auto;
      max-height: 320px;
      font-size: 0.72rem;
      line-height: 1.35;
      background: #0a0e14;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.75rem;
      margin-top: 0.5rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Explainer metrics and mask visualizations</h1>
    <p class="muted">Fold <strong>{html.escape(str(fold_index))}</strong> · HTML generated <code>{html.escape(gen_at)}</code></p>
    {comp_block}
    <nav><strong>Jump:</strong> <a href="#valid-summary">Valid result metrics</a> · <a href="#all-summary">Result metrics</a> · {nav_html}</nav>
    <div class="controls">
      <label for="filter-q">Filter graphs by id</label>
      <input type="search" id="filter-q" placeholder="e.g. 7GAW" autocomplete="off">
    </div>
  </header>

  <section id="valid-summary">
    <h2>Valid result metrics</h2>
    <p class="muted">Means computed only over graphs whose explanation produced a complete metric set (<code>valid == true</code>). Click column headers to sort.</p>
    {valid_table_html}
  </section>

  <section id="all-summary">
    <h2>Result metrics</h2>
    <p class="muted">Means computed over every explained graph in the fold (NaN-skipped). Includes wall-clock runtime for the explainer.</p>
    {result_table_html}
  </section>

  {"".join(sections_html)}

  <script>
(function () {{
  const q = document.getElementById("filter-q");
  const cards = document.querySelectorAll(".graph-card");
  q.addEventListener("input", function () {{
    const needle = (q.value || "").trim().toLowerCase();
    cards.forEach(function (c) {{
      const id = (c.getAttribute("data-graph-id") || "").toLowerCase();
      c.classList.toggle("hidden", needle !== "" && !id.includes(needle));
    }});
  }});

  document.querySelectorAll("table.summary").forEach(function (table) {{
    const theadRow = table.querySelector("thead tr");
    if (!theadRow) return;
    let sortDir = 1;
    let sortCol = null;
    theadRow.querySelectorAll("th").forEach(function (th, idx) {{
      th.addEventListener("click", function () {{
        const col = th.getAttribute("data-sort");
        const type = th.getAttribute("data-type") || "text";
        if (sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}
        const tbody = table.querySelector("tbody");
        const rows = Array.from(tbody.querySelectorAll("tr"));
        const parse = function (td) {{
          if (!td) return {{ v: "", n: NaN }};
          const raw = td.getAttribute("data-sort-value");
          if (raw !== null && raw !== "") {{
            const n = parseFloat(raw);
            return {{ v: td.textContent.trim(), n: n }};
          }}
          return {{ v: td.textContent.trim(), n: NaN }};
        }};
        rows.sort(function (a, b) {{
          const ac = a.children[idx];
          const bc = b.children[idx];
          const A = parse(ac);
          const B = parse(bc);
          let cmp;
          if (type === "num" && !isNaN(A.n) && !isNaN(B.n)) {{
            cmp = A.n - B.n;
          }} else {{
            cmp = A.v.localeCompare(B.v, undefined, {{ sensitivity: "base" }});
          }}
          return sortDir * cmp;
        }});
        rows.forEach(function (r) {{ tbody.appendChild(r); }});
      }});
    }});
  }});
}})();
  </script>
</body>
</html>
"""

    out_dir = fold_root / EXPLANATION_WEB_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Global cross-fold index
# ---------------------------------------------------------------------------


def _nanmean_safe(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


# ---------------------------------------------------------------------------
# Statistical helpers (pure-Python, no numpy -- lists are tiny)
# ---------------------------------------------------------------------------


def _filter_numeric_pairs(
    values: list[float | None],
    weights: list[float | None] | None = None,
) -> tuple[list[float], list[float]]:
    """Return (vals, wts) with all None / non-finite entries dropped."""
    if weights is None:
        weights = [1.0] * len(values)
    out_v: list[float] = []
    out_w: list[float] = []
    for v, w in zip(values, weights):
        if (
            v is not None
            and isinstance(v, (int, float))
            and w is not None
            and isinstance(w, (int, float))
            and w > 0
        ):
            out_v.append(float(v))
            out_w.append(float(w))
    return out_v, out_w


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    tw = sum(weights)
    if tw == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / tw


def _weighted_std(values: list[float], weights: list[float]) -> float | None:
    mu = _weighted_mean(values, weights)
    if mu is None or len(values) < 2:
        return None
    tw = sum(weights)
    var = sum(w * (v - mu) ** 2 for v, w in zip(values, weights)) / tw
    return var ** 0.5


def _weighted_quantile(
    values: list[float], weights: list[float], q: float,
) -> float | None:
    """Weighted quantile via linear interpolation on cumulative weights."""
    if not values:
        return None
    paired = sorted(zip(values, weights))
    cum = []
    s = 0.0
    for _, w in paired:
        s += w
        cum.append(s)
    tw = cum[-1]
    if tw == 0:
        return None
    target = q * tw
    for i, c in enumerate(cum):
        if c >= target:
            if i == 0:
                return paired[0][0]
            lo_c = cum[i - 1]
            frac = (target - lo_c) / (c - lo_c) if c != lo_c else 0.0
            return paired[i - 1][0] + frac * (paired[i][0] - paired[i - 1][0])
    return paired[-1][0]


def _weighted_median(values: list[float], weights: list[float]) -> float | None:
    return _weighted_quantile(values, weights, 0.5)


def _weighted_iqr(
    values: list[float], weights: list[float],
) -> tuple[float | None, float | None]:
    return (
        _weighted_quantile(values, weights, 0.25),
        _weighted_quantile(values, weights, 0.75),
    )


def _unweighted_stats(
    values: list[float | None],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Return (median, q1, q3, mean, std) for *values*, skipping None."""
    nums = sorted(v for v in values if v is not None and isinstance(v, (int, float)))
    if not nums:
        return None, None, None, None, None
    n = len(nums)
    mean = sum(nums) / n
    std = (sum((x - mean) ** 2 for x in nums) / n) ** 0.5 if n > 1 else None

    def _percentile(sorted_vals: list[float], p: float) -> float:
        k = (len(sorted_vals) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = k - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    median = _percentile(nums, 0.5)
    q1 = _percentile(nums, 0.25)
    q3 = _percentile(nums, 0.75)
    return median, q1, q3, mean, std


def _global_cols(table_kind: str) -> list[tuple[str, str, str]]:
    """Columns for the cross-fold tables.

    ``table_kind`` is either ``"valid"`` (uses ``num_valid_graphs``) or
    ``"all"`` (uses ``wall_time_s`` + ``num_graphs``).
    """
    cols: list[tuple[str, str, str]] = [("fold", "Fold", "text")]
    if table_kind == "valid":
        cols.append(_VALID_TABLE_COUNT_COL)
    else:
        cols.extend(_RESULT_TABLE_LEADING_COLS)
    cols.extend((f"mean_{key}", label, "num") for key, label in _METRIC_COLUMNS)
    return cols


def _render_global_table(
    *,
    cols: list[tuple[str, str, str]],
    rows: list[list[Any]],
) -> str:
    thead = "".join(
        f'<th data-sort="{html.escape(k)}" data-type="{t}">{html.escape(l)}</th>'
        for k, l, t in cols
    )
    tbody_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for (key, _label, tp), value in zip(cols, row, strict=True):
            if key == "fold":
                cells.append(str(value))
            else:
                cells.append(
                    f'<td data-sort-value="{_sort_value(value, tp)}">{_fmt_num(value)}</td>'
                )
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = (
        "\n".join(tbody_rows)
        if tbody_rows
        else f'<tr><td colspan="{len(cols)}">No data</td></tr>'
    )
    return (
        f'<table class="summary"><thead><tr>{thead}</tr></thead>'
        f"<tbody>{tbody}</tbody></table>"
    )


def _per_explainer_block_for_global(
    fold_entries_sorted: list[dict],
    explainer_name: str,
    *,
    table_kind: str,
) -> str:
    """One per-fold table for *explainer_name* across all folds, for a given block kind."""
    block_key = "valid_result_metrics" if table_kind == "valid" else "result_metrics"
    cols = _global_cols(table_kind)
    rows: list[list[Any]] = []
    for entry in fold_entries_sorted:
        k = entry["fold_index"]
        per_expl = entry.get("per_explainer_summary", {})
        s = per_expl.get(explainer_name, {})
        block = s.get(block_key, {}) if isinstance(s, dict) else {}
        fold_link = (
            f"folds/fold_{k}/{EXPLANATION_WEB_REPORT_DIR}/index.html"
            f"#explainer-{explainer_name}"
        )
        row: list[Any] = [
            f'<td><a href="{html.escape(fold_link, quote=True)}">Fold {k}</a></td>'
        ]
        for key, _, _tp in cols[1:]:
            row.append(block.get(key) if isinstance(block, dict) else None)
        rows.append(row)
    return _render_global_table(cols=cols, rows=rows)


def _per_fold_avg_table(
    fold_entries_sorted: list[dict],
    *,
    table_kind: str,
) -> str:
    """Cross-explainer averages per fold for the requested block kind."""
    block_key = "valid_result_metrics" if table_kind == "valid" else "result_metrics"
    cols = _global_cols(table_kind)
    rows: list[list[Any]] = []
    for entry in fold_entries_sorted:
        k = entry["fold_index"]
        per_expl = entry.get("per_explainer_summary", {})
        fold_link = f"folds/fold_{k}/{EXPLANATION_WEB_REPORT_DIR}/index.html"
        avg_vals: dict[str, float | None] = {}
        for key, _, _tp in cols[1:]:
            vals = [
                s.get(block_key, {}).get(key)
                for s in per_expl.values()
                if isinstance(s, dict)
            ]
            avg_vals[key] = _nanmean_safe(vals)
        row: list[Any] = [
            f'<td><a href="{html.escape(fold_link, quote=True)}">Fold {k}</a></td>'
        ]
        for key, _, _tp in cols[1:]:
            row.append(avg_vals.get(key))
        rows.append(row)
    return _render_global_table(cols=cols, rows=rows)


def write_global_explanation_index(
    results_root: Path,
    fold_entries: list[dict],
) -> Path:
    """Write ``results_root/explanation_web_report/index.html`` linking all per-fold reports.

    *fold_entries* is a list of dicts, each with keys
    ``fold_index``, ``explainer_names`` and ``per_explainer_summary``. The
    summary's per-explainer dicts must carry both ``valid_result_metrics`` and
    ``result_metrics`` blocks (matching the per-fold ``explanation_report.json``).
    """
    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fold_entries_sorted = sorted(fold_entries, key=lambda e: e["fold_index"])

    all_explainer_names: list[str] = []
    seen: set[str] = set()
    for entry in fold_entries_sorted:
        for name in entry.get("explainer_names", []):
            if name not in seen:
                seen.add(name)
                all_explainer_names.append(name)

    fold_valid_table = _per_fold_avg_table(fold_entries_sorted, table_kind="valid")
    fold_result_table = _per_fold_avg_table(fold_entries_sorted, table_kind="all")

    explainer_sections: list[str] = []
    for explainer_name in all_explainer_names:
        valid_tbl = _per_explainer_block_for_global(
            fold_entries_sorted, explainer_name, table_kind="valid",
        )
        result_tbl = _per_explainer_block_for_global(
            fold_entries_sorted, explainer_name, table_kind="all",
        )
        explainer_sections.append(
            f'<section id="explainer-{html.escape(explainer_name, quote=True)}">'
            f"<h3>{html.escape(explainer_name)}</h3>"
            f'<h4 class="muted">Valid result metrics</h4>{valid_tbl}'
            f'<h4 class="muted">Result metrics</h4>{result_tbl}'
            f"</section>"
        )

    explainer_nav = " &middot; ".join(
        f'<a href="#explainer-{html.escape(n, quote=True)}">{html.escape(n)}</a>'
        for n in all_explainer_names
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Explainer reports &mdash; all folds</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --border: #2d3a4d;
      --text: #e6edf3;
      --muted: #8b9cb3;
      --accent: #58a6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      margin: 0 auto;
      padding: 1rem 1.25rem 3rem;
      max-width: 1200px;
    }}
    a {{ color: var(--accent); }}
    header {{ margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.5rem; }}
    h2 {{ font-size: 1.15rem; margin-top: 2rem; border-top: 1px solid var(--border); padding-top: 1.25rem; }}
    h3 {{ font-size: 1rem; margin-top: 1.5rem; }}
    h4.muted {{ font-size: 0.9rem; color: var(--muted); margin: 0.75rem 0 0.25rem; }}
    .muted {{ color: var(--muted); }}
    nav {{ font-size: 0.9rem; margin: 0.75rem 0; line-height: 1.6; }}
    table.summary {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin: 0.5rem 0 1rem;
    }}
    table.summary th, table.summary td {{
      border: 1px solid var(--border);
      padding: 0.4rem 0.5rem;
      text-align: left;
    }}
    table.summary th {{
      background: var(--surface);
      cursor: pointer;
      user-select: none;
    }}
    table.summary th:hover {{ color: var(--accent); }}
  </style>
</head>
<body>
  <header>
    <h1>Explainer reports &mdash; cross-fold summary</h1>
    <p class="muted">{len(fold_entries_sorted)} fold(s) &middot; HTML generated <code>{html.escape(gen_at)}</code></p>
    <nav>
      <strong>Jump:</strong>
      <a href="#fold-summary">Fold summary</a> &middot;
      <a href="#per-explainer">Per-explainer</a> &middot;
      {explainer_nav}
    </nav>
  </header>

  <section id="fold-summary">
    <h2>Summary by fold</h2>
    <p class="muted">Metrics averaged across all explainers within each fold. Click a fold to open its detailed report.</p>
    <h4 class="muted">Valid result metrics</h4>
    {fold_valid_table}
    <h4 class="muted">Result metrics</h4>
    {fold_result_table}
  </section>

  <section id="per-explainer">
    <h2>Per-explainer across folds</h2>
    <p class="muted">Each explainer is shown in two tables: valid-only aggregates and all-graph aggregates with wall-clock runtime.</p>
    {"".join(explainer_sections)}
  </section>

  <script>
(function () {{
  document.querySelectorAll("table.summary").forEach(function (table) {{
    var theadRow = table.querySelector("thead tr");
    if (!theadRow) return;
    var sortDir = 1;
    var sortCol = null;
    theadRow.querySelectorAll("th").forEach(function (th, idx) {{
      th.addEventListener("click", function () {{
        var col = th.getAttribute("data-sort");
        var type = th.getAttribute("data-type") || "text";
        if (sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}
        var tbody = table.querySelector("tbody");
        var rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort(function (a, b) {{
          var ac = a.children[idx], bc = b.children[idx];
          if (!ac || !bc) return 0;
          var av = ac.getAttribute("data-sort-value"), bv = bc.getAttribute("data-sort-value");
          if (type === "num" && av && bv) {{
            var an = parseFloat(av), bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) return sortDir * (an - bn);
          }}
          return sortDir * (ac.textContent || "").localeCompare(bc.textContent || "", undefined, {{ sensitivity: "base" }});
        }});
        rows.forEach(function (r) {{ tbody.appendChild(r); }});
      }});
    }});
  }});
}})();
  </script>
</body>
</html>
"""

    out_dir = results_root / EXPLANATION_WEB_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Explainer summary page (primary comparative view)
# ---------------------------------------------------------------------------

_SUMMARY_CSS = """\
    :root {
      --bg: #0f1419;
      --surface: #1a2332;
      --border: #2d3a4d;
      --text: #e6edf3;
      --muted: #8b9cb3;
      --accent: #58a6ff;
    }
    * { box-sizing: border-box; }
    body {
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      margin: 0 auto;
      padding: 1rem 1.25rem 3rem;
      max-width: 1400px;
    }
    a { color: var(--accent); }
    header { margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }
    h1 { font-size: 1.35rem; margin: 0 0 0.5rem; }
    h2 { font-size: 1.15rem; margin-top: 2rem; border-top: 1px solid var(--border); padding-top: 1.25rem; }
    h3 { font-size: 1rem; margin-top: 1.5rem; }
    h4.muted { font-size: 0.9rem; color: var(--muted); margin: 0.75rem 0 0.25rem; }
    .muted { color: var(--muted); }
    nav { font-size: 0.9rem; margin: 0.75rem 0; line-height: 1.6; }
    table.summary {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      margin: 0.5rem 0 1rem;
    }
    table.summary th, table.summary td {
      border: 1px solid var(--border);
      padding: 0.35rem 0.45rem;
      text-align: left;
      white-space: nowrap;
    }
    table.summary th {
      background: var(--surface);
      cursor: pointer;
      user-select: none;
    }
    table.summary th:hover { color: var(--accent); }
    .back-link { display: inline-block; margin-bottom: 0.75rem; font-size: 0.9rem; }
"""

_SUMMARY_JS = """\
(function () {
  document.querySelectorAll("table.summary").forEach(function (table) {
    var theadRow = table.querySelector("thead tr");
    if (!theadRow) return;
    var sortDir = 1;
    var sortCol = null;
    theadRow.querySelectorAll("th").forEach(function (th, idx) {
      th.addEventListener("click", function () {
        var col = th.getAttribute("data-sort");
        var type = th.getAttribute("data-type") || "text";
        if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = 1; }
        var tbody = table.querySelector("tbody");
        var rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort(function (a, b) {
          var ac = a.children[idx], bc = b.children[idx];
          if (!ac || !bc) return 0;
          var av = ac.getAttribute("data-sort-value"), bv = bc.getAttribute("data-sort-value");
          if (type === "num" && av && bv) {
            var an = parseFloat(av), bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) return sortDir * (an - bn);
          }
          return sortDir * (ac.textContent || "").localeCompare(bc.textContent || "", undefined, { sensitivity: "base" });
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      });
    });
  });
})();
"""


def _collect_per_explainer_vectors(
    fold_entries_sorted: list[dict],
    all_explainer_names: list[str],
    block_key: str,
) -> dict[str, dict[str, list[float | None]]]:
    """Build ``{explainer: {metric_key: [val_fold0, val_fold1, ...]}}``."""
    vectors: dict[str, dict[str, list[float | None]]] = {}
    for name in all_explainer_names:
        vecs: dict[str, list[float | None]] = {}
        for entry in fold_entries_sorted:
            per_expl = entry.get("per_explainer_summary", {})
            s = per_expl.get(name, {})
            block = s.get(block_key, {}) if isinstance(s, dict) else {}
            for mkey, _ in _METRIC_COLUMNS:
                full_key = f"mean_{mkey}"
                vecs.setdefault(full_key, []).append(
                    block.get(full_key) if isinstance(block, dict) else None,
                )
            for extra_key in ("num_valid_graphs", "num_graphs", "wall_time_s"):
                vecs.setdefault(extra_key, []).append(
                    block.get(extra_key) if isinstance(block, dict) else None,
                )
        vectors[name] = vecs
    return vectors


def _build_weighted_table(
    all_explainer_names: list[str],
    vectors: dict[str, dict[str, list[float | None]]],
    weight_key: str,
) -> str:
    """One table: weighted median, IQR, mean, std per metric per explainer."""
    metric_keys = [f"mean_{k}" for k, _ in _METRIC_COLUMNS]
    metric_labels = [label for _, label in _METRIC_COLUMNS]

    sub_cols: list[str] = ["Median", "Q1", "Q3", "Mean", "Std"]
    header_cells = ['<th data-sort="explainer" data-type="text">Explainer</th>']
    for label in metric_labels:
        for sc in sub_cols:
            col_id = f"{label}_{sc}".replace(" ", "_")
            header_cells.append(
                f'<th data-sort="{html.escape(col_id)}" data-type="num">'
                f"{html.escape(label)}<br><small>{html.escape(sc)}</small></th>"
            )
    thead = "<tr>" + "".join(header_cells) + "</tr>"

    tbody_rows: list[str] = []
    for name in all_explainer_names:
        vecs = vectors[name]
        wts_raw = vecs.get(weight_key, [])
        cells = [f"<td>{html.escape(name)}</td>"]
        for mk in metric_keys:
            vals_raw = vecs.get(mk, [])
            vals, wts = _filter_numeric_pairs(vals_raw, wts_raw)
            med = _weighted_median(vals, wts)
            q1, q3 = _weighted_iqr(vals, wts)
            mean = _weighted_mean(vals, wts)
            std = _weighted_std(vals, wts)
            for stat in (med, q1, q3, mean, std):
                sv = _sort_value(stat, "num")
                cells.append(
                    f'<td data-sort-value="{sv}">{_fmt_num(stat)}</td>'
                )
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    tbody = "\n".join(tbody_rows) if tbody_rows else '<tr><td colspan="1">No data</td></tr>'
    return (
        f'<table class="summary"><thead>{thead}</thead>'
        f"<tbody>{tbody}</tbody></table>"
    )


def _build_unweighted_table(
    all_explainer_names: list[str],
    vectors: dict[str, dict[str, list[float | None]]],
) -> str:
    """One table: unweighted median, IQR, mean, std per metric per explainer."""
    metric_keys = [f"mean_{k}" for k, _ in _METRIC_COLUMNS]
    metric_labels = [label for _, label in _METRIC_COLUMNS]

    sub_cols: list[str] = ["Median", "Q1", "Q3", "Mean", "Std"]
    header_cells = ['<th data-sort="explainer" data-type="text">Explainer</th>']
    for label in metric_labels:
        for sc in sub_cols:
            col_id = f"{label}_{sc}".replace(" ", "_")
            header_cells.append(
                f'<th data-sort="{html.escape(col_id)}" data-type="num">'
                f"{html.escape(label)}<br><small>{html.escape(sc)}</small></th>"
            )
    thead = "<tr>" + "".join(header_cells) + "</tr>"

    tbody_rows: list[str] = []
    for name in all_explainer_names:
        vecs = vectors[name]
        cells = [f"<td>{html.escape(name)}</td>"]
        for mk in metric_keys:
            vals_raw = vecs.get(mk, [])
            med, q1, q3, mean, std = _unweighted_stats(vals_raw)
            for stat in (med, q1, q3, mean, std):
                sv = _sort_value(stat, "num")
                cells.append(
                    f'<td data-sort-value="{sv}">{_fmt_num(stat)}</td>'
                )
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    tbody = "\n".join(tbody_rows) if tbody_rows else '<tr><td colspan="1">No data</td></tr>'
    return (
        f'<table class="summary"><thead>{thead}</thead>'
        f"<tbody>{tbody}</tbody></table>"
    )


def _build_coverage_table(
    all_explainer_names: list[str],
    valid_vectors: dict[str, dict[str, list[float | None]]],
    all_vectors: dict[str, dict[str, list[float | None]]],
    fold_entries_sorted: list[dict],
) -> str:
    """Valid-graph coverage table: total valid / total graphs + per-fold."""
    fold_indices = [e["fold_index"] for e in fold_entries_sorted]

    header_cells = [
        '<th data-sort="explainer" data-type="text">Explainer</th>',
        '<th data-sort="total_valid" data-type="num">Valid</th>',
        '<th data-sort="total_graphs" data-type="num">Total</th>',
        '<th data-sort="coverage_pct" data-type="num">Coverage %</th>',
    ]
    for k in fold_indices:
        header_cells.append(
            f'<th data-sort="fold_{k}" data-type="num">Fold {k}</th>'
        )
    thead = "<tr>" + "".join(header_cells) + "</tr>"

    tbody_rows: list[str] = []
    for name in all_explainer_names:
        v_valid = valid_vectors[name].get("num_valid_graphs", [])
        v_total = all_vectors[name].get("num_graphs", [])
        total_valid = sum(x for x in v_valid if x is not None and isinstance(x, (int, float)))
        total_graphs = sum(x for x in v_total if x is not None and isinstance(x, (int, float)))
        pct = (total_valid / total_graphs * 100) if total_graphs > 0 else None

        cells = [
            f"<td>{html.escape(name)}</td>",
            f'<td data-sort-value="{_sort_value(total_valid, "num")}">{_fmt_num(total_valid)}</td>',
            f'<td data-sort-value="{_sort_value(total_graphs, "num")}">{_fmt_num(total_graphs)}</td>',
            f'<td data-sort-value="{_sort_value(pct, "num")}">{_fmt_num(round(pct, 1) if pct is not None else None)}</td>',
        ]
        for vv, vt in zip(v_valid, v_total):
            vv_n = vv if vv is not None and isinstance(vv, (int, float)) else 0
            vt_n = vt if vt is not None and isinstance(vt, (int, float)) else 0
            label = f"{int(vv_n)}/{int(vt_n)}" if vt_n > 0 else "\u2014"
            fold_pct = vv_n / vt_n * 100 if vt_n > 0 else None
            cells.append(
                f'<td data-sort-value="{_sort_value(fold_pct, "num")}">{label}</td>'
            )
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    tbody = "\n".join(tbody_rows) if tbody_rows else '<tr><td colspan="1">No data</td></tr>'
    return (
        f'<table class="summary"><thead><tr>{"".join(header_cells)}</tr></thead>'
        f"<tbody>{tbody}</tbody></table>"
    )


def _build_mean_across_folds_table(
    all_explainer_names: list[str],
    vectors: dict[str, dict[str, list[float | None]]],
    *,
    table_kind: str,
) -> str:
    """Compact table: one row per explainer, nanmean of each metric across folds."""
    metric_keys = [f"mean_{k}" for k, _ in _METRIC_COLUMNS]
    metric_labels = [label for _, label in _METRIC_COLUMNS]

    header_cells = ['<th data-sort="explainer" data-type="text">Explainer</th>']
    if table_kind == "valid":
        header_cells.append('<th data-sort="num_valid_graphs" data-type="num">Valid graphs (total)</th>')
    else:
        header_cells.append('<th data-sort="wall_time_s" data-type="num">Wall (s) total</th>')
        header_cells.append('<th data-sort="num_graphs" data-type="num">Graphs (total)</th>')
    for label in metric_labels:
        col_id = label.replace(" ", "_")
        header_cells.append(
            f'<th data-sort="{html.escape(col_id)}" data-type="num">{html.escape(label)}</th>'
        )
    thead = "<tr>" + "".join(header_cells) + "</tr>"

    tbody_rows: list[str] = []
    for name in all_explainer_names:
        vecs = vectors[name]
        cells = [f"<td>{html.escape(name)}</td>"]
        if table_kind == "valid":
            total = sum(
                x for x in vecs.get("num_valid_graphs", [])
                if x is not None and isinstance(x, (int, float))
            )
            cells.append(f'<td data-sort-value="{_sort_value(total, "num")}">{_fmt_num(total)}</td>')
        else:
            total_time = sum(
                x for x in vecs.get("wall_time_s", [])
                if x is not None and isinstance(x, (int, float))
            )
            total_g = sum(
                x for x in vecs.get("num_graphs", [])
                if x is not None and isinstance(x, (int, float))
            )
            cells.append(f'<td data-sort-value="{_sort_value(total_time, "num")}">{_fmt_num(round(total_time, 1))}</td>')
            cells.append(f'<td data-sort-value="{_sort_value(total_g, "num")}">{_fmt_num(total_g)}</td>')

        for mk in metric_keys:
            val = _nanmean_safe(vecs.get(mk, []))
            sv = _sort_value(val, "num")
            cells.append(f'<td data-sort-value="{sv}">{_fmt_num(val)}</td>')
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    tbody = "\n".join(tbody_rows) if tbody_rows else '<tr><td colspan="1">No data</td></tr>'
    return (
        f'<table class="summary"><thead><tr>{"".join(header_cells)}</tr></thead>'
        f"<tbody>{tbody}</tbody></table>"
    )


def write_explainer_summary_page(
    results_root: Path,
    fold_entries: list[dict],
) -> Path:
    """Write ``results_root/explanation_web_report/explainer_summary.html``.

    Primary comparative view: one row per explainer with weighted/unweighted
    statistics across folds, valid-graph coverage, and a compact mean table.
    """
    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fold_entries_sorted = sorted(fold_entries, key=lambda e: e["fold_index"])

    all_explainer_names: list[str] = []
    seen: set[str] = set()
    for entry in fold_entries_sorted:
        for name in entry.get("explainer_names", []):
            if name not in seen:
                seen.add(name)
                all_explainer_names.append(name)

    valid_vectors = _collect_per_explainer_vectors(
        fold_entries_sorted, all_explainer_names, "valid_result_metrics",
    )
    all_vectors = _collect_per_explainer_vectors(
        fold_entries_sorted, all_explainer_names, "result_metrics",
    )

    weighted_valid_table = _build_weighted_table(
        all_explainer_names, valid_vectors, "num_valid_graphs",
    )
    weighted_all_table = _build_weighted_table(
        all_explainer_names, all_vectors, "num_graphs",
    )
    unweighted_valid_table = _build_unweighted_table(
        all_explainer_names, valid_vectors,
    )
    unweighted_all_table = _build_unweighted_table(
        all_explainer_names, all_vectors,
    )
    coverage_table = _build_coverage_table(
        all_explainer_names, valid_vectors, all_vectors, fold_entries_sorted,
    )
    mean_valid_table = _build_mean_across_folds_table(
        all_explainer_names, valid_vectors, table_kind="valid",
    )
    mean_all_table = _build_mean_across_folds_table(
        all_explainer_names, all_vectors, table_kind="all",
    )

    doc = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Explainer summary &mdash; across folds</title>
  <style>
{_SUMMARY_CSS}
  </style>
</head>
<body>
  <header>
    <h1>Summary by explainer (across folds)</h1>
    <p class="muted">{len(fold_entries_sorted)} fold(s) &middot; {len(all_explainer_names)} explainer(s) &middot; HTML generated <code>{html.escape(gen_at)}</code></p>
    <a class="back-link" href="index.html">&larr; Detailed per-fold breakdown</a>
    <nav>
      <strong>Jump:</strong>
      <a href="#mean-summary">Mean across folds</a> &middot;
      <a href="#coverage">Coverage</a> &middot;
      <a href="#weighted">Weighted stats</a> &middot;
      <a href="#unweighted">Unweighted stats</a>
    </nav>
  </header>

  <section id="mean-summary">
    <h2>Mean across folds</h2>
    <p class="muted">Simple mean of each metric across folds, per explainer. Each fold counts equally.</p>
    <h4 class="muted">Valid result metrics</h4>
    {mean_valid_table}
    <h4 class="muted">Result metrics</h4>
    {mean_all_table}
  </section>

  <section id="coverage">
    <h2>Valid-graph coverage</h2>
    <p class="muted">Total valid graphs / total graphs per explainer, with per-fold breakdown.</p>
    {coverage_table}
  </section>

  <section id="weighted">
    <h2>Weighted statistics across folds</h2>
    <p class="muted">Weighted by <code>num_valid_graphs</code> (valid table) or <code>num_graphs</code> (all table). Median, IQR (Q1&ndash;Q3), mean, and std.</p>
    <h4 class="muted">Valid result metrics</h4>
    {weighted_valid_table}
    <h4 class="muted">Result metrics</h4>
    {weighted_all_table}
  </section>

  <section id="unweighted">
    <h2>Unweighted statistics across folds</h2>
    <p class="muted">Each fold counts equally. Median, IQR (Q1&ndash;Q3), mean, and std.</p>
    <h4 class="muted">Valid result metrics</h4>
    {unweighted_valid_table}
    <h4 class="muted">Result metrics</h4>
    {unweighted_all_table}
  </section>

  <script>
{_SUMMARY_JS}
  </script>
</body>
</html>
"""

    out_dir = results_root / EXPLANATION_WEB_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "explainer_summary.html"
    out_path.write_text(doc, encoding="utf-8")
    return out_path
