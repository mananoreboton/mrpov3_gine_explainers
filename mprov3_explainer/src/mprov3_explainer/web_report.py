"""Static HTML reports: per-fold explainer metrics and a global cross-fold index."""

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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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

    rows_summary: list[dict[str, Any]] = []
    sections_html: list[str] = []
    nav_links: list[str] = []

    for explainer_name in explainer_names:
        report_path = explanations_base / explainer_name / "explanation_report.json"
        report = _load_json(report_path)
        if not report:
            continue
        rows_summary.append({"explainer": explainer_name, "report": report})

        nav_links.append(
            f'<a href="#explainer-{html.escape(explainer_name, quote=True)}">'
            f"{html.escape(explainer_name)}</a>"
        )

        try:
            blurb = html.escape(get_spec(explainer_name).report_paragraph)
        except ValueError:
            blurb = ""
        status = report.get("run_status", "ok")
        status_note = report.get("run_status_note", "")
        if status != "ok":
            blurb = (
                f"{blurb} "
                f"Run status: {html.escape(str(status))}. "
                f"{html.escape(str(status_note))}"
            ).strip()

        masks_dir = explanations_base / explainer_name / "masks"
        graphs_dir = visualizations_run_dir(fold_root, explainer_name) / "graphs"

        cards: list[str] = []
        for entry in report.get("per_graph", []):
            graph_id = entry.get("graph_id") or ""
            if not graph_id:
                continue
            mask_path = masks_dir / f"{graph_id}.json"
            png_path = graphs_dir / f"mask_{graph_id}.png"
            rel_png = f"../visualizations/{explainer_name}/graphs/mask_{graph_id}.png"
            rel_json = f"../explanations/{explainer_name}/masks/{graph_id}.json"

            if mask_path.is_file():
                raw_json = html.escape(mask_path.read_text(encoding="utf-8"))
                mask_note = ""
            else:
                raw_json = "(mask JSON file missing)"
                mask_note = " <span class=\"warn\">No mask file</span>"

            if png_path.is_file():
                img_block = (
                    f'<img src="{html.escape(rel_png, quote=True)}" '
                    f'alt="Mask visualization {html.escape(graph_id)}" loading="lazy">'
                )
            else:
                img_block = '<p class="muted">PNG not rendered (missing or draw failed).</p>'

            per_keys = [
                ("fidelity_plus", "Fid+ (top-k)"),
                ("fidelity_minus", "Fid− (top-k)"),
                ("pyg_characterization", "PyG char (top-k)"),
                ("fidelity_plus_soft", "Fid+ (soft)"),
                ("fidelity_minus_soft", "Fid− (soft)"),
                ("pyg_characterization_soft", "PyG char (soft)"),
                ("paper_sufficiency", "Fsuf (raw)"),
                ("paper_comprehensiveness", "Fcom (raw)"),
                ("paper_f1_fidelity", "Ff1 (clamped)"),
                ("mask_spread", "Mask spread"),
                ("mask_entropy", "Mask entropy"),
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

            cards.append(
                f'<article class="graph-card" data-graph-id="{html.escape(graph_id, quote=True)}">'
                f'<h4 id="g-{html.escape(explainer_name, quote=True)}-{html.escape(graph_id, quote=True)}">'
                f"{html.escape(graph_id)}</h4>"
                f"{mask_note}"
                f'<table class="mini"><tbody>{tcells}</tbody></table>'
                f'<div class="img-wrap">{img_block}</div>'
                f'<details><summary>Raw mask JSON</summary>'
                f'<p><a href="{html.escape(rel_json, quote=True)}">Open mask file path</a> (local)</p>'
                f'<pre class="json">{raw_json}</pre></details>'
                f"</article>"
            )

        cards_joined = "\n".join(cards) if cards else "<p class=\"muted\">No per-graph entries.</p>"
        sections_html.append(
            f'<section class="explainer-section" id="explainer-{html.escape(explainer_name, quote=True)}">'
            f"<h2>{html.escape(explainer_name)}</h2>"
            f'<p class="blurb">{blurb}</p>'
            f'<div class="graph-grid">{cards_joined}</div>'
            f"</section>"
        )

    # Summary table
    #
    # The first 10 columns keep their original key/label/order so existing
    # dashboards, screenshots and bookmarks still match. New diagnostic
    # columns (top-k soft fidelity, degenerate mask count, mean mask spread)
    # are appended at the end and fall back to "—" via _fmt_num if the
    # JSON does not carry the key (i.e. for older reports).
    sum_cols = [
        ("explainer", "Explainer", "text"),
        ("run_status", "Status", "text"),
        ("mean_fidelity_plus", "Mean Fid+", "num"),
        ("mean_fidelity_minus", "Mean Fid−", "num"),
        ("mean_pyg_characterization", "Mean PyG char", "num"),
        ("mean_paper_sufficiency", "Mean Fsuf (raw)", "num"),
        ("mean_paper_comprehensiveness", "Mean Fcom (raw)", "num"),
        ("mean_paper_f1_fidelity", "Mean Ff1 (clamped)", "num"),
        ("mean_fidelity_plus_all_graphs", "All Fid+", "num"),
        ("mean_fidelity_minus_all_graphs", "All Fid−", "num"),
        ("mean_pyg_characterization_all_graphs", "All PyG char", "num"),
        ("mean_paper_sufficiency_all_graphs", "All Fsuf (raw)", "num"),
        ("mean_paper_comprehensiveness_all_graphs", "All Fcom (raw)", "num"),
        ("mean_paper_f1_fidelity_all_graphs", "All Ff1 (clamped)", "num"),
        ("num_graphs", "Graphs", "num"),
        ("num_valid", "Valid", "num"),
        ("wall_time_s", "Wall (s)", "num"),
        ("mean_fidelity_plus_soft", "Mean Fid+ (soft)", "num"),
        ("mean_fidelity_minus_soft", "Mean Fid− (soft)", "num"),
        ("mean_pyg_characterization_soft", "Mean PyG char (soft)", "num"),
        ("num_degenerate_mask", "Degen.", "num"),
        ("num_misclassified", "Misclass.", "num"),
        ("num_prediction_baseline_mismatch", "Pred. drift", "num"),
        ("mean_mask_spread", "Spread", "num"),
        ("mean_mask_entropy", "Entropy", "num"),
    ]
    thead = "".join(
        f'<th data-sort="{html.escape(key)}" data-type="{tp}">{html.escape(label)}</th>'
        for key, label, tp in sum_cols
    )
    tbody_rows: list[str] = []
    for row in rows_summary:
        en = row["explainer"]
        rep = row["report"]
        cells = [f'<td data-col="explainer">{html.escape(en)}</td>']
        for key, _, tp in sum_cols[1:]:
            val = rep.get(key)
            cells.append(
                f'<td data-col="{html.escape(key)}" data-sort-value="{_sort_value(val, tp)}">'
                f"{_fmt_num(val)}</td>"
            )
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = (
        "\n".join(tbody_rows)
        if tbody_rows
        else f'<tr><td colspan="{len(sum_cols)}">No data</td></tr>'
    )

    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comp_block = ""
    if comparison:
        seed_val = comparison.get("seed")
        topk_val = comparison.get("top_k_fraction")
        seed_dt = (
            f"<dt>seed</dt><dd>{html.escape(str(seed_val))}</dd>"
            if seed_val is not None
            else ""
        )
        topk_dt = (
            f"<dt>top_k_fraction</dt><dd>{html.escape(str(topk_val))}</dd>"
            if topk_val is not None
            else ""
        )
        comp_block = (
            "<h3>Comparison report (on disk)</h3>"
            "<dl class=\"meta-dl\">"
            f"<dt>generated_at</dt><dd>{html.escape(str(comparison.get('generated_at', '')))}</dd>"
            f"<dt>fold_index</dt><dd>{html.escape(str(comparison.get('fold_index', '')))}</dd>"
            f"<dt>fold_metric</dt><dd>{html.escape(str(comparison.get('fold_metric', '')))}</dd>"
            f"<dt>split</dt><dd>{html.escape(str(comparison.get('split', '')))}</dd>"
            f"{seed_dt}{topk_dt}"
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
    <nav><strong>Jump:</strong> <a href="#summary">Metrics table</a> · {nav_html}</nav>
    <div class="controls">
      <label for="filter-q">Filter graphs by id</label>
      <input type="search" id="filter-q" placeholder="e.g. 7GAW" autocomplete="off">
    </div>
  </header>

  <section id="summary">
    <h2>Metrics by explainer</h2>
    <p class="muted">Click column headers to sort. Mean Fid+ and Mean Fid&minus; are PyG class-decision GraphFramEx rates, not probability-drop ratios. Paths are relative to this HTML file.</p>
    <table class="summary" id="summary-table">
      <thead><tr>{thead}</tr></thead>
      <tbody>{tbody}</tbody>
    </table>
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

  const table = document.getElementById("summary-table");
  if (!table) return;
  const theadRow = table.querySelector("thead tr");
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


def _sort_value(val: Any, tp: str) -> str:
    if tp != "num":
        return ""
    if val is None:
        return ""
    try:
        return str(float(val))
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Global cross-fold index
# ---------------------------------------------------------------------------

_GLOBAL_METRIC_KEYS = [
    ("mean_fid_plus", "Mean Fid+"),
    ("mean_fid_minus", "Mean Fid\u2212"),
    ("mean_paper_sufficiency", "Mean Fsuf (raw)"),
    ("mean_paper_comprehensiveness", "Mean Fcom (raw)"),
    ("mean_paper_f1_fidelity", "Mean Ff1 (clamped)"),
    ("mean_fid_plus_all_graphs", "All Fid+"),
    ("mean_fid_minus_all_graphs", "All Fid\u2212"),
    ("mean_paper_sufficiency_all_graphs", "All Fsuf (raw)"),
    ("mean_paper_comprehensiveness_all_graphs", "All Fcom (raw)"),
    ("mean_paper_f1_fidelity_all_graphs", "All Ff1 (clamped)"),
    ("num_valid", "Valid"),
    ("wall_time_s", "Wall (s)"),
]


def _nanmean_safe(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def write_global_explanation_index(
    results_root: Path,
    fold_entries: list[dict],
) -> Path:
    """Write ``results_root/explanation_web_report/index.html`` linking all per-fold reports.

    *fold_entries* is a list of dicts, each with keys:
    ``fold_index``, ``explainer_names``, ``per_explainer_summary``.
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

    # --- Per-fold summary table ---
    fold_table_rows: list[str] = []
    for entry in fold_entries_sorted:
        k = entry["fold_index"]
        per_expl = entry.get("per_explainer_summary", {})
        fold_link = f"folds/fold_{k}/{EXPLANATION_WEB_REPORT_DIR}/index.html"

        avg_vals: dict[str, float | None] = {}
        for mkey, _ in _GLOBAL_METRIC_KEYS:
            vals = [
                s.get(mkey)
                for s in per_expl.values()
                if isinstance(s, dict)
            ]
            avg_vals[mkey] = _nanmean_safe(vals)

        cells = [
            f'<td><a href="{html.escape(fold_link, quote=True)}">Fold {k}</a></td>'
        ]
        for mkey, _ in _GLOBAL_METRIC_KEYS:
            v = avg_vals.get(mkey)
            sv = _sort_value(v, "num")
            cells.append(
                f'<td data-sort-value="{sv}">{_fmt_num(v)}</td>'
            )
        fold_table_rows.append("<tr>" + "".join(cells) + "</tr>")

    fold_thead = (
        '<th data-sort="fold" data-type="text">Fold</th>'
        + "".join(
            f'<th data-sort="{html.escape(mk)}" data-type="num">{html.escape(ml)}</th>'
            for mk, ml in _GLOBAL_METRIC_KEYS
        )
    )
    fold_tbody = "\n".join(fold_table_rows) if fold_table_rows else "<tr><td>No data</td></tr>"

    # --- Per-explainer cross-fold table ---
    explainer_sections: list[str] = []
    for explainer_name in all_explainer_names:
        rows: list[str] = []
        for entry in fold_entries_sorted:
            k = entry["fold_index"]
            per_expl = entry.get("per_explainer_summary", {})
            s = per_expl.get(explainer_name, {})
            fold_link = f"folds/fold_{k}/{EXPLANATION_WEB_REPORT_DIR}/index.html#explainer-{explainer_name}"
            cells = [
                f'<td><a href="{html.escape(fold_link, quote=True)}">Fold {k}</a></td>'
            ]
            for mkey, _ in _GLOBAL_METRIC_KEYS:
                v = s.get(mkey) if isinstance(s, dict) else None
                sv = _sort_value(v, "num")
                cells.append(f'<td data-sort-value="{sv}">{_fmt_num(v)}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")

        tbody_expl = "\n".join(rows) if rows else "<tr><td>No data</td></tr>"
        explainer_sections.append(
            f'<section id="explainer-{html.escape(explainer_name, quote=True)}">'
            f"<h3>{html.escape(explainer_name)}</h3>"
            f'<table class="summary"><thead><tr>{fold_thead}</tr></thead>'
            f"<tbody>{tbody_expl}</tbody></table>"
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
    .muted {{ color: var(--muted); }}
    nav {{ font-size: 0.9rem; margin: 0.75rem 0; line-height: 1.6; }}
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
    <p class="muted">Metrics averaged across all explainers in each fold. Headline means are valid-only; columns prefixed with All are diagnostics over every explained graph. Click a fold to open its detailed report.</p>
    <table class="summary" id="fold-table">
      <thead><tr>{fold_thead}</tr></thead>
      <tbody>{fold_tbody}</tbody>
    </table>
  </section>

  <section id="per-explainer">
    <h2>Per-explainer across folds</h2>
    <p class="muted">Each table shows one explainer&rsquo;s metrics across all folds. Compare valid-only headline means with All diagnostics before drawing final conclusions.</p>
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
