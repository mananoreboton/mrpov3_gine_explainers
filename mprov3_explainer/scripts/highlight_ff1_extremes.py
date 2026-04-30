#!/usr/bin/env python3
"""Generate copies of HTML report files with min/max highlighting on Ff1 columns.

For every ``*.html`` file found recursively under ``--root``, this script:

1. Parses the HTML with BeautifulSoup.
2. For each ``<table>``, identifies columns whose header text contains ``Ff1``
   (case-insensitive substring match).
3. Within each such column, finds the minimum and maximum numeric cell values
   and applies CSS classes (``ff1-min`` / ``ff1-max``) so they are visually
   highlighted.
4. Writes a sibling copy with a configurable suffix (default
   ``.ff1_extremes``), e.g. ``index.html`` → ``index.ff1_extremes.html``.

Usage (from ``mprov3_explainer/``):

    uv run python scripts/highlight_ff1_extremes.py
    uv run python scripts/highlight_ff1_extremes.py --root results/explanation_web_report --suffix .highlighted
    uv run python scripts/highlight_ff1_extremes.py --dry-run
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import re

from bs4 import BeautifulSoup, Tag

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = _SCRIPT_DIR.parent / "results"

_HIGHLIGHT_CSS = """\
.ff1-min {
  background: rgba(248, 81, 73, 0.25) !important;
  box-shadow: inset 0 0 0 1.5px #f85149;
}
.ff1-max {
  background: rgba(63, 185, 80, 0.25) !important;
  box-shadow: inset 0 0 0 1.5px #3fb950;
}
"""


def _parse_cell_value(td: Tag) -> float | None:
    """Extract a numeric value from a ``<td>``, returning *None* if missing."""
    raw = td.get("data-sort-value")
    if raw is not None and raw != "":
        try:
            v = float(raw)
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            pass
    text = td.get_text(strip=True)
    if not text or text in ("\u2014", "—", "---", "nan"):
        return None
    try:
        v = float(text)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _inject_css(soup: BeautifulSoup) -> None:
    """Add the highlight CSS to ``<head>`` if not already present."""
    head = soup.find("head")
    if head is None:
        return
    for style in head.find_all("style"):
        if "ff1-min" in (style.string or ""):
            return
    new_style = soup.new_tag("style")
    new_style.string = _HIGHLIGHT_CSS
    head.append(new_style)


def _rewrite_html_links(soup: BeautifulSoup, *, suffix: str) -> bool:
    """Rewrite relative ``.html`` links to point to their suffixed copies."""
    changed = False
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href or not isinstance(href, str):
            continue
        if href.startswith("#"):
            continue

        parts = urlsplit(href)
        if parts.scheme or parts.netloc:
            # Leave absolute URLs (http(s), file://, etc.) untouched.
            continue

        path = parts.path
        if not path or not path.endswith(".html"):
            continue

        base = path[:-5]  # strip ".html"
        if base.endswith(suffix):
            continue

        new_path = f"{base}{suffix}.html"
        new_href = urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))
        if new_href != href:
            a["href"] = new_href
            changed = True
    return changed


def _latex_escape(s: str) -> str:
    # Keep it minimal; these pages mostly contain plain ASCII + underscores.
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def _looks_numeric(s: str) -> bool:
    s = s.strip()
    if not s or s in ("\u2014", "—", "---"):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _table_to_latex(
    table: Tag,
    *,
    caption: str,
    label: str,
) -> str:
    thead = table.find("thead")
    header_cells: list[str] = []
    if thead:
        tr = thead.find("tr")
        if tr:
            header_cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
    if not header_cells:
        # Fallback: first row
        tr = table.find("tr")
        if tr:
            header_cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]

    tbody = table.find("tbody")
    rows = (tbody.find_all("tr") if tbody else table.find_all("tr")[1:])
    body: list[list[str]] = []
    for r in rows:
        cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
        if cells:
            body.append(cells)

    ncols = max(len(header_cells), max((len(r) for r in body), default=0))
    if ncols == 0:
        return ""

    # Heuristic column spec: first col left, rest right if numeric-ish.
    col_specs: list[str] = []
    for j in range(ncols):
        if j == 0:
            col_specs.append("l")
            continue
        # Check a few rows to see if column is numeric
        samples = [r[j] for r in body if j < len(r)][:5]
        is_num = bool(samples) and all(_looks_numeric(x) for x in samples)
        col_specs.append("r" if is_num else "l")

    def _row_to_line(vals: list[str]) -> str:
        padded = (vals + [""] * ncols)[:ncols]
        return " & ".join(_latex_escape(v) for v in padded) + r" \\"

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{''.join(col_specs)}}}",
        r"\toprule",
    ]
    if header_cells:
        lines.append(_row_to_line(header_cells))
        lines.append(r"\midrule")
    for r in body:
        lines.append(_row_to_line(r))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _class_id_from_filename(name: str) -> int | None:
    m = re.match(r"explainer_summary_class_(\d+)\.html$", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _find_ff1_col_idx(table: Tag) -> int | None:
    thead = table.find("thead")
    tr = thead.find("tr") if thead else table.find("tr")
    if not tr:
        return None
    headers = tr.find_all(["th", "td"])
    for idx, th in enumerate(headers):
        if "ff1" in th.get_text(" ", strip=True).lower():
            return idx
    return None


def _inject_benchmark_section_for_class_page(soup: BeautifulSoup) -> bool:
    """Add Benchmark section/table to explainer_summary_class_* pages."""
    # Discover folds present
    fold_sections = soup.find_all("section", id=re.compile(r"^fold-\d+$"))
    if not fold_sections:
        return False

    rows: list[dict[str, object]] = []
    for sec in fold_sections:
        sec_id = sec.get("id", "")
        try:
            fold_idx = int(str(sec_id).split("-", 1)[1])
        except Exception:
            continue

        # Find the valid metrics table within this fold section.
        # Structure: <h4 class="muted">Valid result metrics</h4><table class="summary">...</table>
        valid_h4 = None
        for h4 in sec.find_all("h4"):
            if "valid result metrics" in h4.get_text(" ", strip=True).lower():
                valid_h4 = h4
                break
        if valid_h4 is None:
            continue
        valid_table = valid_h4.find_next("table", class_="summary")
        if valid_table is None:
            continue

        ff1_idx = _find_ff1_col_idx(valid_table)
        if ff1_idx is None:
            continue

        tbody = valid_table.find("tbody")
        trs = tbody.find_all("tr") if tbody else valid_table.find_all("tr")[1:]
        vals: list[tuple[str, float]] = []
        for tr in trs:
            cells = tr.find_all(["td", "th"])
            if len(cells) <= ff1_idx:
                continue
            expl = cells[0].get_text(" ", strip=True)
            v = _parse_cell_value(cells[ff1_idx])
            if expl and v is not None:
                vals.append((expl, float(v)))
        if len(vals) < 2:
            continue

        min_val = min(v for _, v in vals)
        max_val = max(v for _, v in vals)
        best = sorted({e for e, v in vals if v == max_val})
        worst = sorted({e for e, v in vals if v == min_val})

        rows.append({
            "fold": fold_idx,
            "best_explainers": ", ".join(best),
            "best_val": max_val,
            "worst_explainers": ", ".join(worst),
            "worst_val": min_val,
        })

    if not rows:
        return False

    # Build HTML table
    def _td_num(val: float, cls: str) -> Tag:
        td = soup.new_tag("td")
        td["data-sort-value"] = str(float(val))
        td["class"] = [cls]
        td.string = f"{val:.6g}"
        return td

    table = soup.new_tag("table")
    table["class"] = ["summary"]
    thead = soup.new_tag("thead")
    trh = soup.new_tag("tr")
    headers = [
        ("Fold", "text"),
        ("Best explainer(s)", "text"),
        ("Best Mean Ff1", "num"),
        ("Worst explainer(s)", "text"),
        ("Worst Mean Ff1", "num"),
    ]
    for label, tp in headers:
        th = soup.new_tag("th")
        th["data-type"] = tp
        th["data-sort"] = label.replace(" ", "_")
        th.string = label
        trh.append(th)
    thead.append(trh)
    table.append(thead)

    tbody = soup.new_tag("tbody")
    for r in sorted(rows, key=lambda x: int(x["fold"])):  # type: ignore[index]
        tr = soup.new_tag("tr")
        td_fold = soup.new_tag("td")
        td_fold.string = f"Fold {r['fold']}"
        tr.append(td_fold)

        td_best = soup.new_tag("td")
        td_best.string = str(r["best_explainers"])
        tr.append(td_best)

        tr.append(_td_num(float(r["best_val"]), "ff1-max"))

        td_worst = soup.new_tag("td")
        td_worst.string = str(r["worst_explainers"])
        tr.append(td_worst)

        tr.append(_td_num(float(r["worst_val"]), "ff1-min"))

        tbody.append(tr)
    table.append(tbody)

    bench_sec = soup.new_tag("section")
    bench_sec["id"] = "benchmark"
    h2 = soup.new_tag("h2")
    h2.string = "Benchmark"
    p = soup.new_tag("p")
    p["class"] = ["muted"]
    p.string = "Best/worst Mean Ff1 per fold (Valid result metrics)."
    bench_sec.append(h2)
    bench_sec.append(p)
    bench_sec.append(table)

    # Insert after header
    header = soup.find("header")
    if header and header.parent:
        header.insert_after(bench_sec)
    else:
        body = soup.find("body")
        if body:
            body.insert(0, bench_sec)

    # Add to nav if present
    nav = soup.find("nav")
    if nav:
        # Insert before LaTeX if possible, else append at end.
        a = soup.new_tag("a", href="#benchmark")
        a.string = "Benchmark"
        nav.append(" · ")
        nav.append(a)

    return True


def _update_latex_export_for_class_page(
    soup: BeautifulSoup,
    *,
    class_id: int | None,
) -> bool:
    """Replace the LaTeX export block with LaTeX matching visible tables (plus Benchmark)."""
    details = soup.find("details", class_="latex-export")
    if details is None:
        return False
    pre = details.find("pre")
    if pre is None:
        return False

    # Collect the tables we want (exclude any tables inside the latex-export block itself).
    body = soup.find("body")
    if body is None:
        return False

    tables: list[tuple[str, str, Tag]] = []

    def _add_table_if_found(section_id: str, table: Tag | None, caption: str, label: str) -> None:
        if table is None:
            return
        if details in table.parents:
            return
        tables.append((caption, label, table))

    cid = f"class{class_id}" if class_id is not None else "classX"

    # Benchmark
    bench = soup.find("section", id="benchmark")
    if bench:
        _add_table_if_found(
            "benchmark",
            bench.find("table", class_="summary"),
            f"Benchmark (best/worst Mean Ff1 per fold) [{cid}]",
            f"tab:{cid}-benchmark",
        )

    # Mean summary (valid)
    mean = soup.find("section", id="mean-summary")
    if mean:
        # First summary table is the mean-valid table on class pages.
        _add_table_if_found(
            "mean-summary",
            mean.find("table", class_="summary"),
            f"Mean across folds (valid result metrics) [{cid}]",
            f"tab:{cid}-mean-valid",
        )

    # Coverage
    cov = soup.find("section", id="coverage")
    if cov:
        _add_table_if_found(
            "coverage",
            cov.find("table", class_="summary"),
            f"Valid-graph coverage [{cid}]",
            f"tab:{cid}-coverage",
        )

    # Weighted / Unweighted
    weighted = soup.find("section", id="weighted")
    if weighted:
        _add_table_if_found(
            "weighted",
            weighted.find("table", class_="summary"),
            f"Weighted statistics across folds (valid) [{cid}]",
            f"tab:{cid}-weighted-valid",
        )
    unweighted = soup.find("section", id="unweighted")
    if unweighted:
        _add_table_if_found(
            "unweighted",
            unweighted.find("table", class_="summary"),
            f"Unweighted statistics across folds (valid) [{cid}]",
            f"tab:{cid}-unweighted-valid",
        )

    # Per-fold tables
    per_fold = soup.find("section", id="per-fold")
    if per_fold:
        for fold_sec in per_fold.find_all("section", id=re.compile(r"^fold-\d+$")):
            sec_id = str(fold_sec.get("id", ""))
            try:
                k = int(sec_id.split("-", 1)[1])
            except Exception:
                continue
            # Grab the table after the "Valid result metrics" h4.
            valid_h4 = None
            for h4 in fold_sec.find_all("h4"):
                if "valid result metrics" in h4.get_text(" ", strip=True).lower():
                    valid_h4 = h4
                    break
            if valid_h4 is None:
                continue
            tbl = valid_h4.find_next("table", class_="summary")
            _add_table_if_found(
                sec_id,
                tbl,
                f"Fold {k} — Valid result metrics [{cid}]",
                f"tab:{cid}-fold{k}-valid",
            )

    latex_parts: list[str] = []
    for caption, label, table in tables:
        tex = _table_to_latex(table, caption=caption, label=label)
        if tex:
            latex_parts.append(tex)
            latex_parts.append("")  # spacing

    latex_content = "\n".join(latex_parts).strip()
    if not latex_content:
        return False

    # Replace pre contents
    pre.clear()
    pre.string = latex_content
    return True


def _ff1_column_indices(header_row: Tag) -> list[int]:
    """Return 0-based column indices whose header text contains 'Ff1'."""
    indices: list[int] = []
    for idx, th in enumerate(header_row.find_all(["th", "td"])):
        if "ff1" in th.get_text().lower():
            indices.append(idx)
    return indices


def _highlight_table(table: Tag) -> bool:
    """Apply min/max classes to Ff1 columns in *table*. Return True if any cell was touched."""
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
    else:
        header_row = table.find("tr")
    if header_row is None:
        return False

    ff1_cols = _ff1_column_indices(header_row)
    if not ff1_cols:
        return False

    tbody = table.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
    if not body_rows:
        return False

    touched = False
    for col_idx in ff1_cols:
        cells: list[tuple[Tag, float]] = []
        for row in body_rows:
            tds = row.find_all(["td", "th"])
            if col_idx >= len(tds):
                continue
            td = tds[col_idx]
            val = _parse_cell_value(td)
            if val is not None:
                cells.append((td, val))

        if len(cells) < 2:
            continue

        min_val = min(v for _, v in cells)
        max_val = max(v for _, v in cells)
        if min_val == max_val:
            continue

        for td, val in cells:
            existing = td.get("class", [])
            if isinstance(existing, str):
                existing = existing.split()
            new_classes = list(existing)
            if val == min_val:
                new_classes.append("ff1-min")
            if val == max_val:
                new_classes.append("ff1-max")
            if new_classes != existing:
                td["class"] = new_classes
                touched = True

    return touched


def process_file(html_path: Path, suffix: str, *, dry_run: bool = False) -> Path | None:
    """Process one HTML file and write the highlighted copy. Return output path or None."""
    raw = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    tables = soup.find_all("table")
    any_change = False
    for table in tables:
        if _highlight_table(table):
            any_change = True

    if not any_change:
        return None

    _inject_css(soup)
    _rewrite_html_links(soup, suffix=suffix)

    class_id = _class_id_from_filename(html_path.name)
    if class_id is not None:
        _inject_benchmark_section_for_class_page(soup)
        _update_latex_export_for_class_page(soup, class_id=class_id)

    stem = html_path.stem
    out_name = f"{stem}{suffix}{html_path.suffix}"
    out_path = html_path.parent / out_name

    if dry_run:
        print(f"[dry-run] Would write: {out_path}")
        return out_path

    out_path.write_text(str(soup), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy HTML reports with min/max highlighting on Ff1 columns.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="Root directory to scan recursively for *.html files "
             f"(default: {_DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".ff1_extremes",
        help='Suffix inserted before .html in output filenames (default: ".ff1_extremes").',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing anything.",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Root directory does not exist: {root}")

    html_files = sorted(root.rglob("*.html"))
    # Skip files that are themselves highlighted copies
    html_files = [f for f in html_files if args.suffix not in f.stem]

    if not html_files:
        print(f"No HTML files found under {root}")
        return

    print(f"Scanning {len(html_files)} HTML file(s) under {root} ...\n")

    written = 0
    skipped = 0
    for path in html_files:
        result = process_file(path, args.suffix, dry_run=args.dry_run)
        if result:
            rel = result.relative_to(root) if result.is_relative_to(root) else result
            print(f"  {'[dry-run] ' if args.dry_run else ''}Written: {rel}")
            written += 1
        else:
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            print(f"  Skipped (no Ff1 columns): {rel}")
            skipped += 1

    action = "Would write" if args.dry_run else "Wrote"
    print(f"\nDone. {action} {written} file(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
