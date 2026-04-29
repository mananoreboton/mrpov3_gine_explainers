"""
mprov3-ui — zero-dependency HTTP server for GINE + Explainer result sites.

Routes
------
/                          landing page (dynamically lists available folds)
/gine/                     mprov3_gine/results/visualizations/
/classifications/          mprov3_gine/results/classifications/
/explainer/                global cross-fold index (or single-fold fallback)
/explainer/explainer_summary.html  explainer summary (primary comparative view)
/explainer/fold_K/         per-fold root (redirects to explanation_web_report/)
/explainer/fold_K/<path>   serves files from results/folds/fold_K/<path>
"""

from __future__ import annotations

import argparse
import mimetypes
import posixpath
import re
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths resolved relative to this file so the server works regardless of cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

_EXPLAINER_RESULTS = _ROOT / "mprov3_explainer" / "results"
_EXPLAINER_FOLDS = _EXPLAINER_RESULTS / "folds"
_EXPLAINER_GLOBAL_REPORT = _EXPLAINER_RESULTS / "explanation_web_report"

_STATIC_ROUTES: dict[str, Path] = {
    "/gine/":             _ROOT / "mprov3_gine" / "results" / "visualizations",
    "/classifications/":  _ROOT / "mprov3_gine" / "results" / "classifications",
}

_FOLD_ROUTE_RE = re.compile(
    r"^/explainer/(?:folds/)?fold_(?P<fold>\d+)/(?P<rest>.*)$"
)


def _discover_explainer_folds() -> list[int]:
    """Scan results/folds/ for fold directories that contain explanation_web_report/."""
    if not _EXPLAINER_FOLDS.is_dir():
        return []
    folds: list[int] = []
    for sub in sorted(_EXPLAINER_FOLDS.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("fold_"):
            continue
        try:
            fi = int(sub.name.removeprefix("fold_"))
        except ValueError:
            continue
        report = sub / "explanation_web_report" / "index.html"
        explanations = sub / "explanations"
        if report.is_file() or explanations.is_dir():
            folds.append(fi)
    return folds


def _build_landing_html(fold_indices: list[int]) -> str:
    if fold_indices:
        fold_links = "\n".join(
            f'        <a class="sub-link" href="/explainer/fold_{k}/">'
            f'<span class="sl-icon">&#128203;</span>'
            f"Fold {k}"
            f'<span class="sl-desc">metrics &amp; mask visualizations</span>'
            f"</a>"
            for k in fold_indices
        )
        explainer_card = (
            '    <div class="card-group">\n'
            '      <div class="card-icon">&#128269;</div>\n'
            '      <h2>Explainer Reports</h2>\n'
            f'      <p class="card-desc">Per-explainer fidelity metrics across {len(fold_indices)} fold(s).</p>\n'
            '      <div class="sub-links">\n'
            '        <a class="sub-link" href="/explainer/explainer_summary.html">'
            '<span class="sl-icon">&#128202;</span>'
            'Explainer summary'
            '<span class="sl-desc">comparative view across folds</span>'
            '</a>\n'
            '        <a class="sub-link" href="/explainer/">'
            '<span class="sl-icon">&#128203;</span>'
            'Per-fold breakdown'
            '<span class="sl-desc">detailed cross-fold index</span>'
            '</a>\n'
            f'{fold_links}\n'
            '      </div>\n'
            '    </div>'
        )
    else:
        explainer_card = (
            '    <a class="card" href="/explainer/">\n'
            '      <div class="card-icon">&#128269;</div>\n'
            '      <h2>Explainer Report</h2>\n'
            '      <p>Per-explainer fidelity metrics.</p>\n'
            '    </a>'
        )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MPro v3 Results</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: #f5f7fa;
      color: #1f2328;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 3rem 1.5rem;
    }}
    header {{ text-align: center; margin-bottom: 2.5rem; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; color: #1f2328; }}
    p.subtitle {{ margin-top: 0.5rem; color: #656d76; font-size: 0.95rem; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.5rem;
      width: 100%;
      max-width: 900px;
    }}
    a.card {{
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 10px;
      padding: 1.75rem 2rem;
      text-decoration: none;
      color: inherit;
      transition: box-shadow 0.15s, border-color 0.15s;
      display: block;
    }}
    a.card:hover {{
      box-shadow: 0 4px 16px rgba(0,0,0,0.10);
      border-color: #0969da;
    }}
    .card-group {{
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 10px;
      padding: 1.75rem 2rem;
    }}
    .card-icon {{ font-size: 2rem; margin-bottom: 0.75rem; }}
    .card-group h2, a.card h2 {{ font-size: 1.15rem; font-weight: 600; color: #1f2328; margin-bottom: 0.25rem; }}
    .card-group p.card-desc, a.card p {{ margin-top: 0.4rem; font-size: 0.875rem; color: #656d76; line-height: 1.5; margin-bottom: 1rem; }}
    .sub-links {{ display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.75rem; }}
    .sub-link {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      padding: 0.6rem 0.85rem;
      background: #f5f7fa;
      border: 1px solid #d0d7de;
      border-radius: 7px;
      text-decoration: none;
      color: #0969da;
      font-size: 0.9rem;
      font-weight: 500;
      transition: background 0.12s, border-color 0.12s;
    }}
    .sub-link:hover {{ background: #eaf0fb; border-color: #0969da; }}
    .sub-link .sl-icon {{ font-size: 1.1rem; }}
    .sub-link .sl-desc {{ font-size: 0.78rem; color: #656d76; font-weight: 400; margin-left: auto; }}
    a.card h2 {{ color: #0969da; }}
    footer {{
      margin-top: 3rem;
      font-size: 0.8rem;
      color: #656d76;
    }}
  </style>
</head>
<body>
  <header>
    <h1>MPro v3 &mdash; Results Dashboard</h1>
    <p class="subtitle">Browse the GNN training results and the explainer evaluation reports.</p>
  </header>
  <div class="cards">
    <div class="card-group">
      <div class="card-icon">&#128202;</div>
      <h2>GINE Results</h2>
      <p class="card-desc">GNN training outputs: ligand-graph visualizations and per-fold classification performance.</p>
      <div class="sub-links">
        <a class="sub-link" href="/gine/">
          <span class="sl-icon">&#128444;</span>
          Visualizations
          <span class="sl-desc">graph gallery &amp; per-PDB pages</span>
        </a>
        <a class="sub-link" href="/classifications/">
          <span class="sl-icon">&#9989;</span>
          Classifications
          <span class="sl-desc">fold accuracy &amp; correct/wrong grid</span>
        </a>
      </div>
    </div>
{explainer_card}
  </div>
  <footer>mprov3-ui &mdash; served locally</footer>
</body>
</html>
"""


def _resolve_explainer_root(url_path: str) -> Path | None:
    """Map /explainer/ to global index or single-fold fallback."""
    if _EXPLAINER_GLOBAL_REPORT.is_dir():
        return _EXPLAINER_GLOBAL_REPORT

    folds = _discover_explainer_folds()
    if len(folds) == 1:
        return _EXPLAINER_FOLDS / f"fold_{folds[0]}" / "explanation_web_report"
    return None


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        print(f"  {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        raw = urllib.parse.unquote(parsed.path)
        url_path = posixpath.normpath(raw)
        if raw.endswith("/") and not url_path.endswith("/"):
            url_path += "/"

        # Landing page
        if url_path in ("/", ""):
            fold_indices = _discover_explainer_folds()
            html_body = _build_landing_html(fold_indices)
            self._send_html(html_body.encode())
            return

        # Redirect bare prefix → trailing-slash version
        for prefix in list(_STATIC_ROUTES) + ["/explainer/"]:
            if url_path == prefix.rstrip("/"):
                self._redirect(prefix)
                return

        # Per-fold routes: /explainer/fold_K/...
        # Maps to the fold root so that relative paths from
        # explanation_web_report/index.html (e.g. ../visualizations/...)
        # resolve correctly within the same URL subtree.
        m = _FOLD_ROUTE_RE.match(url_path)
        if m:
            fold_k = int(m.group("fold"))
            rest = m.group("rest")
            fold_dir = _EXPLAINER_FOLDS / f"fold_{fold_k}"
            if not rest:
                self._redirect(f"/explainer/fold_{fold_k}/explanation_web_report/")
                return
            fs_path = fold_dir / rest
            if fs_path.is_dir():
                fs_path = fs_path / "index.html"
            self._serve_file(fs_path)
            return

        # Bare /explainer/fold_K or /explainer/folds/fold_K redirect
        bare_expl_match = re.match(r"^/explainer/(?:folds/)?fold_(\d+)$", url_path)
        if bare_expl_match:
            self._redirect(f"/explainer/fold_{bare_expl_match.group(1)}/")
            return

        # Global /explainer/ route
        if url_path.startswith("/explainer/"):
            expl_root = _resolve_explainer_root(url_path)
            if expl_root:
                rel = url_path[len("/explainer/"):]
                fs_path = expl_root / rel if rel else expl_root / "index.html"
                if fs_path.is_dir():
                    fs_path = fs_path / "index.html"
                self._serve_file(fs_path)
                return
            self._send_error(404, "No explainer reports found")
            return

        # Static file routes (GINE, classifications)
        for prefix, fs_root in _STATIC_ROUTES.items():
            if url_path.startswith(prefix):
                rel = url_path[len(prefix):]
                fs_path = fs_root / rel if rel else fs_root / "index.html"
                if fs_path.is_dir():
                    fs_path = fs_path / "index.html"
                self._serve_file(fs_path)
                return

        self._send_error(404, "Not found")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _redirect(self, location: str) -> None:
        self.send_response(301)
        self.send_header("Location", location)
        self.end_headers()

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        body = message.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        if not path.exists():
            self._send_error(404, f"File not found: {path.name}")
            return
        if not path.is_file():
            self._send_error(400, "Not a file")
            return

        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"

        try:
            data = path.read_bytes()
        except OSError as exc:
            self._send_error(500, str(exc))
            return

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mprov3-ui",
        description="Serve MPro v3 result sites locally.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        metavar="PORT",
        help="TCP port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically (headless / CI mode)",
    )
    args = parser.parse_args()

    fold_indices = _discover_explainer_folds()

    # Validate result directories exist (warn but don't abort)
    for prefix, fs_path in _STATIC_ROUTES.items():
        if not fs_path.exists():
            print(f"  WARNING: results directory for '{prefix}' not found: {fs_path}")

    url = f"http://localhost:{args.port}"
    server = ThreadingHTTPServer(("localhost", args.port), _Handler)

    print(f"mprov3-ui  →  {url}")
    print(f"  /gine/             → {_STATIC_ROUTES['/gine/']}")
    print(f"  /classifications/  → {_STATIC_ROUTES['/classifications/']}")
    if _EXPLAINER_GLOBAL_REPORT.is_dir():
        print(f"  /explainer/        → {_EXPLAINER_GLOBAL_REPORT} (global index)")
    elif len(fold_indices) == 1:
        print(f"  /explainer/        → fold_{fold_indices[0]} (single fold)")
    else:
        print(f"  /explainer/        → (no reports found)")
    for k in fold_indices:
        fold_dir = _EXPLAINER_FOLDS / f"fold_{k}"
        print(f"  /explainer/fold_{k}/ → {fold_dir}")
    print("  Press Ctrl-C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
