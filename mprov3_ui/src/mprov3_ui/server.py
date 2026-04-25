"""
mprov3-ui — zero-dependency HTTP server for GINE + Explainer result sites.

Routes
------
/                landing page (inline HTML)
/gine/           mprov3_gine/results/visualizations/
/classifications/ mprov3_gine/results/classifications/
/explainer/      mprov3_explainer/results/folds/fold_0/explanation_web_report/
/visualizations/ mprov3_explainer/results/folds/fold_0/visualizations/
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import posixpath
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths resolved relative to this file so the server works regardless of cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
# server.py  →  mprov3_ui/  →  src/  →  mprov3_ui/  →  from_scratch/
_ROOT = _HERE.parents[2]

_EXPLAINER_FOLD = _ROOT / "mprov3_explainer" / "results" / "folds" / "fold_0"

_ROUTES: dict[str, Path] = {
    "/gine/":             _ROOT / "mprov3_gine" / "results" / "visualizations",
    "/classifications/":  _ROOT / "mprov3_gine" / "results" / "classifications",
    "/explainer/":        _EXPLAINER_FOLD / "explanation_web_report",
    # The explainer HTML uses absolute paths like /visualizations/… for images
    "/visualizations/":   _EXPLAINER_FOLD / "visualizations",
}

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
_LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MPro v3 Results</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: #f5f7fa;
      color: #1f2328;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 3rem 1.5rem;
    }
    header { text-align: center; margin-bottom: 2.5rem; }
    h1 { font-size: 1.75rem; font-weight: 700; color: #1f2328; }
    p.subtitle { margin-top: 0.5rem; color: #656d76; font-size: 0.95rem; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.5rem;
      width: 100%;
      max-width: 900px;
    }
    /* Plain link card (Explainer) */
    a.card {
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 10px;
      padding: 1.75rem 2rem;
      text-decoration: none;
      color: inherit;
      transition: box-shadow 0.15s, border-color 0.15s;
      display: block;
    }
    a.card:hover {
      box-shadow: 0 4px 16px rgba(0,0,0,0.10);
      border-color: #0969da;
    }
    /* Group card (GINE) — not a link itself */
    .card-group {
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 10px;
      padding: 1.75rem 2rem;
    }
    .card-icon { font-size: 2rem; margin-bottom: 0.75rem; }
    .card-group h2, a.card h2 { font-size: 1.15rem; font-weight: 600; color: #1f2328; margin-bottom: 0.25rem; }
    .card-group p.card-desc, a.card p { margin-top: 0.4rem; font-size: 0.875rem; color: #656d76; line-height: 1.5; margin-bottom: 1rem; }
    .sub-links { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.75rem; }
    .sub-link {
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
    }
    .sub-link:hover { background: #eaf0fb; border-color: #0969da; }
    .sub-link .sl-icon { font-size: 1.1rem; }
    .sub-link .sl-desc { font-size: 0.78rem; color: #656d76; font-weight: 400; margin-left: auto; }
    a.card h2 { color: #0969da; }
    footer {
      margin-top: 3rem;
      font-size: 0.8rem;
      color: #656d76;
    }
  </style>
</head>
<body>
  <header>
    <h1>MPro v3 — Results Dashboard</h1>
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
    <a class="card" href="/explainer/">
      <div class="card-icon">&#128269;</div>
      <h2>Explainer Report</h2>
      <p>Per-explainer fidelity metrics (fidelity+/&minus;, sufficiency, comprehensiveness) for fold 0.</p>
    </a>
  </div>
  <footer>mprov3-ui &mdash; served locally</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        print(f"  {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        raw = urllib.parse.unquote(parsed.path)
        # Normalise without stripping a meaningful trailing slash
        url_path = posixpath.normpath(raw)
        if raw.endswith("/") and not url_path.endswith("/"):
            url_path += "/"

        # Landing page
        if url_path in ("/", ""):
            self._send_html(_LANDING_HTML.encode())
            return

        # Redirect bare prefix (no trailing slash) → trailing-slash version
        for prefix in _ROUTES:
            if url_path == prefix.rstrip("/"):
                self._redirect(prefix)
                return

        # Static file routes
        for prefix, fs_root in _ROUTES.items():
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

    # Validate result directories exist (warn but don't abort)
    for prefix, fs_path in _ROUTES.items():
        if not fs_path.exists():
            print(f"  WARNING: results directory for '{prefix}' not found: {fs_path}")

    url = f"http://localhost:{args.port}"
    server = ThreadingHTTPServer(("localhost", args.port), _Handler)

    print(f"mprov3-ui  →  {url}")
    print(f"  /gine/             → {_ROUTES['/gine/']}")
    print(f"  /classifications/  → {_ROUTES['/classifications/']}")
    print(f"  /explainer/        → {_ROUTES['/explainer/']}")
    print(f"  /visualizations/   → {_ROUTES['/visualizations/']}")
    print("  Press Ctrl-C to stop.\n")

    if not args.no_browser:
        # Open after a short delay so the server socket is ready
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
