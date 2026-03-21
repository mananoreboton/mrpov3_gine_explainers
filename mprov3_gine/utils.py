"""
Shared utilities: run timestamps, latest-run resolution, logging, and HTML helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Timestamp format for result subfolders (e.g. 2025-03-14_120000)
RUN_TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"


def run_timestamp() -> str:
    """Return current UTC timestamp string for use in output paths."""
    return datetime.now(timezone.utc).strftime(RUN_TIMESTAMP_FMT)


def get_latest_timestamp_dir(base_path: Path) -> Optional[Path]:
    """
    Return the path to the most recent timestamp-named subfolder under base_path.
    Expects subfolder names like 2025-03-14_120000. Returns None if no valid subfolder exists.
    """
    if not base_path.exists() or not base_path.is_dir():
        return None
    candidates: List[Path] = []
    for p in base_path.iterdir():
        if p.is_dir() and len(p.name) == 17 and p.name[4] == "-" and p.name[7] == "-" and p.name[10] == "_":
            try:
                datetime.strptime(p.name, RUN_TIMESTAMP_FMT)
                candidates.append(p)
            except ValueError:
                pass
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def html_escape(text: str) -> str:
    """Escape text for safe use in HTML content and attributes."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def html_document(
    title: str,
    body_lines: List[str],
    *,
    style: Optional[str] = None,
    lang: str = "en",
) -> str:
    """Build a full HTML5 document string from title, optional style, and body lines."""
    lines: List[str] = [
        "<!DOCTYPE html>",
        f"<html lang='{html_escape(lang)}'>",
        "<head>",
        "<meta charset='utf-8' />",
        f"<title>{html_escape(title)}</title>",
    ]
    if style:
        lines.append("<style>")
        lines.append(style)
        lines.append("</style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.extend(body_lines)
    lines.append("</body></html>")
    return "\n".join(lines)


class RunLogger:
    """
    Context manager that writes each log message to both stdout and a log file.
    Use for capturing the main terminal output of a script into a file.
    """

    def __init__(self, log_path: Path):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
