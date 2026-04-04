"""
Shared utilities: logging, overwrite notices, and HTML helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional


def log_overwrite_if_exists(path: Path, log: Callable[[str], None]) -> None:
    """If *path* exists, log a single info line before replacing it."""
    if path.exists():
        log(f"[INFO] Output exists; overwriting: {path}")


def log_overwrite_dir_if_nonempty(path: Path, log: Callable[[str], None]) -> None:
    """If *path* is a non-empty directory, log once before writing into it."""
    if path.is_dir() and any(path.iterdir()):
        log(f"[INFO] Output directory exists with prior files; overwriting under: {path}")


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
