"""Parse command-line arguments for compare_explainers."""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check that registered explainer classes exist and expose build_explainer.",
    )
    p.add_argument(
        "--explainers",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Canonical explainer names (e.g. GNNEXPL). "
            "Multiple values and comma-separated tokens are accepted. "
            "If omitted, uses the default eight straight-forward explainers."
        ),
    )
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def explainer_names_from_args(args: argparse.Namespace) -> Optional[List[str]]:
    """
    Return None to mean 'use defaults'; otherwise a non-empty list of names.
    """
    raw = args.explainers
    if raw is None:
        return None
    names: List[str] = []
    for chunk in raw:
        for part in chunk.split(","):
            s = part.strip()
            if s:
                names.append(s)
    if not names:
        return None
    return names
