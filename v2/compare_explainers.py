#!/usr/bin/env python3
"""
App entrypoint: verify explainer modules/classes exist and expose build_explainer.

Run from this directory (v2/):  uv run python compare_explainers.py
"""

from __future__ import annotations

from typing import List, Optional

from cli.compare_explainers_cli import explainer_names_from_args, parse_args
from explainers import (
    DEFAULT_STRAIGHTFORWARD_EXPLAINERS,
    is_registered,
    resolve_explainer_class,
)


def run_compare_explainers(explainer_names: Optional[List[str]] = None) -> int:
    if explainer_names is None:
        names = list(DEFAULT_STRAIGHTFORWARD_EXPLAINERS)
    else:
        names = explainer_names

    exit_code = 0
    for name in names:
        key = name.upper()
        if not is_registered(key):
            print(f"{name}: NOT REGISTERED (unknown canonical name)")
            exit_code = 1
            continue
        cls = resolve_explainer_class(key)
        if cls is None:
            print(f"{name}: FAIL (import error or missing class)")
            exit_code = 1
            continue
        if not callable(getattr(cls, "build_explainer", None)):
            print(f"{name}: FAIL (no build_explainer)")
            exit_code = 1
            continue
        mod = cls.__module__
        print(f"{name}: OK (class {cls.__name__} in {mod})")
    return exit_code


def main() -> None:
    args = parse_args()
    names = explainer_names_from_args(args)
    code = run_compare_explainers(names)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
