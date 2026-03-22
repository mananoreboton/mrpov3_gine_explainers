"""Validate canonical explainer names against a caller-supplied registry (e.g. v2 explainers)."""

from __future__ import annotations

from typing import Any, Callable, Sequence


def validate_explainer_names(
    names: Sequence[str],
    *,
    is_registered: Callable[[str], bool],
    resolve_explainer_class: Callable[[str], Any],
) -> bool:
    """
    Return True if any name is invalid. Prints errors to stdout.

    ``is_registered`` / ``resolve_explainer_class`` are typically from the v2 ``explainers`` package.
    """
    bad = False
    for raw in names:
        key = raw.upper()
        if not is_registered(key):
            print(f"{raw}: NOT REGISTERED")
            bad = True
            continue
        cls = resolve_explainer_class(key)
        if cls is None or not callable(getattr(cls, "build_explainer", None)):
            print(f"{raw}: FAIL (class or build_explainer)")
            bad = True
    return bad
