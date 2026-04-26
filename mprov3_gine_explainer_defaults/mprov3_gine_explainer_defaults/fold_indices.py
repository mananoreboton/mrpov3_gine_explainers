"""Resolve which CV fold indices to run from CLI-style arguments."""

from __future__ import annotations


def resolve_fold_indices(
    num_folds: int,
    *,
    fold_indices: list[int] | None = None,
) -> list[int]:
    """
    fold_indices: explicit subset; order preserved, duplicates dropped.
    If None, returns ``list(range(num_folds))``.
    """
    if num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}")

    raw = list(range(num_folds)) if fold_indices is None else fold_indices

    seen: set[int] = set()
    out: list[int] = []
    for i in raw:
        if i < 0 or i >= num_folds:
            raise ValueError(
                f"fold index {i} out of range for num_folds={num_folds} (valid: 0..{num_folds - 1})"
            )
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
