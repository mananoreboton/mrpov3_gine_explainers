"""Resolve which CV fold indices to run from CLI-style arguments."""

from __future__ import annotations


def resolve_fold_indices(
    num_folds: int,
    *,
    fold_index: int | None = None,
    fold_indices: list[int] | None = None,
) -> list[int]:
    """
    fold_index: single fold (backward compatible).
    fold_indices: explicit subset; order preserved, duplicates dropped.
    If neither is set, returns ``list(range(num_folds))``.
    """
    if fold_index is not None and fold_indices is not None:
        raise ValueError("Pass at most one of fold_index and fold_indices")
    if num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}")

    if fold_index is not None:
        raw = [fold_index]
    elif fold_indices is not None:
        raw = fold_indices
    else:
        raw = list(range(num_folds))

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
