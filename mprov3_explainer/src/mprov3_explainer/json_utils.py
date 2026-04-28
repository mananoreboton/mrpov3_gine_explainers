"""Strict JSON serialization helpers for explainer artifacts."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any


def to_strict_jsonable(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Python's standard JSON encoder emits bare ``NaN`` and ``Infinity`` tokens by
    default. Those tokens are not valid JSON, and the encoder's ``default``
    hook is not called for floats. Sanitizing before serialization lets us use
    ``allow_nan=False`` as an enforcement guard.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_strict_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_strict_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_strict_jsonable(v) for v in value]
    return value


def dumps_strict_json(value: Any, **kwargs: Any) -> str:
    """Serialize as standards-compliant JSON, converting NaN/inf to ``null``."""
    return json.dumps(to_strict_jsonable(value), allow_nan=False, **kwargs)
