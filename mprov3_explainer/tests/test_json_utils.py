"""Tests for strict JSON artifact serialization."""

from __future__ import annotations

import json
import math

from mprov3_explainer.json_utils import dumps_strict_json, to_strict_jsonable


def test_to_strict_jsonable_recursively_replaces_non_finite_floats():
    payload = {
        "top": float("nan"),
        "nested": [1.0, float("inf"), {"x": -float("inf")}],
    }

    out = to_strict_jsonable(payload)

    assert out == {"top": None, "nested": [1.0, None, {"x": None}]}


def test_dumps_strict_json_emits_null_not_nan_tokens():
    payload = {
        "nan": math.nan,
        "pos_inf": math.inf,
        "neg_inf": -math.inf,
        "finite": 0.5,
    }

    text = dumps_strict_json(payload, indent=2)

    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {
        "nan": None,
        "pos_inf": None,
        "neg_inf": None,
        "finite": 0.5,
    }
