"""
PyG Explainer task settings that must match the trained GINE graph classifier.

Use with torch_geometric.explain.Explainer ``model_config`` for graph-level
multiclass logits (``return_type="raw"``).
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_EXPLANATION_TYPE: str = "model"
PHENOMENON_EXPLANATION_TYPE: str = "phenomenon"

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "mode": "multiclass_classification",
    "task_level": "graph",
    "return_type": "raw",
}
