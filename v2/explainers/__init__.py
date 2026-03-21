"""
Registry of straight-forward PyG explainers (see doc/table_of_explainer_implementations.md).
"""

from __future__ import annotations

from typing import Any, Dict, Final, List, Optional, Tuple, Type

DEFAULT_STRAIGHTFORWARD_EXPLAINERS: Final[Tuple[str, ...]] = (
    "GRADEXPINODE",
    "GRADEXPLEDGE",
    "GUIDEDBP",
    "IGEDGE",
    "IGNODE",
    "GNNEXPL",
    "PGEXPL",
    "PGMEXPL",
)

# canonical_name -> (importlib module path, class name)
EXPLAINER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "GRADEXPINODE": ("explainers.gradexpinode", "GradExpINodeExplainer"),
    "GRADEXPLEDGE": ("explainers.gradexpledge", "GradExpLEdgeExplainer"),
    "GUIDEDBP": ("explainers.guidedbp", "GuidedBpExplainer"),
    "IGEDGE": ("explainers.igedge", "IgEdgeExplainer"),
    "IGNODE": ("explainers.ignode", "IgNodeExplainer"),
    "GNNEXPL": ("explainers.gnnexpl", "GnnExplExplainer"),
    "PGEXPL": ("explainers.pgexpl", "PgExplExplainer"),
    "PGMEXPL": ("explainers.pgmexpl", "PgmExplExplainer"),
}


def resolve_explainer_class(name: str) -> Optional[Type[Any]]:
    """Import and return the explainer class, or None if unknown / import fails."""
    import importlib

    entry = EXPLAINER_REGISTRY.get(name.upper())
    if entry is None:
        return None
    mod_path, cls_name = entry
    try:
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name, None)
    except Exception:
        return None


def is_registered(name: str) -> bool:
    return name.upper() in EXPLAINER_REGISTRY


__all__ = [
    "DEFAULT_STRAIGHTFORWARD_EXPLAINERS",
    "EXPLAINER_REGISTRY",
    "resolve_explainer_class",
    "is_registered",
]
