"""Build PyG ``Explainer`` instances for canonical explainer names and a given model."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch.nn as nn
from mprov3_gine_explainer_defaults import (
    DEFAULT_GNN_EXPLAINER_EPOCHS,
    DEFAULT_GNN_EXPLAINER_LR,
)

from explainers import DEFAULT_STRAIGHTFORWARD_EXPLAINERS
from explainers.gnnexpl import GnnExplExplainer
from explainers.gradexpinode import GradExpINodeExplainer
from explainers.gradexpledge import GradExpLEdgeExplainer
from explainers.guidedbp import GuidedBpExplainer
from explainers.igedge import IgEdgeExplainer
from explainers.ignode import IgNodeExplainer
from explainers.pgexpl import PgExplExplainer
from explainers.pgmexpl import PgmExplExplainer

_ExplainerFactory = Callable[[nn.Module], Any]


def _build_gnnexpl(model: nn.Module) -> Any:
    return GnnExplExplainer.build_explainer(
        model,
        epochs=DEFAULT_GNN_EXPLAINER_EPOCHS,
        lr=DEFAULT_GNN_EXPLAINER_LR,
    )


_EXPLAINER_FACTORIES: Dict[str, _ExplainerFactory] = {
    "GRADEXPINODE": lambda m: GradExpINodeExplainer.build_explainer(m),
    "GRADEXPLEDGE": lambda m: GradExpLEdgeExplainer.build_explainer(m),
    "GUIDEDBP": lambda m: GuidedBpExplainer.build_explainer(m),
    "IGEDGE": lambda m: IgEdgeExplainer.build_explainer(m),
    "IGNODE": lambda m: IgNodeExplainer.build_explainer(m),
    "GNNEXPL": _build_gnnexpl,
    "PGEXPL": lambda m: PgExplExplainer.build_explainer(m),
    "PGMEXPL": lambda m: PgmExplExplainer.build_explainer(m),
}

if set(_EXPLAINER_FACTORIES) != set(DEFAULT_STRAIGHTFORWARD_EXPLAINERS):
    raise RuntimeError(
        "_EXPLAINER_FACTORIES keys must match DEFAULT_STRAIGHTFORWARD_EXPLAINERS"
    )


def build_explainers_for_model(
    model: nn.Module, names: Sequence[str]
) -> List[Tuple[str, Any]]:
    """
    Return ``(canonical_upper, explainer)`` in the same order as ``names``.

    Each name must appear in ``DEFAULT_STRAIGHTFORWARD_EXPLAINERS`` / the registry.
    """
    out: List[Tuple[str, Any]] = []
    for raw in names:
        key = raw.upper()
        factory = _EXPLAINER_FACTORIES.get(key)
        if factory is None:
            raise KeyError(f"No explainer factory for {key!r}")
        out.append((key, factory(model)))
    return out
