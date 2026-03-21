"""MPro-GINE Explainer - PyTorch package."""

__version__ = "0.1.0"

from mprov3_explainer.device import get_device
from mprov3_explainer.explainers import (
    AVAILABLE_EXPLAINERS,
    get_builder,
    validate_explainer,
)
from mprov3_explainer.paths import (
    explanations_run_dir,
    get_latest_timestamp_dir,
    resolve_checkpoint_path,
    resolve_dataset_dir,
    run_timestamp,
    visualizations_run_dir,
)
from mprov3_explainer.pipeline import (
    ExplanationResult,
    aggregate_fidelity,
    run_explanations,
)
from mprov3_explainer.preprocessing import (
    PreprocessedExplanation,
    apply_preprocessing,
    edge_mask_to_node_mask,
    normalize_mask,
)

__all__ = [
    "get_device",
    "__version__",
    "AVAILABLE_EXPLAINERS",
    "get_builder",
    "validate_explainer",
    "explanations_run_dir",
    "get_latest_timestamp_dir",
    "resolve_checkpoint_path",
    "resolve_dataset_dir",
    "run_timestamp",
    "visualizations_run_dir",
    "ExplanationResult",
    "aggregate_fidelity",
    "run_explanations",
    "PreprocessedExplanation",
    "apply_preprocessing",
    "edge_mask_to_node_mask",
    "normalize_mask",
]
