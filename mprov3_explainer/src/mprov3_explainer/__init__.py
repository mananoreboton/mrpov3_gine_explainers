"""MPro-GINE Explainer - PyTorch package."""

__version__ = "0.1.0"

from mprov3_explainer.comparison_report_html import write_comparison_report_html
from mprov3_explainer.device import get_device
from mprov3_explainer.explainers import (
    AVAILABLE_EXPLAINERS,
    ExplainerSpec,
    explainer_report_meta,
    get_builder,
    get_spec,
    validate_explainer,
)
from mprov3_explainer.paths import (
    explanations_run_dir,
    resolve_checkpoint_path,
    resolve_dataset_dir,
    resolve_training_checkpoint_and_dataset_name,
    visualizations_run_dir,
)
from mprov3_explainer.pipeline import (
    ExplanationResult,
    aggregate_fidelity,
    run_explanations,
    train_explainer,
)
from mprov3_explainer.preprocessing import (
    PreprocessedExplanation,
    apply_preprocessing,
    edge_mask_to_node_mask,
    normalize_mask,
    reduce_node_mask,
)

__all__ = [
    "write_comparison_report_html",
    "get_device",
    "__version__",
    "AVAILABLE_EXPLAINERS",
    "ExplainerSpec",
    "explainer_report_meta",
    "get_builder",
    "get_spec",
    "validate_explainer",
    "explanations_run_dir",
    "resolve_checkpoint_path",
    "resolve_dataset_dir",
    "resolve_training_checkpoint_and_dataset_name",
    "visualizations_run_dir",
    "ExplanationResult",
    "aggregate_fidelity",
    "run_explanations",
    "train_explainer",
    "PreprocessedExplanation",
    "apply_preprocessing",
    "edge_mask_to_node_mask",
    "normalize_mask",
    "reduce_node_mask",
]
