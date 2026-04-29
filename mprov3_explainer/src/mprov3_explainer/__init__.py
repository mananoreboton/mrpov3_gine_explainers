"""MPro-GINE Explainer - PyTorch package."""

__version__ = "0.1.0"

from mprov3_explainer.device import get_device
from mprov3_explainer.explainers import (
    AVAILABLE_EXPLAINERS,
    ExplainerSpec,
    explainer_report_meta,
    get_builder,
    get_spec,
    validate_explainer,
)
from mprov3_explainer.json_utils import dumps_strict_json, to_strict_jsonable
from mprov3_explainer.paths import (
    explanations_run_dir,
    resolve_checkpoint_path,
    resolve_dataset_dir,
    resolve_training_checkpoint_and_dataset_name,
    visualizations_run_dir,
)
from mprov3_explainer.pipeline import (
    DEFAULT_FIDELITY_CURVE_TOP_K,
    DEFAULT_PAPER_N_THRESHOLDS,
    ExplanationResult,
    PredictionBaselineEntry,
    collect_prediction_baseline,
    diagnose_explanation_run,
    nanmean,
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
from mprov3_explainer.web_report import (
    write_explainer_summary_page,
    write_fold_explanation_web_report,
    write_global_explanation_index,
)

__all__ = [
    "get_device",
    "__version__",
    "AVAILABLE_EXPLAINERS",
    "ExplainerSpec",
    "explainer_report_meta",
    "get_builder",
    "get_spec",
    "validate_explainer",
    "dumps_strict_json",
    "to_strict_jsonable",
    "explanations_run_dir",
    "resolve_checkpoint_path",
    "resolve_dataset_dir",
    "resolve_training_checkpoint_and_dataset_name",
    "visualizations_run_dir",
    "DEFAULT_FIDELITY_CURVE_TOP_K",
    "DEFAULT_PAPER_N_THRESHOLDS",
    "ExplanationResult",
    "PredictionBaselineEntry",
    "collect_prediction_baseline",
    "diagnose_explanation_run",
    "nanmean",
    "run_explanations",
    "train_explainer",
    "PreprocessedExplanation",
    "apply_preprocessing",
    "edge_mask_to_node_mask",
    "normalize_mask",
    "reduce_node_mask",
    "write_explainer_summary_page",
    "write_fold_explanation_web_report",
    "write_global_explanation_index",
]
