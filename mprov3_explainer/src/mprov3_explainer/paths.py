"""Path resolution for explainer inputs under mprov3_gine/results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mprov3_gine_explainer_defaults import (
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    PYG_DATA_FILENAME,
    RESULTS_DATASETS,
    RESULTS_EXPLANATIONS,
    RESULTS_TRAININGS,
    RESULTS_VISUALIZATIONS,
)

RUN_TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"


def run_timestamp() -> str:
    """Return current UTC timestamp string for use in output paths."""
    return datetime.now(timezone.utc).strftime(RUN_TIMESTAMP_FMT)


def get_latest_timestamp_dir(base_path: Path) -> Optional[Path]:
    """
    Return the most recent timestamp-named subfolder under base_path.
    Expects subfolder names like 2025-03-14_120000. Returns None if none exist.
    """
    if not base_path.exists() or not base_path.is_dir():
        return None
    candidates: list[Path] = []
    for p in base_path.iterdir():
        if p.is_dir() and len(p.name) == 17 and p.name[4] == "-" and p.name[7] == "-" and p.name[10] == "_":
            try:
                datetime.strptime(p.name, RUN_TIMESTAMP_FMT)
                candidates.append(p)
            except ValueError:
                pass
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def resolve_checkpoint_path(
    results_root: Path, checkpoint_name: str = DEFAULT_TRAINING_CHECKPOINT_FILENAME
) -> Path:
    """Resolve path to checkpoint from results_root/trainings/<latest>/<checkpoint_name>."""
    trainings_base = results_root / RESULTS_TRAININGS
    latest = get_latest_timestamp_dir(trainings_base)
    if latest is None:
        raise FileNotFoundError(f"No training run found under {trainings_base}. Run train.py first.")
    path = latest / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def resolve_dataset_dir(results_root: Path) -> Path:
    """Resolve path to latest dataset dir: results_root/datasets/<latest>/ (contains data.pt)."""
    datasets_base = results_root / RESULTS_DATASETS
    latest = get_latest_timestamp_dir(datasets_base)
    if latest is None:
        raise FileNotFoundError(f"No dataset found under {datasets_base}. Run build_dataset.py first.")
    if not (latest / PYG_DATA_FILENAME).exists():
        raise FileNotFoundError(f"{PYG_DATA_FILENAME} not found in {latest}")
    return latest


def explanations_run_dir(results_root: Path, timestamp: str, explainer_name: str) -> Path:
    """Path for writing/reading explanation outputs: results_root/explanations/<timestamp>/<explainer_name>/."""
    return results_root / RESULTS_EXPLANATIONS / timestamp / explainer_name


def visualizations_run_dir(results_root: Path, timestamp: str, explainer_name: str) -> Path:
    """Path for writing/reading visualization outputs: results_root/visualizations/<timestamp>/<explainer_name>/."""
    return results_root / RESULTS_VISUALIZATIONS / timestamp / explainer_name
