"""
Resolve paths under a GNN `results/` tree (trainings, datasets, explanations, visualizations).

Uses segment constants from `data_path_defaults`; callers supply concrete `Path` roots
(e.g. mprov3_gine/results, mprov3_explainer/results).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from mprov3_gine_explainer_defaults.data_path_defaults import (
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


def _parse_timestamp_dir_name(name: str) -> None:
    """Raise ValueError if *name* is not a valid RUN_TIMESTAMP_FMT folder name."""
    if len(name) != 17 or name[4] != "-" or name[7] != "-" or name[10] != "_":
        raise ValueError(f"Invalid trainings timestamp folder name: {name!r}")
    datetime.strptime(name, RUN_TIMESTAMP_FMT)


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
    results_root: Path,
    checkpoint_name: str = DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    *,
    trainings_timestamp: Optional[str] = None,
) -> Path:
    """
    Resolve path to checkpoint under results_root/trainings/.

    If *trainings_timestamp* is set, use ``trainings/<trainings_timestamp>/<checkpoint_name>``.
    Otherwise use the latest timestamp-named subfolder (by mtime).
    """
    trainings_base = results_root / RESULTS_TRAININGS
    if trainings_timestamp is not None:
        _parse_timestamp_dir_name(trainings_timestamp)
        run_dir = trainings_base / trainings_timestamp
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Training run folder not found: {run_dir}")
        path = run_dir / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    latest = get_latest_timestamp_dir(trainings_base)
    if latest is None:
        raise FileNotFoundError(f"No training run found under {trainings_base}. Run train.py first.")
    path = latest / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def read_dataset_name_from_train_log(training_run_dir: Path) -> Optional[str]:
    """
    Parse ``train.log`` written by ``train.py`` (``Dataset: <path> (latest)``) and return
    the dataset folder name (last path segment). Returns None if missing or unparsable.
    """
    log_path = training_run_dir / "train.log"
    if not log_path.is_file():
        return None
    prefix = "Dataset:"
    suffix = " (latest)"
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix) :].strip()
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)].strip()
        name = Path(rest).name
        return name if name else None
    return None


def resolve_training_checkpoint_and_dataset_name(
    results_root: Path,
    *,
    trainings_timestamp: Optional[str] = None,
) -> Tuple[Path, str]:
    """
    ``results/trainings/<ts>/`` checkpoint plus the PyG dataset folder name to load.

    Prefers the dataset recorded in that run's ``train.log`` when the corresponding
    ``results/datasets/<name>/data.pt`` still exists; otherwise falls back to
    ``resolve_dataset_dir`` (latest dataset).
    """
    checkpoint_path = resolve_checkpoint_path(
        results_root, trainings_timestamp=trainings_timestamp
    )
    run_dir = checkpoint_path.parent
    logged = read_dataset_name_from_train_log(run_dir)
    if logged is not None:
        ds_pt = results_root / RESULTS_DATASETS / logged / PYG_DATA_FILENAME
        if ds_pt.is_file():
            return checkpoint_path, logged
    dataset_dir = resolve_dataset_dir(results_root)
    return checkpoint_path, dataset_dir.name


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
