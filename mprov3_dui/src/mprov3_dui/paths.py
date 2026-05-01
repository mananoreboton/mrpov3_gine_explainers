from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Workspace root: parent of the ``mprov3_dui`` project directory."""
    return Path(__file__).resolve().parents[3]


def default_labeled_sample_csv() -> Path:
    return repo_root() / "mprov3_explainer" / "results" / "labeled_explanation_sample.csv"
