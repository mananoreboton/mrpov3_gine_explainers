"""
Monorepo layout: workspace root is the parent of the ``mprov3_gine_explainer_defaults`` project folder.

Expected siblings under that root include ``mprov3_gine``, ``mprov3_data`` (name from
``DEFAULT_MPRO_SNAPSHOT_DIR_NAME``), ``mprov3_explainer``, etc.
"""

from __future__ import annotations

from pathlib import Path

from mprov3_gine_explainer_defaults.data_path_defaults import (
    DEFAULT_MPRO_SNAPSHOT_DIR_NAME,
    RESULTS_DIR_NAME,
)

# .../mprov3_gine_explainer_defaults/mprov3_gine_explainer_defaults/gine_project_paths.py -> workspace
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

WORKSPACE_ROOT: Path = _WORKSPACE_ROOT
"""Repository root containing ``mprov3_gine_explainer_defaults``, ``mprov3_gine``, …"""

GINE_PROJECT_DIR: Path = _WORKSPACE_ROOT / "mprov3_gine"
MPRO_EXPLAINER_PROJECT_DIR: Path = _WORKSPACE_ROOT / "mprov3_explainer"

DEFAULT_DATA_ROOT: str = str(_WORKSPACE_ROOT / DEFAULT_MPRO_SNAPSHOT_DIR_NAME)
"""Default raw MPro snapshot path (sibling of ``mprov3_gine``)."""

DEFAULT_RESULTS_ROOT: str = str(GINE_PROJECT_DIR / RESULTS_DIR_NAME)
"""Default ``mprov3_gine/results/`` path."""
