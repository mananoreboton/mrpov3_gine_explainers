"""
Validate that a pre-built PyG dataset (data.pt) is compatible with this project.

Expects ``results/datasets/data.pt`` (flat layout). Legacy timestamped subfolders under
``results/datasets/<timestamp>/`` are not used; migrate or rebuild with build_dataset.py.

Checks performed:
- dataset_root / dataset_name / data.pt exists and can be loaded via MProV3Dataset
- By default, every graph is checked for expected attributes and shapes (optional
  ``--max_samples`` to limit). Expected layout:
  - x: float32, shape (N, 4)  (x, y, z, atomic_number)
  - edge_index: long, shape (2, E)
  - edge_attr: float32, shape (E, 1)
  - pIC50: float32 tensor with a single value
  - category: long tensor with a single class index in [0, num_classes-1]
  - pdb_id attribute is present (each graph is checked; OK lines default to the log file only)
- pdb_order.txt exists and matches dataset length; ``pdb_id`` on each checked graph
  matches the corresponding line (case-insensitive)
- Train/val/test indices derived from the split files are within dataset bounds; the
  log groups output **by fold**, and under each fold **by split** (train / val / test:
  member list, then per-split range check)

Use ``--verbose`` to print every per-graph OK line on stdout. Use ``--quiet`` to
keep fold sections (split membership and per-split checks) in the log file only.

Exit code:
- 0 if all checks pass
- 1 if any check fails
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from mprov3_gine_explainer_defaults import (
    CHECK_FORMAT_DATASETS_SUBDIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_NUM_FOLDS,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    PYG_DATA_FILENAME,
    PYG_PDB_ORDER_FILENAME,
    RESULTS_CHECK_FORMAT,
    RESULTS_DATASETS,
    resolve_fold_indices,
)
from dataset import (
    MProV3Dataset,
    get_train_val_test_indices,
    load_dataset_pdb_order,
)
from utils import RunLogger

# Max PDB IDs (or indices) listed per split line under each fold section; remainder summarized.
_FOLD_SPLIT_MEMBER_PREVIEW_MAX = 80

EXPECTED_PYG_GRAPH_SCHEMA = """Expected PyG Data attributes (per graph):
  x          torch.float32   shape (N, 4)     columns: x, y, z, atomic_number
  edge_index torch.long      shape (2, E)     directed edges
  edge_attr  torch.float32   shape (E, 1)     bond-type scalar per directed edge
  pIC50      torch.float32   shape () or (1,) single regression label
  category   torch.long      shape () or (1,) single class index in [0, num_classes - 1]
  pdb_id     str or scalar   identifies the structure (must match pdb_order.txt order)
Consistency: edge_index.size(1) must equal edge_attr.size(0)."""


@dataclass
class CheckResult:
    ok: bool
    message: str


def _check_dataset_file_exists(
    data_root: Path,
    dataset_name: str,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    dataset_path = data_root / dataset_name / PYG_DATA_FILENAME

    if not data_root.exists():
        results.append(CheckResult(False, f"Data root does not exist: {data_root}"))
        return results

    if dataset_path.exists():
        results.append(CheckResult(True, f"Found PyG dataset file at {dataset_path}"))
    else:
        results.append(
            CheckResult(
                False,
                "Missing PyG dataset file. Expected to find "
                f"{dataset_path}. Run build_dataset.py first.",
            )
        )
    return results


def _load_dataset(
    data_root: Path,
    dataset_name: str,
) -> Tuple[List[CheckResult], MProV3Dataset | None]:
    results: List[CheckResult] = []
    try:
        ds = MProV3Dataset(
            root=str(data_root),
            dataset_name=dataset_name,
        )
    except Exception as exc:
        results.append(
            CheckResult(
                False,
                "Failed to load dataset via MProV3Dataset. "
                f"This dataset is not compatible with the project as-is. Error: {exc}",
            )
        )
        return results, None

    if len(ds) == 0:
        results.append(
            CheckResult(
                False,
                "Loaded MProV3Dataset successfully, but it is empty (len(dataset) == 0).",
            )
        )
    else:
        results.append(
            CheckResult(
                True,
                f"Successfully loaded MProV3Dataset with {len(ds)} graphs.",
            )
        )
    return results, ds


def _check_sample_graphs(
    dataset: MProV3Dataset,
    num_classes: int,
    max_samples: int | None = None,
) -> Tuple[List[CheckResult], List[str], List[CheckResult]]:
    """
    Validate graphs. If max_samples is None, checks every graph; otherwise at most that many.

    Returns (detail_results, per_graph_status_lines, summary_results).
    ``detail_results`` are per-attribute failures (log to file only in main to avoid
    duplicating the one-line status). ``summary_results`` is one pass/fail rollup.
    """
    detail_results: List[CheckResult] = []
    status_lines: List[str] = []
    if len(dataset) == 0:
        return [], [], [
            CheckResult(
                False,
                "Graph validation skipped: dataset has no graphs (length 0).",
            )
        ]

    n_total = len(dataset)
    n_samples = n_total if max_samples is None else min(n_total, max_samples)
    for idx in range(n_samples):
        try:
            g = dataset[idx]
        except Exception as exc:
            detail_results.append(
                CheckResult(
                    False,
                    f"Graph {idx}: failed to load from dataset: {exc}",
                )
            )
            status_lines.append(f"Graph {idx} (pdb=?): FAIL — could not index dataset ({exc})")
            continue

        pdb_display = getattr(g, "pdb_id", None)
        pdb_str = str(pdb_display) if pdb_display is not None else "<missing pdb_id>"
        _detail_offset = len(detail_results)

        # x: (N, 4) float32
        if not hasattr(g, "x"):
            detail_results.append(
                CheckResult(
                    False,
                    f"Graph {idx} (pdb={pdb_str}) is missing attribute 'x'.",
                )
            )
        else:
            if g.x.dtype != torch.float32:
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'x' has dtype {g.x.dtype}, "
                        "expected torch.float32.",
                    )
                )
            if g.x.dim() != 2 or g.x.size(1) != 4:
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'x' has shape {tuple(g.x.shape)}, "
                        "expected (N, 4) with features [x, y, z, atomic_number].",
                    )
                )

        # edge_index: (2, E) long
        if not hasattr(g, "edge_index"):
            detail_results.append(
                CheckResult(
                    False,
                    f"Graph {idx} (pdb={pdb_str}) is missing attribute 'edge_index'.",
                )
            )
        else:
            if g.edge_index.dtype != torch.long:
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'edge_index' has dtype "
                        f"{g.edge_index.dtype}, expected torch.long.",
                    )
                )
            if g.edge_index.dim() != 2 or g.edge_index.size(0) != 2:
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'edge_index' has shape "
                        f"{tuple(g.edge_index.shape)}, expected (2, E).",
                    )
                )

        # edge_attr: (E, 1) float32
        if not hasattr(g, "edge_attr"):
            detail_results.append(
                CheckResult(
                    False,
                    f"Graph {idx} (pdb={pdb_str}) is missing attribute 'edge_attr'.",
                )
            )
        else:
            if g.edge_attr.dtype != torch.float32:
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'edge_attr' has dtype "
                        f"{g.edge_attr.dtype}, expected torch.float32.",
                    )
                )
            if g.edge_attr.dim() != 2 or g.edge_attr.size(1) != 1:
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'edge_attr' has shape "
                        f"{tuple(g.edge_attr.shape)}, expected (E, 1).",
                    )
                )

            # Consistency between edge_index and edge_attr
            if hasattr(g, "edge_index"):
                e_index_e = int(g.edge_index.size(1))
                e_attr_e = int(g.edge_attr.size(0))
                if e_index_e != e_attr_e:
                    detail_results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} (pdb={pdb_str}) has {e_index_e} edges in "
                            f"edge_index but {e_attr_e} rows in edge_attr.",
                        )
                    )

        # pIC50: regression label (1,)
        if not hasattr(g, "pIC50"):
            detail_results.append(
                CheckResult(
                    False,
                    f"Graph {idx} (pdb={pdb_str}) is missing 'pIC50' label.",
                )
            )
        else:
            if not isinstance(g.pIC50, torch.Tensor):
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'pIC50' is not a tensor "
                        f"(type={type(g.pIC50)}).",
                    )
                )
            else:
                if g.pIC50.dtype != torch.float32:
                    detail_results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} (pdb={pdb_str}) 'pIC50' has dtype "
                            f"{g.pIC50.dtype}, expected torch.float32.",
                        )
                    )
                if g.pIC50.numel() != 1:
                    detail_results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} (pdb={pdb_str}) 'pIC50' has shape "
                            f"{tuple(g.pIC50.shape)}, expected a single scalar value.",
                        )
                    )

        # category: classification label (1,) in [0, num_classes-1]
        if not hasattr(g, "category"):
            detail_results.append(
                CheckResult(
                    False,
                    f"Graph {idx} (pdb={pdb_str}) is missing 'category' label.",
                )
            )
        else:
            if not isinstance(g.category, torch.Tensor):
                detail_results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} (pdb={pdb_str}) 'category' is not a tensor "
                        f"(type={type(g.category)}).",
                    )
                )
            else:
                if g.category.dtype != torch.long:
                    detail_results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} (pdb={pdb_str}) 'category' has dtype "
                            f"{g.category.dtype}, expected torch.long.",
                        )
                    )
                if g.category.numel() != 1:
                    detail_results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} (pdb={pdb_str}) 'category' has shape "
                            f"{tuple(g.category.shape)}, expected a single class index.",
                        )
                    )
                else:
                    val = int(g.category.item())
                    if not (0 <= val < num_classes):
                        detail_results.append(
                            CheckResult(
                                False,
                                f"Graph {idx} (pdb={pdb_str}) 'category' value {val} "
                                f"is out of expected range [0, {num_classes - 1}].",
                            )
                        )

        # pdb_id: for split mapping and traceability
        if not hasattr(g, "pdb_id"):
            detail_results.append(
                CheckResult(
                    False,
                    f"Graph {idx} (pdb={pdb_str}) is missing attribute 'pdb_id' "
                    "(required for mapping splits to dataset indices).",
                )
            )

        chunk = detail_results[_detail_offset:]
        graph_failed = any(not r.ok for r in chunk)
        if graph_failed:
            brief = "; ".join(r.message for r in chunk if not r.ok)
            status_lines.append(f"Graph {idx} (pdb={pdb_str}): FAIL — {brief}")
        else:
            status_lines.append(f"Graph {idx} (pdb={pdb_str}): OK")

    summary_results: List[CheckResult] = []
    if not any(not r.ok for r in detail_results):
        summary_results.append(
            CheckResult(
                True,
                f"All checked graphs (n={n_samples}) have the expected attributes "
                "and basic shapes.",
            )
        )
    else:
        n_failed_graphs = sum(1 for line in status_lines if ": FAIL" in line)
        summary_results.append(
            CheckResult(
                False,
                f"{n_failed_graphs} of {n_samples} checked graphs failed attribute/shape validation "
                "(see per-graph lines and log file for details).",
            )
        )

    return detail_results, status_lines, summary_results


def _normalize_pdb_token(value: object) -> str:
    return str(value).strip().upper()


def _split_member_body(
    idx: torch.Tensor,
    pdb_order: Optional[List[str]],
    dataset_len: int,
    max_show: int = _FOLD_SPLIT_MEMBER_PREVIEW_MAX,
) -> Tuple[int, str]:
    """
    Member list string for one split. Returns (count, body) where body is PDB IDs
    (or indices) comma-separated, possibly truncated.
    """
    if idx.numel() == 0:
        return 0, "(no dataset indices in this split)"

    raw = sorted({int(x.item()) for x in idx.view(-1)})
    labels: List[str] = []
    for i in raw:
        if pdb_order is not None and 0 <= i < len(pdb_order) and i < dataset_len:
            labels.append(str(pdb_order[i]))
        else:
            labels.append(str(i))

    n = len(labels)
    if n <= max_show:
        body = ", ".join(labels)
    else:
        body = ", ".join(labels[:max_show]) + f" … (+{n - max_show} more)"
    return n, body


def _check_pdb_id_alignment(
    dataset: MProV3Dataset,
    pdb_order: List[str],
    n_samples: int,
) -> List[CheckResult]:
    """Ensure ``data[i].pdb_id`` matches ``pdb_order[i]`` for checked indices (case-insensitive)."""
    results: List[CheckResult] = []
    for i in range(n_samples):
        g = dataset[i]
        expected = _normalize_pdb_token(pdb_order[i])
        got = _normalize_pdb_token(getattr(g, "pdb_id", ""))
        if got != expected:
            results.append(
                CheckResult(
                    False,
                    f"Graph {i}: pdb_id mismatch — data has {got!r}, "
                    f"{PYG_PDB_ORDER_FILENAME} line {i + 1} has {expected!r}.",
                )
            )
    if results:
        return results
    return [
        CheckResult(
            True,
            f"pdb_id matches {PYG_PDB_ORDER_FILENAME} for all {n_samples} checked graphs.",
        )
    ]


def _check_pdb_order_file(
    data_root: Path,
    dataset_name: str,
    dataset_len: int,
    pdb_order_cached: Optional[List[str]] = None,
) -> Tuple[List[CheckResult], Optional[List[str]]]:
    """
    Validate ``pdb_order.txt``. If ``pdb_order_cached`` is set, use it instead of reading disk again.
    Returns ``(check_results, pdb_order_or_none)``; the list is only returned when present and
    its length matches ``dataset_len``.
    """
    results: List[CheckResult] = []
    pdb_order = (
        pdb_order_cached
        if pdb_order_cached is not None
        else load_dataset_pdb_order(data_root, dataset_name)
    )
    if pdb_order is None:
        results.append(
            CheckResult(
                False,
                f"Missing {PYG_PDB_ORDER_FILENAME} in the dataset folder. This file is required to "
                "map split PDB IDs to dataset indices (it is written by build_dataset.py).",
            )
        )
        return results, None

    if len(pdb_order) != dataset_len:
        results.append(
            CheckResult(
                False,
                f"{PYG_PDB_ORDER_FILENAME} contains {len(pdb_order)} entries, but dataset has "
                f"{dataset_len} graphs; they must match.",
            )
        )
        return results, None

    results.append(
        CheckResult(
            True,
            f"{PYG_PDB_ORDER_FILENAME} exists and its length matches the dataset size.",
        )
    )
    return results, pdb_order


def _check_split_indices_in_range(
    dataset_root: Path,
    splits_root: Path,
    dataset_name: str,
    dataset_len: int,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    num_folds: int,
    fold_index: int,
    dataset_pdb_order: Optional[List[str]] = None,
) -> Tuple[List[CheckResult], List[str]]:
    results: List[CheckResult] = []
    lines: List[str] = []
    pdb_order = (
        dataset_pdb_order
        if dataset_pdb_order is not None
        else load_dataset_pdb_order(dataset_root, dataset_name)
    )

    try:
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            data_root=splits_root,
            train_file=train_split_file,
            val_file=val_split_file,
            test_file=test_split_file,
            num_folds=num_folds,
            fold_index=fold_index,
            dataset_pdb_order=pdb_order,
        )
    except Exception as exc:
        msg = (
            "Failed to compute train/val/test indices via get_train_val_test_indices; "
            f"dataset or splits may be incompatible. Error: {exc}"
        )
        results.append(CheckResult(False, msg))
        lines.append(f"  [ERROR] {msg}")
        return results, lines

    splits_spec: List[Tuple[str, torch.Tensor]] = [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]

    for split_name, idx in splits_spec:
        lines.append(f"  --- {split_name.upper()} ---")
        n_mem, body = _split_member_body(idx, pdb_order, dataset_len)
        lines.append(f"    Members (n={n_mem}, dataset index order; PDB when known): {body}")

        if idx.numel() == 0:
            err = (
                f"{split_name} indices tensor is empty; check that splits reference "
                "valid PDB IDs and that corresponding graphs exist in the dataset."
            )
            lines.append(f"    [ERROR] {err}")
            results.append(CheckResult(False, err))
            continue

        min_idx = int(idx.min().item())
        max_idx = int(idx.max().item())
        if min_idx < 0 or max_idx >= dataset_len:
            err = (
                f"{split_name} indices out of range [0, {dataset_len - 1}]: "
                f"min={min_idx}, max={max_idx}."
            )
            lines.append(f"    [ERROR] {err}")
            results.append(CheckResult(False, err))
            continue

        lines.append(
            f"    [OK] {split_name}: {int(idx.numel())} indices, all in "
            f"[0, {dataset_len - 1}] (min={min_idx}, max={max_idx})."
        )
        results.append(
            CheckResult(
                True,
                f"{split_name}: {int(idx.numel())} indices in range.",
            )
        )

    return results, lines


def run_graph_level_checks(
    dataset_root: Path,
    dataset_name: str,
    num_classes: int,
    max_samples: int | None,
) -> Tuple[
    bool,
    List[CheckResult],
    List[CheckResult],
    List[str],
    List[CheckResult],
    List[CheckResult],
    List[CheckResult],
    int | None,
    Optional[List[str]],
]:
    """
    File presence, load dataset, validate graphs, ``pdb_order.txt``, pdb_id alignment.

    Returns:
        all_ok,
        prefix_results (file exists + dataset load messages),
        graph_detail (per-attribute; log to file only in main),
        graph_status_lines,
        graph_summary (single rollup),
        pdb_order_results,
        pdb_alignment_results,
        dataset_len or None,
        pdb_order list or None (for fold checks; only when length matches dataset).
    """
    prefix_results: List[CheckResult] = []
    prefix_results.extend(_check_dataset_file_exists(dataset_root, dataset_name))

    if any(not r.ok for r in prefix_results):
        return False, prefix_results, [], [], [], [], [], None, None

    load_results, dataset = _load_dataset(dataset_root, dataset_name)
    prefix_results.extend(load_results)

    if dataset is None:
        return False, prefix_results, [], [], [], [], [], None, None

    dataset_len = len(dataset)
    graph_detail, graph_status_lines, graph_summary = _check_sample_graphs(
        dataset=dataset,
        num_classes=num_classes,
        max_samples=max_samples,
    )
    n_samples = len(graph_status_lines)

    pdb_order_results, pdb_order = _check_pdb_order_file(
        data_root=dataset_root,
        dataset_name=dataset_name,
        dataset_len=dataset_len,
    )

    pdb_alignment_results: List[CheckResult] = []
    if pdb_order is not None and n_samples > 0:
        pdb_alignment_results = _check_pdb_id_alignment(dataset, pdb_order, n_samples)

    all_ok = (
        all(r.ok for r in prefix_results)
        and all(r.ok for r in graph_detail)
        and all(r.ok for r in graph_summary)
        and all(r.ok for r in pdb_order_results)
        and all(r.ok for r in pdb_alignment_results)
    )
    return (
        all_ok,
        prefix_results,
        graph_detail,
        graph_status_lines,
        graph_summary,
        pdb_order_results,
        pdb_alignment_results,
        dataset_len,
        pdb_order,
    )


def run_fold_split_checks(
    dataset_root: Path,
    splits_root: Path,
    dataset_name: str,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    num_folds: int,
    fold_index: int,
    dataset_len: int,
    dataset_pdb_order: Optional[List[str]],
) -> Tuple[bool, List[CheckResult], List[str]]:
    """Train/val/test index range check for one fold; third value is structured log lines (by split)."""
    results, preview_lines = _check_split_indices_in_range(
        dataset_root=dataset_root,
        splits_root=splits_root,
        dataset_name=dataset_name,
        dataset_len=dataset_len,
        train_split_file=train_split_file,
        val_split_file=val_split_file,
        test_split_file=test_split_file,
        num_folds=num_folds,
        fold_index=fold_index,
        dataset_pdb_order=dataset_pdb_order,
    )
    return all(r.ok for r in results), results, preview_lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that results/datasets/data.pt (flat layout) is compatible with "
            "training and evaluation. Optional --data_root points at the folder that "
            "directly contains data.pt, or at a legacy per-run subfolder under datasets/."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            f"Folder that directly contains data.pt (e.g. {DEFAULT_RESULTS_ROOT}/datasets), "
            "or a legacy folder like results/datasets/<timestamp>/. "
            f"Default: {DEFAULT_RESULTS_ROOT}/datasets"
        ),
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default=None,
        help=f"Path to raw MPro snapshot (Splits/). Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--train_split_file",
        type=str,
        default=DEFAULT_TRAIN_SPLIT_FILE,
        help=(
            f"Train split file name under splits_root/Splits/ "
            f"(default: {DEFAULT_TRAIN_SPLIT_FILE})"
        ),
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=(
            f"Validation split file name under splits_root/Splits "
            f"(default: {DEFAULT_VAL_SPLIT_FILE})"
        ),
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=(
            f"Test split file name under splits_root/Splits "
            f"(default: {DEFAULT_TEST_SPLIT_FILE})"
        ),
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help=f"Number of folds expected in the split files (default: {DEFAULT_NUM_FOLDS}).",
    )
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument(
        "--fold_index",
        type=int,
        default=None,
        help="Check split indices for a single fold only. Default: all folds.",
    )
    fold_group.add_argument(
        "--fold_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Check these fold indices only. Default: all folds.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of classification classes (Category: default 3).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Check at most N graphs (in dataset index order). "
            "Default: check every graph."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print each per-graph result line on stdout (default: OK lines are only "
            "in the log file; failures always print here)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Write per-fold / per-split sections (members + index checks) to the log "
            "file only (stdout still gets the one-line per-graph summary and any failures)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_root = Path(DEFAULT_RESULTS_ROOT)
    dataset_root: Path | None = None
    dataset_name: str | None = None

    if args.data_root:
        p = Path(args.data_root)
        if (p / PYG_DATA_FILENAME).exists():
            dataset_root = p.parent
            dataset_name = p.name
    else:
        dataset_base = results_root / RESULTS_DATASETS
        if (dataset_base / PYG_DATA_FILENAME).is_file():
            dataset_root = dataset_base.parent
            dataset_name = dataset_base.name

    if dataset_root is None or dataset_name is None:
        missing_base = Path(args.data_root) if args.data_root else results_root / RESULTS_DATASETS
        msg = (
            f"No PyG dataset at {missing_base / PYG_DATA_FILENAME}. "
            "Run build_dataset.py (writes results/datasets/data.pt). "
            "Legacy results/datasets/<timestamp>/ layouts are no longer auto-detected."
        )
        log_dir = results_root / RESULTS_CHECK_FORMAT / CHECK_FORMAT_DATASETS_SUBDIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "check_output.log"
        with RunLogger(log_path) as log:
            log.log(f"[ERROR] {msg}")
        print(msg, file=sys.stderr)
        sys.exit(1)

    splits_root = Path(args.splits_root or DEFAULT_DATA_ROOT)

    fold_list = resolve_fold_indices(
        args.num_folds,
        fold_index=args.fold_index,
        fold_indices=args.fold_indices,
    )

    log_dir = results_root / RESULTS_CHECK_FORMAT / CHECK_FORMAT_DATASETS_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "check_output.log"

    all_ok = True
    with RunLogger(log_path) as log:
        log.log(EXPECTED_PYG_GRAPH_SCHEMA)
        log.log("")
        log.log(
            f"Checking PyG output dataset: {dataset_root} / {dataset_name} "
            f"(splits from {splits_root}, folds={fold_list}, num_folds={args.num_folds}, "
            f"max_samples={args.max_samples}, verbose={args.verbose}, quiet={args.quiet})"
        )

        (
            ok_g,
            prefix_results,
            graph_detail,
            per_graph_lines,
            graph_summary,
            pdb_order_results,
            pdb_alignment_results,
            dataset_len,
            pdb_order,
        ) = run_graph_level_checks(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            num_classes=args.num_classes,
            max_samples=args.max_samples,
        )
        all_ok = all_ok and ok_g

        for r in prefix_results:
            status = "OK" if r.ok else "ERROR"
            log.log(f"[{status}] {r.message}")

        for r in graph_detail:
            status = "OK" if r.ok else "ERROR"
            log.log_file_only(f"[{status}] [GRAPH_DETAIL] {r.message}")

        n_ok = sum(1 for line in per_graph_lines if line.endswith(": OK"))
        n_fail = sum(1 for line in per_graph_lines if ": FAIL" in line)
        for line in per_graph_lines:
            is_fail = ": FAIL" in line
            if is_fail or args.verbose:
                log.log(line)
            else:
                log.log_file_only(line)
        if per_graph_lines and not args.verbose:
            log.log(
                f"Per-graph summary: {n_ok} OK, {n_fail} failed "
                f"(each graph line is in the log file)."
            )

        for r in graph_summary:
            status = "OK" if r.ok else "ERROR"
            log.log(f"[{status}] {r.message}")

        for r in pdb_order_results:
            status = "OK" if r.ok else "ERROR"
            log.log(f"[{status}] {r.message}")

        for r in pdb_alignment_results:
            status = "OK" if r.ok else "ERROR"
            log.log(f"[{status}] {r.message}")

        if dataset_len is not None:
            log.log("")
            log.log("=" * 72)
            log.log(
                f"Split checks by fold (splits under {splits_root / 'Splits'}; "
                f"{len(fold_list)} fold(s))"
            )
            log.log("=" * 72)
            for k in fold_list:
                log.log("")
                sep = "=" * 72
                log.log(sep)
                log.log(f"FOLD {k}  —  train / val / test")
                log.log(sep)
                ok_f, _fold_results, fold_section_lines = run_fold_split_checks(
                    dataset_root=dataset_root,
                    splits_root=splits_root,
                    dataset_name=dataset_name,
                    train_split_file=args.train_split_file,
                    val_split_file=args.val_split_file,
                    test_split_file=args.test_split_file,
                    num_folds=args.num_folds,
                    fold_index=k,
                    dataset_len=dataset_len,
                    dataset_pdb_order=pdb_order,
                )
                for pl in fold_section_lines:
                    if args.quiet:
                        log.log_file_only(pl)
                    else:
                        log.log(pl)
                if args.quiet and fold_section_lines:
                    log.log(
                        f"(Fold {k}: full per-split listing is in the log file when using --quiet.)"
                    )
                all_ok = all_ok and ok_f

        if all_ok:
            log.log("All output-data-format checks passed.")
        else:
            log.log(
                "One or more output-data-format checks FAILED. "
                "See messages above and the log file for details."
            )
        log.log(f"Log written to {log_path}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

