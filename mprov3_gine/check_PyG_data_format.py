"""
Validate that a pre-built PyG dataset (data.pt) is compatible with this project.

Expects ``results/datasets/data.pt`` (flat layout). Legacy timestamped subfolders under
``results/datasets/<timestamp>/`` are not used; migrate or rebuild with build_dataset.py.

Checks performed:
- dataset_root / dataset_name / data.pt exists and can be loaded via MProV3Dataset
- A small sample of graphs has the expected attributes and shapes:
  - x: float32, shape (N, 4)  (x, y, z, atomic_number)
  - edge_index: long, shape (2, E)
  - edge_attr: float32, shape (E, 1)
  - pIC50: float32 tensor with a single value
  - category: long tensor with a single class index in [0, num_classes-1]
  - pdb_id attribute is present
- pdb_order.txt exists and matches dataset length
- Train/val/test indices derived from the split files are within dataset bounds

Exit code:
- 0 if all checks pass
- 1 if any check fails
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
    SplitConfig,
    resolve_fold_indices,
)
from dataset import (
    MProV3Dataset,
    get_train_val_test_indices,
    load_dataset_pdb_order,
)
from utils import RunLogger


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
    max_samples: int = 10,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    if len(dataset) == 0:
        # Already reported as error in _load_dataset.
        return results

    n_samples = min(len(dataset), max_samples)
    for idx in range(n_samples):
        try:
            g = dataset[idx]
        except Exception as exc:
            results.append(
                CheckResult(
                    False,
                    f"Failed to index dataset at position {idx}: {exc}",
                )
            )
            continue

        # x: (N, 4) float32
        if not hasattr(g, "x"):
            results.append(CheckResult(False, f"Graph {idx} is missing attribute 'x'."))
        else:
            if g.x.dtype != torch.float32:
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'x' has dtype {g.x.dtype}, expected torch.float32.",
                    )
                )
            if g.x.dim() != 2 or g.x.size(1) != 4:
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'x' has shape {tuple(g.x.shape)}, "
                        "expected (N, 4) with features [x, y, z, atomic_number].",
                    )
                )

        # edge_index: (2, E) long
        if not hasattr(g, "edge_index"):
            results.append(
                CheckResult(False, f"Graph {idx} is missing attribute 'edge_index'.")
            )
        else:
            if g.edge_index.dtype != torch.long:
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'edge_index' has dtype {g.edge_index.dtype}, "
                        "expected torch.long.",
                    )
                )
            if g.edge_index.dim() != 2 or g.edge_index.size(0) != 2:
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'edge_index' has shape {tuple(g.edge_index.shape)}, "
                        "expected (2, E).",
                    )
                )

        # edge_attr: (E, 1) float32
        if not hasattr(g, "edge_attr"):
            results.append(
                CheckResult(False, f"Graph {idx} is missing attribute 'edge_attr'.")
            )
        else:
            if g.edge_attr.dtype != torch.float32:
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'edge_attr' has dtype {g.edge_attr.dtype}, "
                        "expected torch.float32.",
                    )
                )
            if g.edge_attr.dim() != 2 or g.edge_attr.size(1) != 1:
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'edge_attr' has shape {tuple(g.edge_attr.shape)}, "
                        "expected (E, 1).",
                    )
                )

            # Consistency between edge_index and edge_attr
            if hasattr(g, "edge_index"):
                e_index_e = int(g.edge_index.size(1))
                e_attr_e = int(g.edge_attr.size(0))
                if e_index_e != e_attr_e:
                    results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} has {e_index_e} edges in edge_index "
                            f"but {e_attr_e} rows in edge_attr.",
                        )
                    )

        # pIC50: regression label (1,)
        if not hasattr(g, "pIC50"):
            results.append(CheckResult(False, f"Graph {idx} is missing 'pIC50' label."))
        else:
            if not isinstance(g.pIC50, torch.Tensor):
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'pIC50' is not a tensor (type={type(g.pIC50)}).",
                    )
                )
            else:
                if g.pIC50.dtype != torch.float32:
                    results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} 'pIC50' has dtype {g.pIC50.dtype}, "
                            "expected torch.float32.",
                        )
                    )
                if g.pIC50.numel() != 1:
                    results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} 'pIC50' has shape {tuple(g.pIC50.shape)}, "
                            "expected a single scalar value.",
                        )
                    )

        # category: classification label (1,) in [0, num_classes-1]
        if not hasattr(g, "category"):
            results.append(
                CheckResult(False, f"Graph {idx} is missing 'category' label.")
            )
        else:
            if not isinstance(g.category, torch.Tensor):
                results.append(
                    CheckResult(
                        False,
                        f"Graph {idx} 'category' is not a tensor (type={type(g.category)}).",
                    )
                )
            else:
                if g.category.dtype != torch.long:
                    results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} 'category' has dtype {g.category.dtype}, "
                            "expected torch.long.",
                        )
                    )
                if g.category.numel() != 1:
                    results.append(
                        CheckResult(
                            False,
                            f"Graph {idx} 'category' has shape {tuple(g.category.shape)}, "
                            "expected a single class index.",
                        )
                    )
                else:
                    val = int(g.category.item())
                    if not (0 <= val < num_classes):
                        results.append(
                            CheckResult(
                                False,
                                f"Graph {idx} 'category' value {val} is out of expected "
                                f"range [0, {num_classes - 1}].",
                            )
                        )

        # pdb_id: for split mapping and traceability
        if not hasattr(g, "pdb_id"):
            results.append(
                CheckResult(
                    False,
                    f"Graph {idx} is missing attribute 'pdb_id' (required for "
                    "mapping splits to dataset indices).",
                )
            )

    if not any(not r.ok for r in results):
        results.append(
            CheckResult(
                True,
                f"All checked graphs (n={n_samples}) have the expected attributes "
                "and basic shapes.",
            )
        )

    return results


def _check_pdb_order_file(
    data_root: Path,
    dataset_name: str,
    dataset_len: int,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    pdb_order = load_dataset_pdb_order(data_root, dataset_name)
    if pdb_order is None:
        results.append(
            CheckResult(
                False,
                f"Missing {PYG_PDB_ORDER_FILENAME} in the dataset folder. This file is required to "
                "map split PDB IDs to dataset indices (it is written by build_dataset.py).",
            )
        )
        return results

    if len(pdb_order) != dataset_len:
        results.append(
            CheckResult(
                False,
                f"{PYG_PDB_ORDER_FILENAME} contains {len(pdb_order)} entries, but dataset has "
                f"{dataset_len} graphs; they must match.",
            )
        )
    else:
        results.append(
            CheckResult(
                True,
                f"{PYG_PDB_ORDER_FILENAME} exists and its length matches the dataset size.",
            )
        )
    return results


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
) -> List[CheckResult]:
    results: List[CheckResult] = []
    pdb_order = load_dataset_pdb_order(dataset_root, dataset_name)

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
        results.append(
            CheckResult(
                False,
                "Failed to compute train/val/test indices via get_train_val_test_indices; "
                f"dataset or splits may be incompatible. Error: {exc}",
            )
        )
        return results

    def _check_subset(name: str, idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            results.append(
                CheckResult(
                    False,
                    f"{name} indices tensor is empty; check that splits reference "
                    "valid PDB IDs and that corresponding graphs exist in the dataset.",
                )
            )
            return
        min_idx = int(idx.min().item())
        max_idx = int(idx.max().item())
        if min_idx < 0 or max_idx >= dataset_len:
            results.append(
                CheckResult(
                    False,
                    f"{name} indices out of range [0, {dataset_len - 1}]: "
                    f"min={min_idx}, max={max_idx}.",
                )
            )
        else:
            results.append(
                CheckResult(
                    True,
                    f"{name} indices are within dataset bounds "
                    f"[0, {dataset_len - 1}] (n={idx.numel()}).",
                )
            )

    _check_subset("train", train_idx)
    _check_subset("val", val_idx)
    _check_subset("test", test_idx)

    return results


def run_checks(
    dataset_root: Path,
    splits_root: Path,
    dataset_name: str,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    num_folds: int,
    fold_index: int,
    num_classes: int,
) -> Tuple[bool, List[CheckResult]]:
    all_results: List[CheckResult] = []

    all_results.extend(_check_dataset_file_exists(dataset_root, dataset_name))

    # If the dataset file is missing, deeper checks will just fail noisily; bail out early.
    if any(not r.ok for r in all_results):
        return False, all_results

    load_results, dataset = _load_dataset(dataset_root, dataset_name)
    all_results.extend(load_results)

    if dataset is None:
        return False, all_results

    all_results.extend(
        _check_sample_graphs(
            dataset=dataset,
            num_classes=num_classes,
            max_samples=10,
        )
    )
    all_results.extend(
        _check_pdb_order_file(
            data_root=dataset_root,
            dataset_name=dataset_name,
            dataset_len=len(dataset),
        )
    )
    all_results.extend(
        _check_split_indices_in_range(
            dataset_root=dataset_root,
            splits_root=splits_root,
            dataset_name=dataset_name,
            dataset_len=len(dataset),
            train_split_file=train_split_file,
            val_split_file=val_split_file,
            test_split_file=test_split_file,
            num_folds=num_folds,
            fold_index=fold_index,
        )
    )

    all_ok = all(r.ok for r in all_results)
    return all_ok, all_results


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
        help=f"Train split file name under data_root/Splits (default: {DEFAULT_TRAIN_SPLIT_FILE})",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=f"Validation split file name under data_root/Splits (default: {DEFAULT_VAL_SPLIT_FILE})",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=f"Test split file name under data_root/Splits (default: {DEFAULT_TEST_SPLIT_FILE})",
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
        log.log(
            f"Checking PyG output dataset: {dataset_root} / {dataset_name} "
            f"(splits from {splits_root}, folds={fold_list}, num_folds={args.num_folds})"
        )

        for k in fold_list:
            log.log(f"--- fold_index={k} ---")
            ok, results = run_checks(
                dataset_root=dataset_root,
                splits_root=splits_root,
                dataset_name=dataset_name,
                train_split_file=args.train_split_file,
                val_split_file=args.val_split_file,
                test_split_file=args.test_split_file,
                num_folds=args.num_folds,
                fold_index=k,
                num_classes=args.num_classes,
            )
            all_ok = all_ok and ok
            for r in results:
                status = "OK" if r.ok else "ERROR"
                log.log(f"[{status}] {r.message}")

        if all_ok:
            log.log("All output-data-format checks passed.")
        else:
            log.log(
                "One or more output-data-format checks FAILED. "
                "See messages above for details."
            )
        log.log(f"Log written to {log_path}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

