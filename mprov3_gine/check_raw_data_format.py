"""
Validate that an MPro Version 3 (or compatible) raw dataset at --data_root
has the expected structure and content to be used with this project.

Checks performed:
- Required files and folders exist (Info.csv, Ligand/Ligand_SDF, Splits/...)
- Info.csv has the expected columns and types
- Split files can be parsed and have the requested number of folds
- A sample of PDB IDs has corresponding SDF files
- A small sample of SDFs can be parsed into PyG graphs

Exit code:
- 0 if all checks pass
- 1 if any check fails
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from config import (
    CHECK_FORMAT_RAW_DATA_SUBDIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    MPRO_INFO_CSV,
    MPRO_LIGAND_DIR,
    MPRO_LIGAND_SDF_SUBDIR,
    MPRO_SPLITS_DIR,
    RESULTS_CHECK_FORMAT,
)
from dataset import (
    load_activity_and_category,
    load_splits,
    sdf_to_graph,
)
from utils import RunLogger, run_timestamp


@dataclass
class CheckResult:
    ok: bool
    message: str


def _check_paths_exist(data_root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []

    if not data_root.exists():
        return [CheckResult(False, f"Data root does not exist: {data_root}")]

    info_path = data_root / MPRO_INFO_CSV
    if info_path.exists():
        results.append(CheckResult(True, f"Found Info.csv at {info_path}"))
    else:
        results.append(
            CheckResult(False, f"Missing Info.csv at expected location: {info_path}")
        )

    sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
    if sdf_dir.exists() and sdf_dir.is_dir():
        results.append(CheckResult(True, f"Found SDF directory at {sdf_dir}"))
    else:
        results.append(
            CheckResult(
                False,
                f"Missing SDF directory at expected location: {sdf_dir} "
                "(expected folder with *_ligand.sdf files)",
            )
        )

    splits_dir = data_root / MPRO_SPLITS_DIR
    if splits_dir.exists() and splits_dir.is_dir():
        results.append(CheckResult(True, f"Found Splits directory at {splits_dir}"))
    else:
        results.append(
            CheckResult(
                False,
                f"Missing Splits directory at expected location: {splits_dir}",
            )
        )

    return results


def _check_info_csv(data_root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    info_path = data_root / MPRO_INFO_CSV
    if not info_path.exists():
        # This is already reported by _check_paths_exist, avoid duplicate error here.
        return results

    try:
        df = pd.read_csv(info_path, sep=";")
    except Exception as exc:  # pragma: no cover - defensive
        results.append(
            CheckResult(
                False,
                f"Failed to read Info.csv ({info_path}): {exc}",
            )
        )
        return results

    required_cols = {"PDB_ID", "pIC50", "Category"}
    missing = required_cols.difference(df.columns)
    if missing:
        results.append(
            CheckResult(
                False,
                f"Info.csv is missing required columns: {sorted(missing)} "
                f"(found columns: {sorted(df.columns.tolist())})",
            )
        )
        return results

    try:
        # Validate that conversions used by load_activity_and_category will succeed.
        _ = df["PDB_ID"].astype(str)
        _ = df["pIC50"].astype(float)
        _ = df["Category"].astype(int)
    except Exception as exc:  # pragma: no cover - defensive
        results.append(
            CheckResult(
                False,
                "Failed to convert Info.csv columns to expected types "
                f"(PDB_ID->str, pIC50->float, Category->int): {exc}",
            )
        )
        return results

    # Reuse project helper to ensure no surprises.
    try:
        pIC50_dict, category_dict = load_activity_and_category(data_root)
        if not pIC50_dict:
            results.append(
                CheckResult(
                    False,
                    "load_activity_and_category returned an empty pIC50 dict; "
                    "Info.csv may be empty or malformed.",
                )
            )
        else:
            results.append(
                CheckResult(
                    True,
                    f"Loaded {len(pIC50_dict)} entries from Info.csv via "
                    "load_activity_and_category (pIC50 and Category OK).",
                )
            )
        if not category_dict:
            results.append(
                CheckResult(
                    False,
                    "Category mapping from load_activity_and_category is empty; "
                    "Category column may be malformed.",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive
        results.append(
            CheckResult(
                False,
                f"load_activity_and_category raised an exception: {exc}",
            )
        )

    return results


def _check_splits(
    data_root: Path,
    train_file: str,
    val_file: str,
    test_file: str,
    num_folds: int,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    splits_dir = data_root / MPRO_SPLITS_DIR

    train_path = splits_dir / train_file
    val_path = splits_dir / val_file
    test_path = splits_dir / test_file

    for p, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not p.exists():
            results.append(
                CheckResult(False, f"Missing {name} split file at expected location: {p}")
            )

    # If any split file is missing, skip deeper checks (already reported).
    if any(not r.ok for r in results):
        return results

    try:
        folds = load_splits(
            data_root,
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            num_folds=num_folds,
        )
    except Exception as exc:
        results.append(
            CheckResult(
                False,
                "Failed to load splits via dataset.load_splits; "
                f"please verify the content of the split files. Error: {exc}",
            )
        )
        return results

    if len(folds) != num_folds:
        results.append(
            CheckResult(
                False,
                f"Expected {num_folds} folds, but load_splits returned {len(folds)}.",
            )
        )
    else:
        results.append(
            CheckResult(
                True,
                f"Successfully loaded {len(folds)} folds from split files.",
            )
        )

    # Sanity check: ensure PDB IDs referenced in splits exist in Info.csv.
    try:
        pIC50_dict, _ = load_activity_and_category(data_root)
        valid_ids = set(pIC50_dict.keys())
        missing_any = False
        for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds):
            for subset_name, subset_ids in [
                ("train", train_ids),
                ("val", val_ids),
                ("test", test_ids),
            ]:
                missing_ids = [p for p in subset_ids if p not in valid_ids]
                if missing_ids:
                    missing_any = True
                    results.append(
                        CheckResult(
                            False,
                            f"Fold {fold_idx} {subset_name} split contains "
                            f"{len(missing_ids)} PDB IDs not present in Info.csv; "
                            "for example: "
                            f"{missing_ids[:5]}",
                        )
                    )
        if not missing_any:
            results.append(
                CheckResult(
                    True,
                    "All PDB IDs referenced in split files are present in Info.csv.",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive
        results.append(
            CheckResult(
                False,
                f"While cross-checking splits with Info.csv, an error occurred: {exc}",
            )
        )

    return results


def _sample_pdb_ids_for_sdf_check(
    data_root: Path,
    max_samples: int = 20,
) -> List[str]:
    """Return up to max_samples PDB IDs from Info.csv to test SDF parsing."""
    info_path = data_root / MPRO_INFO_CSV
    if not info_path.exists():
        return []
    try:
        df = pd.read_csv(info_path, sep=";")
    except Exception:  # pragma: no cover - defensive
        return []
    pdb_ids = df["PDB_ID"].astype(str).tolist()
    return pdb_ids[:max_samples]


def _check_sdf_files_and_graphs(data_root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []

    sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
    if not sdf_dir.exists():
        # Already flagged in _check_paths_exist; nothing to add.
        return results

    sample_ids = _sample_pdb_ids_for_sdf_check(data_root, max_samples=20)
    if not sample_ids:
        results.append(
            CheckResult(
                False,
                "Could not determine any PDB IDs from Info.csv to test SDF files.",
            )
        )
        return results

    missing_sdf = []
    parsed_graphs = 0
    for pdb_id in sample_ids:
        sdf_path = sdf_dir / f"{pdb_id}_ligand.sdf"
        if not sdf_path.exists():
            missing_sdf.append(str(sdf_path))
            continue
        g = sdf_to_graph(sdf_path)
        if g is None:
            results.append(
                CheckResult(
                    False,
                    f"Failed to parse SDF into graph for PDB_ID {pdb_id} at {sdf_path}.",
                )
            )
        else:
            parsed_graphs += 1

    if missing_sdf:
        results.append(
            CheckResult(
                False,
                f"Missing SDF files for {len(missing_sdf)} of the first "
                f"{len(sample_ids)} PDB IDs. Example missing paths: {missing_sdf[:5]}",
            )
        )

    if parsed_graphs == 0:
        results.append(
            CheckResult(
                False,
                "None of the sampled SDF files could be parsed into PyG graphs. "
                "Check that SDF files are present and valid.",
            )
        )
    else:
        results.append(
            CheckResult(
                True,
                f"Successfully parsed {parsed_graphs} SDF files from the sample "
                "into PyG graphs (sdf_to_graph OK).",
            )
        )

    return results


def run_checks(
    data_root: Path,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    num_folds: int,
) -> Tuple[bool, List[CheckResult]]:
    all_results: List[CheckResult] = []
    # Basic presence checks
    all_results.extend(_check_paths_exist(data_root))
    # Info.csv content checks
    all_results.extend(_check_info_csv(data_root))
    # Split files and their consistency with Info.csv
    all_results.extend(
        _check_splits(
            data_root=data_root,
            train_file=train_split_file,
            val_file=val_split_file,
            test_file=test_split_file,
            num_folds=num_folds,
        )
    )
    # SDF + graph sanity checks
    all_results.extend(_check_sdf_files_and_graphs(data_root))

    all_ok = all(r.ok for r in all_results)
    return all_ok, all_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that a raw MPro Version 3 dataset at --data_root has the "
            "expected structure and can be used to build / load the PyG dataset."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"Path to raw MPro snapshot (Info.csv, Ligand/, Splits/). Default: {DEFAULT_DATA_ROOT}",
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
        default=5,
        help="Number of folds expected in the split files (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)

    ts = run_timestamp()
    log_dir = (
        Path(DEFAULT_RESULTS_ROOT) / RESULTS_CHECK_FORMAT / CHECK_FORMAT_RAW_DATA_SUBDIR / ts
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "check_input.log"

    with RunLogger(log_path) as log:
        log.log(f"Checking raw dataset at: {data_root}")
        ok, results = run_checks(
            data_root=data_root,
            train_split_file=args.train_split_file,
            val_split_file=args.val_split_file,
            test_split_file=args.test_split_file,
            num_folds=args.num_folds,
        )

        for r in results:
            status = "OK" if r.ok else "ERROR"
            log.log(f"[{status}] {r.message}")

        if ok:
            log.log("All input-data-format checks passed.")
        else:
            log.log(
                "One or more input-data-format checks FAILED. "
                "See messages above for details."
            )
        log.log(f"Log written to {log_path}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

