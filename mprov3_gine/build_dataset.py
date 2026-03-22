"""
Build the PyG dataset from SDFs and Info.csv. Run this once before training.
The resulting dataset is saved under results/datasets/<timestamp>/.
"""

import argparse
from pathlib import Path

import torch

from mprov3_gine_explainer_defaults import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    MPRO_LIGAND_DIR,
    MPRO_LIGAND_SDF_SUBDIR,
    PYG_DATA_FILENAME,
    PYG_PDB_ORDER_FILENAME,
    RESULTS_DATASETS,
)
from dataset import load_activity_and_category, sdf_to_graph
from tqdm import tqdm
from utils import RunLogger, run_timestamp


def build_and_save_pyg_dataset(
    data_root: Path,
    out_dir: Path,
) -> Path:
    """
    Build PyG graph list from SDFs and Info.csv and save to out_dir (e.g. results/datasets/<timestamp>/).
    Returns the path to the saved data.pt.
    """
    sdf_dir = data_root / MPRO_LIGAND_DIR / MPRO_LIGAND_SDF_SUBDIR
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")
    pIC50_dict, category_dict = load_activity_and_category(data_root)
    pdb_ids = sorted(pIC50_dict.keys())
    data_list = []
    dataset_pdb_order = []
    for pdb_id in tqdm(pdb_ids, desc="Building PyG dataset"):
        sdf_path = sdf_dir / f"{pdb_id}_ligand.sdf"
        if not sdf_path.exists():
            continue
        g = sdf_to_graph(sdf_path)
        if g is None:
            continue
        g.pIC50 = torch.tensor([pIC50_dict[pdb_id]], dtype=torch.float32)
        g.category = torch.tensor([category_dict.get(pdb_id, 0)], dtype=torch.long)
        g.pdb_id = pdb_id
        data_list.append(g)
        dataset_pdb_order.append(pdb_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / PYG_DATA_FILENAME
    from torch_geometric.data import InMemoryDataset

    InMemoryDataset.save(data_list, str(out_path))
    pdb_order_path = out_dir / PYG_PDB_ORDER_FILENAME
    pdb_order_path.write_text("\n".join(dataset_pdb_order))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PyG dataset from SDFs and Info.csv. Run before training."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to raw MPro snapshot (Ligand/, Info.csv); default: DEFAULT_DATA_ROOT",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); dataset written to results_root/datasets/<timestamp>/.",
    )
    args = parser.parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    results_root = Path(args.results_root or DEFAULT_RESULTS_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    ts = run_timestamp()
    out_dir = results_root / RESULTS_DATASETS / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build.log"

    with RunLogger(log_path) as log:
        log.log(f"Building PyG dataset from {data_root}")
        log.log(f"Output directory: {out_dir}")
        path = build_and_save_pyg_dataset(data_root, out_dir)
        log.log(f"PyG dataset saved to {path}")
        log.log(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
