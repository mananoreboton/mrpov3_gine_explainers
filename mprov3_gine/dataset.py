"""
MPro Version 3 dataset: load SDF ligands and build PyTorch Geometric graphs.
Node features: (x, y, z, atomic_number). Edges from bonds.
Labels: pIC50 (regression) and Category (classification: -1, 0, 1 -> 0, 1, 2).
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from mprov3_gine_explainer_defaults import (
    MPRO_INFO_CSV,
    MPRO_SPLITS_DIR,
    PYG_DATA_FILENAME,
    PYG_PDB_ORDER_FILENAME,
)
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


# Atomic number lookup (common elements in MPro ligands)
ATOMIC_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17, "Br": 35,
    "I": 53, "P": 15, "B": 5, "H": 1,
}


def _bond_type_to_scalar(bond) -> float:
    """Map RDKit bond type to scalar for GINE edge_attr. Single=1, Double=2, Triple=3, Aromatic=1.5."""
    from rdkit.Chem import BondType
    bt = bond.GetBondType()
    if bt == BondType.SINGLE:
        return 1.0
    if bt == BondType.DOUBLE:
        return 2.0
    if bt == BondType.TRIPLE:
        return 3.0
    if bt == BondType.AROMATIC:
        return 1.5
    return 1.0


def sdf_to_graph(sdf_path: Path) -> Optional[Data]:
    """Load one SDF and return a PyG Data with x (N,4): [x,y,z, atomic_num], edge_index (2,E), edge_attr (E,1) for GINE."""
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        return None

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pos = conf.GetPositions()  # (n, 3)
    node_feats = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        anum = ATOMIC_NUM.get(sym, 6)  # default C if unknown
        node_feats.append([pos[i, 0], pos[i, 1], pos[i, 2], float(anum)])
    x = torch.tensor(node_feats, dtype=torch.float32)

    edge_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _bond_type_to_scalar(bond)
        edge_list.append([u, v])
        edge_attr_list.append([bt])
        edge_list.append([v, u])
        edge_attr_list.append([bt])
    if len(edge_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


def load_activity_and_category(
    data_root: Path,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Load pIC50 and Category from Info.csv. Return dicts PDB_ID -> value."""
    info_path = data_root / MPRO_INFO_CSV
    df = pd.read_csv(info_path, sep=";")
    pIC50 = dict(zip(df["PDB_ID"].astype(str), df["pIC50"].astype(float)))
    # Category: -1 -> 0, 0 -> 1, 1 -> 2 for class indices
    cat_map = {-1: 0, 0: 1, 1: 2}
    category = {}
    for _, row in df.iterrows():
        c = int(row["Category"])
        category[str(row["PDB_ID"])] = cat_map.get(c, 0)
    return pIC50, category


# Map class index (0, 1, 2) back to original Category (-1, 0, 1) for reporting.
ORIGINAL_CATEGORY_FROM_CLASS: Dict[int, int] = {0: -1, 1: 0, 2: 1}


def _parse_split_file(path: Path) -> List[List[str]]:
    """Parse a single Splits file (Python list of lists of PDB IDs)."""
    import ast
    text = path.read_text()
    folds = ast.literal_eval(text)
    if isinstance(folds, list) and len(folds) > 0:
        if isinstance(folds[0], list):
            return folds
        return [folds]
    return []


def load_splits(
    data_root: Path,
    train_file: str,
    val_file: str,
    test_file: str,
    num_folds: int,
    *,
    use_validation: bool = True,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    Load train/val/test splits from the Splits folder.
    Each file must contain num_folds lists of PDB IDs. Returns one (train_ids, val_ids, test_ids) per fold.
    If use_validation is False, only train and test files are read; val_ids are always [] per fold.
    """
    splits_dir = data_root / MPRO_SPLITS_DIR
    train_path = splits_dir / train_file
    test_path = splits_dir / test_file
    for p, name in [(train_path, "train"), (test_path, "test")]:
        if not p.exists():
            raise FileNotFoundError(f"Split file not found: {p}")
    train_folds = _parse_split_file(train_path)
    test_folds = _parse_split_file(test_path)
    if use_validation:
        val_path = splits_dir / val_file
        if not val_path.exists():
            raise FileNotFoundError(f"Split file not found: {val_path}")
        val_folds = _parse_split_file(val_path)
        n = min(len(train_folds), len(val_folds), len(test_folds))
    else:
        val_folds = None
        n = min(len(train_folds), len(test_folds))
    if n < num_folds:
        raise ValueError(
            f"Split files have {n} folds but num_folds={num_folds}. "
            f"Each file must contain at least {num_folds} lists."
        )
    if use_validation:
        return [
            (train_folds[k], val_folds[k], test_folds[k])
            for k in range(num_folds)
        ]
    return [
        (train_folds[k], [], test_folds[k])
        for k in range(num_folds)
    ]


def _pyg_dataset_not_found_message(data_root: Path, dataset_name: str) -> str:
    return (
        f"PyG dataset not found at {data_root / dataset_name / PYG_DATA_FILENAME}. "
        f"Create it first with: uv run python build_dataset.py --data_root {data_root} [--dataset_name {dataset_name}]"
    )


class MProV3Dataset(InMemoryDataset):
    """
    Load a pre-built PyG dataset from data_root/dataset_name/data.pt.
    Does not build from SDFs; if the file is missing, raises an error. Use build_dataset.py to create it.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self._data_root = Path(root)
        self._dataset_name = dataset_name
        dataset_path = self._data_root / dataset_name / PYG_DATA_FILENAME
        if not dataset_path.exists():
            raise FileNotFoundError(_pyg_dataset_not_found_message(self._data_root, dataset_name))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(self._data_root)

    @property
    def processed_dir(self) -> str:
        return str(self._data_root / self._dataset_name)

    @property
    def raw_file_names(self) -> List[str]:
        return [MPRO_INFO_CSV]

    @property
    def processed_file_names(self) -> List[str]:
        return [PYG_DATA_FILENAME]

    def process(self):
        raise FileNotFoundError(
            _pyg_dataset_not_found_message(self._data_root, self._dataset_name)
        )


def load_dataset_pdb_order(data_root: Path, dataset_name: str) -> Optional[List[str]]:
    """Load PDB ID order from dataset folder (written by build_dataset). Returns None if missing."""
    path = data_root / dataset_name / PYG_PDB_ORDER_FILENAME
    if not path.exists():
        return None
    text = path.read_text().strip()
    return text.split("\n") if text else []


def get_train_val_test_indices(
    data_root: Path,
    train_file: str,
    val_file: str,
    test_file: str,
    num_folds: int,
    fold_index: int,
    dataset_pdb_order: Optional[List[str]] = None,
    *,
    use_validation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load train/val/test split from three files and map PDB IDs to dataset indices.
    If dataset_pdb_order is provided (PDB IDs in same order as the built dataset), indices
    are in [0, len(dataset)-1]. Otherwise uses sorted(Info.csv) order (may be out of range
    if the dataset has fewer samples than Info.csv).
    If use_validation is False, val_file is not read and val indices are empty.
    """
    folds_tuples = load_splits(
        data_root,
        train_file,
        val_file,
        test_file,
        num_folds,
        use_validation=use_validation,
    )
    k = min(fold_index, num_folds - 1) if num_folds > 0 else 0
    train_ids = set(folds_tuples[k][0])
    val_ids = set(folds_tuples[k][1])
    test_ids = set(folds_tuples[k][2])
    if dataset_pdb_order is not None:
        pdb_order = dataset_pdb_order
    else:
        pIC50_dict, _ = load_activity_and_category(data_root)
        pdb_order = sorted(pIC50_dict.keys())
    pdb_to_idx = {p: i for i, p in enumerate(pdb_order)}
    train_idx = [pdb_to_idx[p] for p in train_ids if p in pdb_to_idx]
    val_idx = [pdb_to_idx[p] for p in val_ids if p in pdb_to_idx]
    test_idx = [pdb_to_idx[p] for p in test_ids if p in pdb_to_idx]
    return (
        torch.tensor(train_idx),
        torch.tensor(val_idx),
        torch.tensor(test_idx),
    )
