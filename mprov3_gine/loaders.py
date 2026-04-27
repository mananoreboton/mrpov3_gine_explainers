"""
PyG DataLoaders for train, validation, and test splits.

Expects a built dataset under results/datasets/ and split PDB lists from the raw snapshot.
See README.md and ``create_data_loaders`` docstring for parameters.
"""

from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from dataset import MProV3Dataset, get_train_val_test_indices, load_dataset_pdb_order
from mprov3_gine_explainer_defaults import SplitConfig


def collate_batch(batch: List):
    """Collate a list of PyG Data objects into a single PyG Batch.

    The returned Batch keeps graph-level attributes (for example ``category``
    and ``pIC50``) as concatenated tensors, so training/evaluation loops can
    consume it directly via ``batch.to(device)``.
    """
    from torch_geometric.data import Batch

    return Batch.from_data_list([b for b in batch])


def create_data_loaders(
    dataset_root: Path,
    data_root: Path,
    split_config: SplitConfig,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders. Loads the PyG dataset from
    dataset_root/split_config.dataset_name (typically dataset_root=.../results/datasets
    and dataset_name=\".\" / BUILT_DATASET_FOLDER_NAME; run build_dataset.py first).
    Splits are read from data_root/Splits/ (raw MPro snapshot).
    """
    dataset = MProV3Dataset(
        root=str(dataset_root),
        dataset_name=split_config.dataset_name,
    )
    dataset_pdb_order = load_dataset_pdb_order(dataset_root, split_config.dataset_name)
    train_idx, val_idx, test_idx = get_train_val_test_indices(
        data_root,
        split_config.train_file,
        split_config.val_file,
        split_config.test_file,
        split_config.num_folds,
        split_config.fold_index,
        dataset_pdb_order=dataset_pdb_order,
        use_validation=split_config.use_validation,
    )
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    test_dataset = Subset(dataset, test_idx.tolist())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader, test_loader
