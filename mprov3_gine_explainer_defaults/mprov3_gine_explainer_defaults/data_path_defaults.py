"""
Shared filesystem layout for the MPro v3 snapshot, GNN `results/`, and explainer outputs.

Constants are path *segments* or filenames; callers join with their project roots.
"""

# --- Repository layout: default raw snapshot is a sibling of mprov3_gine / mprov3_explainer ---
DEFAULT_MPRO_SNAPSHOT_DIR_NAME = "mprov3_data"

# --- Inside the raw MPro snapshot ---
MPRO_INFO_CSV = "Info.csv"
MPRO_SPLITS_DIR = "Splits"
MPRO_LIGAND_DIR = "Ligand"
MPRO_LIGAND_SDF_SUBDIR = "Ligand_SDF"

# --- Split filenames (under MPRO_SPLITS_DIR) ---
DEFAULT_TRAIN_SPLIT_FILE = "train_index_folder.txt"
DEFAULT_VAL_SPLIT_FILE = "valid_index_folder.txt"
DEFAULT_TEST_SPLIT_FILE = "test_index_folder.txt"

# --- Package-local results directory (e.g. mprov3_gine/results, mprov3_explainer/results) ---
RESULTS_DIR_NAME = "results"

# --- First-level segments under a results root ---
RESULTS_TRAININGS = "trainings"
RESULTS_DATASETS = "datasets"
RESULTS_CLASSIFICATIONS = "classifications"
RESULTS_VISUALIZATIONS = "visualizations"
RESULTS_EXPLANATIONS = "explanations"
RESULTS_CHECK_FORMAT = "check_format"

# --- Under results/<RESULTS_CHECK_FORMAT>/ ---
CHECK_FORMAT_DATASETS_SUBDIR = "datasets"
CHECK_FORMAT_RAW_DATA_SUBDIR = "raw_data"

# --- Built PyG artifacts (under results/datasets/<timestamp>/ or similar) ---
PYG_DATA_FILENAME = "data.pt"
PYG_PDB_ORDER_FILENAME = "pdb_order.txt"
DEFAULT_TRAINING_CHECKPOINT_FILENAME = "best_gnn.pt"

# --- Default dataset folder name when using MProV3Dataset(root, dataset_name, ...) ---
DEFAULT_PYG_DATASET_NAME = "processed_pyg"
