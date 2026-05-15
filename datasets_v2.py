"""
Sample datasets for testing various chunking strategies - Version 2
Loads datasets from the datasets/ directory
"""

import os
from pathlib import Path

# Path to datasets directory
DATASETS_DIR = Path(__file__).parent / "datasets"

def load_dataset(name: str) -> str:
    """Load a dataset from file by name"""
    file_path = DATASETS_DIR / f"{name}.txt"
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"Dataset '{name}' not found in {DATASETS_DIR}")

def list_datasets():
    """List all available datasets in the datasets directory"""
    if not DATASETS_DIR.exists():
        return []
    return [f.stem for f in DATASETS_DIR.glob("*.txt")]

# Load all datasets into a dictionary for backward compatibility
DATASETS = {}
for dataset_name in list_datasets():
    try:
        DATASETS[dataset_name] = load_dataset(dataset_name)
    except FileNotFoundError:
        pass  # Skip if file can't be loaded

def get_dataset(name: str) -> str:
    """Retrieve a dataset by name (backward compatibility)"""
    return DATASETS.get(name, "")