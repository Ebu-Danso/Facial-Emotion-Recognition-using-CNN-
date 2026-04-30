"""
utils.py — Shared utility functions for FER project
"""

import os
from pathlib import Path
import yaml


def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """Load YAML config file and return as dictionary."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: dict = None) -> None:
    """Create output directories if they don't exist."""
    os.makedirs("results", exist_ok=True)
    print("Output directories ready.")


def get_class_names(config: dict) -> list:
    """Return list of emotion class names from config."""
    return config.get("class_names", [])


def get_num_classes(config: dict) -> int:
    """Return number of emotion classes from config."""
    return len(config.get("class_names", []))
