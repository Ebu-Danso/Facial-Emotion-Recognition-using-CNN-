"""
utils.py — Shared utility functions for FER project
"""

import os
import yaml


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load YAML config file and return as dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: dict) -> None:
    """Create output directories if they don't exist."""
    os.makedirs("results", exist_ok=True)
    print("Output directories ready.")


def get_class_names(config: dict) -> list:
    """Return list of emotion class names from config."""
    return config["classes"]
