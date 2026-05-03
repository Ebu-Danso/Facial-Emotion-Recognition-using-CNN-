"""
data_exploration.py — Dataset exploration and visualisation
Saves all plots to results/ folder (no interactive Plotly iframe needed)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.dataset import build_generators
from src.utils import load_config, ensure_dirs


# ─────────────────────────────────────────────────────────────
# CONFIG NORMALISATION
# ─────────────────────────────────────────────────────────────

def prepare_config(config: dict) -> dict:
    """
    Makes the config compatible with dataset.py.

    dataset.py expects:
    - config["data"]["train_dir"]
    - config["data"]["test_dir"]
    - config["image"]["img_size"]
    - config["training"]["batch_size"]

    This function allows data_exploration.py to still work even if
    configs/data_config.yaml uses flatter keys like train_path, test_path,
    image_size, and batch_size.
    """

    # Data paths
    if "data" not in config:
        config["data"] = {}

    config["data"]["train_dir"] = config["data"].get(
        "train_dir",
        config.get("train_path", "Data/train")
    )

    config["data"]["test_dir"] = config["data"].get(
        "test_dir",
        config.get("test_path", "Data/test")
    )

    config["data"]["class_names"] = config["data"].get(
        "class_names",
        config.get(
            "class_names",
            ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        )
    )

    # Image settings expected by dataset.py
    if "image" not in config:
        config["image"] = {}

    config["image"]["img_size"] = config["image"].get(
        "img_size",
        config.get("image_size", 48)
    )

    config["image"]["channels"] = config["image"].get("channels", 1)
    config["image"]["color_mode"] = config["image"].get("color_mode", "grayscale")

    # Training settings expected by dataset.py
    if "training" not in config:
        config["training"] = {}

    config["training"]["batch_size"] = config["training"].get(
        "batch_size",
        config.get("batch_size", 32)
    )

    # Output paths for saved plots
    if "output" not in config:
        config["output"] = {}

    config["output"]["distribution_plot"] = config["output"].get(
        "distribution_plot",
        "results/class_distribution.png"
    )

    config["output"]["sample_images_plot"] = config["output"].get(
        "sample_images_plot",
        "results/sample_images.png"
    )

    os.makedirs("results", exist_ok=True)

    return config


# ─────────────────────────────────────────────────────────────
# 1. CLASS DISTRIBUTION BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_class_distribution(train_counts: dict, test_counts: dict, save_path: str) -> None:
    emotions = sorted(train_counts.keys())
    train_vals = [train_counts[e] for e in emotions]
    test_vals = [test_counts.get(e, 0) for e in emotions]

    x = range(len(emotions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        train_vals,
        width,
        label="Train",
        color="#378ADD",
        alpha=0.85,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        test_vals,
        width,
        label="Test",
        color="#EF9F27",
        alpha=0.85,
    )

    ax.set_xlabel("Emotion Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Training Set vs Test Set: Class Distribution", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels([e.upper() for e in emotions])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Class distribution chart saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# 2. SAMPLE IMAGES
# ─────────────────────────────────────────────────────────────

def plot_sample_images(train_generator, save_path: str) -> None:
    images, labels = next(train_generator)

    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Top Row: Raw Display  |  Bottom Row: Normalised Data",
        fontsize=14,
    )

    for i in range(4):
        img_normalised = images[i].squeeze()

        if img_normalised.max() <= 1.0:
            img_raw = (img_normalised * 255).astype(np.uint8)
        else:
            img_raw = img_normalised.astype(np.uint8)

        label_idx = int(np.argmax(labels[i]))
        emotion_name = class_names[label_idx]

        axes[0, i].imshow(img_raw, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(
            f"Class: {emotion_name.upper()}\nMax Pixel: {img_raw.max()} (RAW)"
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(img_normalised, cmap="gray")
        axes[1, i].set_title(
            f"Class: {emotion_name.upper()}\nMax Pixel: {img_normalised.max():.2f}"
        )
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sample images saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# 3. DATASET SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_dataset_summary(train_counts: dict, test_counts: dict) -> None:
    df = pd.DataFrame({"Train": train_counts, "Test": test_counts})
    df.loc["Total"] = df.sum()

    print("\nDataset Structure (Number of Images):")
    print("-" * 40)
    print(df.to_string())
    print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config("configs/data_config.yaml")
    config = prepare_config(config)

    ensure_dirs(config)

    train_gen, test_gen, train_counts, test_counts = build_generators(config)

    print_dataset_summary(train_counts, test_counts)

    plot_class_distribution(
        train_counts,
        test_counts,
        save_path=config["output"]["distribution_plot"],
    )

    plot_sample_images(
        train_gen,
        save_path=config["output"]["sample_images_plot"],
    )