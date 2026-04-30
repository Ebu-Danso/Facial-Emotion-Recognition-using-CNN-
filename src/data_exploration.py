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
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from src.dataset import get_dataloaders, count_images_in_dir
from src.utils import load_config, ensure_dirs, get_class_names


# ─────────────────────────────────────────────────────────────
# 1. CLASS DISTRIBUTION BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_class_distribution(train_counts: dict, test_counts: dict, save_path: str) -> None:
    emotions = sorted(train_counts.keys())
    train_vals = [train_counts[e] for e in emotions]
    test_vals  = [test_counts.get(e, 0) for e in emotions]

    x = range(len(emotions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], train_vals, width, label="Train",
                   color="#378ADD", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], test_vals, width, label="Test",
                   color="#EF9F27", alpha=0.85)

    ax.set_xlabel("Emotion Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Training Set vs Test Set: Class Distribution", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels([e.upper() for e in emotions])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Class distribution chart saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# 2. SAMPLE IMAGES FROM PYTORCH DATALOADER
# ─────────────────────────────────────────────────────────────

def plot_sample_images(train_loader, class_names: list, save_path: str) -> None:
    """Plot sample images from PyTorch DataLoader."""
    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Sample Training Images (Normalized Grayscale, 48×48)", fontsize=14
    )

    for i in range(4):
        img_tensor = images[i].squeeze()  # Remove channel dimension
        img_numpy = img_tensor.cpu().numpy()

        # Denormalize from [-1, 1] to [0, 1] for display
        img_display = (img_numpy + 1) / 2
        img_display = np.clip(img_display, 0, 1)

        label_idx = labels[i].item()
        emotion_name = class_names[label_idx]

        # Raw (rescaled for display)
        axes[0, i].imshow(img_display, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Class: {emotion_name.upper()}\n(Raw Display)")
        axes[0, i].axis("off")

        # Normalized tensor view
        img_norm = img_numpy
        axes[1, i].imshow(img_norm, cmap="gray", vmin=-1, vmax=1)
        axes[1, i].set_title(f"Class: {emotion_name.upper()}\n(Normalized [-1,1])")
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
    ensure_dirs(config)

    train_dir = config.get("train_path", "Data/train")
    test_dir = config.get("test_path", "Data/test")
    class_names = get_class_names(config)

    # Count images per class
    train_counts = count_images_in_dir(train_dir)
    test_counts = count_images_in_dir(test_dir)

    print_dataset_summary(train_counts, test_counts)

    # Plot class distribution
    plot_class_distribution(
        train_counts, test_counts, save_path="results/class_distribution.png"
    )

    # Load data and plot samples
    print("\nLoading training data for sample visualization...")
    train_loader, _, class_names = get_dataloaders(batch_size=8)

    plot_sample_images(train_loader, class_names, save_path="results/sample_images.png")
