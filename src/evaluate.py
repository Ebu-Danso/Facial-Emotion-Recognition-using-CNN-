"""
evaluate.py — Model evaluation with confusion matrix for FER
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

from src.dataset import build_generators
from src.utils   import load_config, ensure_dirs


# ─────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────

def evaluate(config: dict) -> None:
    ensure_dirs(config)

    model_path = config["output"]["model_path"]

    # ── Load model ──
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # ── Load test data ──
    _, test_gen, _, _ = build_generators(config)

    # ── Final test accuracy ──
    batch_size = config["training"]["batch_size"]
    loss, accuracy = model.evaluate(
        test_gen,
        steps=test_gen.samples // batch_size,
        verbose=1,
    )
    print(f"\nFinal Test Accuracy : {accuracy * 100:.2f}%")
    print(f"Final Test Loss     : {loss:.4f}")

    # ── Predictions ──
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())

    # ── Classification report ──
    print("\nClassification Report:")
    print("-" * 55)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── Confusion matrix ──
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=config["output"]["confusion_matrix"]
    )


# ─────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str) -> None:
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13)
    axes[0].set_xlabel("Predicted Emotion")
    axes[0].set_ylabel("True Emotion")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], vmin=0, vmax=1
    )
    axes[1].set_title("Confusion Matrix (Normalised)", fontsize=13)
    axes[1].set_xlabel("Predicted Emotion")
    axes[1].set_ylabel("True Emotion")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("FER2013 CNN Model — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config("configs/base.yaml")
    evaluate(config)
