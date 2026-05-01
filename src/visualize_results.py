"""
visualize_results.py — Results visualisation for FER2013
Produces:
  1. Confusion Matrix (counts + normalised)
  2. Per-class metrics bar chart
  3. Training history curves (Loss + Accuracy)
  4. Sample predictions grid

Usage:
  python src/visualize_results.py
  python src/visualize_results.py --config configs/base.yaml
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
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
# 1. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13)
    axes[0].set_xlabel("Predicted Emotion")
    axes[0].set_ylabel("True Emotion")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalised)", fontsize=13)
    axes[1].set_xlabel("Predicted Emotion")
    axes[1].set_ylabel("True Emotion")
    axes[1].tick_params(axis="x", rotation=45)

    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    plt.suptitle(f"FER2013 CNN — Confusion Matrix  |  Accuracy: {acc:.2f}%", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✓ Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 2. PER-CLASS METRICS BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_classification_report(y_true, y_pred, class_names, save_path):
    report  = classification_report(y_true, y_pred,
                                    target_names=class_names, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    x       = np.arange(len(class_names))
    w       = 0.25
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report[c][metric] for c in class_names]
        bars = ax.bar(x + i * w, vals, w, label=metric.capitalize(),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1-Score", fontsize=13)
    ax.axhline(report["accuracy"], color="red", linestyle="--", linewidth=1,
               label=f"Overall Acc {report['accuracy']*100:.2f}%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✓ Classification report saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 3. TRAINING HISTORY
# ─────────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str):
    """
    history dict keys: accuracy, val_accuracy, loss, val_loss
    (same format as Keras history.history)
    """
    acc      = history.get("accuracy",     [])
    val_acc  = history.get("val_accuracy", [])
    loss     = history.get("loss",         [])
    val_loss = history.get("val_loss",     [])
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc,     "b-o", markersize=4, label="Train Accuracy")
    ax1.plot(epochs, val_acc, "r-o", markersize=4, label="Val Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss,     "b-o", markersize=4, label="Train Loss")
    ax2.plot(epochs, val_loss, "r-o", markersize=4, label="Val Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("CNN Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✓ Training history saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 4. SAMPLE PREDICTIONS GRID
# ─────────────────────────────────────────────────────────────

def plot_sample_predictions(model, test_gen, class_names, num_samples, save_path):
    test_gen.reset()
    images_batch, labels_batch = next(iter(test_gen))

    n       = min(num_samples, len(images_batch))
    indices = random.sample(range(len(images_batch)), n)
    cols    = 4
    rows    = (n + cols - 1) // cols

    preds = model.predict(images_batch, verbose=0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for plot_i, ds_i in enumerate(indices):
        img        = images_batch[ds_i].squeeze()
        true_idx   = int(np.argmax(labels_batch[ds_i]))
        pred_idx   = int(np.argmax(preds[ds_i]))
        confidence = float(preds[ds_i][pred_idx])
        correct    = (true_idx == pred_idx)

        ax = axes[plot_i]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

        color = "green" if correct else "red"
        mark  = "✓" if correct else "✗"
        title = (f"{mark} Pred: {class_names[pred_idx]}\n"
                 f"True: {class_names[true_idx]} ({confidence*100:.1f}%)")
        ax.set_title(title, color=color, fontsize=8.5)

    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.suptitle("Sample Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✓ Sample predictions saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise FER2013 results")
    parser.add_argument("--config",      type=str, default="configs/base.yaml")
    parser.add_argument("--num-samples", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args        = parse_args()
    config      = load_config(args.config)
    class_names = config["classes"]
    model_path  = config["output"]["model_path"]

    ensure_dirs(config)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("✓ Model loaded.\n")

    # Build test generator & get predictions
    _, test_gen, _, _ = build_generators(config)
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred      = np.argmax(predictions, axis=1)
    y_true      = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    output_dir = "results"

    # 1. Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=config["output"]["confusion_matrix"],
    )

    # 2. Per-class metrics
    plot_classification_report(
        y_true, y_pred, class_names,
        save_path=os.path.join(output_dir, "classification_report.png"),
    )

    # 3. Sample predictions
    plot_sample_predictions(
        model, test_gen, class_names,
        num_samples=args.num_samples,
        save_path=os.path.join(output_dir, "sample_predictions.png"),
    )

    print(f"\n✓ All plots saved in: {output_dir}/")
