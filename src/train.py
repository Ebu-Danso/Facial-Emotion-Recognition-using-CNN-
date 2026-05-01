"""
train.py — Training pipeline for Facial Emotion Recognition
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.dataset import build_generators
from src.model   import build_model
from src.utils   import load_config, ensure_dirs


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train(config: dict):
    ensure_dirs(config)

    # ── Data ──
    train_gen, test_gen, _, _ = build_generators(config)

    # ── Model ──
    model = build_model(config)

    # ── Callbacks ──
    model_path = config["output"]["model_path"]
    patience   = config["training"]["patience"]

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    # ── Train ──
    batch_size = config["training"]["batch_size"]
    epochs     = config["training"]["epochs"]

    print("\nStarting training...")
    print(f"  Epochs (max)  : {epochs}")
    print(f"  Early stopping: patience={patience} on val_accuracy")
    print(f"  Best model    : saved to {model_path}\n")

    history = model.fit(
        train_gen,
        steps_per_epoch  = train_gen.samples  // batch_size,
        epochs           = epochs,
        validation_data  = test_gen,
        validation_steps = test_gen.samples   // batch_size,
        callbacks        = [early_stop, checkpoint],
    )

    print("\nTraining complete!")

    # ── Plot & save training curves ──
    plot_training_history(history, config["output"]["training_plot"])

    return model, history


# ─────────────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────────────────────────

def plot_training_history(history, save_path: str) -> None:
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, acc,     "b-o", markersize=4, label="Train Accuracy")
    ax1.plot(epochs, val_acc, "r-o", markersize=4, label="Test Accuracy")
    ax1.set_title("Training and Test Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, loss,     "b-o", markersize=4, label="Train Loss")
    ax2.plot(epochs, val_loss, "r-o", markersize=4, label="Test Loss")
    ax2.set_title("Training and Test Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("CNN Model Training Performance", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training history saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config("configs/base.yaml")
    train(config)
