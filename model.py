"""
model.py — CNN model definition for Facial Emotion Recognition
Exact same architecture as the original notebook.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Flatten, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from src.utils import load_config


def build_model(config: dict) -> Sequential:
    """
    Build and compile the CNN model for 7-class emotion classification.
    Architecture matches the original notebook exactly.
    """
    img_size = config["image"]["img_size"]
    lr       = config["training"]["learning_rate"]

    model = Sequential()

    # ── Block 1 ──────────────────────────────
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu",
                     input_shape=(img_size, img_size, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 2 ──────────────────────────────
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 3 ──────────────────────────────
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Fully Connected Head ──────────────────
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.50))
    model.add(Dense(7, activation="softmax"))   # 7 emotion classes

    # ── Compile ──────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


if __name__ == "__main__":
    config = load_config("configs/base.yaml")
    model  = build_model(config)
    print(f"\nTotal parameters: {model.count_params():,}")
