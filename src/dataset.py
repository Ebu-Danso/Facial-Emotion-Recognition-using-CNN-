"""
dataset.py — Data loading and preprocessing for FER2013
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils import load_config


def count_images_in_dir(directory: str) -> dict:
    """Count number of images in each emotion subfolder."""
    counts = {}
    if os.path.exists(directory):
        for emotion in sorted(os.listdir(directory)):
            emotion_path = os.path.join(directory, emotion)
            if os.path.isdir(emotion_path):
                counts[emotion] = len(os.listdir(emotion_path))
    else:
        print(f"Folder not found: {directory}")
    return counts


def build_generators(config: dict):
    """
    Build and return train and test ImageDataGenerators.

    Returns:
        train_generator, test_generator, train_counts, test_counts
    """
    train_dir = config["data"]["train_dir"]
    test_dir  = config["data"]["test_dir"]
    img_size  = config["image"]["img_size"]
    batch     = config["training"]["batch_size"]
    color     = config["image"]["color_mode"]

    # Count images per class
    train_counts = count_images_in_dir(train_dir)
    test_counts  = count_images_in_dir(test_dir)

    # Normalise pixel values to [0, 1]
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    print("Preparing Training Data...")
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode=color,
        batch_size=batch,
        class_mode="categorical",
        shuffle=True,
    )

    print("\nPreparing Testing Data...")
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        color_mode=color,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, test_generator, train_counts, test_counts


if __name__ == "__main__":
    config = load_config("configs/base.yaml")
    train_gen, test_gen, train_counts, test_counts = build_generators(config)
    print(f"\nTrain classes : {train_gen.class_indices}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Test samples  : {test_gen.samples}")
