"""
dataset.py — PyTorch data loading and preprocessing for FER2013
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils import load_config, get_class_names


def get_dataloaders(
    config_path: str = "configs/data_config.yaml",
    batch_size: int = None,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> tuple:
    """
    Load FER2013 dataset and return PyTorch DataLoaders.

    Args:
        config_path: Path to YAML config file
        batch_size: Batch size for DataLoaders (if None, use config value)
        num_workers: Number of workers for DataLoader
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, test_loader, class_names)

    Example:
        train_loader, test_loader, class_names = get_dataloaders()
        for images, labels in train_loader:
            print(images.shape)  # [batch_size, 1, 48, 48]
            break
    """
    config = load_config(config_path)

    train_dir = config.get("train_path", "Data/train")
    test_dir = config.get("test_path", "Data/test")
    img_size = int(config.get("image_size", 48))
    batch = batch_size or int(config.get("batch_size", 64))
    class_names = get_class_names(config)

    # Define transforms for grayscale images
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ]
    )

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, class_names


def count_images_in_dir(directory: str) -> dict:
    """Count number of images in each emotion subfolder."""
    counts = {}
    if os.path.exists(directory):
        for emotion in sorted(os.listdir(directory)):
            emotion_path = os.path.join(directory, emotion)
            if os.path.isdir(emotion_path):
                # Count only image files
                image_files = [
                    f
                    for f in os.listdir(emotion_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                ]
                counts[emotion] = len(image_files)
    else:
        print(f"Folder not found: {directory}")
    return counts


if __name__ == "__main__":
    config = load_config("configs/data_config.yaml")
    train_dir = config.get("train_path", "Data/train")
    test_dir = config.get("test_path", "Data/test")

    print("Loading dataloaders...")
    train_loader, test_loader, class_names = get_dataloaders()

    print(f"Class names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Show sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape: {images.shape}")
    print(f"Sample batch labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")

    # Count images per class
    print("\nImage counts:")
    train_counts = count_images_in_dir(train_dir)
    test_counts = count_images_in_dir(test_dir)
    for emotion in sorted(train_counts.keys()):
        print(
            f"  {emotion}: {train_counts.get(emotion, 0)} train, {test_counts.get(emotion, 0)} test"
        )
