from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image
except ImportError:
    print("Missing dependency: Pillow is required to run this script. Install with `pip install pillow`.")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Missing dependency: PyYAML is required to read the config file. Install with `pip install pyyaml`.")
    sys.exit(1)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".pgm", ".ppm"}


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid config format. Expected a YAML mapping.")

    return config


def validate_folder(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected {description} to be a directory: {path}")


def build_class_paths(base_path: Path, class_names: List[str]) -> Dict[str, Path]:
    return {name: base_path / name for name in class_names}


def scan_classes(class_paths: Dict[str, Path]) -> Tuple[Dict[str, int], List[Path], List[Path]]:
    class_counts: Dict[str, int] = {}
    unsupported_files: List[Path] = []
    missing_or_empty: List[Path] = []

    for class_name, class_path in class_paths.items():
        if not class_path.exists():
            missing_or_empty.append(class_path)
            class_counts[class_name] = 0
            continue
        if not class_path.is_dir():
            unsupported_files.append(class_path)
            class_counts[class_name] = 0
            continue

        files = [path for path in class_path.iterdir() if path.is_file()]
        valid_files = [path for path in files if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
        invalid_files = [path for path in files if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS]

        class_counts[class_name] = len(valid_files)
        unsupported_files.extend(invalid_files)

        if len(valid_files) == 0:
            missing_or_empty.append(class_path)

    return class_counts, unsupported_files, missing_or_empty


def inspect_image(image_path: Path) -> Tuple[bool, str, str, Tuple[int, int]]:
    try:
        with Image.open(image_path) as image:
            original_mode = image.mode
            original_size = image.size
            accepted = image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            if not accepted:
                return False, original_mode, "unsupported format", original_size
            return True, original_mode, "ok", original_size
    except Exception as exc:
        return False, "unknown", f"cannot open image: {exc}", (0, 0)


def print_summary(train_counts: Dict[str, int], test_counts: Dict[str, int], class_names: List[str]) -> None:
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())

    print("\nDataset validation summary")
    print("===========================")
    print(f"{'Class':<10} {'Train':>7} {'Test':>7}")
    print("---------------------------")
    for name in class_names:
        print(f"{name:<10} {train_counts.get(name, 0):7d} {test_counts.get(name, 0):7d}")
    print("---------------------------")
    print(f"{'TOTAL':<10} {total_train:7d} {total_test:7d}\n")


def process_image(source_path: Path, destination_path: Path, size: int) -> None:
    with Image.open(source_path) as image:
        image = image.convert("L")
        if image.size != (size, size):
            image = image.resize((size, size), Image.BILINEAR)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination_path)


def build_processed_dataset(
    train_paths: Dict[str, Path],
    test_paths: Dict[str, Path],
    processed_root: Path,
    image_size: int,
    class_names: List[str],
    rebuild: bool = False,
) -> None:
    if rebuild and processed_root.exists():
        print(f"Rebuilding processed dataset in: {processed_root}")

    for split_name, paths in [("train", train_paths), ("test", test_paths)]:
        for class_name in class_names:
            source_path = paths[class_name]
            destination_dir = processed_root / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            if not source_path.exists():
                print(f"WARNING: Skipping missing source class folder: {source_path}")
                continue

            for image_path in sorted(source_path.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue

                destination_path = destination_dir / image_path.name
                try:
                    process_image(image_path, destination_path, image_size)
                except Exception as exc:
                    print(f"Failed to process {image_path}: {exc}")

    print(f"\nProcessed dataset created at: {processed_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the FER2013 dataset folder structure and optionally build a processed dataset."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the data config YAML file. Default: configs/data_config.yaml",
    )
    parser.add_argument(
        "--build-processed",
        action="store_true",
        help="Create a processed dataset in the configured processed data path.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the processed dataset by overwriting files in the processed folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else root_dir / "configs" / "data_config.yaml"

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Error reading config: {exc}")
        sys.exit(1)

    raw_dir = root_dir / Path(config.get("raw_data_path", "Data"))
    train_dir = root_dir / Path(config.get("train_path", "Data/train"))
    test_dir = root_dir / Path(config.get("test_path", "Data/test"))
    processed_dir = root_dir / Path(config.get("processed_data_path", "Data/processed"))
    image_size = int(config.get("image_size", 48))
    class_names = list(config.get("class_names", []))

    try:
        validate_folder(raw_dir, "raw data folder")
        validate_folder(train_dir, "train folder")
        validate_folder(test_dir, "test folder")
    except Exception as exc:
        print(f"Dataset validation error: {exc}")
        sys.exit(1)

    if len(class_names) != 7:
        print("Warning: Expected 7 emotion classes in config. Using class names from config anyway.")

    train_class_paths = build_class_paths(train_dir, class_names)
    test_class_paths = build_class_paths(test_dir, class_names)

    train_counts, train_unsupported, train_empty = scan_classes(train_class_paths)
    test_counts, test_unsupported, test_empty = scan_classes(test_class_paths)

    print_summary(train_counts, test_counts, class_names)

    if train_empty or test_empty:
        print("WARNING: One or more class folders are missing or contain no valid images.")
        for folder in train_empty + test_empty:
            print(f"  - {folder}")

    if train_unsupported or test_unsupported:
        print("\nWARNING: Unsupported files were found in the dataset folders:")
        for file_path in train_unsupported + test_unsupported:
            print(f"  - {file_path}")

    print("\nChecking image files for correct dimensions and grayscale compatibility...")
    bad_images: List[Path] = []
    processed_image_shapes: Dict[str, int] = {"wrong_size": 0, "not_grayscale": 0, "bad_files": 0}

    for image_path in sorted(train_dir.rglob("*")) + sorted(test_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        valid, mode, status, size = inspect_image(image_path)
        if not valid or status != "ok":
            processed_image_shapes["bad_files"] += 1
            bad_images.append(image_path)
            continue
        if size != (image_size, image_size):
            processed_image_shapes["wrong_size"] += 1
        if mode not in {"L", "LA", "I;16", "I", "1", "P"}:
            processed_image_shapes["not_grayscale"] += 1

    if bad_images:
        print("\nERROR: Found unreadable or unsupported image files:")
        for bad_path in bad_images:
            print(f"  - {bad_path}")
        print("Please remove or replace these files before processing.")

    if processed_image_shapes["wrong_size"] == 0 and processed_image_shapes["not_grayscale"] == 0 and not bad_images:
        print("All detected images are valid and readable.")
    else:
        print("\nImage check results:")
        print(f"  - Images requiring resize: {processed_image_shapes['wrong_size']}")
        print(f"  - Images that may need grayscale conversion: {processed_image_shapes['not_grayscale']}")
        print(f"  - Bad or unreadable files: {processed_image_shapes['bad_files']}")

    if args.build_processed:
        build_processed_dataset(
            train_class_paths,
            test_class_paths,
            processed_dir,
            image_size,
            class_names,
            rebuild=args.rebuild,
        )
    else:
        print("\nNo processed dataset was created. Re-run with --build-processed to generate Data/processed.")


if __name__ == "__main__":
    main()
