"""
run_pipeline.py — Run the complete Facial Emotion Recognition pipeline

Pipeline order:
  1. Data Preparation
  2. Data Exploration
  3. Model Training
  4. Evaluation

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-prep
  python run_pipeline.py --skip-explore
  python run_pipeline.py --skip-train
  python run_pipeline.py --skip-evaluate
  python run_pipeline.py --skip-train --skip-evaluate
"""

import argparse
import subprocess
import sys
from pathlib import Path


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}\n")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def validate_project_files(project_root: Path) -> bool:
    """
    Check required project files before running the pipeline.
    The project now uses configs/base.yaml as the main config file.
    """
    required_files = [
        project_root / "configs" / "base.yaml",
        project_root / "src" / "__init__.py",
        project_root / "src" / "data_preparation.py",
        project_root / "src" / "data_exploration.py",
        project_root / "src" / "dataset.py",
        project_root / "src" / "model.py",
        project_root / "src" / "train.py",
    ]

    missing_files = [file for file in required_files if not file.exists()]

    if missing_files:
        print_error("Required project files are missing:")
        for file in missing_files:
            print_error(f"  {file}")
        return False

    print_success("Required project files found")
    return True


def validate_data_folders(project_root: Path) -> bool:
    """
    Check that Data/train and Data/test exist.
    This does not verify every image. data_preparation.py handles that.
    """
    train_dir = project_root / "Data" / "train"
    test_dir = project_root / "Data" / "test"

    if not train_dir.exists():
        print_error("Data/train folder not found")
        print_warning("Expected structure:")
        print_info("  Data/train/{angry,disgust,fear,happy,neutral,sad,surprise}/")
        return False

    if not test_dir.exists():
        print_error("Data/test folder not found")
        print_warning("Expected structure:")
        print_info("  Data/test/{angry,disgust,fear,happy,neutral,sad,surprise}/")
        return False

    print_success("Data/train found")
    print_success("Data/test found")
    return True


def ensure_results_folder(project_root: Path) -> None:
    """Create results folder if it does not exist."""
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    print_success("results/ folder is ready")


def run_step(project_root: Path, module_name: str, step_name: str, step_args=None) -> bool:
    """
    Run a pipeline step using the current Python interpreter.

    Example:
      python -m src.data_preparation --config configs/base.yaml
    """
    print_header(f"Step: {step_name}")

    cmd = [sys.executable, "-m", module_name]

    if step_args:
        cmd.extend(step_args)

    print_info("Running command:")
    print_info("  " + " ".join(cmd))

    try:
        subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
        )
        print_success(f"{step_name} completed successfully")
        return True

    except subprocess.CalledProcessError as error:
        print_error(f"{step_name} failed with return code {error.returncode}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Facial Emotion Recognition pipeline"
    )

    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip data preparation",
    )

    parser.add_argument(
        "--skip-explore",
        action="store_true",
        help="Skip data exploration",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training",
    )

    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip evaluation",
    )

    parser.add_argument(
        "--build-processed",
        action="store_true",
        help="Build Data/processed during data preparation",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()
    config_path = "configs/base.yaml"

    print(f"\n{Colors.BOLD}{Colors.GREEN}")
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Facial Emotion Recognition - Full Pipeline Runner   ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    print_header("Pre-flight Checks")
    print_info(f"Project root: {project_root}")
    print_info(f"Config file: {config_path}")

    if not validate_project_files(project_root):
        print_error("Pipeline aborted because required files are missing.")
        sys.exit(1)

    if not validate_data_folders(project_root):
        print_error("Pipeline aborted because dataset folders are missing.")
        sys.exit(1)

    ensure_results_folder(project_root)

    data_prep_args = ["--config", config_path]

    if args.build_processed:
        data_prep_args.append("--build-processed")

    steps = [
        (
            "src.data_preparation",
            "Data Preparation",
            args.skip_prep,
            data_prep_args,
        ),
        (
            "src.data_exploration",
            "Data Exploration",
            args.skip_explore,
            None,
        ),
        (
            "src.train",
            "Model Training",
            args.skip_train,
            None,
        ),
        (
            "src.evaluate",
            "Evaluation",
            args.skip_evaluate,
            None,
        ),
    ]

    failed_step = None
    skipped_steps = []

    for module_name, step_name, skip_step, step_args in steps:
        if skip_step:
            print_header(f"Step: {step_name}")
            print_warning(f"{step_name} skipped")
            skipped_steps.append(step_name)
            continue

        success = run_step(project_root, module_name, step_name, step_args)

        if not success:
            failed_step = step_name
            break

    print_header("Pipeline Summary")

    if failed_step:
        print_error(f"Pipeline failed at: {failed_step}")
        print_info("Fix the error above and run the pipeline again.")
        sys.exit(1)

    if skipped_steps:
        print_warning(f"Steps skipped: {', '.join(skipped_steps)}")

    print_success("All selected pipeline steps completed successfully.")
    print_info("Check the results/ folder for generated outputs.")


if __name__ == "__main__":
    main()