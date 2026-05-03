"""
run_pipeline.py — Run the complete Facial Emotion Recognition pipeline

This script orchestrates the entire project workflow in the correct order:
  1. Data Preparation
  2. Data Exploration
  3. Model Training
  4. Evaluation

Usage:
  python run_pipeline.py                          # Run all steps
  python run_pipeline.py --skip-explore           # Skip data exploration
  python run_pipeline.py --skip-train             # Skip training
  python run_pipeline.py --skip-evaluate          # Skip evaluation
  python run_pipeline.py --skip-train --skip-evaluate  # Multiple skips
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# COLOR CODES FOR TERMINAL OUTPUT
# ─────────────────────────────────────────────────────────────

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print a colorful section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print an error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


# ─────────────────────────────────────────────────────────────
# VALIDATION & SETUP
# ─────────────────────────────────────────────────────────────

def validate_data_folders(project_root: Path) -> bool:
    """
    Check that Data/train and Data/test exist.
    Return True if valid, False otherwise.
    """
    train_dir = project_root / "Data" / "train"
    test_dir = project_root / "Data" / "test"

    if not train_dir.exists():
        print_error("Data/train folder not found!")
        print_warning("Please download the FER2013 dataset from Kaggle:")
        print_info("  https://www.kaggle.com/datasets/msambare/fer2013")
        print_warning("Extract images into Data/train and Data/test with emotion subdirectories:")
        print_info("  Data/train/{angry,disgust,fear,happy,neutral,sad,surprise}/")
        print_info("  Data/test/{angry,disgust,fear,happy,neutral,sad,surprise}/")
        return False

    if not test_dir.exists():
        print_error("Data/test folder not found!")
        print_warning("Please download the FER2013 dataset from Kaggle:")
        print_info("  https://www.kaggle.com/datasets/msambare/fer2013")
        print_warning("Extract images into Data/train and Data/test with emotion subdirectories:")
        print_info("  Data/train/{angry,disgust,fear,happy,neutral,sad,surprise}/")
        print_info("  Data/test/{angry,disgust,fear,happy,neutral,sad,surprise}/")
        return False

    print_success(f"Data/train found")
    print_success(f"Data/test found")
    return True


def ensure_results_folder(project_root: Path) -> None:
    """Create results folder if it doesn't exist"""
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    print_success("Results folder is ready")


def run_step(project_root: Path, module_name: str, step_name: str, args: list = None) -> bool:
    """
    Run a Python module as a subprocess.
    Returns True if successful, False if failed.
    
    Args:
        project_root: Root directory of the project
        module_name: Module name to run (e.g., "src.data_preparation")
        step_name: Human-readable step name for output
        args: Optional list of additional arguments to pass to the script
    """
    print_header(f"Step: {step_name}")

    try:
        cmd = [sys.executable, "-m", module_name]
        if args:
            cmd.extend(args)

        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
            text=True,
            capture_output=False,
        )

        if result.returncode == 0:
            print_success(f"{step_name} completed successfully")
            return True

    except subprocess.CalledProcessError as e:
        print_error(f"{step_name} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"{step_name} encountered an error: {e}")
        return False

    return False


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the Facial Emotion Recognition pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                              Run all steps
  python run_pipeline.py --skip-explore               Skip data exploration
  python run_pipeline.py --skip-train                 Skip training
  python run_pipeline.py --skip-evaluate              Skip evaluation
  python run_pipeline.py --skip-train --skip-evaluate Skip training and evaluation
        """
    )

    parser.add_argument(
        "--skip-explore",
        action="store_true",
        help="Skip data exploration step"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training step"
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip evaluation step"
    )

    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parent.resolve()

    # Print welcome message
    print(f"\n{Colors.BOLD}{Colors.GREEN}")
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Facial Emotion Recognition - Full Pipeline Runner    ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    # Pre-flight checks
    print_header("Pre-flight Checks")
    print_info(f"Project root: {project_root}")

    if not validate_data_folders(project_root):
        print_error("Pipeline aborted: Missing dataset folders")
        sys.exit(1)

    ensure_results_folder(project_root)

    # Pipeline steps
    steps = [
        ("src.data_preparation", "Data Preparation", False, ["--config", "configs/base.yaml", "--build-processed"]),
        ("src.data_exploration", "Data Exploration", args.skip_explore, None),
        ("src.train", "Model Training", args.skip_train, None),
        ("src.evaluate", "Evaluation", args.skip_evaluate, None),
    ]

    failed_steps = []
    skipped_steps = []

    for module_name, step_name, skip, step_args in steps:
        if skip:
            print_header(f"Step: {step_name}")
            print_warning(f"{step_name} skipped (--skip flag)")
            skipped_steps.append(step_name)
            continue

        success = run_step(project_root, module_name, step_name, step_args)
        if not success:
            failed_steps.append(step_name)
            break  # Stop on first failure

    # Summary
    print_header("Pipeline Summary")

    if failed_steps:
        print_error(f"Pipeline failed at: {failed_steps[0]}")
        print_info("Fix the error and run the pipeline again.")
        sys.exit(1)

    if skipped_steps:
        print_warning(f"Steps skipped: {', '.join(skipped_steps)}")

    print_success("All pipeline steps completed successfully!")
    print_info("Check the results/ folder for model and outputs.")
    print(f"\n{Colors.GREEN}{Colors.BOLD}Pipeline execution successful!{Colors.END}\n")


if __name__ == "__main__":
    main()
