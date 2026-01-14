#!/usr/bin/env python3
"""Human motion capture with SMPL-X topology.

Reconstructs people from monocular video with:
- SMPL-X body model (standard topology + UVs)
- World-space motion tracking via WHAM

Pipeline:
1. Motion tracking (WHAM) → skeleton animation in world space

Usage:
    python run_mocap.py <project_dir> [options]

Example:
    python run_mocap.py /path/to/projects/My_Shot
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Environment check and configuration
from env_config import require_conda_env, INSTALL_DIR


# Dependencies check results (cached)
_DEPS_CHECKED = {}


def check_dependency(name: str, import_path: str = None, command: str = None) -> bool:
    """Check if a dependency is available.

    Args:
        name: Dependency name for caching
        import_path: Python import path (e.g., "smplx")
        command: Shell command to check (e.g., ["wham", "--help"])

    Returns:
        True if dependency is available
    """
    if name in _DEPS_CHECKED:
        return _DEPS_CHECKED[name]

    available = False

    if import_path:
        try:
            __import__(import_path)
            available = True
        except ImportError:
            available = False

    elif command:
        try:
            result = subprocess.run(
                command if isinstance(command, list) else [command],
                capture_output=True,
                timeout=5
            )
            available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available = False

    _DEPS_CHECKED[name] = available
    return available


def check_all_dependencies() -> Dict[str, bool]:
    """Check all required dependencies.

    Returns:
        Dict mapping dependency name to availability status
    """
    deps = {
        "numpy": check_dependency("numpy", "numpy"),
        "pytorch": check_dependency("pytorch", "torch"),
        "smplx": check_dependency("smplx", "smplx"),
        "trimesh": check_dependency("trimesh", "trimesh"),
        "opencv": check_dependency("opencv", "cv2"),
        "pillow": check_dependency("pillow", "PIL"),
    }

    # Optional dependencies (for specific methods)
    deps["wham"] = check_dependency("wham", command=["python", "-c", "import wham"])

    return deps


def print_dependency_status():
    """Print status of all dependencies."""
    deps = check_all_dependencies()

    print("\nDependency Status:")
    print("=" * 60)

    # Core dependencies
    print("\nCore (required):")
    for name in ["numpy", "pytorch", "smplx", "trimesh", "opencv", "pillow"]:
        status = "✓" if deps[name] else "✗"
        print(f"  {status} {name}")

    # Optional dependencies
    print("\nOptional (for specific methods):")
    for name in ["wham"]:
        status = "✓" if deps[name] else "✗"
        print(f"  {status} {name}")

    print()


def install_instructions():
    """Print installation instructions for missing dependencies."""
    print("\n" + "=" * 60)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 60)

    print("\nCore dependencies:")
    print("  pip install numpy torch smplx trimesh opencv-python pillow")

    print("\nWHAM (world-grounded motion):")
    print("  git clone https://github.com/yohanshin/WHAM.git")
    print("  cd WHAM && pip install -e .")
    print("  # Download checkpoints from project page")

    print("\nSMPL-X body model:")
    print("  1. Register at https://smpl-x.is.tue.mpg.de/")
    print(f"  2. Download models → place in {INSTALL_DIR}/smplx_models/")
    print()


def run_wham_motion_tracking(
    project_dir: Path,
    person_mask_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> bool:
    """Run WHAM for world-grounded motion tracking.

    Args:
        project_dir: Project directory containing source/frames/
        person_mask_dir: Optional person segmentation masks
        output_dir: Output directory for results

    Returns:
        True if successful
    """
    if not check_dependency("wham"):
        print("Error: WHAM not available", file=sys.stderr)
        print("Run with --check to see installation instructions", file=sys.stderr)
        return False

    output_dir = output_dir or project_dir / "mocap" / "wham"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}", file=sys.stderr)
        return False

    print(f"\n{'=' * 60}")
    print("WHAM Motion Tracking")
    print("=" * 60)
    print(f"Frames: {frames_dir}")
    if person_mask_dir:
        print(f"Masks: {person_mask_dir}")
    print(f"Output: {output_dir}")
    print()

    try:
        # WHAM command - adjust based on actual WHAM CLI
        cmd = [
            "python", "-m", "wham.run",
            "--input", str(frames_dir),
            "--output", str(output_dir),
        ]

        if person_mask_dir and person_mask_dir.exists():
            cmd.extend(["--mask", str(person_mask_dir)])

        print(f"  → Running WHAM...")
        print(f"    $ {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"Error: WHAM failed", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False

        # Check output files
        motion_file = output_dir / "motion.pkl"
        if not motion_file.exists():
            print(f"Error: WHAM output not found: {motion_file}", file=sys.stderr)
            return False

        print(f"  ✓ Motion tracking complete")
        print(f"    Output: {motion_file}")

        return True

    except subprocess.TimeoutExpired:
        print("Error: WHAM timed out", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error running WHAM: {e}", file=sys.stderr)
        return False


def run_mocap_pipeline(
    project_dir: Path,
    skip_texture: bool = False,
) -> bool:
    """Run motion capture pipeline (WHAM).

    Args:
        project_dir: Project directory
        skip_texture: Skip texture projection (unused, kept for API compatibility)

    Returns:
        True if successful
    """
    # Check dependencies
    deps = check_all_dependencies()
    required = ["numpy", "pytorch", "smplx", "trimesh"]
    missing = [name for name in required if not deps[name]]

    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}", file=sys.stderr)
        print_dependency_status()
        install_instructions()
        return False

    print(f"\n{'=' * 60}")
    print("Motion Capture Pipeline (WHAM)")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print()

    # Setup paths
    mocap_dir = project_dir / "mocap"
    mocap_dir.mkdir(parents=True, exist_ok=True)

    person_mask_dir = project_dir / "roto"
    if not person_mask_dir.exists():
        person_mask_dir = None

    # Run WHAM motion tracking
    if not run_wham_motion_tracking(
        project_dir,
        person_mask_dir=person_mask_dir,
        output_dir=mocap_dir / "wham"
    ):
        print("Motion tracking failed", file=sys.stderr)
        return False

    # Final output summary
    motion_file = mocap_dir / "wham" / "motion.pkl"
    print(f"\n{'=' * 60}")
    print("Motion Capture Complete")
    print("=" * 60)
    print(f"Output directory: {mocap_dir}")
    print(f"Motion data: {motion_file}")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Human motion capture with SMPL-X topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        nargs="?",
        help="Project directory containing source/frames/"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and exit"
    )
    parser.add_argument(
        "--skip-texture",
        action="store_true",
        help="Skip texture projection (unused, kept for compatibility)"
    )

    args = parser.parse_args()

    if args.check:
        print_dependency_status()
        install_instructions()
        sys.exit(0)

    # Require correct conda environment for actual processing
    require_conda_env()

    if not args.project_dir:
        parser.print_help()
        print("\nError: project_dir required (or use --check)", file=sys.stderr)
        sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_mocap_pipeline(
        project_dir=project_dir,
        skip_texture=args.skip_texture,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
