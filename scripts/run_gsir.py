#!/usr/bin/env python3
"""GS-IR (Gaussian Splatting Inverse Rendering) wrapper for material decomposition.

Runs GS-IR on a COLMAP-reconstructed scene to extract:
  - Albedo maps
  - Roughness maps
  - Metallic maps
  - Normal maps
  - Environment lighting

Requires:
  - GS-IR installed (https://github.com/lzhnb/GS-IR)
  - COLMAP reconstruction completed (sparse model)
  - CUDA-capable GPU with sufficient VRAM (12GB+ recommended)

Usage:
    python run_gsir.py <project_dir> [options]

Example:
    python run_gsir.py /path/to/projects/My_Shot --iterations 30000
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Environment check and configuration
from env_config import require_conda_env, INSTALL_DIR

# Log capture for debugging
from log_manager import LogCapture


# Default training parameters
DEFAULT_ITERATIONS_STAGE1 = 30000
DEFAULT_ITERATIONS_STAGE2 = 35000
DEFAULT_SH_DEGREE = 3


def check_gsir_available() -> tuple[bool, Optional[Path]]:
    """Check if GS-IR is installed and find its location.

    Returns:
        Tuple of (is_available, gsir_path or None)
        The path returned is the GS-IR repo root containing train.py
    """
    # Check GSIR_PATH environment variable first (most reliable)
    gsir_env = os.environ.get("GSIR_PATH")
    if gsir_env:
        gsir_path = Path(gsir_env)
        if (gsir_path / "train.py").exists():
            return True, gsir_path

    # Check common installation locations
    common_paths = [
        INSTALL_DIR / "GS-IR",
        Path.cwd() / "GS-IR",
        Path("/opt/GS-IR"),
    ]

    for path in common_paths:
        if (path / "train.py").exists():
            return True, path

    # Check if gsir module is importable and find repo root from module path
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import gs_ir; print(gs_ir.__file__)"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            module_path = Path(result.stdout.strip()).parent
            for parent in [module_path] + list(module_path.parents):
                if (parent / "train.py").exists():
                    return True, parent
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return False, None


def run_colmap_undistorter(
    colmap_dir: Path,
    output_dir: Path,
    image_path: Path
) -> bool:
    """Run COLMAP image_undistorter to convert to PINHOLE camera model.

    GS-IR only supports undistorted datasets with PINHOLE or SIMPLE_PINHOLE
    camera models. This function converts distorted COLMAP reconstructions.

    Args:
        colmap_dir: Path to COLMAP reconstruction (containing sparse/0/)
        output_dir: Path for undistorted output
        image_path: Path to original images

    Returns:
        True if undistortion succeeded
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "colmap", "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(colmap_dir / "sparse" / "0"),
        "--output_path", str(output_dir),
        "--output_type", "COLMAP",
    ]

    print(f"    Running COLMAP image_undistorter...")
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=1800
        )
        if result.returncode != 0:
            print(f"    Error: image_undistorter failed: {result.stderr}", file=sys.stderr)
            return False

        # COLMAP outputs to sparse/ directly, but GS-IR expects sparse/0/
        # Move files to create the expected structure
        sparse_dir = output_dir / "sparse"
        sparse_0_dir = sparse_dir / "0"
        if sparse_dir.exists() and not sparse_0_dir.exists():
            temp_dir = output_dir / "sparse_temp"
            sparse_dir.rename(temp_dir)
            sparse_dir.mkdir()
            temp_dir.rename(sparse_0_dir)
            print(f"    Reorganized sparse model to sparse/0/ structure")

        return True
    except subprocess.TimeoutExpired:
        print(f"    Error: image_undistorter timed out", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"    Error: colmap command not found", file=sys.stderr)
        return False


def setup_gsir_data_structure(
    project_dir: Path,
    gsir_data_dir: Path
) -> bool:
    """Set up the data structure expected by GS-IR.

    GS-IR expects:
      - source_path/sparse/0/ - COLMAP sparse reconstruction (PINHOLE cameras)
      - source_path/images/ - Undistorted input images

    If the COLMAP reconstruction uses distorted camera models, this function
    will run image_undistorter to convert to PINHOLE format.

    Args:
        project_dir: Our project directory
        gsir_data_dir: Directory to set up for GS-IR

    Returns:
        True if setup succeeded
    """
    gsir_data_dir.mkdir(parents=True, exist_ok=True)

    colmap_dir = project_dir / "colmap"
    colmap_sparse_model = colmap_dir / "sparse" / "0"
    source_frames = project_dir / "source" / "frames"
    undistorted_dir = colmap_dir / "undistorted"

    if not colmap_sparse_model.exists():
        print(f"    Error: COLMAP sparse model not found: {colmap_sparse_model}", file=sys.stderr)
        print(f"    Run the colmap stage first", file=sys.stderr)
        return False

    if not source_frames.exists():
        print(f"    Error: Source frames not found: {source_frames}", file=sys.stderr)
        return False

    # GS-IR expects sparse/0/ structure
    use_sparse_parent = undistorted_dir / "sparse"
    use_sparse_model = use_sparse_parent / "0"
    use_images = undistorted_dir / "images"

    # Check if undistorted data exists with correct structure
    needs_undistort = (
        not use_sparse_model.exists() or
        not use_images.exists() or
        not (use_sparse_model / "cameras.bin").exists()
    )

    if needs_undistort:
        print(f"    Undistorting images for GS-IR (PINHOLE camera required)...")
        if not run_colmap_undistorter(colmap_dir, undistorted_dir, source_frames):
            print(f"    Error: Failed to undistort images", file=sys.stderr)
            return False

    if not use_sparse_model.exists():
        print(f"    Error: Undistorted sparse model not found: {use_sparse_model}", file=sys.stderr)
        return False

    sparse_link = gsir_data_dir / "sparse"
    images_link = gsir_data_dir / "images"

    for link in [sparse_link, images_link]:
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(link)

    try:
        # Link to sparse parent so sparse/0/ is accessible
        sparse_link.symlink_to(use_sparse_parent.resolve())
        images_link.symlink_to(use_images.resolve())
        print(f"    Created symlinks for GS-IR data structure")
    except OSError as e:
        print(f"    Warning: Could not create symlinks ({e}), copying data...")
        shutil.copytree(use_sparse_parent, sparse_link)
        shutil.copytree(use_images, images_link)

    return True


def run_gsir_command(
    gsir_path: Path,
    script: str,
    args: dict,
    description: str,
    timeout: int = 7200  # 2 hours default
) -> subprocess.CompletedProcess:
    """Run a GS-IR script with the given arguments.

    Args:
        gsir_path: Path to GS-IR installation
        script: Script name (e.g., 'train.py')
        args: Dictionary of argument name -> value
        description: Human-readable description
        timeout: Timeout in seconds

    Returns:
        CompletedProcess result
    """
    script_path = gsir_path / script
    cmd = [sys.executable, str(script_path)]

    for key, value in args.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value is not False and value is not None:
            cmd.extend([f"--{key}" if not key.startswith("-") else key, str(value)])

    import re
    print(f"  → {description}")
    print(f"    $ {' '.join(cmd)}")

    # Stream output to show training progress
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(gsir_path)  # Run from GS-IR directory
    )

    stdout_lines = []
    # Pattern: iteration progress (various formats)
    iter_pattern = re.compile(r'[Ii]teration\s*[:\s]*(\d+)\s*[/|of]\s*(\d+)')
    last_reported = 0
    report_interval = 500  # Report every 500 iterations

    # Use iter(readline, '') to avoid Python's internal buffering
    for line in iter(process.stdout.readline, ''):
        stdout_lines.append(line)
        line = line.strip()

        # Check for iteration progress
        match = iter_pattern.search(line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if current - last_reported >= report_interval or current == total:
                print(f"    Iteration {current}/{total}")
                sys.stdout.flush()
                last_reported = current

    process.wait()

    stdout = ''.join(stdout_lines)
    if process.returncode != 0:
        print(f"    Error: {stdout[:500]}", file=sys.stderr)
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, "")

    # Return a result-like object
    class Result:
        def __init__(self):
            self.returncode = process.returncode
            self.stdout = stdout
            self.stderr = ""
    return Result()


def run_gsir_training(
    gsir_path: Path,
    source_path: Path,
    model_path: Path,
    iterations_stage1: int = DEFAULT_ITERATIONS_STAGE1,
    iterations_stage2: int = DEFAULT_ITERATIONS_STAGE2,
    sh_degree: int = DEFAULT_SH_DEGREE,
    eval_mode: bool = True
) -> bool:
    """Run the two-stage GS-IR training process.

    Stage 1: Initial Gaussian Splatting reconstruction
    Baking: Cache occlusions for indirect lighting
    Stage 2: Material decomposition with PBR

    Args:
        gsir_path: Path to GS-IR installation
        source_path: Path to prepared data (with sparse/ and images/)
        model_path: Output directory for trained model
        iterations_stage1: Iterations for stage 1
        iterations_stage2: Total iterations (stage 1 + stage 2)
        sh_degree: Spherical harmonics degree
        eval_mode: Enable evaluation mode

    Returns:
        True if training succeeded
    """
    model_path.mkdir(parents=True, exist_ok=True)

    checkpoint_stage1 = model_path / f"chkpnt{iterations_stage1}.pth"
    checkpoint_stage2 = model_path / f"chkpnt{iterations_stage2}.pth"

    # Stage 1: Initial reconstruction
    print("\n[GS-IR Stage 1: Gaussian Splatting Reconstruction]")
    if checkpoint_stage1.exists():
        print(f"  → Skipping (checkpoint exists: {checkpoint_stage1.name})")
    else:
        args = {
            "-m": str(model_path),
            "-s": str(source_path),
            "iterations": iterations_stage1,
            "sh_degree": sh_degree,
        }
        if eval_mode:
            args["eval"] = True

        run_gsir_command(
            gsir_path, "train.py", args,
            f"Training stage 1 ({iterations_stage1} iterations)"
        )

    # Baking: Cache occlusions
    print("\n[GS-IR Baking: Caching Occlusions]")
    baked_marker = model_path / ".baked"
    if baked_marker.exists():
        print(f"  → Skipping (already baked)")
    else:
        args = {
            "-m": str(model_path),
            "checkpoint": str(checkpoint_stage1),
        }

        run_gsir_command(
            gsir_path, "baking.py", args,
            "Baking occlusions for indirect lighting"
        )
        baked_marker.touch()

    # Stage 2: Material decomposition
    print("\n[GS-IR Stage 2: Material Decomposition]")
    if checkpoint_stage2.exists():
        print(f"  → Skipping (checkpoint exists: {checkpoint_stage2.name})")
    else:
        args = {
            "-m": str(model_path),
            "-s": str(source_path),
            "start_checkpoint": str(checkpoint_stage1),
            "iterations": iterations_stage2,
            "gamma": True,
            "indirect": True,
        }
        if eval_mode:
            args["eval"] = True

        run_gsir_command(
            gsir_path, "train.py", args,
            f"Training stage 2 with PBR ({iterations_stage2} iterations)"
        )

    return checkpoint_stage2.exists()


def export_materials(
    gsir_path: Path,
    source_path: Path,
    model_path: Path,
    output_dir: Path,
    checkpoint: Path
) -> bool:
    """Export material maps from trained GS-IR model.

    Args:
        gsir_path: Path to GS-IR installation
        source_path: Path to data
        model_path: Path to trained model
        output_dir: Output directory for material maps
        checkpoint: Path to checkpoint file

    Returns:
        True if export succeeded
    """
    print("\n[GS-IR Export: Rendering Material Maps]")

    # Run render with PBR and BRDF evaluation
    args = {
        "-m": str(model_path),
        "-s": str(source_path),
        "checkpoint": str(checkpoint),
        "pbr": True,
        "brdf_eval": True,
    }

    run_gsir_command(
        gsir_path, "render.py", args,
        "Rendering PBR material maps"
    )

    # Find the output directory created by GS-IR
    # Format: {model_path}/test/ours_{iteration}/
    iteration = int(checkpoint.stem.replace("chkpnt", ""))
    render_output = model_path / "test" / f"ours_{iteration}"

    if not render_output.exists():
        # Try train split
        render_output = model_path / "train" / f"ours_{iteration}"

    if not render_output.exists():
        print(f"    Warning: Render output not found at expected location", file=sys.stderr)
        return False

    # Copy/organize materials to our output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy PBR outputs
    pbr_dir = render_output / "pbr"
    if pbr_dir.exists():
        materials_dir = output_dir / "materials"
        materials_dir.mkdir(exist_ok=True)

        for f in pbr_dir.glob("*.png"):
            shutil.copy(f, materials_dir / f.name)
        print(f"    Copied PBR materials to {materials_dir}")

    # Copy normals
    normal_dir = render_output / "normal"
    if normal_dir.exists():
        normals_dir = output_dir / "normals"
        normals_dir.mkdir(exist_ok=True)

        for f in normal_dir.glob("*.png"):
            shutil.copy(f, normals_dir / f.name)
        print(f"    Copied normals to {normals_dir}")

    # Copy depth
    depth_dir = render_output / "depth"
    if depth_dir.exists():
        depths_dir = output_dir / "depth_gsir"
        depths_dir.mkdir(exist_ok=True)

        for f in depth_dir.glob("*.png"):
            shutil.copy(f, depths_dir / f.name)
        print(f"    Copied depth maps to {depths_dir}")

    # Copy environment map if exists
    envmap = render_output / "envmap.png"
    if envmap.exists():
        shutil.copy(envmap, output_dir / "environment.png")
        print(f"    Copied environment map")

    # Create metadata
    metadata = {
        "source": "gs-ir",
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "outputs": {
            "materials": "materials/",
            "normals": "normals/",
            "depth": "depth_gsir/",
            "environment": "environment.png" if envmap.exists() else None,
        }
    }
    with open(output_dir / "gsir_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return True


def run_gsir_pipeline(
    project_dir: Path,
    iterations_stage1: int = DEFAULT_ITERATIONS_STAGE1,
    iterations_stage2: int = DEFAULT_ITERATIONS_STAGE2,
    skip_training: bool = False,
    gsir_path: Optional[Path] = None
) -> bool:
    """Run the complete GS-IR material decomposition pipeline.

    Args:
        project_dir: Project directory with COLMAP output
        iterations_stage1: Training iterations for stage 1
        iterations_stage2: Total training iterations
        skip_training: Skip training if checkpoint exists
        gsir_path: Path to GS-IR installation (auto-detect if None)

    Returns:
        True if pipeline succeeded
    """
    # Check GS-IR availability
    if gsir_path:
        available = (gsir_path / "train.py").exists()
    else:
        available, gsir_path = check_gsir_available()

    if not available or gsir_path is None:
        print("Error: GS-IR not found. Install from:", file=sys.stderr)
        print("  https://github.com/lzhnb/GS-IR", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or set GSIR_PATH environment variable to installation directory", file=sys.stderr)
        return False

    print(f"\n{'='*60}")
    print(f"GS-IR Material Decomposition")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")
    print(f"GS-IR: {gsir_path}")
    print(f"Iterations: {iterations_stage1} (stage1) → {iterations_stage2} (stage2)")
    print()

    # Setup paths
    gsir_data_dir = project_dir / "gsir" / "data"
    gsir_model_dir = project_dir / "gsir" / "model"
    materials_output = project_dir / "camera"  # Output alongside camera data

    # Setup data structure
    print("[Setup] Preparing data structure")
    if not setup_gsir_data_structure(project_dir, gsir_data_dir):
        return False

    # Run training
    try:
        checkpoint = gsir_model_dir / f"chkpnt{iterations_stage2}.pth"

        if skip_training and checkpoint.exists():
            print("\n[Training] Skipping (checkpoint exists)")
        else:
            if not run_gsir_training(
                gsir_path=gsir_path,
                source_path=gsir_data_dir,
                model_path=gsir_model_dir,
                iterations_stage1=iterations_stage1,
                iterations_stage2=iterations_stage2,
            ):
                print("GS-IR training failed", file=sys.stderr)
                return False

        # Export materials
        if not export_materials(
            gsir_path=gsir_path,
            source_path=gsir_data_dir,
            model_path=gsir_model_dir,
            output_dir=materials_output,
            checkpoint=checkpoint,
        ):
            print("GS-IR export failed", file=sys.stderr)
            return False

    except subprocess.CalledProcessError as e:
        print(f"\nGS-IR command failed: {e}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"\nGS-IR command timed out", file=sys.stderr)
        return False

    print(f"\n{'='*60}")
    print(f"GS-IR Material Decomposition Complete")
    print(f"{'='*60}")
    print(f"Model: {gsir_model_dir}")
    print(f"Materials: {materials_output}")
    print()

    return True


def main():
    # Require correct conda environment for GS-IR (needs CUDA packages)
    require_conda_env()

    parser = argparse.ArgumentParser(
        description="Run GS-IR material decomposition on a COLMAP reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory with COLMAP output"
    )
    parser.add_argument(
        "--iterations-stage1",
        type=int,
        default=DEFAULT_ITERATIONS_STAGE1,
        help=f"Training iterations for stage 1 (default: {DEFAULT_ITERATIONS_STAGE1})"
    )
    parser.add_argument(
        "--iterations-stage2",
        type=int,
        default=DEFAULT_ITERATIONS_STAGE2,
        help=f"Total training iterations (default: {DEFAULT_ITERATIONS_STAGE2})"
    )
    parser.add_argument(
        "--gsir-path",
        type=Path,
        default=None,
        help="Path to GS-IR installation (default: auto-detect)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training if checkpoint exists"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if GS-IR is available and exit"
    )

    args = parser.parse_args()

    if args.check:
        available, path = check_gsir_available()
        if available:
            print(f"GS-IR is available at: {path}")
            sys.exit(0)
        else:
            print("GS-IR is not available")
            sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_gsir_pipeline(
        project_dir=project_dir,
        iterations_stage1=args.iterations_stage1,
        iterations_stage2=args.iterations_stage2,
        skip_training=args.skip_training,
        gsir_path=args.gsir_path,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
