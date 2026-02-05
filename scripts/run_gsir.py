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
import time
from pathlib import Path

from run_colmap import get_colmap_executable
from typing import Optional

from env_config import require_conda_env, INSTALL_DIR
from log_manager import LogCapture
from subprocess_utils import (
    ProcessResult,
    ProcessRunner,
    ProgressTracker,
    create_training_patterns,
)
from validate_gsir import validate_gsir_output


# Default training parameters
DEFAULT_ITERATIONS_STAGE1 = 30000
DEFAULT_ITERATIONS_STAGE2 = 35000
DEFAULT_SH_DEGREE = 3

# Frame skip factors for fallback (try denser first)
FRAME_SKIP_FACTORS = [1, 4, 8, 16]

# Error patterns indicating insufficient camera baseline
BASELINE_ERROR_PATTERNS = [
    "invalid gradient",
    "got [0, 0, 3] but expected shape compatible with [0,",
    "got [0, 16, 3]",
]


def _verify_gsir_module_importable() -> bool:
    """Check if the gs_ir CUDA module is importable.

    Returns:
        True if gs_ir can be imported successfully
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import gs_ir"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_gsir_available() -> tuple[bool, Optional[Path]]:
    """Check if GS-IR is installed and find its location.

    Returns:
        Tuple of (is_available, gsir_path or None)
        The path returned is the GS-IR repo root containing train.py
    """
    gsir_path = None

    gsir_env = os.environ.get("GSIR_PATH")
    if gsir_env:
        candidate = Path(gsir_env)
        if (candidate / "train.py").exists():
            gsir_path = candidate

    if not gsir_path:
        common_paths = [
            INSTALL_DIR / "GS-IR",
            Path.cwd() / "GS-IR",
            Path("/opt/GS-IR"),
        ]
        for path in common_paths:
            if (path / "train.py").exists():
                gsir_path = path
                break

    if not gsir_path:
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
                        gsir_path = parent
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not gsir_path:
        return False, None

    if not _verify_gsir_module_importable():
        print(f"Warning: GS-IR directory found at {gsir_path}", file=sys.stderr)
        print(f"         but gs_ir CUDA module is not importable.", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"To fix, install the gs_ir module:", file=sys.stderr)
        print(f"  cd {gsir_path}/gs-ir && pip install -e .", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"Or re-run the installation wizard:", file=sys.stderr)
        print(f"  python scripts/install_wizard/wizard.py", file=sys.stderr)
        return False, None

    return True, gsir_path


def is_baseline_error(error_output: str) -> bool:
    """Check if error output indicates insufficient camera baseline.

    This error occurs when cameras are too close together (e.g., slow-moving
    shots) causing all Gaussian points to be pruned during training.

    Args:
        error_output: stderr/stdout from GS-IR training

    Returns:
        True if the error is due to insufficient baseline
    """
    error_lower = error_output.lower()
    return any(pattern.lower() in error_lower for pattern in BASELINE_ERROR_PATTERNS)


def get_frame_subset_dir(project_dir: Path, skip_factor: int) -> Path:
    """Get the frame directory for a given skip factor.

    Args:
        project_dir: Project directory
        skip_factor: Frame skip factor (1 = all frames, 4 = every 4th, etc.)

    Returns:
        Path to frame directory
    """
    if skip_factor == 1:
        return project_dir / "source" / "frames"
    return project_dir / "source" / f"frames_{skip_factor}s"


def get_colmap_sparse_dir(project_dir: Path, skip_factor: int) -> Path:
    """Get the COLMAP sparse directory for a given skip factor.

    Args:
        project_dir: Project directory
        skip_factor: Frame skip factor

    Returns:
        Path to COLMAP sparse directory
    """
    if skip_factor == 1:
        return project_dir / "colmap" / "sparse" / "0"
    return project_dir / "colmap" / f"sparse_{skip_factor}s" / "0"


def cleanup_colmap_subset(project_dir: Path, skip_factor: int) -> None:
    """Clean up COLMAP and GS-IR files for a failed subset reconstruction.

    Removes all artifacts created for the given skip factor:
    - database_Xs.db
    - sparse_Xs/
    - undistorted_Xs/
    - gsir/data_Xs/

    Does nothing for skip_factor=1 (the original reconstruction).

    Args:
        project_dir: Project directory
        skip_factor: Frame skip factor
    """
    if skip_factor == 1:
        return

    colmap_base = project_dir / "colmap"
    database_path = colmap_base / f"database_{skip_factor}s.db"
    sparse_path = colmap_base / f"sparse_{skip_factor}s"
    undistorted_path = colmap_base / f"undistorted_{skip_factor}s"
    gsir_data_path = project_dir / "gsir" / f"data_{skip_factor}s"

    if database_path.exists():
        database_path.unlink()
    if sparse_path.exists():
        shutil.rmtree(sparse_path)
    if undistorted_path.exists():
        shutil.rmtree(undistorted_path)
    if gsir_data_path.exists():
        shutil.rmtree(gsir_data_path)


def run_colmap_on_subset(project_dir: Path, skip_factor: int) -> bool:
    """Run COLMAP reconstruction on a frame subset.

    Runs COLMAP in completely isolated directories without touching the
    original 1s reconstruction. Creates separate database and sparse
    model directories for each skip factor.

    Args:
        project_dir: Project directory
        skip_factor: Frame skip factor

    Returns:
        True if COLMAP succeeded
    """
    from run_colmap import (
        extract_features,
        match_features,
        run_sparse_reconstruction,
        VirtualDisplay,
        QUALITY_PRESETS,
    )

    frames_dir = get_frame_subset_dir(project_dir, skip_factor)
    if not frames_dir.exists():
        print(f"    Error: Frame subset not found: {frames_dir}", file=sys.stderr)
        return False

    frame_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("*.jpg")))
    if frame_count == 0:
        print(f"    Error: No images found in {frames_dir}", file=sys.stderr)
        return False

    print(f"  → Running COLMAP on {skip_factor}s subset ({frame_count} frames)")

    colmap_base = project_dir / "colmap"
    database_path = colmap_base / f"database_{skip_factor}s.db"
    sparse_path = colmap_base / f"sparse_{skip_factor}s"

    preset = QUALITY_PRESETS["medium"]

    if database_path.exists():
        database_path.unlink()
    if sparse_path.exists():
        shutil.rmtree(sparse_path)

    with VirtualDisplay():
        try:
            print(f"    [1/3] Feature Extraction ({skip_factor}s)")
            extract_features(
                database_path=database_path,
                image_path=frames_dir,
                camera_model="OPENCV",
                max_features=preset["sift_max_features"],
                single_camera=True,
                mask_path=None,
                use_gpu=True,
            )

            print(f"    [2/3] Feature Matching ({skip_factor}s)")
            match_features(
                database_path=database_path,
                matcher_type=preset["matcher"],
                sequential_overlap=preset.get("sequential_overlap", 10),
                use_gpu=True,
            )

            print(f"    [3/3] Sparse Reconstruction ({skip_factor}s)")
            success = run_sparse_reconstruction(
                database_path=database_path,
                image_path=frames_dir,
                output_path=sparse_path,
                refine_focal=preset["ba_refine_focal"],
                min_tri_angle=preset.get("min_tri_angle", 1.5),
                min_num_inliers=preset.get("min_num_inliers", 15),
            )

            if success:
                print(f"    COLMAP succeeded for {skip_factor}s sampling")
                return True
            else:
                print(f"    COLMAP reconstruction failed for {skip_factor}s sampling")
                return False

        except subprocess.CalledProcessError as e:
            print(f"    COLMAP command failed: {e}", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print(f"    COLMAP command timed out", file=sys.stderr)
            return False


def run_colmap_undistorter(
    sparse_model_path: Path,
    output_dir: Path,
    image_path: Path
) -> bool:
    """Run COLMAP image_undistorter to convert to PINHOLE camera model.

    GS-IR only supports undistorted datasets with PINHOLE or SIMPLE_PINHOLE
    camera models. This function converts distorted COLMAP reconstructions.

    Args:
        sparse_model_path: Path to COLMAP sparse model directory (e.g., colmap/sparse/0/)
        output_dir: Path for undistorted output
        image_path: Path to original images

    Returns:
        True if undistortion succeeded
    """
    colmap_exe = get_colmap_executable()
    if not colmap_exe:
        print(f"    Error: COLMAP not found. Install via: conda create -n colmap -c conda-forge colmap", file=sys.stderr)
        return False

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        colmap_exe, "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(sparse_model_path),
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
        print(f"    Error: COLMAP executable not found at: {colmap_exe}", file=sys.stderr)
        return False


def setup_gsir_data_structure(
    project_dir: Path,
    gsir_data_dir: Path,
    skip_factor: int = 1
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
        skip_factor: Frame skip factor (1 = all frames, 4 = every 4th, etc.)

    Returns:
        True if setup succeeded
    """
    gsir_data_dir.mkdir(parents=True, exist_ok=True)

    colmap_dir = project_dir / "colmap"
    colmap_sparse_model = get_colmap_sparse_dir(project_dir, skip_factor)
    source_frames = get_frame_subset_dir(project_dir, skip_factor)
    undistorted_dir = colmap_dir / f"undistorted_{skip_factor}s" if skip_factor > 1 else colmap_dir / "undistorted"

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

    # Count images in source COLMAP model
    def count_colmap_images(sparse_model: Path) -> int:
        """Count registered images in a COLMAP sparse model."""
        images_bin = sparse_model / "images.bin"
        if images_bin.exists():
            try:
                import struct
                with open(images_bin, "rb") as f:
                    return struct.unpack("<Q", f.read(8))[0]
            except (IOError, struct.error):
                pass
        return 0

    source_image_count = count_colmap_images(colmap_sparse_model)
    undistorted_image_count = count_colmap_images(use_sparse_model)

    # Check if undistorted data exists with correct structure AND matching image count
    needs_undistort = (
        not use_sparse_model.exists() or
        not use_images.exists() or
        not (use_sparse_model / "cameras.bin").exists() or
        undistorted_image_count != source_image_count
    )

    if needs_undistort:
        if undistorted_image_count > 0 and undistorted_image_count != source_image_count:
            print(f"    Undistorted data is stale ({undistorted_image_count} vs {source_image_count} images), regenerating...")
        else:
            print(f"    Undistorting images for GS-IR (PINHOLE camera required)...")
        if not run_colmap_undistorter(colmap_sparse_model, undistorted_dir, source_frames):
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
    timeout: int = 7200,
) -> ProcessResult:
    """Run a GS-IR script with the given arguments.

    Args:
        gsir_path: Path to GS-IR installation
        script: Script name (e.g., 'train.py')
        args: Dictionary of argument name -> value
        description: Human-readable description
        timeout: Timeout in seconds

    Returns:
        ProcessResult with captured output
    """
    script_path = gsir_path / script

    args_list = []
    for key, value in args.items():
        if value is True:
            args_list.append(f"--{key}")
        elif value is not False and value is not None:
            args_list.extend([f"--{key}" if not key.startswith("-") else key, str(value)])

    args_str = " ".join(f'"{a}"' if " " in a else a for a in args_list)
    shell_cmd = f'cd "{gsir_path}" && PYTHONPATH="{gsir_path}" "{sys.executable}" "{script_path}" {args_str}'

    tracker = ProgressTracker(
        patterns=create_training_patterns(),
        throttle_interval=2.0,
        min_total=100,
        report_interval=2500,
    )
    runner = ProcessRunner(progress_tracker=tracker, shell=True)

    return runner.run([shell_cmd], description=description, timeout=timeout)


def run_gsir_training(
    gsir_path: Path,
    source_path: Path,
    model_path: Path,
    iterations_stage1: int = DEFAULT_ITERATIONS_STAGE1,
    iterations_stage2: int = DEFAULT_ITERATIONS_STAGE2,
    sh_degree: int = DEFAULT_SH_DEGREE,
    eval_mode: bool = True
) -> tuple[bool, str]:
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
        Tuple of (success, error_output) where error_output contains
        stderr if training failed
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

        result = run_gsir_command(
            gsir_path, "train.py", args,
            f"Training stage 1 ({iterations_stage1} iterations)"
        )
        if result.returncode != 0:
            return False, result.stderr + result.stdout

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

        result = run_gsir_command(
            gsir_path, "baking.py", args,
            "Baking occlusions for indirect lighting"
        )
        if result.returncode != 0:
            return False, result.stderr + result.stdout
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

        result = run_gsir_command(
            gsir_path, "train.py", args,
            f"Training stage 2 with PBR ({iterations_stage2} iterations)"
        )
        if result.returncode != 0:
            return False, result.stderr + result.stdout

    return checkpoint_stage2.exists(), ""


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

    # Run render with PBR enabled to export material maps
    # Skip test cameras - COLMAP datasets typically only have training cameras
    # Skip brdf_eval - requires synthetic data with ground truth albedo
    args = {
        "-m": str(model_path),
        "-s": str(source_path),
        "checkpoint": str(checkpoint),
        "pbr": True,
        "skip_test": True,
    }

    run_gsir_command(
        gsir_path, "render.py", args,
        "Rendering PBR material maps"
    )

    # Find the output directory created by GS-IR
    # Format: {model_path}/train/ours_{iteration}/ (COLMAP uses train split only)
    # Note: GS-IR may output to ours_None if loaded_iter isn't set
    iteration = int(checkpoint.stem.replace("chkpnt", ""))
    candidate_dirs = [
        model_path / "train" / f"ours_{iteration}",
        model_path / "train" / "ours_None",
        model_path / "test" / f"ours_{iteration}",
        model_path / "test" / "ours_None",
    ]

    render_output = None
    for candidate in candidate_dirs:
        if candidate.exists():
            render_output = candidate
            break

    if render_output is None:
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
    with open(output_dir / "gsir_metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return True


def run_gsir_pipeline(
    project_dir: Path,
    iterations_stage1: int = DEFAULT_ITERATIONS_STAGE1,
    iterations_stage2: int = DEFAULT_ITERATIONS_STAGE2,
    skip_training: bool = False,
    gsir_path: Optional[Path] = None,
    validate: bool = True,
    export_usd: bool = False,
) -> bool:
    """Run the complete GS-IR material decomposition pipeline.

    Args:
        project_dir: Project directory with COLMAP output
        iterations_stage1: Training iterations for stage 1
        iterations_stage2: Total training iterations
        skip_training: Skip training if checkpoint exists
        gsir_path: Path to GS-IR installation (auto-detect if None)
        validate: Run output validation after export
        export_usd: Export materials to USD format using Blender

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

    pipeline_start = time.time()

    print(f"\n{'='*60}")
    print(f"GS-IR Material Decomposition")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")
    print(f"GS-IR: {gsir_path}")
    print(f"Iterations: {iterations_stage1} (stage1) → {iterations_stage2} (stage2)")
    print()

    # Setup paths
    materials_output = project_dir / "camera"
    successful_skip = None

    # Try training with fallback to sparser sampling
    for skip_factor in FRAME_SKIP_FACTORS:
        gsir_data_dir = project_dir / "gsir" / f"data_{skip_factor}s" if skip_factor > 1 else project_dir / "gsir" / "data"
        gsir_model_dir = project_dir / "gsir" / f"model_{skip_factor}s" if skip_factor > 1 else project_dir / "gsir" / "model"

        # Check if COLMAP exists for this subset
        colmap_sparse = get_colmap_sparse_dir(project_dir, skip_factor)
        frames_dir = get_frame_subset_dir(project_dir, skip_factor)

        if not frames_dir.exists():
            print(f"  → Frame subset {skip_factor}s not found, skipping")
            continue

        # Run COLMAP on subset if needed (skip_factor > 1 and no sparse model)
        if skip_factor > 1 and not colmap_sparse.exists():
            print(f"\n[Fallback] Attempting with {skip_factor}s frame sampling")
            if not run_colmap_on_subset(project_dir, skip_factor):
                print(f"  → COLMAP failed on {skip_factor}s subset, trying next")
                cleanup_colmap_subset(project_dir, skip_factor)
                continue

        if not colmap_sparse.exists():
            print(f"  → COLMAP sparse model not found for {skip_factor}s, skipping")
            continue

        # Setup data structure for this subset
        print(f"\n[Setup] Preparing data structure ({skip_factor}s sampling)")
        if not setup_gsir_data_structure(project_dir, gsir_data_dir, skip_factor=skip_factor):
            cleanup_colmap_subset(project_dir, skip_factor)
            continue

        checkpoint = gsir_model_dir / f"chkpnt{iterations_stage2}.pth"

        if skip_training and checkpoint.exists():
            print("\n[Training] Skipping (checkpoint exists)")
            successful_skip = skip_factor
            break

        try:
            success, error_output = run_gsir_training(
                gsir_path=gsir_path,
                source_path=gsir_data_dir,
                model_path=gsir_model_dir,
                iterations_stage1=iterations_stage1,
                iterations_stage2=iterations_stage2,
            )

            if success:
                successful_skip = skip_factor
                print(f"  → GS-IR training succeeded with {skip_factor}s sampling")
                break

            if is_baseline_error(error_output):
                print(f"  → Insufficient camera baseline with {skip_factor}s sampling")
                print(f"  → Trying sparser frame sampling...")
                # Clean up failed GS-IR model and COLMAP subset
                if gsir_model_dir.exists():
                    shutil.rmtree(gsir_model_dir)
                cleanup_colmap_subset(project_dir, skip_factor)
                continue
            else:
                print(f"GS-IR training failed (non-baseline error)", file=sys.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"\nGS-IR command failed: {e}", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print(f"\nGS-IR command timed out", file=sys.stderr)
            return False

    if successful_skip is None:
        print("GS-IR training failed with all frame sampling rates", file=sys.stderr)
        return False

    # Use the successful model and data directories
    gsir_data_dir = project_dir / "gsir" / f"data_{successful_skip}s" if successful_skip > 1 else project_dir / "gsir" / "data"
    gsir_model_dir = project_dir / "gsir" / f"model_{successful_skip}s" if successful_skip > 1 else project_dir / "gsir" / "model"
    checkpoint = gsir_model_dir / f"chkpnt{iterations_stage2}.pth"

    # Record which sampling rate succeeded
    skip_marker = project_dir / "gsir" / "selected_skip.txt"
    skip_marker.write_text(f"{successful_skip}s\n")

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

    # Validate outputs
    validation_passed = True
    if validate:
        print("\n[Validation] Checking output quality")
        report = validate_gsir_output(project_dir)
        if report.valid:
            print("  → All validation checks passed")
        else:
            print("  → Validation issues detected:")
            for warning in report.warnings:
                print(f"      {warning}")
            validation_passed = False

    # Export to USD if requested
    usd_output = None
    if export_usd:
        print("\n[USD Export] Creating USD material file")
        try:
            from blender import export_gsir_materials_to_usd

            usd_output = materials_output / "gsir_materials.usd"
            export_gsir_materials_to_usd(
                camera_dir=materials_output,
                output_path=usd_output,
                material_name="gsir_material",
                create_geometry=True,
                export_textures=True,
            )
            print(f"  → USD exported: {usd_output}")
        except FileNotFoundError as e:
            print(f"  → USD export skipped: {e}")
        except Exception as e:
            print(f"  → USD export failed: {e}")

    # Calculate timing
    pipeline_end = time.time()
    total_seconds = pipeline_end - pipeline_start
    total_minutes = total_seconds / 60

    print(f"\n{'='*60}")
    print(f"GS-IR Material Decomposition Complete")
    print(f"{'='*60}")
    print(f"Model: {gsir_model_dir}")
    print(f"Materials: {materials_output}")
    if usd_output and usd_output.exists():
        print(f"USD: {usd_output}")
    if validate:
        print(f"Validation: {'PASSED' if validation_passed else 'ISSUES DETECTED'}")
    print()
    print(f"TOTAL TIME: {total_minutes:.1f} minutes")
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
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip output validation after export"
    )
    parser.add_argument(
        "--export-usd",
        action="store_true",
        help="Export materials to USD format using Blender"
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
        validate=not args.skip_validation,
        export_usd=args.export_usd,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
