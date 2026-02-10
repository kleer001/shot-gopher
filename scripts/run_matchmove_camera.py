#!/usr/bin/env python3
"""COLMAP reconstruction wrapper for automated SfM/MVS pipeline.

Runs COLMAP Structure-from-Motion and Multi-View Stereo reconstruction
on a frame sequence to produce:
  - Accurate camera poses (intrinsics + extrinsics)
  - Sparse 3D point cloud
  - Dense 3D point cloud (optional)
  - Mesh reconstruction (optional)

Usage:
    python run_colmap.py <project_dir> [options]

Example:
    python run_colmap.py /path/to/projects/My_Shot --dense --mesh
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from env_config import require_conda_env
from log_manager import LogCapture
from install_wizard.platform import PlatformManager
from subprocess_utils import (
    ProcessResult,
    ProcessRunner,
    ProgressTracker,
    create_mmcam_patterns,
)
from transforms import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    slerp,
    cubic_bezier,
)

# Cache for COLMAP executable path
_colmap_path: Optional[str] = None

# Dedicated conda environment for COLMAP (avoids solver conflicts)
COLMAP_CONDA_ENV = "colmap"


def _find_conda_base() -> Optional[Path]:
    """Find the conda base directory."""
    # Check CONDA_EXE first (most reliable)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        # CONDA_EXE is like /home/user/anaconda3/bin/conda
        return Path(conda_exe).parent.parent

    # Check common locations
    for base_name in ["anaconda3", "miniconda3", "miniforge3"]:
        conda_base = Path.home() / base_name
        if conda_base.exists():
            return conda_base

    return None


def _get_conda_colmap_path(env_path: Path) -> Optional[Path]:
    """Get the colmap executable path within a conda environment."""
    if sys.platform == "win32":
        candidates = [
            env_path / "Scripts" / "colmap.exe",
            env_path / "Library" / "bin" / "colmap.exe",
        ]
    else:
        candidates = [env_path / "bin" / "colmap"]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_colmap_executable() -> Optional[str]:
    """Get the path to COLMAP executable.

    Search order:
    1. Dedicated 'colmap' conda environment (recommended)
    2. Active conda environment (CONDA_PREFIX)
    3. PlatformManager.find_tool() (repo-local, system PATH excluding snap)

    Returns:
        Path to COLMAP executable as string, or None if not found
    """
    global _colmap_path
    if _colmap_path is not None:
        return _colmap_path

    # 1. Check dedicated colmap conda environment
    conda_base = _find_conda_base()
    if conda_base:
        dedicated_env = conda_base / "envs" / COLMAP_CONDA_ENV
        colmap_path = _get_conda_colmap_path(dedicated_env)
        if colmap_path:
            _colmap_path = str(colmap_path)
            return _colmap_path

    # 2. Check active conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        colmap_path = _get_conda_colmap_path(Path(conda_prefix))
        if colmap_path:
            _colmap_path = str(colmap_path)
            return _colmap_path

    # 3. Use PlatformManager which skips snap for COLMAP
    found = PlatformManager.find_tool("colmap")
    if found:
        _colmap_path = str(found)
        return _colmap_path

    # Don't fall back to bare "colmap" - it would find snap
    return None


# COLMAP quality presets
QUALITY_PRESETS = {
    "low": {
        "sift_max_features": 4096,
        "matcher": "sequential",
        "sequential_overlap": 10,
        "ba_refine_focal": True,
        "dense_max_size": 1000,
        "min_tri_angle": 1.5,  # Default
    },
    "medium": {
        "sift_max_features": 8192,
        "matcher": "sequential",
        "sequential_overlap": 10,
        "ba_refine_focal": True,
        "dense_max_size": 2000,
        "min_tri_angle": 1.5,
    },
    "high": {
        "sift_max_features": 16384,
        "matcher": "exhaustive",
        "sequential_overlap": 10,
        "ba_refine_focal": True,
        "dense_max_size": -1,  # No limit
        "min_tri_angle": 1.5,
    },
    # For slow/minimal camera motion - more aggressive matching
    "slow": {
        "sift_max_features": 16384,
        "matcher": "exhaustive",  # Check all frame pairs
        "sequential_overlap": 50,  # Higher overlap for sequential fallback
        "ba_refine_focal": True,
        "dense_max_size": 2000,
        "min_tri_angle": 0.5,  # Accept smaller triangulation angles
        "min_num_inliers": 10,  # Lower threshold (default 15)
    },
}


class VirtualDisplay:
    """Context manager that starts Xvfb for headless GPU OpenGL support.

    In headless environments without a display, COLMAP's GPU SIFT extraction
    needs an X server for OpenGL context. This starts a virtual framebuffer.
    """

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._display: Optional[str] = None
        self._env_modified: bool = False

    def _needs_virtual_display(self) -> bool:
        """Check if we need to start a virtual display.

        In headless environments without a display, COLMAP's GPU SIFT extraction
        needs an X server for OpenGL context. This starts a virtual framebuffer.
        """
        return not os.environ.get("DISPLAY")

    def _find_free_display(self) -> Optional[int]:
        """Find an unused display number."""
        for display_num in range(99, 199):
            lock_file = Path(f"/tmp/.X{display_num}-lock")
            socket_file = Path(f"/tmp/.X11-unix/X{display_num}")
            if not lock_file.exists() and not socket_file.exists():
                return display_num
        return None

    def __enter__(self) -> "VirtualDisplay":
        if not self._needs_virtual_display():
            return self

        display_num = self._find_free_display()
        if display_num is None:
            print("    Warning: No free display numbers (99-198), GPU may not work")
            return self

        self._display = f":{display_num}"

        try:
            self._process = subprocess.Popen(
                ["Xvfb", self._display, "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            if self._process.poll() is not None:
                print("    Warning: Xvfb failed to start, GPU may not work")
                self._process = None
            else:
                os.environ["DISPLAY"] = self._display
                self._env_modified = True
                print(f"    Started virtual display {self._display} for GPU OpenGL")
        except FileNotFoundError:
            print("    Warning: Xvfb not found, GPU may not work in headless mode")
            self._process = None

        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> bool:
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except ProcessLookupError:
                pass

        if self._env_modified and "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]

        return False


def check_colmap_available() -> bool:
    """Check if COLMAP is installed and accessible."""
    try:
        colmap_exe = get_colmap_executable()
        if colmap_exe is None:
            return False
        result = subprocess.run(
            [colmap_exe, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=(colmap_exe.lower().endswith('.bat'))
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def diagnose_colmap_environment(verbose: bool = False) -> dict:
    """Diagnose COLMAP installation and GPU support.

    Returns:
        Dict with diagnostic info: version, gpu_available, cuda_available, etc.
    """
    info = {
        "colmap_available": False,
        "gpu_sift_available": False,
        "cuda_available": False,
        "opengl_available": False,
        "display_set": bool(os.environ.get("DISPLAY")),
    }

    try:
        colmap_exe = get_colmap_executable()
        if colmap_exe is None:
            if verbose:
                print("    DIAG: COLMAP executable not found")
            return info

        is_bat = colmap_exe.lower().endswith('.bat')
        result = subprocess.run(
            [colmap_exe, "feature_extractor", "--help"],
            capture_output=True, text=True, timeout=10,
            shell=is_bat
        )
        if result.returncode == 0:
            info["colmap_available"] = True
            output = result.stdout + result.stderr
            info["gpu_sift_available"] = "SiftExtraction.use_gpu" in output

        if verbose:
            print(f"    DIAG: COLMAP path: {colmap_exe}")
            print(f"    DIAG: COLMAP available: {info['colmap_available']}")
            print(f"    DIAG: GPU SIFT option available: {info['gpu_sift_available']}")
            print(f"    DIAG: DISPLAY={os.environ.get('DISPLAY', 'not set')}")

        nvidia_smi = PlatformManager.find_tool("nvidia-smi")
        nvidia_cmd = str(nvidia_smi) if nvidia_smi else "nvidia-smi"
        result = subprocess.run(
            [nvidia_cmd, "-L"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            info["cuda_available"] = True
            if verbose:
                for line in result.stdout.strip().split('\n'):
                    print(f"    DIAG: GPU: {line}")

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        if verbose:
            print(f"    DIAG: Error during diagnosis: {e}")

    return info


def verify_database_has_features(database_path: Path, verbose: bool = False) -> tuple[int, int]:
    """Verify COLMAP database has extracted features.

    Args:
        database_path: Path to COLMAP database.db
        verbose: Print diagnostic information

    Returns:
        Tuple of (num_images_with_features, total_keypoints)
    """
    if not database_path.exists():
        if verbose:
            print(f"    DEBUG: Database does not exist: {database_path}")
        return 0, 0

    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM images")
        num_images = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM keypoints")
        num_keypoint_rows = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(rows) FROM keypoints WHERE rows IS NOT NULL")
        result = cursor.fetchone()[0]
        total_keypoints = result if result else 0

        cursor.execute("SELECT COUNT(*) FROM descriptors")
        num_descriptor_rows = cursor.fetchone()[0]

        if verbose:
            print(f"    DEBUG: Database path: {database_path}")
            print(f"    DEBUG: Database size: {database_path.stat().st_size:,} bytes")
            print(f"    DEBUG: images table: {num_images} rows")
            print(f"    DEBUG: keypoints table: {num_keypoint_rows} rows")
            print(f"    DEBUG: descriptors table: {num_descriptor_rows} rows")
            print(f"    DEBUG: total keypoints (SUM): {total_keypoints}")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"    DEBUG: tables in database: {tables}")

        conn.close()
        return num_keypoint_rows, total_keypoints
    except sqlite3.Error as e:
        print(f"    Warning: Could not read database: {e}")
        return 0, 0


def run_colmap_command(
    command: str,
    args: dict,
    description: str,
    timeout: int = None,
) -> ProcessResult:
    """Run a COLMAP command with streaming output and progress tracking.

    Args:
        command: COLMAP subcommand (e.g., 'feature_extractor')
        args: Dictionary of argument name -> value
        description: Human-readable description for logging
        timeout: Timeout in seconds (None = no timeout)

    Returns:
        ProcessResult with captured output
    """
    colmap_exe = get_colmap_executable()
    cmd = [colmap_exe, command]
    for key, value in args.items():
        if value is True:
            cmd.extend([f"--{key}", "true"])
        elif value is False:
            cmd.extend([f"--{key}", "false"])
        elif value is not None:
            cmd.extend([f"--{key}", str(value)])

    is_bat = colmap_exe.lower().endswith('.bat')
    tracker = ProgressTracker(patterns=create_mmcam_patterns())
    runner = ProcessRunner(
        progress_tracker=tracker,
        shell=is_bat,
    )

    return runner.run(cmd, description=description, timeout=timeout)


def prepare_colmap_masks(
    roto_dir: Path,
    frames_dir: Path,
    colmap_dir: Path,
) -> Optional[Path]:
    """Prepare masks with COLMAP-compatible naming convention.

    COLMAP expects masks to be named {image_filename}.png - so for an image
    named 'frame_0001.png', the mask must be 'frame_0001.png.png'.

    This function maps masks from roto/ to the frames in frames_dir by
    sequence order (not by filename matching).

    Args:
        roto_dir: Directory containing mask images (any naming convention)
        frames_dir: Directory containing source images
        colmap_dir: COLMAP working directory (masks will be placed in colmap/masks/)

    Returns:
        Path to prepared masks directory, or None if no masks available
    """
    if not roto_dir.exists():
        return None

    mask_files = sorted(
        list(roto_dir.glob("*.png")) + list(roto_dir.glob("*.jpg"))
    )
    if not mask_files:
        return None

    frame_files = sorted(
        list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg"))
    )
    if not frame_files:
        return None

    if len(mask_files) != len(frame_files):
        print(f"    Warning: Mask count ({len(mask_files)}) != frame count ({len(frame_files)})")
        print(f"    Masks will be matched by sequence order")

    masks_output_dir = colmap_dir / "masks"
    if masks_output_dir.exists():
        shutil.rmtree(masks_output_dir)
    masks_output_dir.mkdir(parents=True)

    copied_count = 0
    for i, frame_file in enumerate(frame_files):
        if i >= len(mask_files):
            break

        colmap_mask_name = f"{frame_file.name}.png"
        dest_path = masks_output_dir / colmap_mask_name
        shutil.copy2(mask_files[i], dest_path)
        copied_count += 1

    if copied_count > 0:
        print(f"    Prepared {copied_count} masks for COLMAP (renamed to {frame_files[0].name}.png format)")
        return masks_output_dir

    return None


def extract_features(
    database_path: Path,
    image_path: Path,
    camera_model: str = "OPENCV",
    max_features: int = 8192,
    single_camera: bool = True,
    mask_path: Optional[Path] = None,
    use_gpu: bool = True,
    max_image_size: int = -1
) -> None:
    """Extract SIFT features from images.

    Args:
        database_path: Path to COLMAP database
        image_path: Path to image directory
        camera_model: Camera model (OPENCV, PINHOLE, RADIAL, etc.)
        max_features: Maximum features per image
        single_camera: If True, assume all images from same camera
        mask_path: Optional path to mask directory (excludes masked regions from feature extraction)
        use_gpu: Whether to use GPU for SIFT extraction (falls back to CPU on failure)
        max_image_size: Maximum image dimension (downscales if larger, -1 for no limit)
    """
    diag = diagnose_colmap_environment(verbose=False)
    gpu_sift_available = diag.get("gpu_sift_available", False)

    def _build_args(gpu: bool) -> dict:
        """Build argument dict with specified GPU setting."""
        args = {
            "database_path": str(database_path),
            "image_path": str(image_path),
            "ImageReader.camera_model": camera_model,
            "ImageReader.single_camera": 1 if single_camera else 0,
            "SiftExtraction.max_num_features": max_features,
        }
        if max_image_size > 0:
            args["ImageReader.max_image_size"] = max_image_size
        if gpu_sift_available:
            args["SiftExtraction.use_gpu"] = 1 if gpu else 0
            if gpu:
                args["SiftExtraction.gpu_index"] = "0"
        if mask_path and mask_path.exists():
            args["ImageReader.mask_path"] = str(mask_path)
        return args

    if mask_path and mask_path.exists():
        print(f"    Using masks from: {mask_path}")

    def _check_features_extracted() -> bool:
        """Check if any features were actually extracted."""
        if not database_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM keypoints")
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except sqlite3.Error:
            return False

    def _run_extraction_with_fallback() -> str:
        """Run extraction with GPU->CPU fallback. Returns last COLMAP output."""
        last_output = ""

        if use_gpu:
            try:
                args = _build_args(gpu=True)
                result = run_colmap_command("feature_extractor", args, "Extracting features (GPU)")
                last_output = result.stdout
                if not _check_features_extracted():
                    print("    GPU extraction produced no features, retrying with CPU...")
                    if database_path.exists():
                        database_path.unlink()
                    args = _build_args(gpu=False)
                    result = run_colmap_command("feature_extractor", args, "Extracting features (CPU)")
                    last_output = result.stdout
            except subprocess.CalledProcessError as e:
                error_output = str(e.stdout) if hasattr(e, 'stdout') and e.stdout else str(e)
                last_output = error_output
                is_gpu_error = (
                    "context" in error_output.lower()
                    or "opengl" in error_output.lower()
                    or e.returncode < 0
                )
                if is_gpu_error:
                    print("    GPU feature extraction failed (OpenGL context error), falling back to CPU...")
                    if database_path.exists():
                        database_path.unlink()
                    args = _build_args(gpu=False)
                    result = run_colmap_command("feature_extractor", args, "Extracting features (CPU)")
                    last_output = result.stdout
                else:
                    raise
        else:
            args = _build_args(gpu=False)
            result = run_colmap_command("feature_extractor", args, "Extracting features (CPU)")
            last_output = result.stdout

        return last_output

    colmap_output = _run_extraction_with_fallback()

    if not _check_features_extracted():
        print("    WARNING: Feature extraction completed but database is empty!")
        print("    Last 1500 chars of COLMAP output:")
        print("-" * 60)
        print(colmap_output[-1500:] if colmap_output else "(no output captured)")
        print("-" * 60)


def match_features(
    database_path: Path,
    matcher_type: str = "sequential",
    sequential_overlap: int = 10,
    use_gpu: bool = True
) -> None:
    """Match features between images.

    Args:
        database_path: Path to COLMAP database
        matcher_type: 'sequential', 'exhaustive', or 'vocab_tree'
        sequential_overlap: Number of overlapping frames for sequential matcher
        use_gpu: Whether to use GPU for SIFT matching (falls back to CPU on failure)
    """
    diag = diagnose_colmap_environment(verbose=False)
    gpu_sift_available = diag.get("gpu_sift_available", False)

    def _run_matcher(gpu: bool):
        """Run the matcher with specified GPU setting."""
        if matcher_type == "sequential":
            args = {
                "database_path": str(database_path),
                "SequentialMatching.overlap": sequential_overlap,
            }
            if gpu_sift_available:
                args["SiftMatching.use_gpu"] = 1 if gpu else 0
                if gpu:
                    args["SiftMatching.gpu_index"] = "0"
            mode = "GPU" if gpu and gpu_sift_available else "CPU"
            run_colmap_command("sequential_matcher", args, f"Matching features (sequential, {mode})")
        elif matcher_type == "exhaustive":
            args = {
                "database_path": str(database_path),
            }
            if gpu_sift_available:
                args["SiftMatching.use_gpu"] = 1 if gpu else 0
                if gpu:
                    args["SiftMatching.gpu_index"] = "0"
            mode = "GPU" if gpu and gpu_sift_available else "CPU"
            run_colmap_command("exhaustive_matcher", args, f"Matching features (exhaustive, {mode})")
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")

    if use_gpu:
        try:
            _run_matcher(gpu=True)
        except subprocess.CalledProcessError as e:
            error_output = str(e.stdout) if hasattr(e, 'stdout') and e.stdout else str(e)
            is_gpu_error = (
                "context" in error_output.lower()
                or "opengl" in error_output.lower()
                or "sift" in error_output.lower()
                or e.returncode < 0
            )
            if is_gpu_error:
                print("    GPU SIFT matching failed, falling back to CPU...")
                _run_matcher(gpu=False)
            else:
                raise
    else:
        _run_matcher(gpu=False)


def count_model_images(model_dir: Path) -> int:
    """Count registered images in a COLMAP model directory."""
    images_bin = model_dir / "images.bin"
    images_txt = model_dir / "images.txt"

    if images_bin.exists():
        try:
            with open(images_bin, "rb") as f:
                return struct.unpack("<Q", f.read(8))[0]
        except (IOError, struct.error):
            pass
    elif images_txt.exists():
        try:
            with open(images_txt, "r") as f:
                lines = [l for l in f if l.strip() and not l.startswith("#")]
                return len(lines) // 2
        except IOError:
            pass
    return 0


def get_sparse_models(sparse_path: Path) -> list[tuple[Path, int]]:
    """Get all sparse models with their image counts, sorted by count descending."""
    if not sparse_path.exists():
        return []

    models = []
    for model_dir in sparse_path.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("merged"):
            count = count_model_images(model_dir)
            if count > 0:
                models.append((model_dir, count))

    return sorted(models, key=lambda x: x[1], reverse=True)


def merge_sparse_models(
    sparse_path: Path,
    output_name: str = "merged"
) -> Optional[Path]:
    """Merge multiple sparse models into one using COLMAP model_merger.

    Args:
        sparse_path: Path containing model subdirectories (0/, 1/, etc.)
        output_name: Name for merged output directory

    Returns:
        Path to merged model, or None if merge failed
    """
    models = get_sparse_models(sparse_path)
    if len(models) < 2:
        return None

    print(f"    Found {len(models)} sub-models, attempting to merge...")
    for model_path, count in models:
        print(f"      Model {model_path.name}: {count} images")

    merged_path = sparse_path / output_name
    merged_path.mkdir(exist_ok=True)

    # Start with the largest model
    largest_model = models[0][0]
    current_merged = merged_path / "current"
    shutil.copytree(largest_model, current_merged)

    total_merged = models[0][1]
    merge_success = False

    # Try to merge each smaller model into the growing merged model
    for model_path, count in models[1:]:
        temp_output = merged_path / "temp_merge"
        if temp_output.exists():
            shutil.rmtree(temp_output)
        temp_output.mkdir()

        try:
            args = {
                "input_path1": str(current_merged),
                "input_path2": str(model_path),
                "output_path": str(temp_output),
            }
            run_colmap_command("model_merger", args, f"Merging model {model_path.name}")

            # Check if merge produced output with more images
            merged_count = count_model_images(temp_output)
            if merged_count > total_merged:
                shutil.rmtree(current_merged)
                temp_output.rename(current_merged)
                total_merged = merged_count
                merge_success = True
                print(f"      Merged model {model_path.name}: now {total_merged} images")
            else:
                shutil.rmtree(temp_output)
                print(f"      Could not merge model {model_path.name} (no improvement)")

        except subprocess.CalledProcessError:
            print(f"      Failed to merge model {model_path.name}")
            if temp_output.exists():
                shutil.rmtree(temp_output)

    if not merge_success:
        shutil.rmtree(merged_path)
        return None

    # Run bundle adjustment on merged model to refine
    print("    Running bundle adjustment on merged model...")
    try:
        args = {
            "input_path": str(current_merged),
            "output_path": str(current_merged),
            "BundleAdjustment.refine_focal_length": 1,
            "BundleAdjustment.refine_principal_point": 0,
            "BundleAdjustment.refine_extra_params": 1,
        }
        run_colmap_command("bundle_adjuster", args, "Bundle adjustment")
    except subprocess.CalledProcessError:
        print("    Warning: Bundle adjustment failed, using unrefined merge")

    # Move to final location
    final_path = merged_path / "final"
    current_merged.rename(final_path)

    print(f"    Merged model has {total_merged} registered images")
    return final_path


def find_best_sparse_model(sparse_path: Path) -> Optional[Path]:
    """Find the best sparse model - merged if possible, otherwise largest.

    Args:
        sparse_path: Path to sparse reconstruction directory

    Returns:
        Path to the best model directory, or None if no valid models found
    """
    models = get_sparse_models(sparse_path)
    if not models:
        return None

    # If only one model, use it
    if len(models) == 1:
        print(f"    Single model with {models[0][1]} registered images")
        return models[0][0]

    # Multiple models - try to merge them
    merged = merge_sparse_models(sparse_path)
    if merged:
        return merged

    # Merging failed - fall back to largest model
    best_model, best_count = models[0]
    print(f"    Using largest model {best_model.name} with {best_count} registered images")
    return best_model


def run_sparse_reconstruction(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    refine_focal: bool = True,
    min_tri_angle: float = 1.5,
    min_num_inliers: int = 15,
) -> bool:
    """Run incremental Structure-from-Motion reconstruction.

    Args:
        database_path: Path to COLMAP database
        image_path: Path to image directory
        output_path: Path to sparse reconstruction output
        refine_focal: Whether to refine focal length during BA
        min_tri_angle: Minimum triangulation angle in degrees (lower = accept smaller motion)
        min_num_inliers: Minimum inliers for valid match (lower = accept weaker matches)

    Returns:
        True if reconstruction succeeded
    """
    output_path.mkdir(parents=True, exist_ok=True)

    args = {
        "database_path": str(database_path),
        "image_path": str(image_path),
        "output_path": str(output_path),
        "Mapper.ba_refine_focal_length": 1 if refine_focal else 0,
        "Mapper.ba_refine_principal_point": 0,
        "Mapper.ba_refine_extra_params": 1,
        "Mapper.init_min_tri_angle": min_tri_angle,
        "Mapper.min_num_matches": min_num_inliers,
    }

    run_colmap_command("mapper", args, "Running sparse reconstruction")

    # Find the best model (most registered images)
    # COLMAP may produce multiple sub-models if it can't connect all images
    best_model = find_best_sparse_model(output_path)

    if best_model is None:
        print("    Warning: No reconstruction model produced", file=sys.stderr)
        return False

    # Reorganize so downstream code always finds the model at "0"
    model_0 = output_path / "0"
    if best_model != model_0:
        # Backup existing model 0 if it exists and isn't the best
        if model_0.exists():
            backup_name = f"0_original"
            backup_path = output_path / backup_name
            if backup_path.exists():
                shutil.rmtree(backup_path)
            model_0.rename(backup_path)
            print(f"    Backed up original model 0 to {backup_name}")

        # Copy best model to 0 (copy instead of move to preserve merged/ structure)
        shutil.copytree(best_model, model_0)
        print(f"    Installed best model as 0 ({count_model_images(model_0)} images)")

    return True


def run_dense_reconstruction(
    image_path: Path,
    sparse_path: Path,
    output_path: Path,
    max_image_size: int = -1
) -> bool:
    """Run Multi-View Stereo dense reconstruction.

    Args:
        image_path: Path to image directory
        sparse_path: Path to sparse model (typically sparse/0/)
        output_path: Path for dense output
        max_image_size: Maximum image dimension (-1 for no limit)

    Returns:
        True if dense reconstruction succeeded
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Undistort images
    args = {
        "image_path": str(image_path),
        "input_path": str(sparse_path),
        "output_path": str(output_path),
        "output_type": "COLMAP",
        "max_image_size": max_image_size if max_image_size > 0 else 2000,
    }
    run_colmap_command("image_undistorter", args, "Undistorting images")

    # Patch match stereo
    args = {
        "workspace_path": str(output_path),
        "workspace_format": "COLMAP",
        "PatchMatchStereo.geom_consistency": True,
    }
    run_colmap_command("patch_match_stereo", args, "Running patch match stereo")

    # Stereo fusion
    args = {
        "workspace_path": str(output_path),
        "workspace_format": "COLMAP",
        "input_type": "geometric",
        "output_path": str(output_path / "fused.ply"),
    }
    run_colmap_command("stereo_fusion", args, "Fusing depth maps")

    return (output_path / "fused.ply").exists()


def run_mesh_reconstruction(
    dense_path: Path,
    output_path: Path
) -> bool:
    """Generate mesh from dense point cloud using Poisson reconstruction.

    Args:
        dense_path: Path to dense reconstruction (containing fused.ply)
        output_path: Output mesh file path

    Returns:
        True if mesh was created
    """
    input_ply = dense_path / "fused.ply"
    if not input_ply.exists():
        print(f"    Error: Dense point cloud not found: {input_ply}", file=sys.stderr)
        return False

    args = {
        "input_path": str(input_ply),
        "output_path": str(output_path),
    }
    run_colmap_command("poisson_mesher", args, "Generating mesh")

    return output_path.exists()


def export_sparse_ply(sparse_model_path: Path, output_path: Path) -> bool:
    """Export sparse reconstruction as PLY point cloud.

    Uses COLMAP's model_converter to convert the sparse model
    (cameras.bin, images.bin, points3D.bin) to a colored PLY file.

    Args:
        sparse_model_path: Path to sparse model directory (e.g., colmap/sparse/0/)
        output_path: Output PLY file path

    Returns:
        True if export succeeded
    """
    if not (sparse_model_path / "points3D.bin").exists():
        txt_path = sparse_model_path / "points3D.txt"
        if not txt_path.exists():
            print(f"    Error: No points3D found in {sparse_model_path}", file=sys.stderr)
            return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = {
        "input_path": str(sparse_model_path),
        "output_path": str(output_path),
        "output_type": "PLY",
    }
    run_colmap_command("model_converter", args, "Exporting sparse point cloud to PLY")

    return output_path.exists()


def read_colmap_array(path: Path) -> np.ndarray:
    """Read a COLMAP dense map binary file (depth or normal map).

    COLMAP binary format: text header 'width&height&channels&' followed
    by float32 data in column-major (Fortran) order.

    Args:
        path: Path to .geometric.bin or .photometric.bin file

    Returns:
        numpy array of shape (height, width) for depth or (height, width, 3) for normals
    """
    with open(path, "rb") as f:
        line = b""
        ampersand_count = 0
        while ampersand_count < 3:
            byte = f.read(1)
            if byte == b"&":
                ampersand_count += 1
            line += byte

        header = line.decode("ascii").strip("&").split("&")
        width = int(header[0])
        height = int(header[1])
        channels = int(header[2])

        data = np.fromfile(f, dtype=np.float32)

    data = data.reshape((width, height, channels), order="F")
    data = data.transpose((1, 0, 2))

    return data.squeeze()


def convert_depth_maps_to_exr(
    dense_path: Path,
    output_dir: Path,
) -> int:
    """Convert COLMAP depth maps from binary format to 32-bit float EXR.

    Reads .geometric.bin files from dense/stereo/depth_maps/ and writes
    single-channel float32 EXR files.

    Args:
        dense_path: Path to COLMAP dense reconstruction directory
        output_dir: Output directory for EXR files

    Returns:
        Number of depth maps converted
    """
    depth_maps_dir = dense_path / "stereo" / "depth_maps"
    if not depth_maps_dir.exists():
        print(f"    No depth maps directory found: {depth_maps_dir}", file=sys.stderr)
        return 0

    geometric_files = sorted(depth_maps_dir.glob("*.geometric.bin"))
    if not geometric_files:
        print(f"    No geometric depth maps found in {depth_maps_dir}", file=sys.stderr)
        return 0

    import OpenEXR

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for bin_path in geometric_files:
        frame_name = bin_path.name.replace(".geometric.bin", "")
        frame_stem = Path(frame_name).stem
        exr_path = output_dir / f"{frame_stem}.exr"

        depth = read_colmap_array(bin_path)
        height, width = depth.shape[:2]

        depth_rgb = np.stack([depth, depth, depth], axis=-1).astype(np.float32)
        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage,
        }
        with OpenEXR.File(header, {"RGB": depth_rgb}) as out:
            out.write(str(exr_path))

        converted += 1

    return converted


def convert_normal_maps_to_exr(
    dense_path: Path,
    output_dir: Path,
) -> int:
    """Convert COLMAP normal maps from binary format to 32-bit float EXR.

    Reads .geometric.bin files from dense/stereo/normal_maps/ and writes
    RGB float32 EXR files where XYZ normals map to RGB channels.

    Args:
        dense_path: Path to COLMAP dense reconstruction directory
        output_dir: Output directory for EXR files

    Returns:
        Number of normal maps converted
    """
    normal_maps_dir = dense_path / "stereo" / "normal_maps"
    if not normal_maps_dir.exists():
        print(f"    No normal maps directory found: {normal_maps_dir}", file=sys.stderr)
        return 0

    geometric_files = sorted(normal_maps_dir.glob("*.geometric.bin"))
    if not geometric_files:
        print(f"    No geometric normal maps found in {normal_maps_dir}", file=sys.stderr)
        return 0

    import OpenEXR

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for bin_path in geometric_files:
        frame_name = bin_path.name.replace(".geometric.bin", "")
        frame_stem = Path(frame_name).stem
        exr_path = output_dir / f"{frame_stem}.exr"

        normals = read_colmap_array(bin_path)
        if normals.ndim == 2:
            normals = np.stack([normals, normals, normals], axis=-1)
        normals = normals.astype(np.float32)

        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage,
        }
        with OpenEXR.File(header, {"RGB": normals}) as out:
            out.write(str(exr_path))

        converted += 1

    return converted


def read_colmap_cameras(cameras_path: Path) -> dict:
    """Read COLMAP cameras.bin or cameras.txt file.

    Returns:
        Dict mapping camera_id -> camera parameters
    """
    cameras = {}

    txt_path = cameras_path.parent / "cameras.txt"
    if txt_path.exists():
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                cameras[camera_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params,
                }
        return cameras

    # Binary format - use COLMAP's model_converter to get text
    bin_path = cameras_path
    if bin_path.exists():
        # Convert to text format temporarily
        temp_dir = cameras_path.parent / "_temp_txt"
        temp_dir.mkdir(exist_ok=True)
        try:
            colmap_exe = get_colmap_executable()
            is_bat = colmap_exe.lower().endswith('.bat')
            subprocess.run([
                colmap_exe, "model_converter",
                "--input_path", str(cameras_path.parent),
                "--output_path", str(temp_dir),
                "--output_type", "TXT"
            ], capture_output=True, check=True, shell=is_bat)

            # Now read the text file
            with open(temp_dir / "cameras.txt") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    camera_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(p) for p in parts[4:]]
                    cameras[camera_id] = {
                        "model": model,
                        "width": width,
                        "height": height,
                        "params": params,
                    }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return cameras


def read_colmap_images(images_path: Path, debug: bool = False) -> dict:
    """Read COLMAP images.bin or images.txt file.

    Returns:
        Dict mapping image_name -> {quat, trans, camera_id}
    """
    images = {}

    if debug:
        print(f"    DEBUG: Reading images from {images_path}")

    txt_path = images_path.parent / "images.txt"
    if txt_path.exists():
        if debug:
            print(f"    DEBUG: Found text file: {txt_path}")
        with open(txt_path) as f:
            all_lines = f.readlines()

        if debug:
            print(f"    DEBUG: Total lines in file: {len(all_lines)}")

        # Images.txt has 2 lines per image: metadata + keypoints
        # Parse only non-comment metadata lines (keypoints line may be empty)
        parsed_count = 0
        skipped_count = 0
        for line in all_lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Metadata lines have at least 10 parts and end with a filename
            parts = line.split()
            if len(parts) >= 10:
                try:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = [float(p) for p in parts[1:5]]
                    tx, ty, tz = [float(p) for p in parts[5:8]]
                    camera_id = int(parts[8])
                    name = parts[9]

                    images[name] = {
                        "image_id": image_id,
                        "quat": [qw, qx, qy, qz],
                        "trans": [tx, ty, tz],
                        "camera_id": camera_id,
                    }
                    parsed_count += 1
                except (ValueError, IndexError) as e:
                    # Not a metadata line (probably keypoints), skip
                    skipped_count += 1
                    if debug:
                        print(f"    DEBUG: Skipped line (parse error): {line[:80]}...")
                    continue
            else:
                skipped_count += 1

        if debug:
            print(f"    DEBUG: Parsed {parsed_count} images, skipped {skipped_count} lines")
        return images

    # Binary format - convert to text
    bin_path = images_path
    if bin_path.exists():
        if debug:
            print(f"    DEBUG: Converting binary file: {bin_path}")
            print(f"    DEBUG: Binary file size: {bin_path.stat().st_size} bytes")
        temp_dir = images_path.parent / "_temp_txt"
        temp_dir.mkdir(exist_ok=True)
        try:
            colmap_exe = get_colmap_executable()
            is_bat = colmap_exe.lower().endswith('.bat')
            result = subprocess.run([
                colmap_exe, "model_converter",
                "--input_path", str(images_path.parent),
                "--output_path", str(temp_dir),
                "--output_type", "TXT"
            ], capture_output=True, text=True, shell=is_bat)

            if debug:
                print(f"    DEBUG: model_converter return code: {result.returncode}")
                if result.stderr:
                    print(f"    DEBUG: model_converter stderr: {result.stderr[:500]}")

            if result.returncode != 0:
                print(f"    Warning: model_converter failed: {result.stderr}", file=sys.stderr)
                return images

            txt_file = temp_dir / "images.txt"
            if debug:
                if txt_file.exists():
                    print(f"    DEBUG: Converted file size: {txt_file.stat().st_size} bytes")
                else:
                    print(f"    DEBUG: Converted file not found!")

            with open(txt_file) as f:
                all_lines = f.readlines()

            if debug:
                print(f"    DEBUG: Total lines in converted file: {len(all_lines)}")
                # Print first few non-comment lines
                data_lines = [l for l in all_lines if l.strip() and not l.strip().startswith("#")]
                print(f"    DEBUG: Non-comment lines: {len(data_lines)}")
                if data_lines:
                    print(f"    DEBUG: First data line: {data_lines[0][:100]}...")

            # Images.txt has 2 lines per image: metadata + keypoints
            # Parse only non-comment metadata lines (keypoints line may be empty)
            parsed_count = 0
            skipped_count = 0
            for line in all_lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Metadata lines have at least 10 parts and end with a filename
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        image_id = int(parts[0])
                        qw, qx, qy, qz = [float(p) for p in parts[1:5]]
                        tx, ty, tz = [float(p) for p in parts[5:8]]
                        camera_id = int(parts[8])
                        name = parts[9]

                        images[name] = {
                            "image_id": image_id,
                            "quat": [qw, qx, qy, qz],
                            "trans": [tx, ty, tz],
                            "camera_id": camera_id,
                        }
                        parsed_count += 1
                    except (ValueError, IndexError) as e:
                        # Not a metadata line (probably keypoints), skip
                        skipped_count += 1
                        if debug:
                            print(f"    DEBUG: Skipped line (parse error): {line[:80]}...")
                        continue
                else:
                    skipped_count += 1

            if debug:
                print(f"    DEBUG: Parsed {parsed_count} images, skipped {skipped_count} lines")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return images


def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_0001.png' or 'depth_00001.png'."""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def interpolate_camera_poses(
    registered_images: dict,
    total_frames: int,
    max_gap: int = 12,
) -> tuple[list[np.ndarray], list[int]]:
    """Interpolate camera poses for missing frames using Bezier curves.

    Args:
        registered_images: Dict of {filename: {quat, trans, ...}} from COLMAP
        total_frames: Total number of frames expected
        max_gap: Maximum gap size to interpolate (larger gaps get static camera)

    Returns:
        Tuple of (list of 4x4 matrices for all frames, list of frame indices that were interpolated)
    """
    if not registered_images:
        return [], []

    # Parse frame numbers and build sparse pose dict
    frame_poses = {}  # frame_idx -> (translation, quaternion)
    for name, data in registered_images.items():
        frame_idx = extract_frame_number(name)
        if frame_idx >= 0:
            # Convert COLMAP to camera-to-world
            quat = np.array(data["quat"])
            trans = np.array(data["trans"])

            R_wc = quaternion_to_rotation_matrix(quat)
            t_wc = trans

            # Camera-to-world
            R_cw = R_wc.T
            t_cw = -R_wc.T @ t_wc
            q_cw = rotation_matrix_to_quaternion(R_cw)

            frame_poses[frame_idx] = (t_cw, q_cw)

    if not frame_poses:
        return [], []

    # Get sorted list of registered frames
    registered_frames = sorted(frame_poses.keys())
    min_frame = min(registered_frames)
    max_frame = max(registered_frames)

    # Determine actual frame range (1-indexed or 0-indexed)
    start_frame = 1 if min_frame >= 1 else 0
    end_frame = start_frame + total_frames - 1

    # Build complete pose list
    all_poses = []
    interpolated_indices = []

    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx in frame_poses:
            # Use registered pose
            t, q = frame_poses[frame_idx]
            matrix = np.eye(4)
            matrix[:3, :3] = quaternion_to_rotation_matrix(q)
            matrix[:3, 3] = t
            all_poses.append(matrix)
        else:
            # Need to interpolate
            # Find surrounding registered frames
            prev_frame = None
            next_frame = None
            for rf in registered_frames:
                if rf < frame_idx:
                    prev_frame = rf
                elif rf > frame_idx and next_frame is None:
                    next_frame = rf
                    break

            # Check if we can interpolate
            if prev_frame is not None and next_frame is not None:
                gap = next_frame - prev_frame
                if gap <= max_gap:
                    # Interpolate
                    t_factor = (frame_idx - prev_frame) / gap
                    t0, q0 = frame_poses[prev_frame]
                    t1, q1 = frame_poses[next_frame]

                    # Linear position interpolation (could use Bezier with tangents)
                    t_interp = t0 + t_factor * (t1 - t0)

                    # SLERP for rotation
                    q_interp = slerp(q0, q1, t_factor)

                    matrix = np.eye(4)
                    matrix[:3, :3] = quaternion_to_rotation_matrix(q_interp)
                    matrix[:3, 3] = t_interp
                    all_poses.append(matrix)
                    interpolated_indices.append(frame_idx)
                else:
                    # Gap too large - use nearest pose
                    if frame_idx - prev_frame <= next_frame - frame_idx:
                        t, q = frame_poses[prev_frame]
                    else:
                        t, q = frame_poses[next_frame]
                    matrix = np.eye(4)
                    matrix[:3, :3] = quaternion_to_rotation_matrix(q)
                    matrix[:3, 3] = t
                    all_poses.append(matrix)
                    interpolated_indices.append(frame_idx)
            elif prev_frame is not None:
                # Extrapolate from previous
                t, q = frame_poses[prev_frame]
                matrix = np.eye(4)
                matrix[:3, :3] = quaternion_to_rotation_matrix(q)
                matrix[:3, 3] = t
                all_poses.append(matrix)
                interpolated_indices.append(frame_idx)
            elif next_frame is not None:
                # Extrapolate from next
                t, q = frame_poses[next_frame]
                matrix = np.eye(4)
                matrix[:3, :3] = quaternion_to_rotation_matrix(q)
                matrix[:3, 3] = t
                all_poses.append(matrix)
                interpolated_indices.append(frame_idx)
            else:
                # No registered frames - identity
                all_poses.append(np.eye(4))
                interpolated_indices.append(frame_idx)

    return all_poses, interpolated_indices


def colmap_to_camera_matrices(images: dict, cameras: dict) -> tuple[list, dict]:
    """Convert COLMAP output to camera matrices in our format.

    COLMAP stores camera-to-world as (R, t) where the world point X
    transforms to camera point as: X_cam = R * X_world + t

    We want camera-to-world 4x4 matrices.

    Returns:
        Tuple of (list of 4x4 matrices sorted by frame, intrinsics dict)
    """
    # Sort images by name (assumes frame_NNNN.png naming)
    sorted_images = sorted(images.items(), key=lambda x: x[0])

    extrinsics = []
    for name, img_data in sorted_images:
        quat = img_data["quat"]
        trans = img_data["trans"]

        # COLMAP's R, t are world-to-camera
        # R_wc @ X_world + t = X_cam
        R_wc = quaternion_to_rotation_matrix(quat)
        t_wc = np.array(trans)

        # Camera-to-world is the inverse
        # R_cw = R_wc.T
        # t_cw = -R_wc.T @ t_wc
        R_cw = R_wc.T
        t_cw = -R_wc.T @ t_wc

        # Build 4x4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = R_cw
        matrix[:3, 3] = t_cw

        extrinsics.append(matrix)

    # Get intrinsics from first camera (assuming single camera)
    if cameras:
        cam = list(cameras.values())[0]
        model = cam["model"]
        params = cam["params"]
        width = cam["width"]
        height = cam["height"]

        # Parse based on camera model
        if model == "PINHOLE":
            fx, fy, cx, cy = params
        elif model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = params
            fx = fy = f
        elif model == "RADIAL":
            f, cx, cy, k1, k2 = params
            fx = fy = f
        elif model in ("OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = params[:4]
        else:
            # Default fallback
            fx = fy = params[0] if params else 1000
            cx = width / 2
            cy = height / 2

        intrinsics = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "model": model,
            "params": params,
        }
    else:
        intrinsics = {}

    return extrinsics, intrinsics


def export_colmap_to_pipeline_format(
    sparse_path: Path,
    output_dir: Path,
    debug: bool = False,
    total_frames: int = 0,
    max_gap: int = 12,
) -> bool:
    """Export COLMAP reconstruction to pipeline camera format.

    Creates extrinsics.json and intrinsics.json compatible with export_camera.py.
    If total_frames is specified and COLMAP didn't register all frames,
    interpolates missing frames using SLERP for rotation and linear for translation.

    Args:
        sparse_path: Path to COLMAP sparse model (e.g., sparse/0/)
        output_dir: Output directory (typically project/camera/)
        debug: Print debug output
        total_frames: Expected total frames (0 = use only registered frames)
        max_gap: Maximum gap to interpolate (larger gaps use nearest pose)

    Returns:
        True if export succeeded
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read COLMAP output
    cameras = read_colmap_cameras(sparse_path / "cameras.bin")
    images = read_colmap_images(sparse_path / "images.bin", debug=debug)

    if not images:
        print("    Error: No images in reconstruction", file=sys.stderr)
        return False

    registered_count = len(images)

    # Check if we need to interpolate
    if total_frames > 0 and registered_count < total_frames:
        print(f"    COLMAP registered {registered_count}/{total_frames} frames")
        print(f"    Interpolating missing frames (max_gap={max_gap})...")

        # Use interpolation to fill gaps
        extrinsics, interpolated = interpolate_camera_poses(
            images, total_frames, max_gap=max_gap
        )

        if interpolated:
            print(f"    Interpolated {len(interpolated)} frames")

        # Get intrinsics from COLMAP
        _, intrinsics = colmap_to_camera_matrices(images, cameras)
    else:
        # Use only registered frames
        extrinsics, intrinsics = colmap_to_camera_matrices(images, cameras)

    # Save extrinsics (list of 4x4 matrices)
    extrinsics_data = [m.tolist() if isinstance(m, np.ndarray) else m for m in extrinsics]
    with open(output_dir / "extrinsics.json", "w", encoding='utf-8') as f:
        json.dump(extrinsics_data, f, indent=2)

    # Save intrinsics
    with open(output_dir / "intrinsics.json", "w", encoding='utf-8') as f:
        json.dump(intrinsics, f, indent=2)

    # Also save the raw COLMAP data for reference
    colmap_data = {
        "cameras": cameras,
        "images": {name: {
            "quat": data["quat"],
            "trans": data["trans"],
            "camera_id": data["camera_id"],
        } for name, data in images.items()},
        "source": "colmap",
        "registered_frames": registered_count,
        "total_frames": total_frames if total_frames > 0 else registered_count,
        "interpolated": total_frames > 0 and registered_count < total_frames,
    }
    with open(output_dir / "colmap_raw.json", "w", encoding='utf-8') as f:
        json.dump(colmap_data, f, indent=2)

    print(f"    Exported {len(extrinsics)} camera frames to {output_dir}")
    return True


def run_colmap_pipeline(
    project_dir: Path,
    quality: str = "medium",
    camera_model: str = "OPENCV",
    use_masks: bool = True,
    max_gap: int = 12,
    max_image_size: int = -1,
) -> bool:
    """Run the COLMAP sparse reconstruction pipeline.

    Produces camera poses (extrinsics/intrinsics) and a sparse point cloud.
    Dense reconstruction is handled separately by the 'dense' pipeline stage.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high', 'slow')
        camera_model: COLMAP camera model to use
        use_masks: If True, automatically use segmentation masks from roto/ (if available)
        max_gap: Maximum frame gap to interpolate if COLMAP misses frames (default: 12)
        max_image_size: Maximum image dimension for feature extraction (-1 for no limit)

    Returns:
        True if reconstruction succeeded
    """
    if not check_colmap_available():
        snap_path = shutil.which("colmap")
        if snap_path and "/snap/" in snap_path:
            print("Error: Only snap COLMAP found, which cannot access mounted drives.", file=sys.stderr)
            print(f"  Found: {snap_path}", file=sys.stderr)
            print("", file=sys.stderr)
            print("Snap apps have confinement that prevents writing to /media/, /mnt/, etc.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Install a non-snap version:", file=sys.stderr)
            print("  GPU support: conda install -c conda-forge colmap", file=sys.stderr)
            print("  CPU only:    sudo apt install colmap", file=sys.stderr)
            print("", file=sys.stderr)
            print("Or move your project to a non-mounted directory (e.g., ~/projects/)", file=sys.stderr)
        else:
            print("Error: COLMAP not found.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Install with:", file=sys.stderr)
            print("  GPU support: conda install -c conda-forge colmap", file=sys.stderr)
            print("  CPU only:    sudo apt install colmap", file=sys.stderr)
        return False

    pipeline_start = time.time()

    diag = diagnose_colmap_environment(verbose=True)
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])

    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}", file=sys.stderr)
        return False

    # Count frames
    frame_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("*.jpg")))
    if frame_count == 0:
        print(f"Error: No images found in {frames_dir}", file=sys.stderr)
        return False

    print(f"\n{'='*60}")
    print(f"COLMAP Reconstruction")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")
    print(f"Frames: {frame_count}")
    print(f"Quality: {quality}")
    print(f"Mode: GPU (with CPU fallback)")
    if max_image_size > 0:
        print(f"Max image size: {max_image_size}px (downscaling enabled)")
    if quality == "slow":
        print(f"  Warning: Slow-camera mode uses aggressive settings for minimal motion")
        print(f"           Results may be jittery due to low parallax")

    # Setup paths
    colmap_dir = project_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Check for segmentation masks and prepare with COLMAP-compatible naming
    roto_dir = project_dir / "roto"
    mask_path = None
    if use_masks and roto_dir.exists():
        roto_mask_count = len(list(roto_dir.glob("*.png"))) + len(list(roto_dir.glob("*.jpg")))
        if roto_mask_count > 0:
            mask_path = prepare_colmap_masks(roto_dir, frames_dir, colmap_dir)
            if mask_path:
                print(f"Dynamic scene segmentation: Enabled ({roto_mask_count} masks)")
                print(f"  → Excluding dynamic regions from feature extraction")
            else:
                print(f"Dynamic scene segmentation: Disabled (mask preparation failed)")
        else:
            print(f"Dynamic scene segmentation: Disabled (no masks found in roto/)")
    else:
        print(f"Dynamic scene segmentation: Disabled")
    print()

    database_path = colmap_dir / "database.db"
    sparse_path = colmap_dir / "sparse"
    dense_path = colmap_dir / "dense"

    # Clean previous run if exists
    if database_path.exists():
        database_path.unlink()
    if sparse_path.exists():
        shutil.rmtree(sparse_path)
    if dense_path.exists():
        shutil.rmtree(dense_path)

    with VirtualDisplay():
        try:
            # Debug: List actual image files
            image_files = sorted(
                list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg"))
            )
            print(f"    Image path: {frames_dir}")
            print(f"    Found {len(image_files)} image files")
            if image_files:
                print(f"    First: {image_files[0].name}, Last: {image_files[-1].name}")

            # Feature extraction
            print("[1/4] Feature Extraction")
            extract_features(
                database_path=database_path,
                image_path=frames_dir,
                camera_model=camera_model,
                max_features=preset["sift_max_features"],
                single_camera=True,
                mask_path=mask_path,
                use_gpu=True,
                max_image_size=max_image_size,
            )

            # Verify features were extracted
            num_images, total_keypoints = verify_database_has_features(database_path)
            if num_images == 0 or total_keypoints == 0:
                print(f"    Error: No features extracted from images", file=sys.stderr)
                verify_database_has_features(database_path, verbose=True)
                print(f"    This can happen if images are too small, dark, or featureless", file=sys.stderr)
                return False
            print(f"    Extracted {total_keypoints:,} keypoints from {num_images} images")

            # Feature matching
            print("\n[2/4] Feature Matching")
            match_features(
                database_path=database_path,
                matcher_type=preset["matcher"],
                sequential_overlap=preset.get("sequential_overlap", 10),
                use_gpu=True,
            )

            # Sparse reconstruction
            print("\n[3/4] Sparse Reconstruction")
            if not run_sparse_reconstruction(
                database_path=database_path,
                image_path=frames_dir,
                output_path=sparse_path,
                refine_focal=preset["ba_refine_focal"],
                min_tri_angle=preset.get("min_tri_angle", 1.5),
                min_num_inliers=preset.get("min_num_inliers", 15),
            ):
                print("Sparse reconstruction failed", file=sys.stderr)
                return False

            # Export camera data
            print("\n[4/4] Exporting Camera Data")
            camera_dir = project_dir / "camera"
            sparse_model = sparse_path / "0"
            if not export_colmap_to_pipeline_format(
                sparse_model,
                camera_dir,
                total_frames=frame_count,
                max_gap=max_gap,
            ):
                print("Camera export failed", file=sys.stderr)
                return False

            # Calculate timing
            pipeline_end = time.time()
            total_seconds = pipeline_end - pipeline_start
            total_minutes = total_seconds / 60
            per_frame_seconds = total_seconds / frame_count if frame_count > 0 else 0

            print(f"\n{'='*60}")
            print(f"COLMAP Reconstruction Complete")
            print(f"{'='*60}")
            print(f"Sparse model: {sparse_model}")
            print(f"Camera data: {camera_dir}")
            print()
            print(f"TOTAL TIME: {total_minutes:.1f} minutes ({per_frame_seconds:.2f}s per frame)")
            print()

            return True

        except subprocess.CalledProcessError as e:
            print(f"\nCOLMAP command failed: {e}", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print(f"\nCOLMAP command timed out", file=sys.stderr)
            return False
        finally:
            temp_masks_dir = colmap_dir / "masks"
            if temp_masks_dir.exists():
                shutil.rmtree(temp_masks_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP reconstruction on a frame sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing source/frames/"
    )
    parser.add_argument(
        "--quality", "-q",
        choices=["low", "medium", "high", "slow"],
        default="medium",
        help="Quality preset: low, medium, high, or 'slow' for minimal camera motion (default: medium)"
    )
    parser.add_argument(
        "--camera-model", "-c",
        choices=["PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"],
        default="OPENCV",
        help="COLMAP camera model (default: OPENCV)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Disable automatic use of segmentation masks from roto/ directory"
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=12,
        help="Maximum frame gap to interpolate if COLMAP misses frames (default: 12)"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=-1,
        help="Maximum image dimension (downscales larger images, -1 for no limit). "
             "Use 1000-2000 for faster processing."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if COLMAP is available and exit"
    )

    args = parser.parse_args()

    require_conda_env()

    if args.check:
        if check_colmap_available():
            print("COLMAP is available")
            sys.exit(0)
        else:
            print("COLMAP is not available")
            sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_colmap_pipeline(
        project_dir=project_dir,
        quality=args.quality,
        camera_model=args.camera_model,
        use_masks=not args.no_masks,
        max_gap=args.max_gap,
        max_image_size=args.max_image_size,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
