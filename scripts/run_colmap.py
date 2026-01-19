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
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Environment check - ensure correct conda environment is active
from env_config import check_conda_env_or_warn, is_in_container


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

    In container environments without a display, COLMAP's GPU SIFT extraction
    needs an X server for OpenGL context. This starts a virtual framebuffer.
    """

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._display: Optional[str] = None
        self._env_modified: bool = False

    def _needs_virtual_display(self) -> bool:
        """Check if we need to start a virtual display."""
        if not is_in_container():
            return False
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
        result = subprocess.run(
            ["colmap", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def verify_database_has_features(database_path: Path) -> tuple[int, int]:
    """Verify COLMAP database has extracted features.

    Args:
        database_path: Path to COLMAP database.db

    Returns:
        Tuple of (num_images_with_features, total_keypoints)
    """
    if not database_path.exists():
        return 0, 0

    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM images")
        num_images = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM keypoints")
        num_keypoint_rows = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(rows) FROM keypoints")
        result = cursor.fetchone()[0]
        total_keypoints = result if result else 0

        conn.close()
        return num_keypoint_rows, total_keypoints
    except sqlite3.Error as e:
        print(f"    Warning: Could not read database: {e}")
        return 0, 0


def run_colmap_command(
    command: str,
    args: dict,
    description: str,
    timeout: int = None,  # No timeout by default
) -> subprocess.CompletedProcess:
    """Run a COLMAP command with streaming output.

    Args:
        command: COLMAP subcommand (e.g., 'feature_extractor')
        args: Dictionary of argument name -> value
        description: Human-readable description for logging
        timeout: Timeout in seconds (None = no timeout)

    Returns:
        CompletedProcess result
    """
    cmd = ["colmap", command]
    for key, value in args.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value is not False and value is not None:
            cmd.extend([f"--{key}", str(value)])

    print(f"  → {description}")
    print(f"    $ {' '.join(cmd)}")
    sys.stdout.flush()

    # Always stream output for visibility
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stdout_lines = []
    last_progress = ""
    is_tty = sys.stdout.isatty()

    # Progress patterns for different COLMAP stages
    # Feature extraction: "Processed file [1/521]"
    extract_pattern = re.compile(r'Processed file \[(\d+)/(\d+)\]')
    # Matching: "Matching block [1/52, 1/52]" or image pairs
    match_pattern = re.compile(r'Matching block \[(\d+)/(\d+)')
    # Mapper: "Registering image #142 (150)"
    register_pattern = re.compile(r'Registering image #(\d+)\s*\((\d+)\)')

    for line in iter(process.stdout.readline, ''):
        stdout_lines.append(line)
        line_stripped = line.strip()

        # Check for feature extraction progress
        match = extract_pattern.search(line_stripped)
        if match:
            current, total = match.group(1), match.group(2)
            progress = f"Extracting features: {current}/{total}"
            if progress != last_progress:
                if is_tty:
                    print(f"\r    {progress}    ", end="")
                else:
                    print(f"    {progress}")
                sys.stdout.flush()
                last_progress = progress
            continue

        # Check for matching progress
        match = match_pattern.search(line_stripped)
        if match:
            current, total = match.group(1), match.group(2)
            progress = f"Matching: block {current}/{total}"
            if progress != last_progress:
                if is_tty:
                    print(f"\r    {progress}    ", end="")
                else:
                    print(f"    {progress}")
                sys.stdout.flush()
                last_progress = progress
            continue

        # Check for registration progress (mapper)
        match = register_pattern.search(line_stripped)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            progress = f"Registered {current}/{total} images"
            if progress != last_progress:
                if is_tty:
                    print(f"\r    {progress}    ", end="")
                else:
                    print(f"    {progress}")
                sys.stdout.flush()
                last_progress = progress
            continue

    # Clear the progress line and print newline (only needed for TTY mode)
    if last_progress and is_tty:
        print()  # newline after progress

    process.wait()

    stdout = ''.join(stdout_lines)
    if process.returncode != 0:
        print(f"    Error output:\n{stdout[-2000:]}", file=sys.stderr)  # Last 2000 chars
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, "")

    # Create a CompletedProcess-like result
    class Result:
        def __init__(self):
            self.returncode = process.returncode
            self.stdout = stdout
            self.stderr = ""
    return Result()


def extract_features(
    database_path: Path,
    image_path: Path,
    camera_model: str = "OPENCV",
    max_features: int = 8192,
    single_camera: bool = True,
    mask_path: Optional[Path] = None,
    use_gpu: bool = True
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
    """
    def _build_args(gpu: bool) -> dict:
        """Build argument dict with specified GPU setting."""
        args = {
            "database_path": str(database_path),
            "image_path": str(image_path),
            "ImageReader.camera_model": camera_model,
            "ImageReader.single_camera": 1 if single_camera else 0,
            "SiftExtraction.max_num_features": max_features,
            "SiftExtraction.use_gpu": 1 if gpu else 0,
        }
        if gpu:
            args["SiftExtraction.gpu_index"] = "0"
        if mask_path and mask_path.exists():
            args["ImageReader.mask_path"] = str(mask_path)
        return args

    if mask_path and mask_path.exists():
        print(f"    Using masks from: {mask_path}")

    if use_gpu:
        try:
            args = _build_args(gpu=True)
            run_colmap_command("feature_extractor", args, "Extracting features (GPU)")
        except subprocess.CalledProcessError as e:
            error_output = str(e.stdout) if hasattr(e, 'stdout') and e.stdout else str(e)
            if "context" in error_output.lower() or "opengl" in error_output.lower() or e.returncode < 0:
                print("    GPU feature extraction failed (OpenGL context error), falling back to CPU...")
                if database_path.exists():
                    database_path.unlink()
                args = _build_args(gpu=False)
                run_colmap_command("feature_extractor", args, "Extracting features (CPU)")
            else:
                raise
    else:
        args = _build_args(gpu=False)
        run_colmap_command("feature_extractor", args, "Extracting features (CPU)")


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
    def _run_matcher(gpu: bool):
        """Run the matcher with specified GPU setting."""
        if matcher_type == "sequential":
            args = {
                "database_path": str(database_path),
                "SequentialMatching.overlap": sequential_overlap,
                "SiftMatching.use_gpu": 1 if gpu else 0,
                "SiftMatching.max_num_matches": 32768,
            }
            # Explicitly set GPU index to avoid multi-GPU initialization issues (Issue #627)
            if gpu:
                args["SiftMatching.gpu_index"] = "0"
            mode = "GPU" if gpu else "CPU"
            run_colmap_command("sequential_matcher", args, f"Matching features (sequential, {mode})")
        elif matcher_type == "exhaustive":
            args = {
                "database_path": str(database_path),
                "SiftMatching.use_gpu": 1 if gpu else 0,
                "SiftMatching.max_num_matches": 32768,
            }
            # Explicitly set GPU index to avoid multi-GPU initialization issues (Issue #627)
            if gpu:
                args["SiftMatching.gpu_index"] = "0"
            mode = "GPU" if gpu else "CPU"
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

    # Check if reconstruction produced output
    model_path = output_path / "0"
    if not model_path.exists():
        print("    Warning: No reconstruction model produced", file=sys.stderr)
        return False

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
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", str(cameras_path.parent),
                "--output_path", str(temp_dir),
                "--output_type", "TXT"
            ], capture_output=True, check=True)

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
            result = subprocess.run([
                "colmap", "model_converter",
                "--input_path", str(images_path.parent),
                "--output_path", str(temp_dir),
                "--output_type", "TXT"
            ], capture_output=True, text=True)

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


def quaternion_to_rotation_matrix(quat: list) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat

    # Normalize
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions.

    Args:
        q1: Start quaternion [w, x, y, z]
        q2: End quaternion [w, x, y, z]
        t: Interpolation factor (0 = q1, 1 = q2)

    Returns:
        Interpolated quaternion
    """
    # Normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If negative dot, negate one quaternion (shortest path)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Compute SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    """Cubic Bezier interpolation.

    Args:
        p0, p1, p2, p3: Control points
        t: Parameter (0 to 1)

    Returns:
        Interpolated point
    """
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3


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
    with open(output_dir / "extrinsics.json", "w") as f:
        json.dump(extrinsics_data, f, indent=2)

    # Save intrinsics
    with open(output_dir / "intrinsics.json", "w") as f:
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
    with open(output_dir / "colmap_raw.json", "w") as f:
        json.dump(colmap_data, f, indent=2)

    print(f"    Exported {len(extrinsics)} camera frames to {output_dir}")
    return True


def run_colmap_pipeline(
    project_dir: Path,
    quality: str = "medium",
    run_dense: bool = False,
    run_mesh: bool = False,
    camera_model: str = "OPENCV",
    use_masks: bool = True,
    max_gap: int = 12,
) -> bool:
    """Run the complete COLMAP reconstruction pipeline.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high', 'slow')
        run_dense: Whether to run dense reconstruction
        run_mesh: Whether to generate mesh (requires dense)
        camera_model: COLMAP camera model to use
        use_masks: If True, automatically use segmentation masks from roto/ (if available)
        max_gap: Maximum frame gap to interpolate if COLMAP misses frames (default: 12)

    Returns:
        True if reconstruction succeeded
    """
    if not check_colmap_available():
        print("Error: COLMAP not found. Install with:", file=sys.stderr)
        print("  Ubuntu: sudo apt install colmap", file=sys.stderr)
        print("  Conda: conda install -c conda-forge colmap", file=sys.stderr)
        return False

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
    if quality == "slow":
        print(f"  Warning: Slow-camera mode uses aggressive settings for minimal motion")
        print(f"           Results may be jittery due to low parallax")

    # Check for segmentation masks
    mask_dir = project_dir / "roto"
    mask_path = None
    if use_masks and mask_dir.exists():
        mask_count = len(list(mask_dir.glob("*.png"))) + len(list(mask_dir.glob("*.jpg")))
        if mask_count > 0:
            mask_path = mask_dir
            print(f"Dynamic scene segmentation: Enabled ({mask_count} masks)")
            print(f"  → Excluding dynamic regions from feature extraction")
        else:
            print(f"Dynamic scene segmentation: Disabled (no masks found)")
    else:
        print(f"Dynamic scene segmentation: Disabled")
    print()

    # Setup paths
    colmap_dir = project_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

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
            )

            # Verify features were extracted
            num_images, total_keypoints = verify_database_has_features(database_path)
            if num_images == 0 or total_keypoints == 0:
                print(f"    Error: No features extracted from images", file=sys.stderr)
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

            # Optional: Dense reconstruction
            if run_dense:
                print("\n[Dense] Running Dense Reconstruction")
                if run_dense_reconstruction(
                    image_path=frames_dir,
                    sparse_path=sparse_model,
                    output_path=dense_path,
                    max_image_size=preset["dense_max_size"],
                ):
                    # Copy point cloud to project
                    if (dense_path / "fused.ply").exists():
                        shutil.copy(
                            dense_path / "fused.ply",
                            project_dir / "camera" / "pointcloud.ply"
                        )
                        print(f"    Point cloud saved to camera/pointcloud.ply")
                else:
                    print("    Dense reconstruction failed (continuing)", file=sys.stderr)

            # Optional: Mesh generation
            if run_mesh and run_dense:
                print("\n[Mesh] Generating Mesh")
                mesh_path = project_dir / "camera" / "mesh.ply"
                if run_mesh_reconstruction(dense_path, mesh_path):
                    print(f"    Mesh saved to camera/mesh.ply")
                else:
                    print("    Mesh generation failed (continuing)", file=sys.stderr)

            print(f"\n{'='*60}")
            print(f"COLMAP Reconstruction Complete")
            print(f"{'='*60}")
            print(f"Sparse model: {sparse_model}")
            print(f"Camera data: {camera_dir}")
            if run_dense:
                print(f"Dense model: {dense_path}")
            print()

            return True

        except subprocess.CalledProcessError as e:
            print(f"\nCOLMAP command failed: {e}", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print(f"\nCOLMAP command timed out", file=sys.stderr)
            return False


def main():
    # Check conda environment (warn but don't exit - allow --help to work)
    check_conda_env_or_warn()

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
        "--dense", "-d",
        action="store_true",
        help="Run dense reconstruction (slower, produces point cloud)"
    )
    parser.add_argument(
        "--mesh", "-m",
        action="store_true",
        help="Generate mesh from dense reconstruction (requires --dense)"
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
        "--check",
        action="store_true",
        help="Check if COLMAP is available and exit"
    )

    args = parser.parse_args()

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
        run_dense=args.dense,
        run_mesh=args.mesh,
        camera_model=args.camera_model,
        use_masks=not args.no_masks,
        max_gap=args.max_gap,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
