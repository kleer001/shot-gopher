#!/usr/bin/env python3
"""Human motion capture with SMPL-X topology using GVHMR.

Reconstructs people from monocular video with:
- SMPL-X body model (standard topology + UVs)
- World-space motion tracking via GVHMR

Pipeline:
1. Motion tracking (GVHMR) -> skeleton animation in world space
2. Output conversion to standardized motion.pkl format

GVHMR features:
- State-of-the-art accuracy (SIGGRAPH Asia 2024)
- Robust camera motion handling (SimpleVO)
- Multi-person support (outputs per-person results)
- Can leverage COLMAP focal length for improved accuracy

Usage:
    python run_mocap.py <project_dir> [options]

Example:
    python run_mocap.py /path/to/projects/My_Shot
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict

from env_config import require_conda_env, INSTALL_DIR
from log_manager import LogCapture


_DEPS_CHECKED = {}


def check_dependency(name: str, import_path: str = None, command: str = None) -> bool:
    """Check if a dependency is available.

    Args:
        name: Dependency name for caching
        import_path: Python import path (e.g., "smplx")
        command: Shell command to check (e.g., ["gvhmr", "--help"])

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

    return deps


def print_dependency_status():
    """Print status of all dependencies."""
    deps = check_all_dependencies()

    print("\nDependency Status:")
    print("=" * 60)

    print("\nCore (required):")
    for name in ["numpy", "pytorch", "smplx", "trimesh", "opencv", "pillow"]:
        status = "OK" if deps[name] else "X"
        print(f"  {status} {name}")

    print()


def install_instructions():
    """Print installation instructions for missing dependencies."""
    print("\n" + "=" * 60)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 60)

    print("\nCore dependencies:")
    print("  pip install numpy torch smplx trimesh opencv-python pillow")

    print("\nGVHMR (world-grounded motion):")
    print("  git clone https://github.com/zju3dv/GVHMR.git")
    print("  cd GVHMR && pip install -e .")
    print("  # Download checkpoints from Google Drive (see GVHMR repo)")

    print("\nSMPL-X body model:")
    print("  1. Register at https://smpl-x.is.tue.mpg.de/")
    print(f"  2. Download models -> place in {INSTALL_DIR}/smplx_models/")
    print()


def colmap_intrinsics_to_focal_mm(
    intrinsics_path: Path,
    sensor_width_mm: Optional[float] = None,
    verbose: bool = True,
) -> Optional[float]:
    """Convert COLMAP intrinsics to focal length in mm.

    Args:
        intrinsics_path: Path to camera/intrinsics.json
        sensor_width_mm: Sensor width in mm (None = assume 36mm full-frame)
        verbose: Print warning messages for errors

    Returns:
        Focal length in millimeters, or None if conversion fails
    """
    import json

    if not intrinsics_path.exists():
        if verbose:
            camera_dir = intrinsics_path.parent
            if camera_dir.exists():
                print(f"  Warning: Camera directory exists but intrinsics.json missing: {camera_dir}")
        return None

    try:
        with open(intrinsics_path, encoding='utf-8') as f:
            intrinsics = json.load(f)

        fx = intrinsics.get("fx", intrinsics.get("focal_x"))
        width = intrinsics.get("width", 1920)

        if fx is None:
            if verbose:
                print(f"  Warning: intrinsics.json missing 'fx' or 'focal_x' field: {intrinsics_path}")
            return None

        if sensor_width_mm is None:
            sensor_width_mm = 36.0

        focal_mm = fx * sensor_width_mm / width
        return focal_mm

    except json.JSONDecodeError as e:
        if verbose:
            print(f"  Warning: Invalid JSON in intrinsics.json: {e}")
        return None
    except (KeyError, TypeError) as e:
        if verbose:
            print(f"  Warning: Error reading intrinsics.json: {e}")
        return None


def detect_static_camera(
    extrinsics_path: Path,
    threshold_meters: float = 0.01,
    verbose: bool = True,
) -> bool:
    """Detect if camera is static from COLMAP extrinsics.

    Analyzes camera translation variance across all frames.

    Args:
        extrinsics_path: Path to extrinsics.json
        threshold_meters: Max camera movement variance to consider "static"
        verbose: Print warning messages for errors

    Returns:
        True if camera appears static
    """
    import json

    if not extrinsics_path.exists():
        if verbose:
            camera_dir = extrinsics_path.parent
            if camera_dir.exists():
                print(f"  Warning: Camera directory exists but extrinsics.json missing: {camera_dir}")
        return False

    try:
        import numpy as np

        with open(extrinsics_path, encoding='utf-8') as f:
            extrinsics = json.load(f)

        if not extrinsics:
            if verbose:
                print(f"  Warning: extrinsics.json is empty: {extrinsics_path}")
            return False

        if not isinstance(extrinsics, list):
            if verbose:
                print(f"  Warning: extrinsics.json should be a list of matrices: {extrinsics_path}")
            return False

        if len(extrinsics) < 2:
            if verbose:
                print(f"  Warning: extrinsics.json has only {len(extrinsics)} frame(s), need 2+ for motion detection")
            return False

        translations = []
        for i, matrix_data in enumerate(extrinsics):
            if isinstance(matrix_data, list) and len(matrix_data) >= 3:
                try:
                    translations.append([matrix_data[0][3], matrix_data[1][3], matrix_data[2][3]])
                except (IndexError, TypeError):
                    if verbose:
                        print(f"  Warning: Invalid matrix format at frame {i} in extrinsics.json")
                    return False

        if not translations:
            if verbose:
                print(f"  Warning: No valid 4x4 matrices found in extrinsics.json")
            return False

        translations = np.array(translations)
        variance = np.var(translations, axis=0).sum()

        return bool(variance < threshold_meters)

    except json.JSONDecodeError as e:
        if verbose:
            print(f"  Warning: Invalid JSON in extrinsics.json: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"  Warning: Error reading extrinsics.json: {e}")
        return False


def export_gvhmr_camera(
    gvhmr_output_dir: Path,
    camera_output_dir: Path,
    n_frames: int,
    image_width: int,
    image_height: int,
) -> bool:
    """Export body-aligned camera from GVHMR SMPL orient params.

    Derives per-frame camera extrinsics algebraically from the
    relationship between smpl_params_global and smpl_params_incam
    orient/transl fields. The resulting camera is consistent with
    the body mesh in GVHMR-global coordinates.

    Creates:
    - extrinsics.json: List of 4x4 camera-to-world matrices
    - intrinsics.json: {fx, fy, cx, cy, width, height, model, params}
    - gvhmr_raw.json: Source metadata

    Args:
        gvhmr_output_dir: Directory containing GVHMR output tree
        camera_output_dir: Output directory (typically project/mocap_camera/)
        n_frames: Total number of source frames
        image_width: Source image width
        image_height: Source image height

    Returns:
        True if export succeeded
    """
    import json

    import numpy as np
    import torch

    from camera_alignment import compute_aligned_camera, compute_pelvis_joint

    def _to_numpy(obj: object) -> object:
        if hasattr(obj, 'numpy'):
            return obj.numpy()
        elif isinstance(obj, dict):
            return {k: _to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_numpy(v) for v in obj]
        return obj

    gvhmr_files = list(gvhmr_output_dir.rglob("hmr4d*.pt"))
    if not gvhmr_files:
        print(f"  No GVHMR output found in {gvhmr_output_dir}", file=sys.stderr)
        return False

    gvhmr_file = gvhmr_files[0]
    gvhmr_data = torch.load(gvhmr_file, map_location='cpu', weights_only=False)
    gvhmr_data = _to_numpy(gvhmr_data)

    smplx_model_path = INSTALL_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    betas = np.asarray(gvhmr_data["smpl_params_global"]["betas"][0], dtype=np.float64)
    pelvis_joint = compute_pelvis_joint(smplx_model_path, betas)

    result = compute_aligned_camera(gvhmr_data, image_width, image_height, pelvis_joint)

    camera_output_dir.mkdir(parents=True, exist_ok=True)

    extrinsics_data = [m.tolist() for m in result["extrinsics"]]
    with open(camera_output_dir / "extrinsics.json", "w", encoding='utf-8') as f:
        json.dump(extrinsics_data, f, indent=2)

    with open(camera_output_dir / "intrinsics.json", "w", encoding='utf-8') as f:
        json.dump(result["intrinsics"], f, indent=2)

    with open(camera_output_dir / "gvhmr_raw.json", "w", encoding='utf-8') as f:
        json.dump(result["metadata"], f, indent=2)

    intr = result["intrinsics"]
    n_exported = len(result["extrinsics"])
    print(f"  -> Exported camera to {camera_output_dir}")
    print(f"    Intrinsics: fx={intr['fx']:.1f}, fy={intr['fy']:.1f}, cx={intr['cx']:.1f}, cy={intr['cy']:.1f}")
    print(f"    Extrinsics: {n_exported} frames (body-aligned)")
    return True


def composite_frames_with_matte(
    frames_dir: Path,
    matte_dir: Path,
    output_dir: Path,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> List[Path]:
    """Composite source frames with roto mattes to isolate a person.

    Multiplies each frame by its corresponding matte, resulting in the
    person isolated on a black background.

    Args:
        frames_dir: Directory containing source frames
        matte_dir: Directory containing matte frames (white=person, black=background)
        output_dir: Directory to write composited frames
        start_frame: Start frame (1-indexed, inclusive)
        end_frame: End frame (1-indexed, inclusive)

    Returns:
        List of paths to composited frames, empty if failed
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("Error: PIL and numpy required for matte compositing", file=sys.stderr)
        return []

    source_frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not source_frames:
        print(f"Error: No source frames in {frames_dir}", file=sys.stderr)
        return []

    matte_frames = sorted(matte_dir.glob("*.png")) + sorted(matte_dir.glob("*.jpg"))
    if not matte_frames:
        print(f"Error: No matte frames in {matte_dir}", file=sys.stderr)
        return []

    if start_frame is not None or end_frame is not None:
        actual_start = (start_frame or 1) - 1
        actual_end = end_frame if end_frame else len(source_frames)
        source_frames = source_frames[actual_start:actual_end]
        matte_frames = matte_frames[actual_start:actual_end]

    if len(source_frames) != len(matte_frames):
        print(f"Error: Frame count mismatch - {len(source_frames)} source vs {len(matte_frames)} matte", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    composited = []

    for i, (src_path, matte_path) in enumerate(zip(source_frames, matte_frames)):
        try:
            src_img = Image.open(src_path).convert("RGB")
            matte_img = Image.open(matte_path).convert("L")

            if src_img.size != matte_img.size:
                matte_img = matte_img.resize(src_img.size, Image.LANCZOS)

            src_array = np.array(src_img, dtype=np.float32)
            matte_array = np.array(matte_img, dtype=np.float32) / 255.0
            matte_array = matte_array[:, :, np.newaxis]

            result_array = (src_array * matte_array).astype(np.uint8)
            result_img = Image.fromarray(result_array)

            output_path = output_dir / f"frame_{i+1:05d}.png"
            result_img.save(output_path)
            composited.append(output_path)

        except Exception as e:
            print(f"Error compositing frame {i+1}: {e}", file=sys.stderr)
            return []

    return composited


def find_or_create_video(
    project_dir: Path,
    fps: float = 24.0,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    mocap_person: Optional[str] = None,
) -> Optional[Path]:
    """Find source video or create one from frames.

    GVHMR requires video input, not frames. This function:
    1. If mocap_person specified: composites frames with roto matte to isolate person
    2. Otherwise looks for original source video (trims if frame range specified)
    3. Falls back to re-encoding frames (with frame range)

    Args:
        project_dir: Project directory
        fps: Frame rate for video encoding
        start_frame: Start frame (1-indexed, inclusive)
        end_frame: End frame (1-indexed, inclusive)
        mocap_person: Roto person folder name (e.g., 'person_00') to isolate

    Returns:
        Path to video file, or None if creation fails
    """
    has_frame_range = start_frame is not None or end_frame is not None
    source_dir = project_dir / "source"

    if mocap_person:
        frames_dir = source_dir / "frames"
        matte_dir = project_dir / "roto" / mocap_person

        if not frames_dir.exists():
            print(f"Error: Source frames required for matte isolation: {frames_dir}", file=sys.stderr)
            return None

        if not matte_dir.exists():
            print(f"Error: Roto matte directory not found: {matte_dir}", file=sys.stderr)
            return None

        masked_dir = source_dir / f"_gvhmr_masked_{mocap_person}"
        range_suffix = ""
        if has_frame_range:
            range_suffix = f"_{start_frame or 1}_{end_frame or 'end'}"

        print(f"  → Isolating {mocap_person} using roto matte...")
        composited = composite_frames_with_matte(
            frames_dir, matte_dir, masked_dir,
            start_frame=start_frame, end_frame=end_frame
        )

        if not composited:
            return None

        print(f"  OK Composited {len(composited)} frames")

        video_path = source_dir / f"_gvhmr_masked_{mocap_person}{range_suffix}.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(masked_dir / "frame_%05d.png"),
            "-frames:v", str(len(composited)),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and video_path.exists():
                print(f"  OK Created masked video: {video_path.name}")
                return video_path
        except Exception as e:
            print(f"  Error creating masked video: {e}", file=sys.stderr)

        return None

    source_video = None
    video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    for ext in video_extensions:
        for video_file in source_dir.glob(f"*{ext}"):
            if not video_file.name.startswith("_"):
                source_video = video_file
                break
        if source_video:
            break
        for video_file in source_dir.glob(f"*{ext.upper()}"):
            if not video_file.name.startswith("_"):
                source_video = video_file
                break
        if source_video:
            break

    if source_video and has_frame_range:
        trimmed_path = source_dir / "_gvhmr_trimmed.mp4"
        start_time = (start_frame - 1) / fps if start_frame else 0
        trim_cmd = ["ffmpeg", "-y", "-i", str(source_video)]
        if start_frame:
            trim_cmd.extend(["-ss", str(start_time)])
        if end_frame:
            duration = (end_frame - (start_frame or 1) + 1) / fps
            trim_cmd.extend(["-t", str(duration)])
        trim_cmd.extend([
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(trimmed_path)
        ])
        range_str = f"{start_frame or 1}-{end_frame or 'end'}"
        print(f"  → Trimming video to frames {range_str}...")
        try:
            result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and trimmed_path.exists():
                print(f"  OK Created trimmed video: {trimmed_path.name}")
                return trimmed_path
        except Exception as e:
            print(f"  Error trimming video: {e}", file=sys.stderr)
        return None

    if source_video and not has_frame_range:
        return source_video

    frames_dir = source_dir / "frames"
    if not frames_dir.exists():
        return None

    frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return None

    if has_frame_range:
        actual_start = (start_frame or 1) - 1
        actual_end = end_frame if end_frame else len(frames)
        frames = frames[actual_start:actual_end]
        if not frames:
            print(f"Error: No frames in range {start_frame}-{end_frame}", file=sys.stderr)
            return None

    range_suffix = ""
    if has_frame_range:
        range_suffix = f"_{start_frame or 1}_{end_frame or 'end'}"
    video_path = source_dir / f"_gvhmr_input{range_suffix}.mp4"

    if video_path.exists() and not has_frame_range:
        return video_path

    range_str = f" (frames {start_frame or 1}-{end_frame or len(frames)})" if has_frame_range else ""
    print(f"  → Creating video from {len(frames)} frames{range_str}...")

    if (frames_dir / "frame_0001.png").exists():
        frame_pattern = frames_dir / "frame_%04d.png"
        pattern_start = 1
    elif (frames_dir / "frame_00001.png").exists():
        frame_pattern = frames_dir / "frame_%05d.png"
        pattern_start = 1
    elif (frames_dir / "frame_0001.jpg").exists():
        frame_pattern = frames_dir / "frame_%04d.jpg"
        pattern_start = 1
    elif (frames_dir / "frame_00001.jpg").exists():
        frame_pattern = frames_dir / "frame_%05d.jpg"
        pattern_start = 1
    else:
        first_frame = frames[0]
        stem = first_frame.stem
        if '_' in stem:
            prefix = stem.rsplit('_', 1)[0]
            num_part = stem.rsplit('_', 1)[1]
            num_digits = len(num_part)
            frame_pattern = first_frame.parent / f"{prefix}_%0{num_digits}d{first_frame.suffix}"
            pattern_start = int(num_part)
        else:
            frame_pattern = first_frame.parent / f"%04d{first_frame.suffix}"
            pattern_start = 1

    cmd = ["ffmpeg", "-y", "-framerate", str(fps)]

    if has_frame_range:
        first_sliced_frame = frames[0]
        stem = first_sliced_frame.stem
        if '_' in stem:
            num_part = stem.rsplit('_', 1)[1]
            try:
                actual_start_number = int(num_part)
            except ValueError:
                actual_start_number = pattern_start
        else:
            actual_start_number = pattern_start
        cmd.extend(["-start_number", str(actual_start_number)])

    cmd.extend(["-i", str(frame_pattern)])

    if has_frame_range:
        frame_count = len(frames)
        cmd.extend(["-frames:v", str(frame_count)])

    cmd.extend([
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and video_path.exists():
            print(f"  OK Created video: {video_path.name}")
            return video_path
    except Exception as e:
        print(f"  Error creating video: {e}", file=sys.stderr)

    return None


def _find_conda() -> Optional[str]:
    """Find conda or mamba executable.

    Returns:
        Path to conda/mamba executable, or None if not found
    """
    import os
    for cmd in ["conda", "mamba"]:
        if shutil.which(cmd):
            return cmd
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return str(conda_exe)
    return None


def _patch_gvhmr_f_mm_type(demo_script: Path) -> None:
    """Patch GVHMR demo.py to accept float focal lengths if needed.

    Upstream defines --f_mm as type=int, but camera solves produce floats.
    The downstream math (create_camera_sensor) is pure multiplication.
    """
    if not demo_script.exists():
        return
    content = demo_script.read_text()
    patched = re.sub(
        r'("--f_mm",\s*)type=int',
        r'\1type=float',
        content,
    )
    if patched != content:
        demo_script.write_text(patched)


def _inject_mmcam_camera(
    mmcam_extrinsics_path: Path,
    slam_output_path: Path,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> bool:
    """Convert mmcam extrinsics to GVHMR's slam_results.pt format.

    Reads camera-to-world 4x4 matrices from mmcam, inverts to
    world-to-camera, slices for frame range, and saves as the
    (L, 4, 4) numpy array that GVHMR's load_data_dict expects.

    Args:
        mmcam_extrinsics_path: Path to mmcam extrinsics.json
        slam_output_path: Where to write slam_results.pt
        start_frame: Start frame (1-indexed, inclusive)
        end_frame: End frame (1-indexed, inclusive)

    Returns:
        True if injection succeeded
    """
    import json
    import numpy as np
    import torch

    with open(mmcam_extrinsics_path, "r", encoding="utf-8") as f:
        c2w_list = json.load(f)

    c2w = np.array(c2w_list, dtype=np.float64)

    if start_frame is not None or end_frame is not None:
        s = (start_frame or 1) - 1
        e = end_frame or len(c2w)
        c2w = c2w[s:e]

    R_c2w = c2w[:, :3, :3]
    t_c2w = c2w[:, :3, 3]
    R_w2c = np.swapaxes(R_c2w, -2, -1)
    t_w2c = -np.einsum("fij,fj->fi", R_w2c, t_c2w)

    T_w2c = np.broadcast_to(np.eye(4), (len(c2w), 4, 4)).copy()
    T_w2c[:, :3, :3] = R_w2c
    T_w2c[:, :3, 3] = t_w2c

    slam_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(T_w2c).float(), slam_output_path)
    print(f"    Injected {len(T_w2c)} mmcam camera poses as slam_results.pt")
    return True


def run_gvhmr_motion_tracking(
    project_dir: Path,
    focal_mm: Optional[float] = None,
    static_camera: bool = False,
    output_dir: Optional[Path] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: float = 24.0,
    mocap_person: Optional[str] = None,
    mmcam_extrinsics_path: Optional[Path] = None,
) -> bool:
    """Run GVHMR for world-grounded motion tracking.

    GVHMR runs in its own dedicated conda environment 'gvhmr' with
    Python 3.10 and specific PyTorch/CUDA versions.

    Args:
        project_dir: Project directory
        focal_mm: Focal length in mm (from COLMAP intrinsics)
        static_camera: Skip visual odometry for static cameras
        output_dir: Output directory for results
        start_frame: Start frame (1-indexed, inclusive)
        end_frame: End frame (1-indexed, inclusive)
        fps: Frames per second (for trimming calculations)
        mocap_person: Roto person folder to isolate (e.g., 'person_00')
        mmcam_extrinsics_path: Path to mmcam extrinsics.json to use
            instead of GVHMR's built-in visual odometry

    Returns:
        True if successful
    """
    gvhmr_dir = INSTALL_DIR / "GVHMR"
    if not gvhmr_dir.exists():
        print("Error: GVHMR not installed", file=sys.stderr)
        print("Run the installation wizard to install GVHMR", file=sys.stderr)
        return False

    conda_exe = _find_conda()
    if not conda_exe:
        print("Error: Conda not found - required for GVHMR", file=sys.stderr)
        return False

    output_dir = output_dir or project_dir / "mocap" / "gvhmr"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_or_create_video(project_dir, fps, start_frame, end_frame, mocap_person)
    if not video_path:
        print("Error: No video file found and could not create from frames", file=sys.stderr)
        return False

    print(f"\n{'=' * 60}")
    print("GVHMR Motion Tracking")
    print("=" * 60)
    print(f"Video: {video_path}")
    if mocap_person:
        print(f"Isolated person: {mocap_person} (via roto matte)")
    if start_frame or end_frame:
        range_str = f"{start_frame or 1}-{end_frame or 'end'}"
        print(f"Frame range: {range_str}")
    if focal_mm:
        print(f"Focal length: {focal_mm:.1f}mm")
    if mmcam_extrinsics_path and not static_camera:
        print(f"Camera source: mmcam ({mmcam_extrinsics_path.parent.name})")
    elif static_camera:
        print(f"Mode: Static camera (skipping visual odometry)")
    print(f"Output: {output_dir}")
    print()

    try:
        demo_script = gvhmr_dir / "tools" / "demo" / "demo.py"
        if not demo_script.exists():
            demo_script = gvhmr_dir / "demo.py"

        _patch_gvhmr_f_mm_type(demo_script)

        if mmcam_extrinsics_path and not static_camera:
            preprocess_dir = output_dir / video_path.stem / "preprocess"
            slam_path = preprocess_dir / "slam_results.pt"
            _inject_mmcam_camera(
                mmcam_extrinsics_path, slam_path,
                start_frame=start_frame, end_frame=end_frame,
            )

        gvhmr_args = [
            "python", str(demo_script),
            "--video", str(video_path),
            "--output_root", str(output_dir),
        ]

        if static_camera:
            gvhmr_args.append("--static_cam")

        if focal_mm:
            gvhmr_args.extend(["--f_mm", str(focal_mm)])

        cmd = [conda_exe, "run", "-n", "gvhmr", "--no-capture-output"] + gvhmr_args

        print(f"  → Running GVHMR in 'gvhmr' conda environment...")
        print(f"    $ {' '.join(gvhmr_args)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=str(gvhmr_dir)
        )

        if result.returncode != 0:
            print(f"Error: GVHMR failed", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            if result.stdout:
                print(result.stdout, file=sys.stderr)
            return False

        output_files = list(output_dir.rglob("hmr4d*.pt"))
        if not output_files:
            print(f"Error: No GVHMR output files found", file=sys.stderr)
            return False

        print(f"  OK Motion tracking complete")
        print(f"    Output files: {len(output_files)}")

        return True

    except subprocess.TimeoutExpired:
        print("Error: GVHMR timed out (>2 hours)", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error running GVHMR: {e}", file=sys.stderr)
        return False


def _convert_single_gvhmr_file(
    gvhmr_file_path: Path,
    output_path: Path,
    np_module,
    gender: str = "neutral"
) -> bool:
    """Convert a single GVHMR output file to motion format.

    Args:
        gvhmr_file_path: Path to GVHMR .pt file (hmr4d_results.pt)
        output_path: Path for output file
        np_module: numpy module reference
        gender: Body model gender (neutral, male, female)

    Returns:
        True if conversion successful
    """
    import pickle
    import torch
    np = np_module

    def to_numpy(obj):
        if hasattr(obj, 'numpy'):
            return obj.numpy()
        elif isinstance(obj, dict):
            return {k: to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_numpy(v) for v in obj]
        return obj

    try:
        gvhmr_data = torch.load(gvhmr_file_path, map_location='cpu', weights_only=False)
        gvhmr_data = to_numpy(gvhmr_data)

        if 'smpl_params_global' in gvhmr_data:
            params = gvhmr_data['smpl_params_global']
        elif 'global_orient' in gvhmr_data:
            params = gvhmr_data
        else:
            params = gvhmr_data

        body_pose = params.get('body_pose', params.get('poses'))
        global_orient = params.get('global_orient')
        transl = params.get('transl', params.get('trans'))
        betas = params.get('betas')

        if body_pose is None:
            print(f"Error: Could not find body_pose in {gvhmr_file_path.name}", file=sys.stderr)
            return False

        body_pose = np.array(body_pose)
        if body_pose.ndim < 2 or body_pose.shape[0] == 0:
            print(f"Error: Invalid body_pose shape in {gvhmr_file_path.name}", file=sys.stderr)
            return False

        n_frames = len(body_pose)

        if global_orient is not None:
            global_orient = np.array(global_orient)
            if global_orient.ndim == 1:
                global_orient = global_orient.reshape(1, -1)
            if len(global_orient) == 1 and n_frames > 1:
                global_orient = np.tile(global_orient, (n_frames, 1))
        else:
            global_orient = np.zeros((n_frames, 3))

        if body_pose.shape[1] == 63:
            poses = np.concatenate([
                global_orient,
                body_pose,
                np.zeros((n_frames, 6))
            ], axis=1)
        elif body_pose.shape[1] > 63:
            poses = np.concatenate([
                global_orient,
                body_pose[:, :63],
                np.zeros((n_frames, 6))
            ], axis=1)
        else:
            padding_needed = 69 - body_pose.shape[1]
            poses = np.concatenate([
                global_orient,
                body_pose,
                np.zeros((n_frames, padding_needed))
            ], axis=1)

        if transl is not None:
            transl = np.array(transl)
        else:
            transl = np.zeros((n_frames, 3))

        if betas is not None:
            betas = np.array(betas)
            if betas.ndim > 1:
                betas = betas[0]
        else:
            betas = np.zeros(10)

        motion_format = {
            'poses': poses,
            'trans': transl,
            'betas': betas,
            'gender': gender
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(motion_format, f)

        return True

    except Exception as e:
        print(f"Error converting {gvhmr_file_path.name}: {e}", file=sys.stderr)
        return False


def list_detected_persons(gvhmr_output_dir: Path) -> List[str]:
    """List all persons detected by GVHMR.

    Args:
        gvhmr_output_dir: Directory containing GVHMR output

    Returns:
        List of person directory names (e.g., ['person_0', 'person_1'])
    """
    if not gvhmr_output_dir.exists():
        return []

    person_dirs = sorted([d.name for d in gvhmr_output_dir.iterdir()
                         if d.is_dir() and d.name.startswith("person_")])
    return person_dirs


def save_motion_output(
    gvhmr_output_dir: Path,
    output_path: Path,
    gender: str = "neutral",
) -> bool:
    """Convert GVHMR output to standardized motion.pkl format.

    Handles both single-person and multi-person GVHMR outputs:
    - Single person: Creates motion.pkl
    - Multi-person: Creates motion_person_0.pkl, motion_person_1.pkl, etc.
                    Also creates motion.pkl as copy of person_0

    Args:
        gvhmr_output_dir: Directory containing GVHMR output
        output_path: Path for output file (motion.pkl)
        gender: Body model gender (neutral, male, female)

    Returns:
        True if conversion successful
    """
    import pickle
    import shutil

    try:
        import numpy as np
    except ImportError:
        print("Error: numpy required for format conversion", file=sys.stderr)
        return False

    if not gvhmr_output_dir.exists():
        print(f"Error: GVHMR output directory not found: {gvhmr_output_dir}", file=sys.stderr)
        return False

    person_dirs = sorted([d for d in gvhmr_output_dir.iterdir()
                         if d.is_dir() and d.name.startswith("person_")])

    if person_dirs:
        print(f"  → Found {len(person_dirs)} person(s) in GVHMR output")
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        person_outputs = {}

        for idx, person_dir in enumerate(person_dirs):
            person_id = person_dir.name
            person_files = list(person_dir.glob("hmr4d*.pt"))

            if not person_files:
                print(f"    Warning: No hmr4d output in {person_id}")
                continue

            person_output = output_dir / f"motion_{person_id}.pkl"
            print(f"  → Converting {person_id}...")

            if _convert_single_gvhmr_file(person_files[0], person_output, np, gender):
                success_count += 1
                person_outputs[idx] = person_output

        if success_count > 0 and person_outputs:
            primary_idx = min(person_outputs.keys())
            primary_output = person_outputs[primary_idx]
            shutil.copy2(primary_output, output_path)
            print(f"  OK Converted {success_count} person(s) to motion format")
            print(f"    Primary output: {output_path} (person_{primary_idx})")
            return True
        else:
            print("Error: No persons converted successfully", file=sys.stderr)
            return False

    gvhmr_files = list(gvhmr_output_dir.rglob("hmr4d*.pt"))
    if not gvhmr_files:
        print(f"Error: No GVHMR output (hmr4d*.pt) found in {gvhmr_output_dir}", file=sys.stderr)
        return False

    gvhmr_output_path = gvhmr_files[0]
    print(f"  → Converting {gvhmr_output_path.name} to motion format...")

    if _convert_single_gvhmr_file(gvhmr_output_path, output_path, np, gender):
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        n_frames = len(data['poses'])
        print(f"  OK Converted {n_frames} frames to motion format")
        print(f"    Output: {output_path}")
        return True

    return False


def run_mocap_pipeline(
    project_dir: Path,
    use_colmap_intrinsics: bool = True,
    gender: str = "neutral",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    mocap_person: Optional[str] = None,
    export_camera: bool = True,
) -> bool:
    """Run motion capture pipeline with GVHMR.

    Args:
        project_dir: Project directory
        use_colmap_intrinsics: Use COLMAP focal length for GVHMR
        gender: Body model gender (neutral, male, female)
        start_frame: Start frame (1-indexed, inclusive)
        end_frame: End frame (1-indexed, inclusive)
        mocap_person: Roto person folder to isolate (e.g., 'person_00')
        export_camera: Export GVHMR camera estimate if COLMAP camera missing

    Returns:
        True if successful
    """
    deps = check_all_dependencies()
    required = ["numpy", "pytorch", "smplx", "trimesh"]
    missing = [name for name in required if not deps[name]]

    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}", file=sys.stderr)
        print_dependency_status()
        install_instructions()
        return False

    person_folder = mocap_person or "person"
    mocap_dir = project_dir / "mocap"
    mocap_person_dir = mocap_dir / person_folder
    mocap_person_dir.mkdir(parents=True, exist_ok=True)

    gvhmr_dir = INSTALL_DIR / "GVHMR"
    gvhmr_checkpoint = gvhmr_dir / "inputs" / "checkpoints" / "gvhmr" / "gvhmr_siga24_release.ckpt"
    gvhmr_available = gvhmr_dir.exists() and gvhmr_checkpoint.exists()

    if not gvhmr_available:
        print("Error: GVHMR not available", file=sys.stderr)
        print("Install GVHMR using the installation wizard", file=sys.stderr)
        return False

    print(f"\n{'=' * 60}")
    print("Motion Capture Pipeline (GVHMR)")
    print("=" * 60)
    print(f"Project: {project_dir}")
    if mocap_person:
        print(f"Isolating: {mocap_person} (via roto matte)")
    if start_frame or end_frame:
        range_str = f"{start_frame or 1}-{end_frame or 'end'}"
        print(f"Frame range: {range_str}")
    print()

    focal_mm = None
    static_camera = False

    intrinsics_path = project_dir / "camera" / "intrinsics.json"
    if use_colmap_intrinsics and intrinsics_path.exists():
        focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path)
        if focal_mm:
            print(f"Using COLMAP focal length: {focal_mm:.1f}mm")

    mmcam_extrinsics_path = None
    extrinsics_path = project_dir / "camera" / "extrinsics.json"
    if extrinsics_path.exists():
        mmcam_extrinsics_path = extrinsics_path
        static_camera = detect_static_camera(extrinsics_path)
        if static_camera:
            print("Detected static camera, skipping visual odometry")
        else:
            print("Using mmcam camera solve for GVHMR")

    success = run_gvhmr_motion_tracking(
        project_dir,
        focal_mm=focal_mm,
        static_camera=static_camera,
        output_dir=mocap_person_dir / "gvhmr",
        start_frame=start_frame,
        end_frame=end_frame,
        mocap_person=mocap_person,
        mmcam_extrinsics_path=mmcam_extrinsics_path,
    )

    if not success:
        print("Motion tracking failed", file=sys.stderr)
        return False

    gvhmr_output = mocap_person_dir / "gvhmr"

    mocap_camera_dir = project_dir / "mocap_camera"

    if export_camera:
        print("\n--- Exporting GVHMR Camera ---")

        source_frames = project_dir / "source" / "frames"
        frame_files = sorted(source_frames.glob("*.png")) + sorted(source_frames.glob("*.jpg"))
        n_frames = len(frame_files)

        image_width, image_height = 1920, 1080
        if frame_files:
            try:
                from PIL import Image
                with Image.open(frame_files[0]) as img:
                    image_width, image_height = img.size
            except Exception as e:
                print(f"  Warning: Could not read image dimensions, using defaults: {e}")

        if n_frames > 0:
            export_gvhmr_camera(
                gvhmr_output_dir=gvhmr_output,
                camera_output_dir=mocap_camera_dir,
                n_frames=n_frames,
                image_width=image_width,
                image_height=image_height,
            )

    print(f"\n{'=' * 60}")
    print("Motion Capture Complete")
    print("=" * 60)
    print(f"Output directory: {mocap_person_dir}")
    print(f"GVHMR output: {gvhmr_output}")

    detected = list_detected_persons(gvhmr_output)
    if len(detected) > 1:
        print(f"\nMulti-person detected in footage ({len(detected)} persons):")
        for i, person_name in enumerate(detected):
            motion_file = mocap_person_dir / f"motion_{person_name}.pkl"
            marker = " <-- primary" if i == 0 else ""
            print(f"  {i}: {motion_file.name}{marker}")
        print("\nTip: Use --mocap-person with roto mattes to isolate individuals")
    print()

    return True


def run_export_pipeline(
    project_dir: Path,
    fps: float = 24.0,
    mocap_person: Optional[str] = None,
) -> bool:
    """Run mesh export after motion capture.

    Runs export_mocap.py in the gvhmr conda environment which has
    chumpy and other SMPL dependencies pre-installed.

    Args:
        project_dir: Project directory
        fps: Frames per second for export
        mocap_person: Person folder name (e.g., 'person_00'), defaults to 'person'

    Returns:
        True if successful
    """
    person_folder = mocap_person or "person"
    mocap_person_dir = project_dir / "mocap" / person_folder
    gvhmr_dir = mocap_person_dir / "gvhmr"

    if not gvhmr_dir.exists():
        print(f"Error: No GVHMR output to export at {gvhmr_dir}", file=sys.stderr)
        return False

    gvhmr_files = list(gvhmr_dir.rglob("hmr4d*.pt"))
    if not gvhmr_files:
        print(f"Error: No GVHMR output (hmr4d*.pt) found in {gvhmr_dir}", file=sys.stderr)
        return False

    conda_exe = _find_conda()
    if not conda_exe:
        print("Error: Conda not found - required for export", file=sys.stderr)
        return False

    scripts_dir = Path(__file__).parent
    export_script = scripts_dir / "export_mocap.py"

    export_args = [
        "python", str(export_script),
        str(project_dir),
        "--fps", str(fps),
        "--format", "abc,usd",
    ]
    if mocap_person:
        export_args.extend(["--mocap-person", mocap_person])

    cmd = [conda_exe, "run", "-n", "gvhmr", "--no-capture-output"] + export_args

    print(f"  → Running export in 'gvhmr' conda environment...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"Error: Export failed with exit code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Error: Export timed out (>30 minutes)", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Warning: Export failed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Human motion capture with SMPL-X topology using GVHMR",
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
        "--no-colmap-intrinsics",
        action="store_true",
        help="Don't use COLMAP intrinsics for GVHMR focal length"
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Body model gender (default: neutral)"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip automatic Alembic/USD export after motion capture"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second for export (default: 24)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start frame for motion capture (1-indexed, inclusive)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame for motion capture (1-indexed, inclusive)"
    )
    parser.add_argument(
        "--mocap-person",
        type=str,
        default=None,
        help="Roto person folder to isolate (e.g., 'person_00'). Composites frames with roto matte."
    )
    parser.add_argument(
        "--list-persons",
        action="store_true",
        help="List detected persons from existing GVHMR output and exit"
    )
    parser.add_argument(
        "--no-camera-export",
        action="store_true",
        help="Don't export GVHMR camera estimate when COLMAP camera is missing"
    )

    args = parser.parse_args()

    if args.check:
        print_dependency_status()
        install_instructions()
        sys.exit(0)

    if args.list_persons:
        if not args.project_dir:
            print("Error: project_dir required for --list-persons", file=sys.stderr)
            sys.exit(1)
        gvhmr_output = args.project_dir.resolve() / "mocap" / "gvhmr"
        detected = list_detected_persons(gvhmr_output)
        if not detected:
            print("No GVHMR output found. Run motion capture first.")
            sys.exit(1)
        print(f"Detected {len(detected)} person(s):")
        mocap_dir = args.project_dir.resolve() / "mocap"
        for i, person_name in enumerate(detected):
            motion_file = mocap_dir / f"motion_{person_name}.pkl"
            exists = "exists" if motion_file.exists() else "not exported"
            print(f"  {i}: {person_name} ({exists})")
        sys.exit(0)

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
        use_colmap_intrinsics=not args.no_colmap_intrinsics,
        gender=args.gender,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        mocap_person=args.mocap_person,
        export_camera=not args.no_camera_export,
    )

    if not success:
        sys.exit(1)

    if not args.no_export:
        export_success = run_export_pipeline(
            project_dir=project_dir,
            fps=args.fps,
            mocap_person=args.mocap_person,
        )
        if not export_success:
            print("Warning: Export failed, but motion data was saved", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    with LogCapture():
        main()
