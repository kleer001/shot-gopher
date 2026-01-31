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

# Log capture for debugging
from log_manager import LogCapture


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
        status = "OK" if deps[name] else "X"
        print(f"  {status} {name}")

    # Optional dependencies
    print("\nOptional (for specific methods):")
    for name in ["wham"]:
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

        print(f"  OK Motion tracking complete")
        print(f"    Output: {motion_file}")

        return True

    except subprocess.TimeoutExpired:
        print("Error: WHAM timed out", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error running WHAM: {e}", file=sys.stderr)
        return False


def colmap_intrinsics_to_focal_mm(
    intrinsics_path: Path,
    sensor_width_mm: Optional[float] = None,
) -> Optional[float]:
    """Convert COLMAP intrinsics to focal length in mm.

    Args:
        intrinsics_path: Path to camera/intrinsics.json
        sensor_width_mm: Sensor width in mm (None = assume 36mm full-frame)

    Returns:
        Focal length in millimeters, or None if conversion fails
    """
    import json

    if not intrinsics_path.exists():
        return None

    try:
        with open(intrinsics_path, encoding='utf-8') as f:
            intrinsics = json.load(f)

        fx = intrinsics.get("fx", intrinsics.get("focal_x"))
        width = intrinsics.get("width", 1920)

        if fx is None:
            return None

        if sensor_width_mm is None:
            sensor_width_mm = 36.0

        focal_mm = fx * sensor_width_mm / width
        return focal_mm

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def detect_static_camera(
    extrinsics_path: Path,
    threshold_meters: float = 0.01
) -> bool:
    """Detect if camera is static from COLMAP extrinsics.

    Analyzes camera translation variance across all frames.

    Args:
        extrinsics_path: Path to extrinsics.json
        threshold_meters: Max camera movement variance to consider "static"

    Returns:
        True if camera appears static
    """
    import json

    if not extrinsics_path.exists():
        return False

    try:
        import numpy as np

        with open(extrinsics_path, encoding='utf-8') as f:
            extrinsics = json.load(f)

        if not extrinsics or len(extrinsics) < 2:
            return False

        translations = []
        for matrix_data in extrinsics:
            if isinstance(matrix_data, list) and len(matrix_data) >= 3:
                translations.append([matrix_data[0][3], matrix_data[1][3], matrix_data[2][3]])

        if not translations:
            return False

        translations = np.array(translations)
        variance = np.var(translations, axis=0).sum()

        return variance < threshold_meters

    except Exception:
        return False


def find_or_create_video(
    project_dir: Path,
    fps: float = 24.0
) -> Optional[Path]:
    """Find source video or create one from frames.

    GVHMR requires video input, not frames. This function:
    1. Looks for original source video
    2. Falls back to re-encoding frames

    Args:
        project_dir: Project directory
        fps: Frame rate for video encoding

    Returns:
        Path to video file, or None if creation fails
    """
    source_dir = project_dir / "source"

    video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    for ext in video_extensions:
        for video_file in source_dir.glob(f"*{ext}"):
            if not video_file.name.startswith("_"):
                return video_file

    frames_dir = source_dir / "frames"
    if not frames_dir.exists():
        return None

    frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return None

    video_path = source_dir / "_gvhmr_input.mp4"

    if video_path.exists():
        return video_path

    print(f"  → Creating video from {len(frames)} frames...")

    if (frames_dir / "frame_0001.png").exists():
        frame_pattern = frames_dir / "frame_%04d.png"
    elif (frames_dir / "frame_00001.png").exists():
        frame_pattern = frames_dir / "frame_%05d.png"
    elif (frames_dir / "frame_0001.jpg").exists():
        frame_pattern = frames_dir / "frame_%04d.jpg"
    elif (frames_dir / "frame_00001.jpg").exists():
        frame_pattern = frames_dir / "frame_%05d.jpg"
    else:
        first_frame = frames[0]
        stem = first_frame.stem
        if '_' in stem:
            prefix = stem.rsplit('_', 1)[0]
            num_part = stem.rsplit('_', 1)[1]
            num_digits = len(num_part)
            frame_pattern = first_frame.parent / f"{prefix}_%0{num_digits}d{first_frame.suffix}"
        else:
            frame_pattern = first_frame.parent / f"%04d{first_frame.suffix}"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_pattern),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and video_path.exists():
            print(f"  OK Created video: {video_path.name}")
            return video_path
    except Exception as e:
        print(f"  Error creating video: {e}", file=sys.stderr)

    return None


def run_gvhmr_motion_tracking(
    project_dir: Path,
    focal_mm: Optional[float] = None,
    static_camera: bool = False,
    output_dir: Optional[Path] = None
) -> bool:
    """Run GVHMR for world-grounded motion tracking.

    Args:
        project_dir: Project directory
        focal_mm: Focal length in mm (from COLMAP intrinsics)
        static_camera: Skip visual odometry for static cameras
        output_dir: Output directory for results

    Returns:
        True if successful
    """
    gvhmr_dir = INSTALL_DIR / "GVHMR"
    if not gvhmr_dir.exists():
        print("Error: GVHMR not installed", file=sys.stderr)
        print("Run the installation wizard to install GVHMR", file=sys.stderr)
        return False

    output_dir = output_dir or project_dir / "mocap" / "gvhmr"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_or_create_video(project_dir)
    if not video_path:
        print("Error: No video file found and could not create from frames", file=sys.stderr)
        return False

    print(f"\n{'=' * 60}")
    print("GVHMR Motion Tracking")
    print("=" * 60)
    print(f"Video: {video_path}")
    if focal_mm:
        print(f"Focal length: {focal_mm:.1f}mm")
    if static_camera:
        print(f"Mode: Static camera (skipping visual odometry)")
    print(f"Output: {output_dir}")
    print()

    try:
        demo_script = gvhmr_dir / "tools" / "demo" / "demo.py"
        if not demo_script.exists():
            demo_script = gvhmr_dir / "demo.py"

        cmd = [
            "python", str(demo_script),
            "--video", str(video_path),
            "--output_root", str(output_dir),
        ]

        if static_camera:
            cmd.append("--static_cam")

        if focal_mm:
            cmd.extend(["--f_mm", str(focal_mm)])

        print(f"  → Running GVHMR...")
        print(f"    $ {' '.join(cmd)}")

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
                print(result.stderr[:500], file=sys.stderr)
            return False

        output_files = list(output_dir.rglob("*.pkl"))
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


def convert_gvhmr_to_wham_format(
    gvhmr_output_dir: Path,
    output_path: Path
) -> bool:
    """Convert GVHMR output to WHAM-compatible motion.pkl format.

    Args:
        gvhmr_output_dir: Directory containing GVHMR output
        output_path: Path for WHAM-compatible output file

    Returns:
        True if conversion successful
    """
    import pickle

    try:
        import numpy as np
    except ImportError:
        print("Error: numpy required for format conversion", file=sys.stderr)
        return False

    gvhmr_files = list(gvhmr_output_dir.rglob("*.pkl"))
    if not gvhmr_files:
        print(f"Error: No GVHMR output files found in {gvhmr_output_dir}", file=sys.stderr)
        return False

    gvhmr_output_path = gvhmr_files[0]
    for f in gvhmr_files:
        if "gvhmr" in f.name.lower() or "global" in f.name.lower():
            gvhmr_output_path = f
            break

    print(f"  → Converting {gvhmr_output_path.name} to WHAM format...")

    try:
        with open(gvhmr_output_path, 'rb') as f:
            gvhmr_data = pickle.load(f)

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
            print("Error: Could not find body_pose in GVHMR output", file=sys.stderr)
            return False

        body_pose = np.array(body_pose)
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

        wham_format = {
            'poses': poses,
            'trans': transl,
            'betas': betas
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(wham_format, f)

        print(f"  OK Converted {n_frames} frames to WHAM format")
        print(f"    Output: {output_path}")

        return True

    except Exception as e:
        print(f"Error converting GVHMR output: {e}", file=sys.stderr)
        return False


def run_mocap_pipeline(
    project_dir: Path,
    method: str = "auto",
    use_colmap_intrinsics: bool = True,
    skip_texture: bool = False,
) -> bool:
    """Run motion capture pipeline with GVHMR (preferred) or WHAM fallback.

    Args:
        project_dir: Project directory
        method: Motion capture method - "auto", "gvhmr", or "wham"
        use_colmap_intrinsics: Use COLMAP focal length for GVHMR
        skip_texture: Skip texture projection (unused, kept for API compatibility)

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

    mocap_dir = project_dir / "mocap"
    mocap_dir.mkdir(parents=True, exist_ok=True)

    gvhmr_dir = INSTALL_DIR / "GVHMR"
    gvhmr_checkpoint = gvhmr_dir / "inputs" / "checkpoints" / "gvhmr" / "gvhmr_siga24_release.ckpt"
    gvhmr_available = gvhmr_dir.exists() and gvhmr_checkpoint.exists()
    wham_available = deps.get("wham", False)

    if method == "auto":
        if gvhmr_available:
            method = "gvhmr"
        elif wham_available:
            method = "wham"
        else:
            print("Error: No motion capture method available", file=sys.stderr)
            print("Install GVHMR or WHAM using the installation wizard", file=sys.stderr)
            return False

    print(f"\n{'=' * 60}")
    print(f"Motion Capture Pipeline ({method.upper()})")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print()

    success = False

    if method == "gvhmr":
        focal_mm = None
        static_camera = False

        intrinsics_path = project_dir / "camera" / "intrinsics.json"
        if use_colmap_intrinsics and intrinsics_path.exists():
            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path)
            if focal_mm:
                print(f"Using COLMAP focal length: {focal_mm:.1f}mm")

        extrinsics_path = project_dir / "camera" / "extrinsics.json"
        if extrinsics_path.exists():
            static_camera = detect_static_camera(extrinsics_path)
            if static_camera:
                print("Detected static camera, skipping visual odometry")

        success = run_gvhmr_motion_tracking(
            project_dir,
            focal_mm=focal_mm,
            static_camera=static_camera,
            output_dir=mocap_dir / "gvhmr"
        )

        if success:
            gvhmr_output = mocap_dir / "gvhmr"
            wham_compat = mocap_dir / "wham" / "motion.pkl"
            if not convert_gvhmr_to_wham_format(gvhmr_output, wham_compat):
                print("Warning: Could not create WHAM-compatible output", file=sys.stderr)

        if not success and wham_available:
            print("\n  → GVHMR failed, falling back to WHAM...")
            method = "wham"

    if method == "wham":
        person_mask_dir = project_dir / "roto"
        if not person_mask_dir.exists():
            person_mask_dir = None

        success = run_wham_motion_tracking(
            project_dir,
            person_mask_dir=person_mask_dir,
            output_dir=mocap_dir / "wham"
        )

    if not success:
        print("Motion tracking failed", file=sys.stderr)
        return False

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
    parser.add_argument(
        "--method", "-m",
        choices=["auto", "gvhmr", "wham"],
        default="auto",
        help="Motion capture method: auto (GVHMR with WHAM fallback), gvhmr, wham (default: auto)"
    )
    parser.add_argument(
        "--no-colmap-intrinsics",
        action="store_true",
        help="Don't use COLMAP intrinsics for GVHMR focal length"
    )

    args = parser.parse_args()

    if args.check:
        print_dependency_status()
        install_instructions()
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
        method=args.method,
        use_colmap_intrinsics=not args.no_colmap_intrinsics,
        skip_texture=args.skip_texture,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
