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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# COLMAP quality presets
QUALITY_PRESETS = {
    "low": {
        "sift_max_features": 4096,
        "matcher": "sequential",
        "ba_refine_focal": True,
        "dense_max_size": 1000,
    },
    "medium": {
        "sift_max_features": 8192,
        "matcher": "sequential",
        "ba_refine_focal": True,
        "dense_max_size": 2000,
    },
    "high": {
        "sift_max_features": 16384,
        "matcher": "exhaustive",
        "ba_refine_focal": True,
        "dense_max_size": -1,  # No limit
    },
}


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


def run_colmap_command(
    command: str,
    args: dict,
    description: str,
    timeout: int = 3600
) -> subprocess.CompletedProcess:
    """Run a COLMAP command with the given arguments.

    Args:
        command: COLMAP subcommand (e.g., 'feature_extractor')
        args: Dictionary of argument name -> value
        description: Human-readable description for logging
        timeout: Timeout in seconds

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

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        print(f"    Error: {result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    return result


def extract_features(
    database_path: Path,
    image_path: Path,
    camera_model: str = "OPENCV",
    max_features: int = 8192,
    single_camera: bool = True
) -> None:
    """Extract SIFT features from images.

    Args:
        database_path: Path to COLMAP database
        image_path: Path to image directory
        camera_model: Camera model (OPENCV, PINHOLE, RADIAL, etc.)
        max_features: Maximum features per image
        single_camera: If True, assume all images from same camera
    """
    args = {
        "database_path": str(database_path),
        "image_path": str(image_path),
        "ImageReader.camera_model": camera_model,
        "ImageReader.single_camera": 1 if single_camera else 0,
        "SiftExtraction.max_num_features": max_features,
        "SiftExtraction.use_gpu": 1,
    }

    run_colmap_command("feature_extractor", args, "Extracting features")


def match_features(
    database_path: Path,
    matcher_type: str = "sequential",
    sequential_overlap: int = 10
) -> None:
    """Match features between images.

    Args:
        database_path: Path to COLMAP database
        matcher_type: 'sequential', 'exhaustive', or 'vocab_tree'
        sequential_overlap: Number of overlapping frames for sequential matcher
    """
    if matcher_type == "sequential":
        args = {
            "database_path": str(database_path),
            "SequentialMatching.overlap": sequential_overlap,
            "SiftMatching.use_gpu": 1,
        }
        run_colmap_command("sequential_matcher", args, "Matching features (sequential)")
    elif matcher_type == "exhaustive":
        args = {
            "database_path": str(database_path),
            "SiftMatching.use_gpu": 1,
        }
        run_colmap_command("exhaustive_matcher", args, "Matching features (exhaustive)")
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")


def run_sparse_reconstruction(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    refine_focal: bool = True
) -> bool:
    """Run incremental Structure-from-Motion reconstruction.

    Args:
        database_path: Path to COLMAP database
        image_path: Path to image directory
        output_path: Path to sparse reconstruction output
        refine_focal: Whether to refine focal length during BA

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


def read_colmap_images(images_path: Path) -> dict:
    """Read COLMAP images.bin or images.txt file.

    Returns:
        Dict mapping image_name -> {quat, trans, camera_id}
    """
    images = {}

    txt_path = images_path.parent / "images.txt"
    if txt_path.exists():
        with open(txt_path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        # Images.txt has 2 lines per image
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
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
        return images

    # Binary format - convert to text
    bin_path = images_path
    if bin_path.exists():
        temp_dir = images_path.parent / "_temp_txt"
        temp_dir.mkdir(exist_ok=True)
        try:
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", str(images_path.parent),
                "--output_path", str(temp_dir),
                "--output_type", "TXT"
            ], capture_output=True, check=True)

            with open(temp_dir / "images.txt") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

            for i in range(0, len(lines), 2):
                parts = lines[i].split()
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
    output_dir: Path
) -> bool:
    """Export COLMAP reconstruction to pipeline camera format.

    Creates extrinsics.json and intrinsics.json compatible with export_camera.py

    Args:
        sparse_path: Path to COLMAP sparse model (e.g., sparse/0/)
        output_dir: Output directory (typically project/camera/)

    Returns:
        True if export succeeded
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read COLMAP output
    cameras = read_colmap_cameras(sparse_path / "cameras.bin")
    images = read_colmap_images(sparse_path / "images.bin")

    if not images:
        print("    Error: No images in reconstruction", file=sys.stderr)
        return False

    # Convert to our format
    extrinsics, intrinsics = colmap_to_camera_matrices(images, cameras)

    # Save extrinsics (list of 4x4 matrices)
    extrinsics_data = [m.tolist() for m in extrinsics]
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
) -> bool:
    """Run the complete COLMAP reconstruction pipeline.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high')
        run_dense: Whether to run dense reconstruction
        run_mesh: Whether to generate mesh (requires dense)
        camera_model: COLMAP camera model to use

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

    try:
        # Feature extraction
        print("[1/4] Feature Extraction")
        extract_features(
            database_path=database_path,
            image_path=frames_dir,
            camera_model=camera_model,
            max_features=preset["sift_max_features"],
            single_camera=True,
        )

        # Feature matching
        print("\n[2/4] Feature Matching")
        match_features(
            database_path=database_path,
            matcher_type=preset["matcher"],
        )

        # Sparse reconstruction
        print("\n[3/4] Sparse Reconstruction")
        if not run_sparse_reconstruction(
            database_path=database_path,
            image_path=frames_dir,
            output_path=sparse_path,
            refine_focal=preset["ba_refine_focal"],
        ):
            print("Sparse reconstruction failed", file=sys.stderr)
            return False

        # Export camera data
        print("\n[4/4] Exporting Camera Data")
        camera_dir = project_dir / "camera"
        sparse_model = sparse_path / "0"
        if not export_colmap_to_pipeline_format(sparse_model, camera_dir):
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
        choices=["low", "medium", "high"],
        default="medium",
        help="Quality preset (default: medium)"
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
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
