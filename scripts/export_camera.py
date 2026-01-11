#!/usr/bin/env python3
"""Export camera data to Alembic format.

Converts camera data (from Depth Anything V3 or COLMAP) to an Alembic (.abc)
camera file for import into Houdini/Nuke/Maya/Blender.

Supports camera data from:
  - Depth Anything V3 (monocular depth estimation with camera)
  - COLMAP (Structure-from-Motion reconstruction)

Usage:
    python export_camera.py <project_dir> [--start-frame 1001] [--fps 24]

Example:
    python export_camera.py /path/to/projects/My_Shot_Name --fps 24
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import alembic
    from alembic import Abc, AbcGeom
    HAS_ALEMBIC = True
except ImportError:
    HAS_ALEMBIC = False


def load_camera_data(camera_dir: Path) -> tuple[list[np.ndarray], dict, str]:
    """Load extrinsics and intrinsics from camera data JSONs.

    Supports both Depth Anything V3 and COLMAP output formats.

    Args:
        camera_dir: Path to camera/ directory containing extrinsics.json and intrinsics.json

    Returns:
        Tuple of (list of 4x4 extrinsic matrices, intrinsics dict, source name)
    """
    extrinsics_path = camera_dir / "extrinsics.json"
    intrinsics_path = camera_dir / "intrinsics.json"
    colmap_raw_path = camera_dir / "colmap_raw.json"

    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics_path}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

    with open(extrinsics_path) as f:
        extrinsics_data = json.load(f)

    with open(intrinsics_path) as f:
        intrinsics_data = json.load(f)

    # Detect source: COLMAP creates colmap_raw.json, DA3 does not
    if colmap_raw_path.exists():
        source = "colmap"
    else:
        source = "da3"

    # Parse extrinsics - expected format: list of 4x4 matrices (one per frame)
    # Both DA3 and COLMAP output as nested lists
    extrinsics = []
    if isinstance(extrinsics_data, list):
        for matrix_data in extrinsics_data:
            if isinstance(matrix_data, list):
                matrix = np.array(matrix_data).reshape(4, 4)
            else:
                matrix = np.eye(4)  # Fallback
            extrinsics.append(matrix)
    elif isinstance(extrinsics_data, dict) and "matrices" in extrinsics_data:
        for matrix_data in extrinsics_data["matrices"]:
            matrix = np.array(matrix_data).reshape(4, 4)
            extrinsics.append(matrix)

    return extrinsics, intrinsics_data, source


def matrix_to_alembic_xform(matrix: np.ndarray) -> list[float]:
    """Convert 4x4 numpy matrix to Alembic's column-major 16-element list.

    Alembic uses column-major order, numpy is row-major by default.
    """
    return matrix.T.flatten().tolist()


def decompose_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 transformation matrix into translation, rotation, scale.

    Returns:
        Tuple of (translation vec3, rotation matrix 3x3, scale vec3)
    """
    translation = matrix[:3, 3]

    # Extract rotation and scale from upper-left 3x3
    m = matrix[:3, :3]
    scale = np.array([
        np.linalg.norm(m[:, 0]),
        np.linalg.norm(m[:, 1]),
        np.linalg.norm(m[:, 2])
    ])

    # Normalize to get rotation
    rotation = m.copy()
    for i in range(3):
        if scale[i] != 0:
            rotation[:, i] /= scale[i]

    return translation, rotation, scale


def rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to Euler angles (XYZ order, in degrees)."""
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        y = np.arctan2(-rotation[2, 0], sy)
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        y = np.arctan2(-rotation[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))


def compute_fov_from_intrinsics(intrinsics: dict, image_width: int, image_height: int) -> tuple[float, float]:
    """Compute horizontal and vertical FOV from camera intrinsics.

    Args:
        intrinsics: Dict with fx, fy, cx, cy values
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Tuple of (horizontal_fov, vertical_fov) in degrees
    """
    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    fy = intrinsics.get("fy", intrinsics.get("focal_y", 1000))

    # FOV = 2 * atan(sensor_size / (2 * focal_length))
    # In pixel units: FOV = 2 * atan(image_dimension / (2 * focal_pixel))
    h_fov = 2 * np.degrees(np.arctan(image_width / (2 * fx)))
    v_fov = 2 * np.degrees(np.arctan(image_height / (2 * fy)))

    return h_fov, v_fov


def export_alembic_camera(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1001,
    fps: float = 24.0,
    image_width: int = 1920,
    image_height: int = 1080
) -> None:
    """Export camera animation to Alembic file.

    Args:
        extrinsics: List of 4x4 camera-to-world matrices per frame
        intrinsics: Camera intrinsics dict (fx, fy, cx, cy)
        output_path: Output .abc file path
        start_frame: Starting frame number (VFX convention: 1001)
        fps: Frames per second
        image_width: Image width for FOV calculation
        image_height: Image height for FOV calculation
    """
    if not HAS_ALEMBIC:
        raise ImportError(
            "Alembic Python bindings not available. "
            "Install with: pip install alembic (or build PyAlembic from source)"
        )

    # Calculate time per frame
    time_per_frame = 1.0 / fps

    # Create Alembic archive
    archive = Abc.OArchive(str(output_path))
    top = archive.getTop()

    # Create time sampling (one sample per frame)
    num_frames = len(extrinsics)
    time_sampling = AbcGeom.TimeSampling(time_per_frame, start_frame / fps)
    ts_index = archive.addTimeSampling(time_sampling)

    # Create camera object
    camera_obj = AbcGeom.OCamera(top, "DA3_Camera", ts_index)
    camera_schema = camera_obj.getSchema()

    # Create xform (transform) for the camera
    xform_obj = AbcGeom.OXform(top, "DA3_Camera_Xform", ts_index)
    xform_schema = xform_obj.getSchema()

    # Calculate FOV from intrinsics
    h_fov, v_fov = compute_fov_from_intrinsics(intrinsics, image_width, image_height)

    # Compute focal length in mm (assuming 36mm sensor width - full frame)
    sensor_width_mm = 36.0
    focal_length_mm = (sensor_width_mm / 2) / np.tan(np.radians(h_fov / 2))

    # Write samples for each frame
    for frame_idx, matrix in enumerate(extrinsics):
        # Set camera intrinsics (same for all frames typically)
        cam_sample = AbcGeom.CameraSample()
        cam_sample.setFocalLength(focal_length_mm)
        cam_sample.setHorizontalAperture(sensor_width_mm / 10)  # Alembic uses cm
        cam_sample.setVerticalAperture((sensor_width_mm * image_height / image_width) / 10)
        cam_sample.setNearClippingPlane(0.1)
        cam_sample.setFarClippingPlane(10000.0)
        camera_schema.set(cam_sample)

        # Set transform
        xform_sample = AbcGeom.XformSample()

        # Decompose matrix for cleaner transform stack
        translation, rotation, scale = decompose_matrix(matrix)
        euler = rotation_matrix_to_euler(rotation)

        # Add transform operations
        xform_sample.addOp(AbcGeom.XformOp(AbcGeom.kTranslateOperation, AbcGeom.kTranslateHint),
                          Abc.V3d(*translation))
        xform_sample.addOp(AbcGeom.XformOp(AbcGeom.kRotateXOperation, AbcGeom.kRotateHint),
                          euler[0])
        xform_sample.addOp(AbcGeom.XformOp(AbcGeom.kRotateYOperation, AbcGeom.kRotateHint),
                          euler[1])
        xform_sample.addOp(AbcGeom.XformOp(AbcGeom.kRotateZOperation, AbcGeom.kRotateHint),
                          euler[2])

        xform_schema.set(xform_sample)

    print(f"Exported {num_frames} frames to {output_path}")
    print(f"  Frame range: {start_frame}-{start_frame + num_frames - 1}")
    print(f"  Focal length: {focal_length_mm:.2f}mm")
    print(f"  H-FOV: {h_fov:.2f} deg, V-FOV: {v_fov:.2f} deg")


def export_json_camera(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1001,
    fps: float = 24.0
) -> None:
    """Export camera data to JSON format (fallback when Alembic unavailable).

    Creates a JSON file that can be imported via scripts in Houdini/Nuke/Blender.
    """
    camera_data = {
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": start_frame + len(extrinsics) - 1,
        "intrinsics": intrinsics,
        "frames": []
    }

    for frame_idx, matrix in enumerate(extrinsics):
        translation, rotation, scale = decompose_matrix(matrix)
        euler = rotation_matrix_to_euler(rotation)

        frame_data = {
            "frame": start_frame + frame_idx,
            "matrix": matrix.tolist(),
            "translation": translation.tolist(),
            "rotation_euler_xyz": euler.tolist(),
            "scale": scale.tolist()
        }
        camera_data["frames"].append(frame_data)

    with open(output_path, 'w') as f:
        json.dump(camera_data, f, indent=2)

    print(f"Exported {len(extrinsics)} frames to {output_path}")
    print(f"  Frame range: {start_frame}-{start_frame + len(extrinsics) - 1}")


def main():
    parser = argparse.ArgumentParser(
        description="Export DA3 camera data to Alembic format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing camera/ subfolder"
    )
    parser.add_argument(
        "--start-frame", "-s",
        type=int,
        default=1001,
        help="Starting frame number (default: 1001)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=24.0,
        help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=1920,
        help="Image width in pixels (default: 1920)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=1080,
        help="Image height in pixels (default: 1080)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: <project_dir>/camera/camera.abc)"
    )
    parser.add_argument(
        "--format",
        choices=["abc", "json", "both"],
        default="both",
        help="Output format (default: both)"
    )

    args = parser.parse_args()

    # Validate project directory
    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    camera_dir = project_dir / "camera"
    if not camera_dir.exists():
        print(f"Error: Camera directory not found: {camera_dir}", file=sys.stderr)
        sys.exit(1)

    # Load camera data
    try:
        extrinsics, intrinsics, source = load_camera_data(camera_dir)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading camera data: {e}", file=sys.stderr)
        sys.exit(1)

    if not extrinsics:
        print("Error: No camera extrinsics found", file=sys.stderr)
        sys.exit(1)

    source_name = "COLMAP (SfM)" if source == "colmap" else "Depth Anything V3"
    print(f"Loaded {len(extrinsics)} camera frames from {source_name}")

    # Determine output path
    output_base = args.output or (camera_dir / "camera")

    # Export based on format
    if args.format in ("abc", "both"):
        abc_path = output_base.with_suffix(".abc")
        if HAS_ALEMBIC:
            try:
                export_alembic_camera(
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    output_path=abc_path,
                    start_frame=args.start_frame,
                    fps=args.fps,
                    image_width=args.width,
                    image_height=args.height
                )
            except Exception as e:
                print(f"Error exporting Alembic: {e}", file=sys.stderr)
                if args.format == "abc":
                    sys.exit(1)
        else:
            print("Warning: Alembic not available, skipping .abc export", file=sys.stderr)
            print("  Install with: pip install PyAlembic", file=sys.stderr)
            if args.format == "abc":
                sys.exit(1)

    if args.format in ("json", "both"):
        json_path = output_base.with_suffix(".camera.json")
        export_json_camera(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            output_path=json_path,
            start_frame=args.start_frame,
            fps=args.fps
        )


if __name__ == "__main__":
    main()
