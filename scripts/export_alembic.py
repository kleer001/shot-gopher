#!/usr/bin/env python3
"""Export camera data to Alembic (.abc) format.

Standalone script to convert camera JSON data to Alembic format
for import into Houdini, Maya, Nuke, etc.

The Alembic file includes:
  - Animated camera transform (position + rotation per frame)
  - Camera intrinsics (focal length, aperture, near/far clip)
  - Proper time sampling at specified FPS

Requirements:
    conda install -c conda-forge alembic

Usage:
    python export_alembic.py <project_dir> [options]
    python export_alembic.py camera_data.json --output camera.abc

Example:
    python export_alembic.py /path/to/project --fps 24 --start-frame 1
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Try to import Alembic (conda install -c conda-forge alembic)
try:
    import alembic.Abc as Abc
    import alembic.AbcGeom as AbcGeom
    import imath
    HAS_ALEMBIC = True
except ImportError:
    HAS_ALEMBIC = False
    Abc = None
    AbcGeom = None
    imath = None


def load_project_metadata(project_dir: Path) -> dict:
    """Load project metadata (fps, resolution, etc.) from project.json."""
    metadata_path = project_dir / "project.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}


def load_camera_data(input_path: Path) -> tuple[list[np.ndarray], dict, str, Path]:
    """Load camera data from project directory or JSON file.

    Args:
        input_path: Either a project directory (with camera/ subfolder)
                   or a direct path to extrinsics.json

    Returns:
        Tuple of (extrinsics matrices, intrinsics dict, source name, project_dir)
    """
    if input_path.is_dir():
        project_dir = input_path
        camera_dir = input_path / "camera"
        if not camera_dir.exists():
            camera_dir = input_path
            project_dir = input_path.parent
        extrinsics_path = camera_dir / "extrinsics.json"
        intrinsics_path = camera_dir / "intrinsics.json"
        colmap_raw_path = camera_dir / "colmap_raw.json"
    else:
        extrinsics_path = input_path
        intrinsics_path = input_path.parent / "intrinsics.json"
        colmap_raw_path = input_path.parent / "colmap_raw.json"
        project_dir = input_path.parent.parent  # camera/ -> project/

    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {extrinsics_path}")

    with open(extrinsics_path) as f:
        extrinsics_data = json.load(f)

    intrinsics = {}
    if intrinsics_path.exists():
        with open(intrinsics_path) as f:
            intrinsics = json.load(f)

    source = "colmap" if colmap_raw_path.exists() else "da3"

    extrinsics = []
    if isinstance(extrinsics_data, list):
        for matrix_data in extrinsics_data:
            if isinstance(matrix_data, list):
                matrix = np.array(matrix_data).reshape(4, 4)
            else:
                matrix = np.eye(4)
            extrinsics.append(matrix)
    elif isinstance(extrinsics_data, dict) and "matrices" in extrinsics_data:
        for matrix_data in extrinsics_data["matrices"]:
            matrix = np.array(matrix_data).reshape(4, 4)
            extrinsics.append(matrix)

    return extrinsics, intrinsics, source, project_dir


def decompose_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 matrix to translation, rotation matrix, and scale."""
    translation = matrix[:3, 3].copy()

    m = matrix[:3, :3]
    scale = np.array([
        np.linalg.norm(m[:, 0]),
        np.linalg.norm(m[:, 1]),
        np.linalg.norm(m[:, 2])
    ])

    rotation = m.copy()
    for i in range(3):
        if scale[i] > 1e-8:
            rotation[:, i] /= scale[i]

    return translation, rotation, scale


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles (XYZ order, radians)."""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def compute_focal_length_mm(intrinsics: dict, sensor_width_mm: float = 36.0) -> float:
    """Convert focal length from pixels to mm."""
    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    width = intrinsics.get("width", 1920)
    focal_mm = fx * sensor_width_mm / width
    return focal_mm


def export_alembic_camera(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1,
    fps: float = 24.0,
    camera_name: str = "camera",
    sensor_width_mm: float = 36.0,
) -> None:
    """Export camera animation to Alembic file.

    Args:
        extrinsics: List of 4x4 camera-to-world matrices
        intrinsics: Camera intrinsics dict
        output_path: Output .abc file path
        start_frame: Starting frame number
        fps: Frames per second
        camera_name: Name of camera in Alembic hierarchy
        sensor_width_mm: Sensor width in mm for focal length conversion
    """
    if not HAS_ALEMBIC:
        print("Error: Alembic module not available.", file=sys.stderr)
        print("Install with: conda install -c conda-forge alembic", file=sys.stderr)
        sys.exit(1)

    num_frames = len(extrinsics)

    # Create archive
    archive = Abc.OArchive(str(output_path))
    top = archive.getTop()

    # Time sampling
    time_per_frame = 1.0 / fps
    start_time = (start_frame - 1) * time_per_frame
    time_sampling = AbcGeom.TimeSampling(time_per_frame, start_time)
    time_sampling_index = archive.addTimeSampling(time_sampling)

    # Create Xform (parent transform node)
    xform_obj = AbcGeom.OXform(top, camera_name, time_sampling_index)
    xform_schema = xform_obj.getSchema()

    # Create Camera under the xform
    camera_obj = AbcGeom.OCamera(xform_obj, f"{camera_name}Shape", time_sampling_index)
    camera_schema = camera_obj.getSchema()

    # Camera properties
    focal_mm = compute_focal_length_mm(intrinsics, sensor_width_mm)
    width = intrinsics.get("width", 1920)
    height = intrinsics.get("height", 1080)

    h_aperture = sensor_width_mm / 10.0  # Alembic uses cm
    v_aperture = h_aperture * height / width

    print(f"  Focal length: {focal_mm:.2f}mm")
    print(f"  Sensor: {sensor_width_mm:.1f}mm x {sensor_width_mm * height / width:.1f}mm")
    print(f"  Resolution: {width}x{height}")

    # Write each frame
    for frame_idx, matrix in enumerate(extrinsics):
        translation, rotation, scale = decompose_matrix(matrix)
        euler = rotation_matrix_to_euler_xyz(rotation)
        euler_deg = np.degrees(euler)

        # Xform sample
        xform_sample = AbcGeom.XformSample()
        xform_sample.setTranslation(imath.V3d(
            float(translation[0]),
            float(translation[1]),
            float(translation[2])
        ))
        xform_sample.setXRotation(float(euler_deg[0]))
        xform_sample.setYRotation(float(euler_deg[1]))
        xform_sample.setZRotation(float(euler_deg[2]))
        xform_sample.setScale(imath.V3d(
            float(scale[0]),
            float(scale[1]),
            float(scale[2])
        ))
        xform_schema.set(xform_sample)

        # Camera sample
        camera_sample = AbcGeom.CameraSample()
        camera_sample.setFocalLength(focal_mm)
        camera_sample.setHorizontalAperture(h_aperture)
        camera_sample.setVerticalAperture(v_aperture)
        camera_sample.setNearClippingPlane(0.1)
        camera_sample.setFarClippingPlane(10000.0)
        camera_schema.set(camera_sample)

    print(f"  Exported {num_frames} frames to {output_path}")
    print(f"  Frame range: {start_frame}-{start_frame + num_frames - 1}")


def main():
    parser = argparse.ArgumentParser(
        description="Export camera data to Alembic (.abc) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Project directory or path to extrinsics.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output .abc file path (default: <project>/camera/camera.abc)"
    )
    parser.add_argument(
        "--start-frame", "-s",
        type=int,
        default=1,
        help="Starting frame number (default: 1)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=24.0,
        help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--camera-name", "-n",
        default="camera",
        help="Camera name in Alembic hierarchy (default: camera)"
    )
    parser.add_argument(
        "--sensor-width",
        type=float,
        default=36.0,
        help="Sensor width in mm (default: 36mm full-frame)"
    )

    args = parser.parse_args()

    if not HAS_ALEMBIC:
        print("Error: Alembic module not available.", file=sys.stderr)
        print("Install with: conda install -c conda-forge alembic", file=sys.stderr)
        sys.exit(1)

    try:
        extrinsics, intrinsics, source, project_dir = load_camera_data(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not extrinsics:
        print("Error: No camera data found", file=sys.stderr)
        sys.exit(1)

    # Load project metadata for defaults
    metadata = load_project_metadata(project_dir)

    # Use metadata values if not overridden by args
    fps = args.fps
    start_frame = args.start_frame
    if fps == 24.0 and "fps" in metadata:  # 24 is the default, check if metadata has actual value
        fps = metadata["fps"]
        print(f"Using FPS from project: {fps}")
    if start_frame == 1 and "start_frame" in metadata:
        start_frame = metadata["start_frame"]

    print(f"Loaded {len(extrinsics)} camera frames from {source}")

    if args.output:
        output_path = args.output
    elif args.input.is_dir():
        output_path = args.input / "camera" / "camera.abc"
    else:
        output_path = args.input.parent / "camera.abc"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_alembic_camera(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        output_path=output_path,
        start_frame=start_frame,
        fps=fps,
        camera_name=args.camera_name,
        sensor_width_mm=args.sensor_width,
    )


if __name__ == "__main__":
    main()
