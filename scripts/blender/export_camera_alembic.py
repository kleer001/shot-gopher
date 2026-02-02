#!/usr/bin/env python3
"""Blender script to export camera animation to Alembic.

This script runs inside Blender's Python environment and exports
camera data from JSON files as an animated Alembic (.abc) file.

Usage (from command line):
    blender -b --python export_camera_alembic.py -- \
        --input /path/to/camera/ \
        --output /path/to/camera.abc \
        --fps 24 \
        --start-frame 1

The script will:
1. Load extrinsics.json and intrinsics.json from the camera directory
2. Create a camera with animated transforms
3. Export to Alembic with proper time sampling
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for camera in list(bpy.data.cameras):
        bpy.data.cameras.remove(camera)


def load_camera_data(camera_dir: Path) -> tuple[list, dict, str]:
    """Load extrinsics and intrinsics from camera data JSONs.

    Args:
        camera_dir: Path to camera/ directory

    Returns:
        Tuple of (extrinsics list, intrinsics dict, source name)
    """
    extrinsics_path = camera_dir / "extrinsics.json"
    intrinsics_path = camera_dir / "intrinsics.json"
    colmap_raw_path = camera_dir / "colmap_raw.json"

    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics_path}")

    with open(extrinsics_path, encoding='utf-8') as f:
        extrinsics_data = json.load(f)

    intrinsics_data = {}
    if intrinsics_path.exists():
        with open(intrinsics_path, encoding='utf-8') as f:
            intrinsics_data = json.load(f)

    source = "colmap" if colmap_raw_path.exists() else "vda"

    extrinsics = []
    if isinstance(extrinsics_data, list):
        extrinsics = extrinsics_data
    elif isinstance(extrinsics_data, dict) and "matrices" in extrinsics_data:
        extrinsics = extrinsics_data["matrices"]

    return extrinsics, intrinsics_data, source


def matrix_to_blender(matrix_data: list) -> tuple[tuple, tuple]:
    """Convert 4x4 matrix to Blender location and rotation.

    Converts from OpenCV convention (Y-down, Z-forward) to
    OpenGL/Blender convention (Y-up, Z-back) via matrix @ flip.

    Args:
        matrix_data: 4x4 matrix as nested list or flat list

    Returns:
        Tuple of (location, euler_rotation_radians) for Blender
    """
    if not matrix_data:
        raise ValueError("Empty matrix data")

    if len(matrix_data) == 16:
        m = [matrix_data[i:i+4] for i in range(0, 16, 4)]
    elif len(matrix_data) == 4 and isinstance(matrix_data[0], list) and len(matrix_data[0]) == 4:
        m = matrix_data
    else:
        raise ValueError(f"Invalid matrix shape: expected 16 elements or 4x4 nested list")

    flip = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += m[i][k] * flip[k][j]

    tx, ty, tz = result[0][3], result[1][3], result[2][3]

    r00, r01, r02 = result[0][0], result[0][1], result[0][2]
    r10, r11, r12 = result[1][0], result[1][1], result[1][2]
    r20, r21, r22 = result[2][0], result[2][1], result[2][2]

    sy = math.sqrt(r00**2 + r10**2)
    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    else:
        rx = math.atan2(-r12, r11)
        ry = math.atan2(-r20, sy)
        rz = 0

    return (tx, ty, tz), (rx, ry, rz)


def create_animated_camera(
    extrinsics: list,
    intrinsics: dict,
    start_frame: int,
    fps: float,
    camera_name: str
) -> bpy.types.Object:
    """Create an animated camera from extrinsics data.

    Args:
        extrinsics: List of 4x4 matrices per frame
        intrinsics: Camera intrinsics dict
        start_frame: Starting frame number
        fps: Frames per second
        camera_name: Name for the camera object

    Returns:
        The created camera object
    """
    cam_data = bpy.data.cameras.new(name=camera_name)
    cam_obj = bpy.data.objects.new(camera_name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    width = intrinsics.get("width", 1920)
    height = intrinsics.get("height", 1080)

    sensor_width_mm = 36.0
    focal_length_mm = fx * sensor_width_mm / width

    cam_data.lens = focal_length_mm
    cam_data.sensor_width = sensor_width_mm
    cam_data.sensor_height = sensor_width_mm * height / width

    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height

    if abs(fps - round(fps)) < 0.001:
        bpy.context.scene.render.fps = int(round(fps))
        bpy.context.scene.render.fps_base = 1.0
    else:
        bpy.context.scene.render.fps = 1000
        bpy.context.scene.render.fps_base = 1000.0 / fps

    cam_obj.rotation_mode = 'XYZ'

    for i, matrix_data in enumerate(extrinsics):
        frame = start_frame + i

        location, rotation = matrix_to_blender(matrix_data)

        cam_obj.location = location
        cam_obj.rotation_euler = rotation

        cam_obj.keyframe_insert(data_path="location", frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    return cam_obj


def export_alembic(
    output_path: Path,
    start_frame: int,
    end_frame: int,
):
    """Export scene to Alembic file.

    Args:
        output_path: Output .abc file path
        start_frame: First frame to export
        end_frame: Last frame to export
    """
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.alembic_export(
        filepath=str(output_path),
        start=start_frame,
        end=end_frame,
        selected=False,
        visible_objects_only=True,
        flatten=False,
        uvs=False,
        normals=False,
        vcolors=False,
        apply_subdiv=False,
        curves_as_mesh=False,
        use_instancing=True,
        global_scale=1.0,
        triangulate=False,
        export_hair=False,
        export_particles=False,
        packuv=False,
        face_sets=False,
    )


def main():
    """Main entry point for the export script."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Export camera animation to Alembic"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directory containing extrinsics.json and intrinsics.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output Alembic file path"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=24.0,
        help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--start-frame", "-s",
        type=int,
        default=1,
        help="Starting frame number (default: 1)"
    )
    parser.add_argument(
        "--camera-name", "-n",
        type=str,
        default=None,
        help="Camera name (default: based on source)"
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.is_dir():
        print(f"Error: Input must be a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading camera data from {args.input}")
    try:
        extrinsics, intrinsics, source = load_camera_data(args.input)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading camera data: {e}", file=sys.stderr)
        sys.exit(1)

    if not extrinsics:
        print("Error: No extrinsics data found", file=sys.stderr)
        sys.exit(1)

    camera_name = args.camera_name or f"{source}_camera"
    print(f"Found {len(extrinsics)} camera frames (source: {source})")

    clear_scene()

    print("Creating animated camera...")
    cam_obj = create_animated_camera(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        start_frame=args.start_frame,
        fps=args.fps,
        camera_name=camera_name
    )

    end_frame = args.start_frame + len(extrinsics) - 1

    print(f"Exporting Alembic...")
    print(f"  Output: {args.output}")
    print(f"  Frames: {args.start_frame}-{end_frame}")
    print(f"  FPS: {args.fps}")
    print(f"  Camera: {camera_name}")

    try:
        export_alembic(
            args.output,
            args.start_frame,
            end_frame,
        )
    except Exception as e:
        print(f"Error exporting Alembic: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
