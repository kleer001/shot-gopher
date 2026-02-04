#!/usr/bin/env python3
"""Common utilities for Blender camera export scripts.

This module contains shared functionality used by both Alembic and USD
camera export scripts, following DRY principles.
"""

import json
import math
from pathlib import Path
from typing import Callable

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

    Raises:
        FileNotFoundError: If extrinsics.json not found
        json.JSONDecodeError: If JSON is malformed
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

    Raises:
        ValueError: If matrix data is empty or invalid shape
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

    Raises:
        ValueError: If intrinsics contain invalid values
    """
    cam_data = bpy.data.cameras.new(name=camera_name)
    cam_obj = bpy.data.objects.new(camera_name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    width = intrinsics.get("width", 1920)
    height = intrinsics.get("height", 1080)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    if fx <= 0:
        raise ValueError(f"Invalid focal length: {fx}")

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


def run_camera_export(
    input_dir: Path,
    output_path: Path,
    fps: float,
    start_frame: int,
    camera_name: str | None,
    format_name: str,
    export_fn: Callable[[Path, int, int], None],
) -> None:
    """Run camera export with common setup and error handling.

    Args:
        input_dir: Directory containing extrinsics.json and intrinsics.json
        output_path: Output file path
        fps: Frames per second
        start_frame: Starting frame number
        camera_name: Optional camera name (default: based on source)
        format_name: Format name for messages (e.g., "Alembic", "USD")
        export_fn: Format-specific export function(output_path, start_frame, end_frame)

    Raises:
        SystemExit: On any error
    """
    import sys

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input must be a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading camera data from {input_dir}")
    try:
        extrinsics, intrinsics, source = load_camera_data(input_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading camera data: {e}", file=sys.stderr)
        sys.exit(1)

    if not extrinsics:
        print("Error: No extrinsics data found", file=sys.stderr)
        sys.exit(1)

    resolved_camera_name = camera_name or f"{source}_camera"
    print(f"Found {len(extrinsics)} camera frames (source: {source})")

    clear_scene()

    print("Creating animated camera...")
    try:
        create_animated_camera(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            start_frame=start_frame,
            fps=fps,
            camera_name=resolved_camera_name
        )
    except ValueError as e:
        print(f"Error creating camera: {e}", file=sys.stderr)
        sys.exit(1)

    end_frame = start_frame + len(extrinsics) - 1

    print(f"Exporting {format_name}...")
    print(f"  Output: {output_path}")
    print(f"  Frames: {start_frame}-{end_frame}")
    print(f"  FPS: {fps}")
    print(f"  Camera: {resolved_camera_name}")

    try:
        export_fn(output_path, start_frame, end_frame)
    except Exception as e:
        print(f"Error exporting {format_name}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully exported: {output_path}")
