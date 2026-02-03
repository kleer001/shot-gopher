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
import sys
from pathlib import Path

import bpy

from camera_common import run_camera_export


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

    run_camera_export(
        input_dir=args.input,
        output_path=args.output,
        fps=args.fps,
        start_frame=args.start_frame,
        camera_name=args.camera_name,
        format_name="Alembic",
        export_fn=export_alembic,
    )


if __name__ == "__main__":
    main()
