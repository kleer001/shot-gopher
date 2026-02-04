#!/usr/bin/env python3
"""Blender script to export camera animation to USD.

This script runs inside Blender's Python environment and exports
camera data from JSON files as an animated USD (.usd/.usda/.usdc) file.

Usage (from command line):
    blender -b --python export_camera_usd.py -- \
        --input /path/to/camera/ \
        --output /path/to/camera.usd \
        --fps 24 \
        --start-frame 1

The script will:
1. Load extrinsics.json and intrinsics.json from the camera directory
2. Create a camera with animated transforms
3. Export to USD with proper time sampling
"""

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from camera_common import run_camera_export


def export_usd(
    output_path: Path,
    start_frame: int,
    end_frame: int,
):
    """Export scene to USD file.

    Args:
        output_path: Output .usd/.usda/.usdc file path
        start_frame: First frame to export
        end_frame: Last frame to export
    """
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=False,
        visible_objects_only=True,
        export_animation=True,
        export_hair=False,
        export_uvmaps=False,
        export_normals=False,
        export_materials=False,
        use_instancing=True,
        evaluation_mode='RENDER',
        generate_preview_surface=False,
        export_textures=False,
        overwrite_textures=False,
        relative_paths=True,
    )


def main():
    """Main entry point for the export script."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Export camera animation to USD"
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
        help="Output USD file path (.usd, .usda, or .usdc)"
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
        format_name="USD",
        export_fn=export_usd,
    )


if __name__ == "__main__":
    main()
