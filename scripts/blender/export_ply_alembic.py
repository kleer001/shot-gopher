#!/usr/bin/env python3
"""Blender script to export a PLY file to Alembic.

This script runs inside Blender's Python environment and exports
a PLY file (point cloud or mesh) as an Alembic (.abc) file.

Usage (from command line):
    blender -b --python export_ply_alembic.py -- \
        --input /path/to/geometry.ply \
        --output /path/to/output.abc \
        --fps 24
"""

import argparse
import sys
from pathlib import Path

import bpy


def clear_scene() -> None:
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def import_ply(filepath: Path) -> bpy.types.Object:
    """Import a PLY file into the scene."""
    bpy.ops.wm.ply_import(filepath=str(filepath))
    if not bpy.context.selected_objects:
        raise ValueError(f"Failed to import: {filepath}")
    return bpy.context.selected_objects[0]


def export_alembic(output_path: Path, fps: float = 24.0) -> None:
    """Export scene to Alembic file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.render.fps_base = fps / int(fps) if int(fps) > 0 else 1.0

    bpy.ops.wm.alembic_export(
        filepath=str(output_path),
        start=1,
        end=1,
        selected=False,
        visible_objects_only=True,
        flatten=False,
        uvs=True,
        normals=True,
        vcolors=True,
        curves_as_mesh=False,
        use_instancing=False,
        global_scale=1.0,
        triangulate=False,
        export_hair=False,
        export_particles=False,
        packuv=True,
        face_sets=False,
        evaluation_mode='RENDER',
    )


def main() -> None:
    """Main entry point for the export script."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Export PLY geometry to Alembic"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input PLY file"
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
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Importing PLY: {args.input}")
    clear_scene()

    obj = import_ply(args.input)
    print(f"  Vertices: {len(obj.data.vertices)}")
    print(f"  Faces: {len(obj.data.polygons)}")

    print(f"Exporting Alembic: {args.output}")
    export_alembic(args.output, args.fps)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
