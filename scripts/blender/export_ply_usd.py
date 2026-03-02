#!/usr/bin/env python3
"""Blender script to export a PLY file to USD.

This script runs inside Blender's Python environment and exports
a PLY file (point cloud or mesh) as a USD (.usd) file.

Usage (from command line):
    blender -b --python export_ply_usd.py -- \
        --input /path/to/geometry.ply \
        --output /path/to/output.usd \
        --fps 24
"""

import argparse
import sys
from pathlib import Path

import bmesh
import bpy


def ensure_faces(obj: bpy.types.Object) -> None:
    """Create triangles from vertex triplets if mesh has no faces.

    USD PolyMesh requires faces — a vertex-only mesh gets discarded
    on export. This creates minimal triangles so all point positions
    survive the USD round-trip.
    """
    if len(obj.data.polygons) > 0:
        return

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    verts = list(bm.verts)
    for i in range(0, len(verts) - 2, 3):
        bm.faces.new((verts[i], verts[i + 1], verts[i + 2]))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    print(f"  Added {len(obj.data.polygons)} faces for USD compatibility")


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
    bpy.ops.wm.ply_import(filepath=str(filepath), forward_axis='NEGATIVE_Z', up_axis='Y')
    if not bpy.context.selected_objects:
        raise ValueError(f"Failed to import: {filepath}")
    return bpy.context.selected_objects[0]


def export_usd(output_path: Path, fps: int = 24) -> None:
    """Export scene to USD file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.render.fps_base = fps / int(fps) if int(fps) > 0 else 1.0

    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=False,
        visible_objects_only=True,
        export_animation=False,
        export_mesh_uv_maps=True,
        export_normals=True,
        export_materials=False,
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
        description="Export PLY geometry to USD"
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
        help="Output USD file path"
    )
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=24,
        help="Integer FPS (default: 24)"
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
    ensure_faces(obj)

    print(f"Exporting USD: {args.output}")
    export_usd(args.output, args.fps)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
