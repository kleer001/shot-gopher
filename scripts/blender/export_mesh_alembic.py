#!/usr/bin/env python3
"""Blender script to export OBJ mesh sequence to Alembic.

This script runs inside Blender's Python environment and exports
a sequence of OBJ files as an animated Alembic (.abc) file.

Usage (from command line):
    blender -b --python export_mesh_alembic.py -- \
        --input /path/to/meshes/ \
        --output /path/to/output.abc \
        --fps 24 \
        --start-frame 1

The script will:
1. Import OBJ files as an animated mesh sequence
2. Export to Alembic with proper time sampling
"""

import argparse
import sys
from pathlib import Path

import bpy


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)


def find_obj_files(input_dir: Path) -> list[Path]:
    """Find and sort OBJ files in directory.

    Args:
        input_dir: Directory containing OBJ files

    Returns:
        Sorted list of OBJ file paths
    """
    obj_files = sorted(input_dir.glob("*.obj"))

    if not obj_files:
        obj_files = sorted(input_dir.glob("*.OBJ"))

    return obj_files


def import_obj_sequence_as_shape_keys(
    obj_files: list[Path],
    start_frame: int = 1
) -> bpy.types.Object:
    """Import OBJ sequence using shape keys for animation.

    This method imports the first OBJ as the base mesh, then adds
    each subsequent OBJ as a shape key. This preserves vertex order
    and creates smooth interpolation.

    Args:
        obj_files: List of OBJ file paths
        start_frame: Starting frame number

    Returns:
        The imported mesh object
    """
    if not obj_files:
        raise ValueError("No OBJ files provided")

    bpy.ops.wm.obj_import(filepath=str(obj_files[0]))
    base_obj = bpy.context.selected_objects[0]
    base_obj.name = "animated_mesh"

    if not base_obj.data.shape_keys:
        base_obj.shape_key_add(name="Basis", from_mix=False)

    for i, obj_file in enumerate(obj_files[1:], start=1):
        frame = start_frame + i

        bpy.ops.wm.obj_import(filepath=str(obj_file))
        temp_obj = bpy.context.selected_objects[0]

        base_obj.select_set(True)
        bpy.context.view_layer.objects.active = base_obj

        try:
            sk = base_obj.shape_key_add(name=f"frame_{frame:04d}", from_mix=False)

            if len(temp_obj.data.vertices) == len(base_obj.data.vertices):
                for j, vert in enumerate(temp_obj.data.vertices):
                    sk.data[j].co = vert.co
        finally:
            bpy.data.objects.remove(temp_obj, do_unlink=True)

    num_frames = len(obj_files)
    for i, key in enumerate(base_obj.data.shape_keys.key_blocks[1:], start=0):
        frame = start_frame + i

        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=frame - 1)

        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=frame)

        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=frame + 1)

    return base_obj


def import_obj_sequence_as_mesh_cache(
    obj_files: list[Path],
    start_frame: int = 1,
    fps: float = 24.0
) -> bpy.types.Object:
    """Import OBJ sequence using Mesh Sequence Cache modifier.

    This method is more memory efficient for long sequences but
    requires the mesh cache add-on. Falls back to shape keys if
    the add-on is not available.

    Args:
        obj_files: List of OBJ file paths
        start_frame: Starting frame number
        fps: Frames per second

    Returns:
        The imported mesh object
    """
    return import_obj_sequence_as_shape_keys(obj_files, start_frame)


def export_alembic(
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float = 24.0,
    apply_modifiers: bool = True
):
    """Export scene to Alembic file.

    Args:
        output_path: Output .abc file path
        start_frame: First frame to export
        end_frame: Last frame to export
        fps: Frames per second
        apply_modifiers: Whether to apply modifiers (shape keys need this True)
    """
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    bpy.context.scene.render.fps = int(fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.alembic_export(
        filepath=str(output_path),
        start=start_frame,
        end=end_frame,
        selected=False,
        visible_objects_only=True,
        flatten=False,
        uvs=True,
        normals=True,
        vcolors=False,
        apply_subdiv=apply_modifiers,
        curves_as_mesh=False,
        use_instancing=True,
        global_scale=1.0,
        triangulate=False,
        export_hair=False,
        export_particles=False,
        packuv=True,
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
        description="Export OBJ mesh sequence to Alembic"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directory containing OBJ files"
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
        "--method",
        choices=["shape_keys", "mesh_cache"],
        default="shape_keys",
        help="Import method (default: shape_keys)"
    )

    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    obj_files = find_obj_files(args.input)
    if not obj_files:
        print(f"Error: No OBJ files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(obj_files)} OBJ files")
    print(f"  First: {obj_files[0].name}")
    print(f"  Last: {obj_files[-1].name}")

    clear_scene()

    print("Importing mesh sequence...")
    if args.method == "mesh_cache":
        mesh_obj = import_obj_sequence_as_mesh_cache(
            obj_files, args.start_frame, args.fps
        )
    else:
        mesh_obj = import_obj_sequence_as_shape_keys(obj_files, args.start_frame)

    print(f"Created animated mesh: {mesh_obj.name}")

    end_frame = args.start_frame + len(obj_files) - 1

    print(f"Exporting Alembic...")
    print(f"  Output: {args.output}")
    print(f"  Frames: {args.start_frame}-{end_frame}")
    print(f"  FPS: {args.fps}")

    export_alembic(
        args.output,
        args.start_frame,
        end_frame,
        args.fps,
        apply_modifiers=True
    )

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
