#!/usr/bin/env python3
"""Blender script to export OBJ mesh sequence to USD.

This script runs inside Blender's Python environment and exports
a sequence of OBJ files as an animated USD (.usd/.usda/.usdc) file.

Usage (from command line):
    blender -b --python export_mesh_usd.py -- \
        --input /path/to/meshes/ \
        --output /path/to/output.usd \
        --fps 24 \
        --start-frame 1

The script will:
1. Import OBJ files as an animated mesh sequence
2. Export to USD with proper time sampling
"""

import argparse
import sys
from pathlib import Path

import bpy


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def find_obj_files(input_dir: Path) -> list[Path]:
    """Find and sort OBJ files in directory.

    Args:
        input_dir: Directory containing OBJ files

    Returns:
        Sorted list of OBJ file paths
    """
    obj_files = list(input_dir.glob("*.obj")) + list(input_dir.glob("*.OBJ"))
    return sorted(set(obj_files))


def get_shape_key_frame(shape_key: bpy.types.ShapeKey) -> int:
    """Extract frame number from shape key name.

    Shape keys are named 'frame_XXXX' where XXXX is the frame number.

    Args:
        shape_key: Blender shape key with name in 'frame_XXXX' format

    Returns:
        Frame number as integer
    """
    return int(shape_key.name.split("_")[1])


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

    Raises:
        ValueError: If no OBJ files provided or import fails
        RuntimeError: If OBJ files have inconsistent vertex counts
    """
    if not obj_files:
        raise ValueError("No OBJ files provided")

    bpy.ops.wm.obj_import(filepath=str(obj_files[0]))
    if not bpy.context.selected_objects:
        raise ValueError(f"Failed to import base mesh: {obj_files[0]}")

    base_obj = bpy.context.selected_objects[0]
    base_obj.name = "animated_mesh"
    base_vertex_count = len(base_obj.data.vertices)

    if not base_obj.data.shape_keys:
        base_obj.shape_key_add(name="Basis", from_mix=False)

    for i, obj_file in enumerate(obj_files[1:], start=1):
        frame = start_frame + i

        bpy.ops.wm.obj_import(filepath=str(obj_file))
        if not bpy.context.selected_objects:
            print(f"Warning: Failed to import {obj_file}, skipping")
            continue

        temp_obj = bpy.context.selected_objects[0]

        base_obj.select_set(True)
        bpy.context.view_layer.objects.active = base_obj

        try:
            temp_vertex_count = len(temp_obj.data.vertices)
            if temp_vertex_count != base_vertex_count:
                raise RuntimeError(
                    f"Vertex count mismatch: {obj_file.name} has {temp_vertex_count} vertices, "
                    f"expected {base_vertex_count}. All OBJ files must have identical topology."
                )

            sk = base_obj.shape_key_add(name=f"frame_{frame:04d}", from_mix=False)
            for j, vert in enumerate(temp_obj.data.vertices):
                sk.data[j].co = vert.co
        finally:
            bpy.data.objects.remove(temp_obj, do_unlink=True)

    if len(obj_files) == 1:
        print("Note: Only 1 OBJ file - exporting static mesh (no animation)")
        return base_obj

    for key in base_obj.data.shape_keys.key_blocks[1:]:
        frame = get_shape_key_frame(key)
        prev_frame = frame - 1

        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=prev_frame)

        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=frame)

        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=frame + 1)

    return base_obj


def export_usd(
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float = 24.0,
):
    """Export scene to USD file.

    Args:
        output_path: Output .usd file path
        start_frame: First frame to export
        end_frame: Last frame to export
        fps: Frames per second
    """
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    bpy.context.scene.render.fps = int(fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=False,
        visible_objects_only=True,
        export_animation=True,
        export_uvmaps=True,
        export_normals=True,
        export_materials=False,
        use_instancing=True,
        evaluation_mode='RENDER',
    )


def main():
    """Main entry point for the export script."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Export OBJ mesh sequence to USD"
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
        help="Output USD file path"
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

    try:
        print("Importing mesh sequence...")
        mesh_obj = import_obj_sequence_as_shape_keys(obj_files, args.start_frame)
    except (ValueError, RuntimeError) as e:
        print(f"Error importing mesh sequence: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Created animated mesh: {mesh_obj.name}")

    if mesh_obj.data.shape_keys and len(mesh_obj.data.shape_keys.key_blocks) > 1:
        end_frame = max(
            get_shape_key_frame(key)
            for key in mesh_obj.data.shape_keys.key_blocks[1:]
        )
    else:
        end_frame = args.start_frame

    print(f"Exporting USD...")
    print(f"  Output: {args.output}")
    print(f"  Frames: {args.start_frame}-{end_frame}")
    print(f"  FPS: {args.fps}")

    try:
        export_usd(
            args.output,
            args.start_frame,
            end_frame,
            args.fps,
        )
    except Exception as e:
        print(f"Error exporting USD: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
