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

Uses shape keys with proper animation setup that Blender's USD
exporter can detect and bake into vertex animation.
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
    """Find and sort OBJ files in directory."""
    obj_files = list(input_dir.glob("*.obj")) + list(input_dir.glob("*.OBJ"))
    return sorted(set(obj_files))


def read_obj_vertices(filepath: Path) -> list[tuple[float, float, float]]:
    """Read vertex positions from OBJ file."""
    vertices = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
    return vertices


def create_mesh_from_obj(filepath: Path) -> bpy.types.Object:
    """Create a mesh object from OBJ file."""
    bpy.ops.wm.obj_import(filepath=str(filepath))
    if not bpy.context.selected_objects:
        raise ValueError(f"Failed to import: {filepath}")

    obj = bpy.context.selected_objects[0]
    obj.name = "animated_mesh"
    return obj


def setup_shape_key_animation(mesh_obj: bpy.types.Object,
                               obj_files: list[Path],
                               start_frame: int) -> int:
    """Set up shape keys and their animation for each frame.

    Creates one shape key per OBJ file and animates them so exactly
    one shape key is active at each frame.

    Args:
        mesh_obj: The base mesh object
        obj_files: List of OBJ file paths
        start_frame: Starting frame number

    Returns:
        End frame number
    """
    if not mesh_obj.data.shape_keys:
        mesh_obj.shape_key_add(name="Basis", from_mix=False)

    base_vertex_count = len(mesh_obj.data.vertices)

    print(f"Creating {len(obj_files)} shape keys...")

    for i, obj_file in enumerate(obj_files):
        frame = start_frame + i

        if i == 0:
            continue

        vertices = read_obj_vertices(obj_file)

        if len(vertices) != base_vertex_count:
            raise RuntimeError(
                f"Vertex count mismatch: {obj_file.name} has {len(vertices)} vertices, "
                f"expected {base_vertex_count}"
            )

        sk = mesh_obj.shape_key_add(name=f"frame_{frame:04d}", from_mix=False)
        for j, (x, y, z) in enumerate(vertices):
            sk.data[j].co = (x, y, z)

        if i % 50 == 0:
            print(f"  Created shape key {i}/{len(obj_files)}...")

    num_shape_keys = len(mesh_obj.data.shape_keys.key_blocks) - 1
    print(f"Created {num_shape_keys} shape keys")

    if num_shape_keys == 0:
        return start_frame

    print("Setting up animation...")

    for key in mesh_obj.data.shape_keys.key_blocks[1:]:
        frame = int(key.name.split("_")[1])

        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=frame - 1)

        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=frame)

        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=frame + 1)

    action = mesh_obj.data.shape_keys.animation_data.action
    for fcurve in action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'CONSTANT'

    end_frame = start_frame + len(obj_files) - 1
    return end_frame


def export_usd(
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float = 24.0,
):
    """Export scene to USD file with animation baked."""
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.render.fps_base = fps / int(fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Calling USD export for frames {start_frame}-{end_frame}...")

    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=False,
        visible_objects_only=True,
        export_animation=True,
        export_uvmaps=True,
        export_normals=True,
        export_materials=False,
        use_instancing=False,
        evaluation_mode='RENDER',
    )


def verify_animation(mesh_obj: bpy.types.Object, start_frame: int, end_frame: int):
    """Verify shape key animation is set up correctly."""
    print("\nVerifying animation setup...")

    bpy.context.scene.frame_set(start_frame)
    bpy.context.view_layer.update()

    first_vert = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices[0].co.copy()
    print(f"  Frame {start_frame}: vertex[0] = ({first_vert.x:.4f}, {first_vert.y:.4f}, {first_vert.z:.4f})")

    mid_frame = (start_frame + end_frame) // 2
    bpy.context.scene.frame_set(mid_frame)
    bpy.context.view_layer.update()

    mid_vert = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices[0].co.copy()
    print(f"  Frame {mid_frame}: vertex[0] = ({mid_vert.x:.4f}, {mid_vert.y:.4f}, {mid_vert.z:.4f})")

    bpy.context.scene.frame_set(end_frame)
    bpy.context.view_layer.update()

    last_vert = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices[0].co.copy()
    print(f"  Frame {end_frame}: vertex[0] = ({last_vert.x:.4f}, {last_vert.y:.4f}, {last_vert.z:.4f})")

    if first_vert == mid_vert == last_vert:
        print("  WARNING: Vertices appear identical across frames - animation may not be working!")
    else:
        print("  OK: Vertices differ across frames - animation appears to be working")

    bpy.context.scene.frame_set(start_frame)


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

    print("Creating base mesh from first OBJ...")
    mesh_obj = create_mesh_from_obj(obj_files[0])

    print(f"Created mesh: {mesh_obj.name}")
    print(f"  Vertices: {len(mesh_obj.data.vertices)}")

    end_frame = setup_shape_key_animation(mesh_obj, obj_files, args.start_frame)

    print(f"Animation range: {args.start_frame}-{end_frame}")

    verify_animation(mesh_obj, args.start_frame, end_frame)

    print(f"\nExporting USD...")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
