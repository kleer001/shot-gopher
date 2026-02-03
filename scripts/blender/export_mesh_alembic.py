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

The script uses a frame change handler to update mesh vertices at each
frame during export, ensuring Alembic properly captures the animation.
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
    """Read vertex positions from OBJ file.

    Args:
        filepath: Path to OBJ file

    Returns:
        List of (x, y, z) vertex positions
    """
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
    return vertices


def load_all_obj_data(obj_files: list[Path]) -> list[list[tuple[float, float, float]]]:
    """Load vertex data from all OBJ files.

    Args:
        obj_files: List of OBJ file paths

    Returns:
        List of vertex position lists, one per frame
    """
    all_vertices = []
    for obj_file in obj_files:
        vertices = read_obj_vertices(obj_file)
        all_vertices.append(vertices)
    return all_vertices


def create_mesh_from_obj(filepath: Path) -> bpy.types.Object:
    """Create a mesh object from OBJ file.

    Args:
        filepath: Path to OBJ file

    Returns:
        Created mesh object
    """
    bpy.ops.wm.obj_import(filepath=str(filepath))
    if not bpy.context.selected_objects:
        raise ValueError(f"Failed to import: {filepath}")

    obj = bpy.context.selected_objects[0]
    obj.name = "animated_mesh"
    return obj


class MeshSequenceUpdater:
    """Updates mesh vertices at each frame from pre-loaded OBJ data."""

    def __init__(self, mesh_obj: bpy.types.Object,
                 all_vertices: list[list[tuple[float, float, float]]],
                 start_frame: int):
        self.mesh_obj = mesh_obj
        self.all_vertices = all_vertices
        self.start_frame = start_frame
        self.num_frames = len(all_vertices)

    def update(self, scene):
        """Frame change handler - updates mesh vertices."""
        frame = scene.frame_current
        frame_index = frame - self.start_frame

        if frame_index < 0 or frame_index >= self.num_frames:
            return

        vertices = self.all_vertices[frame_index]
        mesh = self.mesh_obj.data

        for i, (x, y, z) in enumerate(vertices):
            if i < len(mesh.vertices):
                mesh.vertices[i].co = (x, y, z)

        mesh.update()


def export_alembic(
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float = 24.0,
):
    """Export scene to Alembic file with animation baked.

    Args:
        output_path: Output .abc file path
        start_frame: First frame to export
        end_frame: Last frame to export
        fps: Frames per second
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

    print("Loading all OBJ vertex data...")
    all_vertices = load_all_obj_data(obj_files)

    vertex_count = len(all_vertices[0]) if all_vertices else 0
    for i, verts in enumerate(all_vertices):
        if len(verts) != vertex_count:
            print(f"Error: Vertex count mismatch at frame {i}: {len(verts)} vs {vertex_count}",
                  file=sys.stderr)
            sys.exit(1)

    clear_scene()

    print("Creating base mesh...")
    mesh_obj = create_mesh_from_obj(obj_files[0])

    print(f"Created animated mesh: {mesh_obj.name}")
    print(f"  Vertices: {len(mesh_obj.data.vertices)}")
    print(f"  Frames: {len(obj_files)}")

    updater = MeshSequenceUpdater(mesh_obj, all_vertices, args.start_frame)
    bpy.app.handlers.frame_change_pre.append(updater.update)

    end_frame = args.start_frame + len(obj_files) - 1

    print(f"  Animation range: {args.start_frame}-{end_frame}")

    bpy.context.scene.frame_set(args.start_frame)
    updater.update(bpy.context.scene)

    print(f"Exporting Alembic...")
    print(f"  Output: {args.output}")
    print(f"  Frames: {args.start_frame}-{end_frame}")
    print(f"  FPS: {args.fps}")

    try:
        export_alembic(
            args.output,
            args.start_frame,
            end_frame,
            args.fps,
        )
    except Exception as e:
        print(f"Error exporting Alembic: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if updater.update in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(updater.update)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
