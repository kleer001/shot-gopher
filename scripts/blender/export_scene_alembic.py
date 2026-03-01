#!/usr/bin/env python3
"""Blender script to export animated mesh + camera to a single Alembic file.

Combines OBJ mesh sequence and camera JSON data into one .abc containing
both an animated polymesh and an animated camera node.

Usage (from command line):
    blender -b --python export_scene_alembic.py -- \
        --mesh-dir /path/to/obj_sequence/ \
        --camera-dir /path/to/camera_json/ \
        --output /path/to/scene.abc \
        --fps 24 --start-frame 1 --camera-name mocap_mmcam
"""

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from camera_common import create_animated_camera, load_camera_data
from export_mesh_alembic import (
    create_mesh_from_obj,
    find_obj_files,
    setup_shape_key_animation,
    verify_animation,
)


def clear_scene():
    """Remove all objects, meshes, and cameras from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for camera in list(bpy.data.cameras):
        bpy.data.cameras.remove(camera)


def export_alembic(output_path: Path, start_frame: int, end_frame: int, fps: int):
    """Export entire scene to Alembic."""
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    bpy.context.scene.render.fps = fps
    bpy.context.scene.render.fps_base = 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Calling Alembic export for frames {start_frame}-{end_frame}...")
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
        evaluation_mode="RENDER",
    )


def main():
    """Main entry point."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Export mesh + camera to single Alembic"
    )
    parser.add_argument(
        "--mesh-dir", type=Path, required=True, help="Directory containing OBJ files"
    )
    parser.add_argument(
        "--camera-dir",
        type=Path,
        required=True,
        help="Directory containing extrinsics.json and intrinsics.json",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output Alembic file path"
    )
    parser.add_argument("--fps", type=int, default=24, help="FPS (default: 24)")
    parser.add_argument(
        "--start-frame", type=int, default=1, help="Start frame (default: 1)"
    )
    parser.add_argument(
        "--camera-name", type=str, default="camera", help="Camera name"
    )
    args = parser.parse_args(argv)

    obj_files = find_obj_files(args.mesh_dir)
    if not obj_files:
        print(f"Error: No OBJ files found in {args.mesh_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(obj_files)} OBJ files")

    extrinsics, intrinsics, source = load_camera_data(args.camera_dir)
    if not extrinsics:
        print("Error: No extrinsics data found", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(extrinsics)} camera frames")

    clear_scene()

    print("Creating animated mesh...")
    mesh_obj = create_mesh_from_obj(obj_files[0])
    print(f"  {mesh_obj.name}: {len(mesh_obj.data.vertices)} vertices")
    end_frame = setup_shape_key_animation(mesh_obj, obj_files, args.start_frame)

    print("Creating animated camera...")
    create_animated_camera(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        start_frame=args.start_frame,
        fps=args.fps,
        camera_name=args.camera_name,
    )

    print(f"Animation range: {args.start_frame}-{end_frame}")
    verify_animation(mesh_obj, args.start_frame, end_frame)

    print(f"\nExporting combined scene Alembic...")
    print(f"  Output: {args.output}")
    export_alembic(args.output, args.start_frame, end_frame, args.fps)
    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
