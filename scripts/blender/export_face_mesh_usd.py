#!/usr/bin/env python3
"""Blender script to export animated face mesh landmarks to USD.

Reads face_mesh/landmarks.npz and creates a 478-vertex animated mesh,
driving vertex positions per frame via shape keys, then exports to USD.

Usage (from command line):
    blender -b --python export_face_mesh_usd.py -- \
        --input /path/to/face_mesh/ \
        --output /path/to/face_mesh.usd \
        --fps 24 \
        --start-frame 1
"""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def clear_scene() -> None:
    """Remove all objects and meshes from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def create_landmark_mesh(verts_frame0: np.ndarray) -> bpy.types.Object:
    """Create a mesh object from first-frame landmark positions.

    Adds minimal triangles from consecutive vertex triplets so that
    Blender's USD exporter includes vertex animation in the output.

    Args:
        verts_frame0: (478, 3) array of vertex positions.

    Returns:
        Created Blender mesh object.
    """
    n = len(verts_frame0)
    vertices = [tuple(v.tolist()) for v in verts_frame0]
    faces = [(i, i + 1, i + 2) for i in range(0, n - 2, 3)]

    mesh = bpy.data.meshes.new("face_landmarks")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("face_landmarks", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def setup_shape_key_animation(
    obj: bpy.types.Object,
    landmarks: np.ndarray,
    detected: np.ndarray,
    start_frame: int,
) -> None:
    """Animate vertex positions per frame using shape keys.

    Args:
        obj: Blender mesh object.
        landmarks: (n_frames, 478, 3) landmark array.
        detected: (n_frames,) boolean detection mask.
        start_frame: First Blender frame number.
    """
    n_frames = landmarks.shape[0]
    n_verts = landmarks.shape[1]

    obj.shape_key_add(name="Basis", from_mix=False)
    shape_keys = [obj.shape_key_add(name=f"frame_{start_frame + i:06d}", from_mix=False)
                  for i in range(n_frames)]

    last_valid = landmarks[0] if detected[0] else np.zeros((n_verts, 3), dtype=np.float32)

    for i, sk in enumerate(shape_keys):
        frame_lm = landmarks[i] if detected[i] else last_valid
        if detected[i]:
            last_valid = frame_lm
        for v_idx in range(n_verts):
            sk.data[v_idx].co = frame_lm[v_idx].tolist()

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = start_frame + n_frames - 1

    for i, sk in enumerate(shape_keys):
        blender_frame = start_frame + i
        for j, other_sk in enumerate(shape_keys):
            val = 1.0 if j == i else 0.0
            other_sk.value = val
            other_sk.keyframe_insert(data_path="value", frame=blender_frame)

    key_data = obj.data.shape_keys
    if key_data and key_data.animation_data and key_data.animation_data.action:
        for fc in key_data.animation_data.action.fcurves:
            for kfp in fc.keyframe_points:
                kfp.interpolation = "CONSTANT"


def export_usd(output_path: Path, start_frame: int, end_frame: int, fps: int) -> None:
    """Export the scene to USD.

    Args:
        output_path: Output .usd file path.
        start_frame: First frame to export.
        end_frame: Last frame to export.
        fps: Frames per second.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=False,
        visible_objects_only=True,
        export_animation=True,
        export_hair=False,
        export_uvmaps=False,
        export_normals=False,
        export_materials=False,
        use_instancing=False,
        evaluation_mode="RENDER",
    )


def main() -> None:
    """Entry point for headless Blender execution."""
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Export face mesh landmarks to USD")
    parser.add_argument("--input", "-i", type=Path, required=True, help="face_mesh/ directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .usd file")
    parser.add_argument("--fps", "-f", type=int, default=24, help="Frames per second")
    parser.add_argument("--start-frame", "-s", type=int, default=1, help="First frame number")
    args = parser.parse_args(argv)

    landmarks_path = args.input / "landmarks.npz"
    if not landmarks_path.exists():
        print(f"Error: landmarks.npz not found: {landmarks_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(landmarks_path)
    landmarks = data["landmarks"]
    detected = data["detected"]
    n_frames = landmarks.shape[0]

    first_detected = int(np.argmax(detected)) if detected.any() else 0
    verts_frame0 = landmarks[first_detected]

    clear_scene()
    obj = create_landmark_mesh(verts_frame0)
    setup_shape_key_animation(obj, landmarks, detected, args.start_frame)
    export_usd(args.output, args.start_frame, args.start_frame + n_frames - 1, args.fps)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
