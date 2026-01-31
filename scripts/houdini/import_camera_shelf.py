"""Houdini shelf tool script - paste this into a new shelf tool.

To install:
1. Right-click on shelf → New Tool
2. Name it "Import Pipeline Camera"
3. Paste this entire script into the Script tab
4. Click Accept

Then click the shelf button to import a camera.
"""

import json
import math
import os
import hou


def decompose_matrix(matrix):
    tx, ty, tz = matrix[0][3], matrix[1][3], matrix[2][3]
    m = [[matrix[i][j] for j in range(3)] for i in range(3)]
    sx = math.sqrt(m[0][0]**2 + m[1][0]**2 + m[2][0]**2)
    sy = math.sqrt(m[0][1]**2 + m[1][1]**2 + m[2][1]**2)
    sz = math.sqrt(m[0][2]**2 + m[1][2]**2 + m[2][2]**2)
    if sx > 1e-8: m[0][0] /= sx; m[1][0] /= sx; m[2][0] /= sx
    if sy > 1e-8: m[0][1] /= sy; m[1][1] /= sy; m[2][1] /= sy
    if sz > 1e-8: m[0][2] /= sz; m[1][2] /= sz; m[2][2] /= sz
    sy_check = math.sqrt(m[0][0]**2 + m[1][0]**2)
    if sy_check > 1e-6:
        rx = math.atan2(m[2][1], m[2][2])
        ry = math.atan2(-m[2][0], sy_check)
        rz = math.atan2(m[1][0], m[0][0])
    else:
        rx = math.atan2(-m[1][2], m[1][1])
        ry = math.atan2(-m[2][0], sy_check)
        rz = 0
    return (tx, ty, tz), (math.degrees(rx), math.degrees(ry), math.degrees(rz))


# Prompt for project directory
project_dir = hou.ui.selectFile(
    title="Select Project Directory (containing camera/ folder)",
    file_type=hou.fileType.Directory,
    chooser_mode=hou.fileChooserMode.Read,
)

if project_dir:
    project_dir = os.path.normpath(hou.text.expandString(project_dir))
    camera_dir = os.path.join(project_dir, "camera")

    # Load JSON files
    with open(os.path.join(camera_dir, "extrinsics.json")) as f:
        extrinsics = json.load(f)

    intrinsics = {}
    intrinsics_path = os.path.join(camera_dir, "intrinsics.json")
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path) as f:
            intrinsics = json.load(f)

    metadata = {}
    metadata_path = os.path.join(project_dir, "project.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    fps = metadata.get("fps", 24.0)
    start_frame = metadata.get("start_frame", 1)

    # Compute focal length
    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    width = intrinsics.get("width", 1920)
    height = intrinsics.get("height", 1080)
    sensor_mm = 36.0
    focal_mm = fx * sensor_mm / width

    # Set FPS
    hou.setFps(fps)

    # Create camera
    cam = hou.node("/obj").createNode("cam", "pipeline_cam")
    cam.parm("focal").set(focal_mm)
    cam.parm("aperture").set(sensor_mm)
    cam.parm("resx").set(width)
    cam.parm("resy").set(height)

    # Animate
    for i, matrix in enumerate(extrinsics):
        frame = start_frame + i
        trans, rot = decompose_matrix(matrix)
        cam.parm("tx").setKeyframe(hou.Keyframe(trans[0], frame))
        cam.parm("ty").setKeyframe(hou.Keyframe(trans[1], frame))
        cam.parm("tz").setKeyframe(hou.Keyframe(trans[2], frame))
        cam.parm("rx").setKeyframe(hou.Keyframe(rot[0], frame))
        cam.parm("ry").setKeyframe(hou.Keyframe(rot[1], frame))
        cam.parm("rz").setKeyframe(hou.Keyframe(rot[2], frame))

    # Set frame range
    end_frame = start_frame + len(extrinsics) - 1
    hou.playbar.setFrameRange(start_frame, end_frame)
    hou.playbar.setPlaybackRange(start_frame, end_frame)

    cam.setSelected(True, clear_all_selected=True)

    hou.ui.displayMessage(
        f"Camera imported!\n\nNode: {cam.path()}\nFrames: {start_frame}-{end_frame}\nFocal: {focal_mm:.1f}mm",
        title="Import Complete"
    )
