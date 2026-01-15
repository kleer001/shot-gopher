"""Houdini script to import camera from pipeline JSON format.

Run from Houdini: File → Run Script → select import_camera.cmd
Or paste into Houdini's Python shell.

Creates an animated camera in /obj from extrinsics.json and intrinsics.json
"""

import json
import math
import os

import hou


def decompose_matrix(matrix):
    """Decompose 4x4 matrix to translation, rotation (euler XYZ), scale."""
    # Extract translation
    tx = matrix[0][3]
    ty = matrix[1][3]
    tz = matrix[2][3]

    # Extract 3x3 rotation/scale
    m = [[matrix[i][j] for j in range(3)] for i in range(3)]

    # Compute scale
    sx = math.sqrt(m[0][0]**2 + m[1][0]**2 + m[2][0]**2)
    sy = math.sqrt(m[0][1]**2 + m[1][1]**2 + m[2][1]**2)
    sz = math.sqrt(m[0][2]**2 + m[1][2]**2 + m[2][2]**2)

    # Normalize to get rotation matrix
    if sx > 1e-8:
        m[0][0] /= sx; m[1][0] /= sx; m[2][0] /= sx
    if sy > 1e-8:
        m[0][1] /= sy; m[1][1] /= sy; m[2][1] /= sy
    if sz > 1e-8:
        m[0][2] /= sz; m[1][2] /= sz; m[2][2] /= sz

    # Convert rotation matrix to Euler XYZ (degrees)
    sy_check = math.sqrt(m[0][0]**2 + m[1][0]**2)

    if sy_check > 1e-6:
        rx = math.atan2(m[2][1], m[2][2])
        ry = math.atan2(-m[2][0], sy_check)
        rz = math.atan2(m[1][0], m[0][0])
    else:
        rx = math.atan2(-m[1][2], m[1][1])
        ry = math.atan2(-m[2][0], sy_check)
        rz = 0

    return (tx, ty, tz), (math.degrees(rx), math.degrees(ry), math.degrees(rz)), (sx, sy, sz)


def load_camera_data(project_dir):
    """Load camera extrinsics and intrinsics from project directory."""
    camera_dir = os.path.join(project_dir, "camera")

    extrinsics_path = os.path.join(camera_dir, "extrinsics.json")
    intrinsics_path = os.path.join(camera_dir, "intrinsics.json")
    metadata_path = os.path.join(project_dir, "project.json")

    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"Extrinsics not found: {extrinsics_path}")

    with open(extrinsics_path) as f:
        extrinsics = json.load(f)

    intrinsics = {}
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path) as f:
            intrinsics = json.load(f)

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    return extrinsics, intrinsics, metadata


def compute_focal_length_mm(intrinsics, sensor_width_mm=36.0):
    """Convert focal length from pixels to mm."""
    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    width = intrinsics.get("width", 1920)
    return fx * sensor_width_mm / width


def create_camera(project_dir, camera_name="pipeline_cam", sensor_width_mm=36.0):
    """Create animated camera in /obj from pipeline camera data.

    Args:
        project_dir: Path to project directory containing camera/ subfolder
        camera_name: Name for the camera node
        sensor_width_mm: Sensor width in mm (default 36mm full-frame)

    Returns:
        The created camera node
    """
    # Load data
    extrinsics, intrinsics, metadata = load_camera_data(project_dir)

    fps = metadata.get("fps", 24.0)
    start_frame = metadata.get("start_frame", 1)

    print(f"Loaded {len(extrinsics)} camera frames")
    print(f"FPS: {fps}, Start frame: {start_frame}")

    # Set Houdini FPS to match
    hou.setFps(fps)

    # Create camera node
    obj = hou.node("/obj")
    cam = obj.createNode("cam", camera_name)

    # Set focal length
    focal_mm = compute_focal_length_mm(intrinsics, sensor_width_mm)
    cam.parm("focal").set(focal_mm)
    print(f"Focal length: {focal_mm:.2f}mm")

    # Set aperture (sensor size)
    cam.parm("aperture").set(sensor_width_mm)

    # Set resolution if available
    width = intrinsics.get("width", 1920)
    height = intrinsics.get("height", 1080)
    cam.parm("resx").set(width)
    cam.parm("resy").set(height)

    # Animate the camera
    for frame_idx, matrix in enumerate(extrinsics):
        frame = start_frame + frame_idx

        trans, rot, scale = decompose_matrix(matrix)

        # Set keyframes
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

    # Select and frame the camera
    cam.setSelected(True, clear_all_selected=True)
    cam.setDisplayFlag(True)

    print(f"Created camera: {cam.path()}")
    print(f"Frame range: {start_frame}-{end_frame}")

    return cam


# Main execution when run as script
if __name__ == "__main__" or hou.isUIAvailable():
    # Prompt user for project directory
    project_dir = hou.ui.selectFile(
        title="Select Project Directory",
        file_type=hou.fileType.Directory,
        chooser_mode=hou.fileChooserMode.Read,
    )

    if project_dir:
        # Clean up path (remove trailing slash, expand variables)
        project_dir = os.path.normpath(hou.text.expandString(project_dir))

        try:
            cam = create_camera(project_dir)
            hou.ui.displayMessage(
                f"Camera imported successfully!\n\n"
                f"Node: {cam.path()}\n"
                f"Frames: {int(cam.parm('tx').keyframes()[0].frame())}-"
                f"{int(cam.parm('tx').keyframes()[-1].frame())}",
                title="Camera Import"
            )
        except Exception as e:
            hou.ui.displayMessage(
                f"Error importing camera:\n\n{e}",
                severity=hou.severityType.Error,
                title="Camera Import Error"
            )
    else:
        print("No project directory selected")
