#!/usr/bin/env python3
"""Export camera data to various formats for VFX applications.

Converts camera data (from Video Depth Anything or COLMAP) to formats
importable by Houdini, Nuke, Maya, and Blender.

Supported output formats:
  - Nuke .chan (text file with per-frame transforms)
  - CSV (spreadsheet-compatible camera data)
  - JSON (detailed per-frame transforms)
  - Alembic .abc (requires Blender, auto-installed via wizard)
  - USD .usd/.usda/.usdc (requires Blender, auto-installed via wizard)
  - After Effects .jsx (JavaScript script for camera import)

Supports camera data from:
  - Video Depth Anything (monocular depth estimation with camera)
  - COLMAP (Structure-from-Motion reconstruction)

Usage:
    python export_camera.py <project_dir> [--format chan] [--fps 24]

Example:
    python export_camera.py /path/to/projects/My_Shot --format chan
    python export_camera.py /path/to/projects/My_Shot --format csv
    python export_camera.py /path/to/projects/My_Shot --format all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from transforms import (
    compute_fov_from_intrinsics,
    convert_opencv_to_opengl,
    decompose_matrix,
    rotation_matrix_to_euler,
)

HAS_BLENDER = False
try:
    from blender import export_camera_to_alembic, export_camera_to_usd, check_blender_available
    HAS_BLENDER = True
except ImportError:
    pass


def load_camera_data(
    camera_dir: Path,
    convert_to_opengl: bool = True
) -> tuple[list[np.ndarray], dict, str]:
    """Load extrinsics and intrinsics from camera data JSONs.

    Supports COLMAP output format. Optionally converts from OpenCV to OpenGL
    coordinate convention for DCC compatibility.

    Args:
        camera_dir: Path to camera/ directory containing extrinsics.json and intrinsics.json
        convert_to_opengl: If True, convert from OpenCV (Y-down, Z-forward) to
                           OpenGL (Y-up, Z-back) convention for DCC apps

    Returns:
        Tuple of (list of 4x4 extrinsic matrices, intrinsics dict, source name)
    """
    extrinsics_path = camera_dir / "extrinsics.json"
    intrinsics_path = camera_dir / "intrinsics.json"
    colmap_raw_path = camera_dir / "colmap_raw.json"

    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics_path}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

    with open(extrinsics_path) as f:
        extrinsics_data = json.load(f)

    with open(intrinsics_path) as f:
        intrinsics_data = json.load(f)

    if colmap_raw_path.exists():
        source = "colmap"
    else:
        source = "da3"

    extrinsics = []
    if isinstance(extrinsics_data, list):
        for matrix_data in extrinsics_data:
            if isinstance(matrix_data, list):
                matrix = np.array(matrix_data).reshape(4, 4)
            else:
                matrix = np.eye(4)
            if convert_to_opengl:
                matrix = convert_opencv_to_opengl(matrix)
            extrinsics.append(matrix)
    elif isinstance(extrinsics_data, dict) and "matrices" in extrinsics_data:
        for matrix_data in extrinsics_data["matrices"]:
            matrix = np.array(matrix_data).reshape(4, 4)
            if convert_to_opengl:
                matrix = convert_opencv_to_opengl(matrix)
            extrinsics.append(matrix)

    return extrinsics, intrinsics_data, source


def export_json_camera(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1,
    fps: float = 24.0
) -> None:
    """Export camera data to JSON format (fallback when Alembic unavailable).

    Creates a JSON file that can be imported via scripts in Houdini/Nuke/Blender.
    """
    camera_data = {
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": start_frame + len(extrinsics) - 1,
        "intrinsics": intrinsics,
        "frames": []
    }

    for frame_idx, matrix in enumerate(extrinsics):
        translation, rotation, scale = decompose_matrix(matrix)
        euler = rotation_matrix_to_euler(rotation)

        frame_data = {
            "frame": start_frame + frame_idx,
            "matrix": matrix.tolist(),
            "translation": translation.tolist(),
            "rotation_euler_xyz": euler.tolist(),
            "scale": scale.tolist()
        }
        camera_data["frames"].append(frame_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(camera_data, f, indent=2)

    print(f"Exported {len(extrinsics)} frames to {output_path}")
    print(f"  Frame range: {start_frame}-{start_frame + len(extrinsics) - 1}")


def export_nuke_chan(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1,
    rotation_order: str = "zxy",
) -> None:
    """Export camera to Nuke .chan format.

    The .chan format is a simple text file with one line per frame:
    frame tx ty tz rx ry rz

    Nuke's default rotation order is ZXY. Set rotation order on Camera node
    to match the exported data.

    Import in Nuke: Camera node -> File -> Import chan file

    Args:
        extrinsics: List of 4x4 camera-to-world matrices per frame
        intrinsics: Camera intrinsics (for focal length info in header)
        output_path: Output .chan file path
        start_frame: Starting frame number
        rotation_order: Euler rotation order (default: "zxy" for Nuke)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
        fy = intrinsics.get("fy", intrinsics.get("focal_y", 1000))
        width = intrinsics.get("width", 1920)
        height = intrinsics.get("height", 1080)

        f.write(f"# Nuke camera chan file\n")
        f.write(f"# Frames: {start_frame}-{start_frame + len(extrinsics) - 1}\n")
        f.write(f"# Focal length (pixels): fx={fx:.2f} fy={fy:.2f}\n")
        f.write(f"# Resolution: {width}x{height}\n")
        f.write(f"# Rotation order: {rotation_order.upper()}\n")
        f.write(f"# Format: frame tx ty tz rx ry rz\n")
        f.write(f"#\n")

        for frame_idx, matrix in enumerate(extrinsics):
            frame = start_frame + frame_idx
            translation, rotation, _ = decompose_matrix(matrix)
            euler = rotation_matrix_to_euler(rotation, order=rotation_order)

            f.write(f"{frame} {translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} "
                    f"{euler[0]:.6f} {euler[1]:.6f} {euler[2]:.6f}\n")

    print(f"Exported {len(extrinsics)} frames to {output_path}")
    print(f"  Frame range: {start_frame}-{start_frame + len(extrinsics) - 1}")
    print(f"  Rotation order: {rotation_order.upper()}")
    print(f"  Import in Nuke: Camera → File → Import chan file")


def export_csv(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1,
    rotation_order: str = "xyz",
) -> None:
    """Export camera to CSV format for spreadsheet/scripted import.

    Columns: frame, tx, ty, tz, rx, ry, rz, fx, fy, cx, cy

    Args:
        extrinsics: List of 4x4 camera-to-world matrices per frame
        intrinsics: Camera intrinsics dict
        output_path: Output .csv file path
        start_frame: Starting frame number
        rotation_order: Euler rotation order (default: "xyz")
    """
    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    fy = intrinsics.get("fy", intrinsics.get("focal_y", 1000))
    cx = intrinsics.get("cx", intrinsics.get("principal_x", 960))
    cy = intrinsics.get("cy", intrinsics.get("principal_y", 540))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Rotation order: {rotation_order.upper()}\n")
        f.write("frame,tx,ty,tz,rx,ry,rz,fx,fy,cx,cy\n")

        for frame_idx, matrix in enumerate(extrinsics):
            frame = start_frame + frame_idx
            translation, rotation, _ = decompose_matrix(matrix)
            euler = rotation_matrix_to_euler(rotation, order=rotation_order)

            f.write(f"{frame},{translation[0]:.6f},{translation[1]:.6f},{translation[2]:.6f},"
                    f"{euler[0]:.6f},{euler[1]:.6f},{euler[2]:.6f},"
                    f"{fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f}\n")

    print(f"Exported {len(extrinsics)} frames to {output_path}")
    print(f"  Frame range: {start_frame}-{start_frame + len(extrinsics) - 1}")
    print(f"  Rotation order: {rotation_order.upper()}")


def export_houdini_cmd(
    camera_dir: Path,
    output_path: Path,
    start_frame: int = 1,
    fps: float = 24.0,
) -> None:
    """Export Houdini .cmd file that creates camera from JSON.

    The .cmd file contains embedded Python that reads the extrinsics.json
    and intrinsics.json from the same directory and creates an animated
    camera in /obj.

    Run in Houdini: File → Run Script → select camera.cmd

    Args:
        camera_dir: Directory containing extrinsics.json and intrinsics.json
        output_path: Output .cmd file path
        start_frame: Starting frame number
        fps: Frames per second
    """
    # Get absolute path to camera directory
    camera_dir_abs = camera_dir.resolve()

    # Write a separate .py file with the actual Python code
    # This is more reliable than embedding in python -c
    py_script_path = output_path.with_suffix('.py')

    py_content = f'''#!/usr/bin/env python
# Houdini camera import script
# Auto-generated for project: {camera_dir_abs}
#
# Usage:
#   1. File -> Run Script -> select camera.cmd
#   2. OR Windows -> Python Source Editor -> Open -> camera.py -> Run
#   3. OR paste this into Houdini's Python Shell

import json
import math
import os
import sys

print("=" * 60)
print("Pipeline Camera Import")
print("=" * 60)

camera_dir = r'{camera_dir_abs}'
print(f"Camera directory: {{camera_dir}}")

# Check files exist
extrinsics_path = os.path.join(camera_dir, 'extrinsics.json')
intrinsics_path = os.path.join(camera_dir, 'intrinsics.json')

if not os.path.exists(extrinsics_path):
    print(f"ERROR: Extrinsics not found: {{extrinsics_path}}")
    sys.exit(1)
print(f"Found: extrinsics.json")

# Load data
with open(extrinsics_path) as f:
    extrinsics = json.load(f)
print(f"Loaded {{len(extrinsics)}} camera frames")

intrinsics = {{}}
if os.path.exists(intrinsics_path):
    with open(intrinsics_path) as f:
        intrinsics = json.load(f)
    print(f"Found: intrinsics.json")
else:
    print(f"No intrinsics.json, using defaults")

def decompose(m):
    tx, ty, tz = m[0][3], m[1][3], m[2][3]
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    sx = math.sqrt(r[0][0]**2 + r[1][0]**2 + r[2][0]**2)
    sy = math.sqrt(r[0][1]**2 + r[1][1]**2 + r[2][1]**2)
    sz = math.sqrt(r[0][2]**2 + r[1][2]**2 + r[2][2]**2)
    if sx > 1e-8: r[0][0]/=sx; r[1][0]/=sx; r[2][0]/=sx
    if sy > 1e-8: r[0][1]/=sy; r[1][1]/=sy; r[2][1]/=sy
    if sz > 1e-8: r[0][2]/=sz; r[1][2]/=sz; r[2][2]/=sz
    sc = math.sqrt(r[0][0]**2 + r[1][0]**2)
    if sc > 1e-6:
        rx = math.atan2(r[2][1], r[2][2])
        ry = math.atan2(-r[2][0], sc)
        rz = math.atan2(r[1][0], r[0][0])
    else:
        rx = math.atan2(-r[1][2], r[1][1])
        ry = math.atan2(-r[2][0], sc)
        rz = 0
    return (tx, ty, tz), (math.degrees(rx), math.degrees(ry), math.degrees(rz))

fps = {fps}
start_frame = {start_frame}
fx = intrinsics.get('fx', intrinsics.get('focal_x', 1000))
width = intrinsics.get('width', 1920)
height = intrinsics.get('height', 1080)
sensor_mm = 36.0
focal_mm = fx * sensor_mm / width

print(f"FPS: {{fps}}")
print(f"Start frame: {{start_frame}}")
print(f"Focal length: {{focal_mm:.2f}}mm")
print(f"Resolution: {{width}}x{{height}}")

# Import hou module
try:
    import hou
    print("Houdini module loaded OK")
except ImportError:
    print("ERROR: Cannot import hou module - run this inside Houdini!")
    sys.exit(1)

# Set FPS
hou.setFps(fps)
print(f"Set Houdini FPS to {{fps}}")

# Create camera
cam = hou.node('/obj').createNode('cam', 'pipeline_cam')
print(f"Created camera: {{cam.path()}}")

# Set camera properties
cam.parm('focal').set(focal_mm)
cam.parm('aperture').set(sensor_mm)
cam.parm('resx').set(width)
cam.parm('resy').set(height)
print("Set focal/aperture/resolution")

# Set keyframes
print("Setting keyframes...")
for i, m in enumerate(extrinsics):
    frame = start_frame + i
    t, r = decompose(m)
    # Create keyframes properly - setFrame() sets the frame number
    for parm_name, value in [('tx', t[0]), ('ty', t[1]), ('tz', t[2]),
                              ('rx', r[0]), ('ry', r[1]), ('rz', r[2])]:
        k = hou.Keyframe()
        k.setFrame(frame)
        k.setValue(value)
        cam.parm(parm_name).setKeyframe(k)
    if (i + 1) % 50 == 0 or i == len(extrinsics) - 1:
        print(f"  Processed frame {{frame}} ({{i + 1}}/{{len(extrinsics)}})")

# Set frame range
end_frame = start_frame + len(extrinsics) - 1
hou.playbar.setFrameRange(start_frame, end_frame)
hou.playbar.setPlaybackRange(start_frame, end_frame)

# Select camera
cam.setSelected(True, clear_all_selected=True)

print("=" * 60)
print(f"SUCCESS!")
print(f"Camera: {{cam.path()}}")
print(f"Frames: {{start_frame}}-{{end_frame}} ({{len(extrinsics)}} total)")
print("=" * 60)
'''

    # Write the Python script
    with open(py_script_path, 'w', encoding='utf-8') as f:
        f.write(py_content)

    # Write the .cmd file that runs the .py file
    # In hscript, python command takes code directly, not a file path
    # Use exec(open().read()) pattern to execute the .py file
    py_path_escaped = str(py_script_path).replace('\\', '/')
    cmd_content = f'''# Houdini camera import
# File -> Run Script -> select this file
#
# Alternative: Windows -> Python Source Editor -> Open camera.py -> Run

python -c "exec(open(r'{py_path_escaped}').read())"
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cmd_content)

    print(f"Exported Houdini scripts:")
    print(f"  {output_path} (run via File -> Run Script)")
    print(f"  {py_script_path} (or open in Python Source Editor)")


def export_houdini_clip(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1,
    fps: float = 24.0,
) -> None:
    """Export camera to Houdini .clip format (CHOP channel data).

    This creates a text-based clip file that can be imported directly
    into Houdini via CHOP Import or File CHOP.

    Args:
        extrinsics: List of 4x4 camera-to-world matrices per frame
        intrinsics: Camera intrinsics dict
        output_path: Output .clip file path
        start_frame: Starting frame number
        fps: Frames per second
    """
    num_frames = len(extrinsics)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Clip header
        f.write("{\n")
        f.write(f'  rate = {fps}\n')
        f.write(f'  start = {start_frame / fps}\n')
        f.write(f'  tracklength = {num_frames}\n')
        f.write('  tracks = 6 {\n')

        # Channel names
        channels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

        for ch_idx, ch_name in enumerate(channels):
            f.write(f'    {ch_name} ' + '{\n')
            f.write(f'      data = ')

            values = []
            for matrix in extrinsics:
                translation, rotation, _ = decompose_matrix(matrix)
                euler = rotation_matrix_to_euler(rotation)

                if ch_idx < 3:
                    values.append(translation[ch_idx])
                else:
                    values.append(euler[ch_idx - 3])

            f.write(' '.join(f'{v:.6f}' for v in values))
            f.write('\n    }\n')

        f.write('  }\n')
        f.write('}\n')

    print(f"Exported {len(extrinsics)} frames to {output_path}")
    print(f"  Import in Houdini: File CHOP → select .clip file")


def export_after_effects_jsx(
    extrinsics: list[np.ndarray],
    intrinsics: dict,
    output_path: Path,
    start_frame: int = 1,
    fps: float = 24.0,
    camera_name: str = "pipeline_camera"
) -> None:
    """Export camera to After Effects JSX script.

    Generates a JSX script that creates an animated camera in After Effects
    with position and rotation keyframes. The script can be run via:
    File → Scripts → Run Script File

    Args:
        extrinsics: List of 4x4 camera-to-world matrices per frame
        intrinsics: Camera intrinsics dict (fx, fy, cx, cy, width, height)
        output_path: Output .jsx file path
        start_frame: Starting frame number
        fps: Frames per second
        camera_name: Name for the camera layer
    """
    num_frames = len(extrinsics)

    # Extract intrinsics
    fx = intrinsics.get("fx", intrinsics.get("focal_x", 1000))
    fy = intrinsics.get("fy", intrinsics.get("focal_y", 1000))
    width = intrinsics.get("width", 1920)
    height = intrinsics.get("height", 1080)

    # Compute focal length in mm (assuming 36mm sensor width)
    sensor_width_mm = 36.0
    h_fov, v_fov = compute_fov_from_intrinsics(intrinsics, width, height)
    focal_length_mm = (sensor_width_mm / 2) / np.tan(np.radians(h_fov / 2))

    # Generate JSX script content
    jsx_content = f'''// After Effects Camera Import Script
// Auto-generated camera animation
//
// Usage:
//   1. File → Scripts → Run Script File → select this .jsx file
//   2. Or copy/paste into Adobe ExtendScript Toolkit
//
// Note: Enable "Allow Scripts to Write Files and Access Network"
//       in Edit → Preferences → Scripting & Expressions

(function() {{
    app.beginUndoGroup("Import Pipeline Camera");

    try {{
        // Configuration
        var cameraName = "{camera_name}";
        var fps = {fps};
        var startFrame = {start_frame};
        var numFrames = {num_frames};
        var width = {width};
        var height = {height};
        var focalLengthMm = {focal_length_mm:.4f};

        // Get active composition or create new one
        var comp = app.project.activeItem;
        if (!comp || !(comp instanceof CompItem)) {{
            comp = app.project.items.addComp(
                "Pipeline_Comp",
                width,
                height,
                1.0,  // pixel aspect ratio
                (numFrames / fps),  // duration in seconds
                fps
            );
            alert("Created new composition: " + comp.name);
        }}

        // Set composition frame rate and duration
        comp.frameRate = fps;
        comp.workAreaStart = 0;
        comp.workAreaDuration = numFrames / fps;
        comp.duration = numFrames / fps;

        // Create camera layer
        var camera = comp.layers.addCamera(cameraName, [width/2, height/2]);

        // Set camera properties
        // After Effects focal length is in pixels by default, but we can set zoom
        // Zoom relates to focal length via: zoom = (comp.width * focalLength) / (2 * sensorWidth)
        // For 36mm sensor: zoom = (width * focalLength_px) / width = focalLength_px
        // Convert from mm to pixels: focal_px = focal_mm * width / sensor_mm
        var focalLengthPx = focalLengthMm * width / 36.0;
        camera.property("ADBE Camera Options Group").property("ADBE Camera Zoom").setValue(focalLengthPx);

        // Camera transform data (position and rotation)
        var cameraData = [
'''

    # Add camera transform data for each frame
    for frame_idx, matrix in enumerate(extrinsics):
        translation, rotation, _ = decompose_matrix(matrix)
        euler = rotation_matrix_to_euler(rotation)

        # After Effects coordinate system conversion:
        # OpenGL/COLMAP: +Y up, +Z towards camera (RH)
        # After Effects: +Y down, +Z away from camera (RH)
        # Conversion: flip Y and Z
        ae_x = translation[0]
        ae_y = -translation[1]  # Flip Y
        ae_z = -translation[2]  # Flip Z

        # Rotation conversion - also need to flip rotations around Y and Z axes
        ae_rx = -euler[0]  # Flip X rotation
        ae_ry = euler[1]
        ae_rz = -euler[2]  # Flip Z rotation

        jsx_content += f'            {{ pos: [{ae_x:.6f}, {ae_y:.6f}, {ae_z:.6f}], '
        jsx_content += f'rot: [{ae_rx:.6f}, {ae_ry:.6f}, {ae_rz:.6f}] }}'

        if frame_idx < num_frames - 1:
            jsx_content += ',\n'
        else:
            jsx_content += '\n'

    # Complete the JSX script
    jsx_content += f'''        ];

        // Apply keyframes
        var position = camera.property("ADBE Transform Group").property("ADBE Position");
        var xRotation = camera.property("ADBE Transform Group").property("ADBE Rotate X");
        var yRotation = camera.property("ADBE Transform Group").property("ADBE Rotate Y");
        var zRotation = camera.property("ADBE Transform Group").property("ADBE Rotate Z");

        // Enable keyframing
        position.setValuesAtTimes([], []);
        xRotation.setValuesAtTimes([], []);
        yRotation.setValuesAtTimes([], []);
        zRotation.setValuesAtTimes([], []);

        // Add keyframes for each frame
        for (var i = 0; i < cameraData.length; i++) {{
            var frameNum = startFrame + i;
            var timeInSeconds = frameNum / fps;

            var data = cameraData[i];

            // Set position keyframe
            position.setValueAtTime(timeInSeconds, data.pos);

            // Set rotation keyframes
            xRotation.setValueAtTime(timeInSeconds, data.rot[0]);
            yRotation.setValueAtTime(timeInSeconds, data.rot[1]);
            zRotation.setValueAtTime(timeInSeconds, data.rot[2]);
        }}

        // Select the camera
        camera.selected = true;

        // Summary
        var endFrame = startFrame + numFrames - 1;
        alert(
            "Camera Import Complete!\\n\\n" +
            "Camera: " + cameraName + "\\n" +
            "Frames: " + startFrame + "-" + endFrame + " (" + numFrames + " total)\\n" +
            "FPS: " + fps + "\\n" +
            "Focal Length: " + focalLengthMm.toFixed(2) + "mm\\n" +
            "Resolution: " + width + "x" + height
        );

    }} catch (e) {{
        alert("Error importing camera:\\n" + e.toString());
    }} finally {{
        app.endUndoGroup();
    }}
}})();
'''

    # Write the JSX file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(jsx_content)

    print(f"Exported {num_frames} frames to {output_path}")
    print(f"  Frame range: {start_frame}-{start_frame + num_frames - 1}")
    print(f"  Focal length: {focal_length_mm:.2f}mm")
    print(f"  Import in After Effects: File → Scripts → Run Script File")


def main():
    parser = argparse.ArgumentParser(
        description="Export camera data to various VFX formats (supports DA3 and COLMAP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing camera/ subfolder"
    )
    parser.add_argument(
        "--start-frame", "-s",
        type=int,
        default=1,
        help="Starting frame number (default: 1)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=24.0,
        help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=None,
        help="Image width in pixels (auto-detected from intrinsics or source frames)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=None,
        help="Image height in pixels (auto-detected from intrinsics or source frames)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file base path (default: <project_dir>/camera/camera)"
    )
    parser.add_argument(
        "--format",
        choices=["chan", "csv", "clip", "cmd", "json", "abc", "usd", "jsx", "all"],
        default="all",
        help="Output format: chan (Nuke), csv, clip (Houdini), json, abc (Alembic), usd (USD), jsx (After Effects), all (default: all)"
    )
    parser.add_argument(
        "--rotation-order", "-r",
        choices=["xyz", "zxy", "zyx"],
        default=None,
        help="Euler rotation order: xyz (Maya/Houdini default), zxy (Nuke default), zyx. "
             "If not specified, uses format-appropriate default (zxy for Nuke, xyz for others)"
    )

    args = parser.parse_args()

    # Validate project directory
    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    camera_dir = project_dir / "camera"
    if not camera_dir.exists():
        print(f"Error: Camera directory not found: {camera_dir}", file=sys.stderr)
        sys.exit(1)

    # Load camera data
    try:
        extrinsics, intrinsics, source = load_camera_data(camera_dir)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading camera data: {e}", file=sys.stderr)
        sys.exit(1)

    if not extrinsics:
        print("Error: No camera extrinsics found", file=sys.stderr)
        sys.exit(1)

    source_name = "COLMAP (SfM)" if source == "colmap" else "Video Depth Anything"
    camera_name = "colmap_camera" if source == "colmap" else "vda_camera"
    print(f"Loaded {len(extrinsics)} camera frames from {source_name}")
    print(f"  Coordinate system: OpenGL (Y-up, Z-back) for DCC compatibility")

    image_width = args.width
    image_height = args.height

    if image_width is None or image_height is None:
        if "width" in intrinsics and "height" in intrinsics:
            image_width = image_width or int(intrinsics["width"])
            image_height = image_height or int(intrinsics["height"])
        else:
            source_frames_dir = project_dir / "source" / "frames"
            frames = sorted(source_frames_dir.glob("*.png"))
            if not frames:
                frames = sorted(source_frames_dir.glob("*.jpg"))
            if frames:
                from pipeline_utils import get_image_dimensions
                w, h = get_image_dimensions(frames[0])
                if w > 0 and h > 0:
                    image_width = image_width or w
                    image_height = image_height or h

        if image_width is None or image_height is None:
            image_width = image_width or 1920
            image_height = image_height or 1080
            print(f"  Warning: Could not detect resolution, using default {image_width}x{image_height}")
        else:
            print(f"  Detected resolution: {image_width}x{image_height}")

    # Determine output path
    output_base = args.output or (camera_dir / "camera")

    # Track what we exported
    exported = []

    # Export based on format
    fmt = args.format

    # Nuke .chan format (always available)
    if fmt in ("chan", "all"):
        chan_path = output_base.with_suffix(".chan")
        nuke_rot_order = args.rotation_order if args.rotation_order else "zxy"
        export_nuke_chan(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            output_path=chan_path,
            start_frame=args.start_frame,
            rotation_order=nuke_rot_order,
        )
        exported.append(f".chan (Nuke)")

    # CSV format (always available)
    if fmt in ("csv", "all"):
        csv_path = output_base.with_suffix(".csv")
        csv_rot_order = args.rotation_order if args.rotation_order else "xyz"
        export_csv(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            output_path=csv_path,
            start_frame=args.start_frame,
            rotation_order=csv_rot_order,
        )
        exported.append(f".csv")

    # Houdini .clip format (always available)
    if fmt in ("clip", "all"):
        clip_path = output_base.with_suffix(".clip")
        export_houdini_clip(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            output_path=clip_path,
            start_frame=args.start_frame,
            fps=args.fps,
        )
        exported.append(f".clip (Houdini)")

    # JSON format (always available)
    if fmt in ("json", "all"):
        json_path = output_base.with_suffix(".camera.json")
        export_json_camera(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            output_path=json_path,
            start_frame=args.start_frame,
            fps=args.fps
        )
        exported.append(f".camera.json")

    # Houdini .cmd file (always generate - runs embedded Python to create camera)
    if fmt in ("cmd", "all"):
        cmd_path = output_base.parent / "camera.cmd"
        export_houdini_cmd(
            camera_dir=output_base.parent,
            output_path=cmd_path,
            start_frame=args.start_frame,
            fps=args.fps,
        )
        exported.append(f"camera.cmd (Houdini)")

    # Alembic format (requires Blender)
    if fmt in ("abc", "all"):
        abc_path = output_base.with_suffix(".abc")
        if HAS_BLENDER:
            try:
                export_camera_to_alembic(
                    camera_dir=camera_dir,
                    output_path=abc_path,
                    fps=args.fps,
                    start_frame=args.start_frame,
                    camera_name=camera_name,
                )
                exported.append(f".abc (Alembic)")
            except Exception as e:
                print(f"Error exporting Alembic: {e}", file=sys.stderr)
                if fmt == "abc":
                    sys.exit(1)
        else:
            if fmt == "abc":
                print("Error: Alembic requires Blender. Run the installation wizard.", file=sys.stderr)
                sys.exit(1)
            else:
                print("Note: Alembic not available (requires Blender), skipping .abc")

    # USD format (requires Blender)
    if fmt in ("usd", "all"):
        usd_path = output_base.with_suffix(".usd")
        if HAS_BLENDER:
            try:
                export_camera_to_usd(
                    camera_dir=camera_dir,
                    output_path=usd_path,
                    fps=args.fps,
                    start_frame=args.start_frame,
                    camera_name=camera_name,
                )
                exported.append(f".usd (USD)")
            except Exception as e:
                print(f"Error exporting USD: {e}", file=sys.stderr)
                if fmt == "usd":
                    sys.exit(1)
        else:
            if fmt == "usd":
                print("Error: USD export requires Blender. Run the installation wizard.", file=sys.stderr)
                sys.exit(1)
            else:
                print("Note: USD not available (requires Blender), skipping .usd")

    # After Effects JSX format (always available)
    if fmt in ("jsx", "all"):
        jsx_path = output_base.with_suffix(".jsx")
        export_after_effects_jsx(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            output_path=jsx_path,
            start_frame=args.start_frame,
            fps=args.fps,
            camera_name=camera_name
        )
        exported.append(f".jsx (After Effects)")

    # Summary
    if exported:
        print(f"\nExported formats: {', '.join(exported)}")


if __name__ == "__main__":
    main()
