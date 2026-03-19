#!/usr/bin/env python3
"""Export face mesh detection results to DCC-ready formats.

Reads the npz outputs from run_face_mesh.py and produces:
  - face_mesh/export/face_mesh.json     — blendshape curves + head pose (always)
  - face_mesh/preview/                  — per-frame visualization overlays (always)
  - face_mesh/export/landmarks_cam.npz  — camera-space 3D landmarks (if mmcam available)
  - face_mesh/export/face_mesh.abc      — animated point cloud (if Blender available)
  - face_mesh/export/face_mesh.usd      — animated point cloud (if Blender available)

Camera-space lift (Phase 3a):
  If camera/intrinsics.json exists, the 2D pixel landmarks are unprojected to
  camera-space XY using fx/fy/cx/cy. The MediaPipe z (relative depth in image-width
  units) is preserved as Z — it is proportional to true depth but not metric.

Environment:
    Requires 'vfx-pipeline' conda environment (cv2, numpy).

Usage:
    conda run -p <prefix> python export_face_mesh.py <project_dir> [options]

Example:
    conda run -p <prefix> python export_face_mesh.py /path/to/project --fps 24
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from env_config import require_conda_env, CONDA_ENV_PREFIX
from pipeline_constants import START_FRAME

REQUIRED_ENV = CONDA_ENV_PREFIX

HAS_BLENDER = False
try:
    from blender import export_face_mesh_to_alembic, export_face_mesh_to_usd, check_blender_available
    HAS_BLENDER = True
except ImportError:
    pass

# Iris and eye landmarks used for visualization
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
NOSE_TIP = 4

# Number of top blendshapes shown in the preview overlay
_PREVIEW_TOP_N = 5
# Dot radius for regular landmarks vs. iris center
_LM_RADIUS = 1
_IRIS_RADIUS = 6


def load_face_mesh_data(face_mesh_dir: Path) -> dict:
    """Load all npz outputs from run_face_mesh.py.

    Args:
        face_mesh_dir: Path to face_mesh/ project subdirectory.

    Returns:
        Dict with keys: landmarks, blendshapes, head_pose, eye_gaze, detected, meta.
    """
    with open(face_mesh_dir / "meta.json") as f:
        meta = json.load(f)

    lm_data = np.load(face_mesh_dir / "landmarks.npz")
    bs_data = np.load(face_mesh_dir / "blendshapes.npz")
    hp_data = np.load(face_mesh_dir / "head_pose.npz")
    eg_data = np.load(face_mesh_dir / "eye_gaze.npz")

    return {
        "landmarks": lm_data["landmarks"],
        "blendshapes": bs_data["blendshapes"],
        "head_pose": hp_data["head_pose"],
        "eye_gaze": eg_data["eye_gaze"],
        "detected": lm_data["detected"],
        "meta": meta,
    }


def export_json(data: dict, export_dir: Path, start_frame: int) -> Path:
    """Export blendshape curves and head pose per frame to JSON.

    The JSON is structured for animator-friendly consumption:
    - blendshape_names once at the top
    - per-frame list with frame number, detection flag, blendshape array, 4x4 matrix

    Args:
        data: Dict from load_face_mesh_data.
        export_dir: Output directory.
        start_frame: First frame number (for labeling).

    Returns:
        Path to the written JSON file.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / "face_mesh.json"

    blendshapes = data["blendshapes"]
    head_pose = data["head_pose"]
    eye_gaze = data["eye_gaze"]
    detected = data["detected"]
    meta = data["meta"]
    n_frames = meta["n_frames"]

    frames = []
    for i in range(n_frames):
        frames.append({
            "frame": start_frame + i,
            "detected": bool(detected[i]),
            "blendshapes": blendshapes[i].tolist(),
            "head_pose": head_pose[i].tolist(),
            "eye_gaze_left": eye_gaze[i, 0].tolist(),
            "eye_gaze_right": eye_gaze[i, 1].tolist(),
        })

    doc = {
        "fps": meta["fps"],
        "start_frame": start_frame,
        "n_frames": n_frames,
        "detected_count": meta["detected_count"],
        "blendshape_names": meta.get("blendshape_names", []),
        "frames": frames,
    }

    output_path.write_text(json.dumps(doc, indent=2))
    print(f"  JSON: {output_path}")
    return output_path


def lift_to_camera_space(
    landmarks_px: np.ndarray,
    intrinsics: dict,
    image_w: int = 0,
    image_h: int = 0,
) -> np.ndarray:
    """Unproject 2D pixel landmarks to camera-space XY using pinhole intrinsics.

    The MediaPipe z (lm.z * image_width) is a relative depth value — proportional
    to true depth but not metric. It is used as-is for Z, making X and Y
    camera-relative in the same unit system as Z.

    Args:
        landmarks_px: (n_frames, 478, 3) array in pixel space.
        intrinsics: Dict with fx, fy, cx, cy keys.
        image_w: Image width in pixels (used as cx fallback if intrinsics lacks cx/principal_x).
        image_h: Image height in pixels (used as cy fallback if intrinsics lacks cy/principal_y).

    Returns:
        (n_frames, 478, 3) array in camera space (X, Y in image-width units).
    """
    fx = float(intrinsics.get("fx", intrinsics.get("focal_x", 1000.0)))
    fy = float(intrinsics.get("fy", intrinsics.get("focal_y", fx)))
    cx = float(intrinsics.get("cx", intrinsics.get("principal_x", image_w / 2)))
    cy = float(intrinsics.get("cy", intrinsics.get("principal_y", image_h / 2)))

    out = np.zeros_like(landmarks_px)
    px = landmarks_px[..., 0]
    py = landmarks_px[..., 1]
    z = landmarks_px[..., 2]

    out[..., 0] = (px - cx) * z / fx
    out[..., 1] = (py - cy) * z / fy
    out[..., 2] = z

    return out


def export_camera_space_landmarks(
    data: dict,
    project_dir: Path,
    export_dir: Path,
) -> bool:
    """Lift landmarks to camera space if matchmove intrinsics are available.

    Args:
        data: Dict from load_face_mesh_data.
        project_dir: Project root directory.
        export_dir: Output directory.

    Returns:
        True if camera-space export was written.
    """
    intrinsics_path = project_dir / "camera" / "intrinsics.json"
    if not intrinsics_path.exists():
        return False

    with open(intrinsics_path) as f:
        intrinsics = json.load(f)

    meta = data["meta"]
    landmarks_cam = lift_to_camera_space(
        data["landmarks"], intrinsics,
        image_w=meta.get("width", 0),
        image_h=meta.get("height", 0),
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "landmarks_cam.npz"
    np.savez_compressed(out_path, landmarks_cam=landmarks_cam, detected=data["detected"])
    print(f"  Camera-space landmarks: {out_path}")
    return True


def _draw_frame_overlay(
    source_frame: np.ndarray,
    landmarks_px: np.ndarray,
    blendshapes: np.ndarray,
    blendshape_names: list[str],
) -> np.ndarray:
    """Draw landmarks and blendshape overlay on a source frame.

    Args:
        source_frame: BGR source image.
        landmarks_px: (478, 3) landmark array for this frame.
        blendshapes: (52,) blendshape weights for this frame.
        blendshape_names: List of blendshape name strings.

    Returns:
        Annotated BGR image.
    """
    out = source_frame.copy()

    for j in range(len(landmarks_px)):
        px = int(landmarks_px[j, 0])
        py = int(landmarks_px[j, 1])
        cv2.circle(out, (px, py), _LM_RADIUS, (0, 220, 0), -1)

    for iris_idx, color in [(LEFT_IRIS_CENTER, (255, 140, 0)), (RIGHT_IRIS_CENTER, (0, 140, 255))]:
        px = int(landmarks_px[iris_idx, 0])
        py = int(landmarks_px[iris_idx, 1])
        cv2.circle(out, (px, py), _IRIS_RADIUS, color, 2)

    if len(blendshape_names) > 0:
        sorted_indices = np.argsort(blendshapes)[::-1]
        y = 24
        for rank in range(min(_PREVIEW_TOP_N, len(sorted_indices))):
            idx = sorted_indices[rank]
            score = blendshapes[idx]
            if score < 0.05:
                break
            name = blendshape_names[idx] if idx < len(blendshape_names) else str(idx)
            label = f"{name}: {score:.2f}"
            cv2.putText(out, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

    return out


def generate_preview(
    data: dict,
    project_dir: Path,
    preview_dir: Path,
    start_frame: int,
) -> None:
    """Write per-frame visualization overlays to preview_dir.

    Skips frames where no face was detected (copies source frame unmodified).

    Args:
        data: Dict from load_face_mesh_data.
        project_dir: Project root directory.
        preview_dir: Output directory for overlay PNGs.
        start_frame: First frame number (for input filename lookup).
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = project_dir / "source" / "frames"
    frame_paths = sorted(frames_dir.glob("*.png"))

    landmarks = data["landmarks"]
    blendshapes = data["blendshapes"]
    detected = data["detected"]
    blendshape_names = data["meta"].get("blendshape_names", [])
    n_frames = data["meta"]["n_frames"]

    print(f"  Generating {n_frames} preview frames...")

    for i, frame_path in enumerate(frame_paths[:n_frames]):
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue

        if detected[i]:
            out = _draw_frame_overlay(bgr, landmarks[i], blendshapes[i], blendshape_names)
        else:
            out = bgr

        out_path = preview_dir / f"face_mesh_{start_frame + i:06d}.png"
        cv2.imwrite(str(out_path), out)

    print(f"  Preview frames: {preview_dir}/")


def export_blender_formats(
    face_mesh_dir: Path,
    export_dir: Path,
    fps: int,
    start_frame: int,
) -> None:
    """Export face mesh to Alembic and USD via Blender headless.

    Skips silently if Blender is not available.

    Args:
        face_mesh_dir: face_mesh/ project directory.
        export_dir: Output directory.
        fps: Frames per second.
        start_frame: First frame number.
    """
    if not HAS_BLENDER:
        return

    available, msg = check_blender_available()
    if not available:
        print(f"  Blender not available ({msg}) — skipping abc/usd export")
        return

    export_dir.mkdir(parents=True, exist_ok=True)

    abc_path = export_dir / "face_mesh.abc"
    print("  Exporting face mesh Alembic...")
    try:
        export_face_mesh_to_alembic(
            face_mesh_dir=face_mesh_dir,
            output_path=abc_path,
            fps=fps,
            start_frame=start_frame,
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"  Alembic export failed: {e}", file=sys.stderr)

    usd_path = export_dir / "face_mesh.usd"
    print("  Exporting face mesh USD...")
    try:
        export_face_mesh_to_usd(
            face_mesh_dir=face_mesh_dir,
            output_path=usd_path,
            fps=fps,
            start_frame=start_frame,
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"  USD export failed: {e}", file=sys.stderr)


def export_face_mesh(project_dir: Path, fps: int, start_frame: int) -> None:
    """Run all face mesh export steps.

    Args:
        project_dir: Project root directory.
        fps: Frames per second for Alembic/USD timeline.
        start_frame: First frame number.
    """
    require_conda_env(REQUIRED_ENV)

    face_mesh_dir = project_dir / "face_mesh"
    if not (face_mesh_dir / "meta.json").exists():
        print("Error: face_mesh/meta.json not found — run face_mesh stage first.", file=sys.stderr)
        sys.exit(1)

    data = load_face_mesh_data(face_mesh_dir)
    export_dir = face_mesh_dir / "export"
    preview_dir = face_mesh_dir / "preview"

    export_json(data, export_dir, start_frame)
    export_camera_space_landmarks(data, project_dir, export_dir)
    generate_preview(data, project_dir, preview_dir, start_frame)
    export_blender_formats(face_mesh_dir, export_dir, fps, start_frame)

    print(f"\n  Export complete: {export_dir}/")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Export face mesh detection results to DCC-ready formats"
    )
    parser.add_argument("project_dir", type=Path, help="Project directory")
    parser.add_argument("--fps", type=int, default=24, help="FPS for Alembic/USD timeline")
    parser.add_argument(
        "--start-frame", type=int, default=START_FRAME,
        help=f"First frame number (default: {START_FRAME})"
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    export_face_mesh(project_dir, fps=args.fps, start_frame=args.start_frame)


if __name__ == "__main__":
    main()
