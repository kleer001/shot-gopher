#!/usr/bin/env python3
"""Run MediaPipe Face Mesh on source frames.

Uses the MediaPipe FaceLandmarker Tasks API to detect per-frame:
  - 478 facial landmarks (468 face + 10 iris)
  - 52 ARKit-compatible blendshape coefficients
  - Head pose (4x4 face-to-camera transform matrix)
  - Eye gaze vectors (iris displacement from eye axis, 2D proxy)

Outputs:
    face_mesh/landmarks.npz    float32 (n_frames, 478, 3) — pixel x,y + relative z
    face_mesh/blendshapes.npz  float32 (n_frames, 52)     — ARKit blendshape weights
    face_mesh/head_pose.npz    float32 (n_frames, 4, 4)   — face-to-camera transform
    face_mesh/eye_gaze.npz     float32 (n_frames, 2, 3)   — [left, right] gaze vectors
    face_mesh/meta.json        dict    — frame count, dims, blendshape names, detection stats

Environment:
    Requires 'vfx-pipeline' conda environment with mediapipe >= 0.10.

Usage:
    conda run -p <prefix> python run_face_mesh.py <project_dir>
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from env_config import require_conda_env, CONDA_ENV_PREFIX, MEDIAPIPE_MODELS_DIR

REQUIRED_ENV = CONDA_ENV_PREFIX
FACE_LANDMARKER_MODEL = MEDIAPIPE_MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

N_LANDMARKS = 478
N_BLENDSHAPES = 52

# Iris landmark indices (added when refine_landmarks=True)
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Eye corner indices used as gaze reference axis
RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263


def ensure_model() -> Path:
    """Download FaceLandmarker model if not present.

    Returns:
        Path to the model file.
    """
    MEDIAPIPE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_LANDMARKER_MODEL.exists():
        print(f"  Downloading FaceLandmarker model to {FACE_LANDMARKER_MODEL}...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
        print("  Download complete.")
    return FACE_LANDMARKER_MODEL


def _compute_eye_gaze(landmarks_px: np.ndarray) -> np.ndarray:
    """Compute eye gaze vectors from iris displacement relative to eye axis.

    Returns a (2, 3) array of unit vectors [left, right]. Z is always 0 —
    this is a 2D image-plane proxy, not a metric 3D gaze direction.

    Args:
        landmarks_px: (478, 3) landmark array for a single frame in pixel coords.

    Returns:
        (2, 3) float32 unit vectors.
    """
    result = np.zeros((2, 3), dtype=np.float32)

    for side_idx, (inner, outer, iris_center) in enumerate([
        (LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_IRIS_CENTER),
        (RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS_CENTER),
    ]):
        midpoint = (landmarks_px[inner, :2] + landmarks_px[outer, :2]) / 2.0
        iris = landmarks_px[iris_center, :2]
        disp = iris - midpoint
        norm = np.linalg.norm(disp)
        if norm > 1e-6:
            result[side_idx, :2] = disp / norm

    return result


def _load_fps(project_dir: Path) -> float:
    """Load FPS from project metadata, falling back to 24.

    Args:
        project_dir: Project root directory.

    Returns:
        FPS as float.
    """
    meta_path = project_dir / ".metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return float(json.load(f).get("fps", 24))
    return 24.0


def run_face_mesh(project_dir: Path) -> None:
    """Detect facial landmarks, blendshapes, head pose, and eye gaze.

    Writes results to face_mesh/ inside the project directory.

    Args:
        project_dir: Project root directory containing source/frames/.
    """
    require_conda_env(REQUIRED_ENV)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("Error: mediapipe not installed. Run the installation wizard.", file=sys.stderr)
        sys.exit(1)

    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        print(f"Error: frames directory not found: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        print(f"Error: no PNG frames found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    model_path = ensure_model()
    output_dir = project_dir / "face_mesh"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(frame_paths)
    first_bgr = cv2.imread(str(frame_paths[0]))
    if first_bgr is None:
        print(f"Error: could not read {frame_paths[0]}", file=sys.stderr)
        sys.exit(1)

    h, w = first_bgr.shape[:2]
    fps = _load_fps(project_dir)

    landmarks_px = np.zeros((n_frames, N_LANDMARKS, 3), dtype=np.float32)
    blendshapes = np.zeros((n_frames, N_BLENDSHAPES), dtype=np.float32)
    head_pose = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    eye_gaze = np.zeros((n_frames, 2, 3), dtype=np.float32)
    detected = np.zeros(n_frames, dtype=bool)
    blendshape_names: list[str] = []

    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_score=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    print(f"  Processing {n_frames} frames ({w}x{h}) at {fps} fps...")

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, frame_path in enumerate(frame_paths):
            bgr = cv2.imread(str(frame_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(i * 1000.0 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_landmarks:
                if (i + 1) % 50 == 0 or (i + 1) == n_frames:
                    print(f"  {i + 1}/{n_frames} frames")
                continue

            detected[i] = True

            for j, lm in enumerate(result.face_landmarks[0]):
                landmarks_px[i, j] = [lm.x * w, lm.y * h, lm.z * w]

            if result.face_blendshapes:
                shapes = result.face_blendshapes[0]
                if not blendshape_names:
                    blendshape_names = [c.category_name for c in shapes]
                for j, c in enumerate(shapes):
                    blendshapes[i, j] = c.score

            if result.facial_transformation_matrixes:
                mat = result.facial_transformation_matrixes[0]
                head_pose[i] = np.asarray(mat, dtype=np.float32)

            eye_gaze[i] = _compute_eye_gaze(landmarks_px[i])

            if (i + 1) % 50 == 0 or (i + 1) == n_frames:
                print(f"  {i + 1}/{n_frames} frames")

    np.savez_compressed(output_dir / "landmarks.npz", landmarks=landmarks_px, detected=detected)
    np.savez_compressed(output_dir / "blendshapes.npz", blendshapes=blendshapes, detected=detected)
    np.savez_compressed(output_dir / "head_pose.npz", head_pose=head_pose, detected=detected)
    np.savez_compressed(output_dir / "eye_gaze.npz", eye_gaze=eye_gaze, detected=detected)

    meta = {
        "n_frames": n_frames,
        "n_landmarks": N_LANDMARKS,
        "n_blendshapes": N_BLENDSHAPES,
        "width": w,
        "height": h,
        "fps": fps,
        "detected_count": int(detected.sum()),
        "coordinate_space": "pixel",
        "landmark_description": "468 face + 10 iris (MediaPipe FaceLandmarker)",
        "blendshape_names": blendshape_names,
        "eye_gaze_description": "iris displacement unit vector in image plane (2D proxy)",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Detected faces in {detected.sum()}/{n_frames} frames")
    print(f"  Saved: {output_dir}/{{landmarks,blendshapes,head_pose,eye_gaze}}.npz")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Face Mesh on project source frames"
    )
    parser.add_argument("project_dir", type=Path, help="Project directory")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    run_face_mesh(project_dir)


if __name__ == "__main__":
    main()
