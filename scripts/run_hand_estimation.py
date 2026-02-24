#!/usr/bin/env python3
"""Hand pose estimation using WiLoR-mini.

Estimates per-frame 3D hand poses from the same video that GVHMR
processed, producing axis-angle rotations compatible with SMPLX
left_hand_pose / right_hand_pose parameters.

Environment:
    Requires 'gvhmr' conda environment (has WiLoR-mini, torch, cv2).

Usage:
    conda run -n gvhmr python run_hand_estimation.py <project_dir> [options]

Example:
    conda run -n gvhmr python run_hand_estimation.py /path/to/project --mocap-person person
"""

import argparse
import sys
from pathlib import Path

from env_config import require_conda_env

REQUIRED_ENV = "gvhmr"
SMPLX_HAND_JOINTS = 15
HAND_POSE_DIM = SMPLX_HAND_JOINTS * 3


def find_gvhmr_input_video(gvhmr_dir: Path, project_dir: Path) -> Path:
    """Locate the video that GVHMR processed.

    GVHMR creates output at gvhmr_dir/<video_stem>/hmr4d_results.pt.
    The video lives at source/<video_stem>.mp4.

    Args:
        gvhmr_dir: GVHMR output directory (mocap/<person>/gvhmr/)
        project_dir: Project root directory

    Returns:
        Path to the input video
    """
    hmr4d_files = list(gvhmr_dir.rglob("hmr4d*.pt"))
    if not hmr4d_files:
        print(f"Error: No GVHMR output in {gvhmr_dir}", file=sys.stderr)
        sys.exit(1)

    relative = hmr4d_files[0].relative_to(gvhmr_dir)
    if len(relative.parts) < 2:
        print(
            f"Error: Unexpected GVHMR output structure — "
            f"hmr4d file directly in {gvhmr_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    video_stem = relative.parts[0]
    video_path = project_dir / "source" / f"{video_stem}.mp4"

    if not video_path.exists():
        print(f"Error: GVHMR input video not found: {video_path}", file=sys.stderr)
        print(
            f"  (Derived stem '{video_stem}' from GVHMR output structure)",
            file=sys.stderr,
        )
        sys.exit(1)

    return video_path


def load_gvhmr_frame_count(gvhmr_dir: Path) -> int:
    """Load GVHMR output and return the number of frames.

    Args:
        gvhmr_dir: Directory containing GVHMR output (hmr4d*.pt)

    Returns:
        Number of frames in GVHMR output
    """
    import numpy as np
    import torch

    gvhmr_files = list(gvhmr_dir.rglob("hmr4d*.pt"))
    if not gvhmr_files:
        print(f"Error: No GVHMR output found in {gvhmr_dir}", file=sys.stderr)
        sys.exit(1)

    data = torch.load(gvhmr_files[0], map_location="cpu", weights_only=False)

    if "smpl_params_global" in data:
        body_pose = data["smpl_params_global"].get("body_pose")
    else:
        body_pose = data.get("body_pose", data.get("poses"))

    if body_pose is None:
        print("Error: Cannot determine frame count from GVHMR output", file=sys.stderr)
        sys.exit(1)

    if hasattr(body_pose, "numpy"):
        body_pose = body_pose.numpy()

    return len(np.asarray(body_pose))


def extract_hand_pose(wilor_pred: dict) -> "np.ndarray":
    """Extract SMPLX-compatible hand pose from a WiLoR prediction.

    WiLoR outputs axis-angle rotations for hand joints. We take the
    first 45 values (15 joints x 3) for SMPLX compatibility.

    Args:
        wilor_pred: Single hand prediction dict from WiLoR pipeline

    Returns:
        (45,) float32 axis-angle array
    """
    import numpy as np

    raw = np.asarray(wilor_pred["wilor_preds"]["hand_pose"], dtype=np.float32).ravel()
    return raw[:HAND_POSE_DIM]


def bbox_area(pred: dict) -> float:
    """Compute bounding box area for a WiLoR prediction.

    Args:
        pred: Single hand prediction dict with 'hand_bbox' key

    Returns:
        Bounding box area in pixels
    """
    bbox = pred["hand_bbox"]
    return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def smooth_hand_poses(
    poses: "np.ndarray", detected: "np.ndarray", window: int = 7, poly_order: int = 2
) -> "np.ndarray":
    """Fill detection gaps and smooth hand pose trajectories.

    Interior gaps are linearly interpolated between flanking valid frames.
    Leading/trailing gaps hold the nearest valid pose. Then Savitzky-Golay
    smoothing reduces frame-to-frame jitter.

    Args:
        poses: (N, 45) axis-angle hand poses, zeros where undetected
        detected: (N,) boolean detection mask
        window: Savitzky-Golay window length (must be odd)
        poly_order: Savitzky-Golay polynomial order

    Returns:
        (N, 45) smoothed poses
    """
    import numpy as np
    from scipy.signal import savgol_filter

    n_frames = len(poses)
    valid_idx = np.where(detected)[0]

    if len(valid_idx) < 2:
        return poses.copy()

    smoothed = poses.copy()
    xp = valid_idx.astype(float)
    x_all = np.arange(n_frames, dtype=float)

    for dim in range(poses.shape[1]):
        fp = poses[valid_idx, dim]
        smoothed[:, dim] = np.interp(x_all, xp, fp)

    if n_frames >= window:
        for dim in range(smoothed.shape[1]):
            smoothed[:, dim] = savgol_filter(smoothed[:, dim], window, poly_order)

    return smoothed


def run_hand_estimation(
    project_dir: Path, mocap_person: str = "person", smooth: bool = True
) -> bool:
    """Estimate hand poses for all frames and save to hand_poses.npz.

    Args:
        project_dir: Project directory
        mocap_person: Person folder name under mocap/
        smooth: Apply gap filling and temporal smoothing

    Returns:
        True if successful
    """
    import cv2
    import numpy as np
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    mocap_person_dir = project_dir / "mocap" / mocap_person
    gvhmr_dir = mocap_person_dir / "gvhmr"

    video_path = find_gvhmr_input_video(gvhmr_dir, project_dir)
    n_gvhmr_frames = load_gvhmr_frame_count(gvhmr_dir)

    cap = cv2.VideoCapture(str(video_path))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_video_frames != n_gvhmr_frames:
        print(
            f"Error: Frame count mismatch — video has {n_video_frames}, "
            f"GVHMR has {n_gvhmr_frames}",
            file=sys.stderr,
        )
        cap.release()
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"  Device: {device} ({dtype})")

    pipeline = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)

    left_hand_poses = np.zeros((n_gvhmr_frames, HAND_POSE_DIM), dtype=np.float32)
    right_hand_poses = np.zeros((n_gvhmr_frames, HAND_POSE_DIM), dtype=np.float32)

    print(f"  Processing {n_gvhmr_frames} frames...")

    for i in range(n_gvhmr_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Failed to read frame {i}", file=sys.stderr)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = pipeline.predict(frame_rgb)

        if not predictions:
            continue

        left_candidates = [p for p in predictions if p["is_right"] == 0]
        right_candidates = [p for p in predictions if p["is_right"] == 1]

        if left_candidates:
            best = max(left_candidates, key=bbox_area)
            left_hand_poses[i] = extract_hand_pose(best)

        if right_candidates:
            best = max(right_candidates, key=bbox_area)
            right_hand_poses[i] = extract_hand_pose(best)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"    Frame {i + 1}/{n_gvhmr_frames}")

    cap.release()

    left_detected = np.any(left_hand_poses != 0, axis=1)
    right_detected = np.any(right_hand_poses != 0, axis=1)

    if smooth:
        left_hand_poses = smooth_hand_poses(left_hand_poses, left_detected)
        right_hand_poses = smooth_hand_poses(right_hand_poses, right_detected)
        print("  Smoothing: applied (gap-fill + Savitzky-Golay)")
    else:
        print("  Smoothing: skipped (--no-smooth)")

    output_path = mocap_person_dir / "hand_poses.npz"
    np.savez(
        output_path,
        left_hand_pose=left_hand_poses,
        right_hand_pose=right_hand_poses,
        left_detected=left_detected,
        right_detected=right_detected,
    )

    n_left = int(left_detected.sum())
    n_right = int(right_detected.sum())

    print(f"  OK Saved {output_path}")
    print(f"    Left hand:  {n_left} detected, {n_gvhmr_frames - n_left} filled")
    print(f"    Right hand: {n_right} detected, {n_gvhmr_frames - n_right} filled")

    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Estimate hand poses using WiLoR-mini",
    )
    parser.add_argument("project_dir", type=Path, help="Project directory")
    parser.add_argument(
        "--mocap-person",
        default="person",
        help="Person folder name under mocap/ (default: person)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Skip gap filling and temporal smoothing",
    )

    args = parser.parse_args()

    require_conda_env(REQUIRED_ENV)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Hand Pose Estimation (WiLoR-mini)")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print(f"Person: {args.mocap_person}")
    print()

    success = run_hand_estimation(project_dir, args.mocap_person, smooth=not args.no_smooth)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
