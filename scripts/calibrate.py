#!/usr/bin/env python3
"""Camera trajectory alignment between matchmove (COLMAP) and mocap (GVHMR).

Compares two camera trajectories and computes the rigid+scale transform
(Umeyama alignment) that maps one coordinate space to the other. This
allows the mocap animated mesh to be repositioned into the matchmove
camera's world space, so both align with the original plate.

Workflow:
    1. Run mmcam stage → produces camera/extrinsics.json (COLMAP)
    2. Run mocap stage → produces mocap/camera/extrinsics.json (GVHMR)
    3. Run calibrate → compares trajectories, computes alignment transform
    4. Apply transform to mesh or re-export with corrected positioning

Usage:
    python calibrate.py <project_dir> [options]
    python calibrate.py <project_dir> --apply   # write aligned extrinsics

Example:
    python calibrate.py /path/to/projects/My_Shot
    python calibrate.py /path/to/projects/My_Shot --apply --output mocap/camera_aligned/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from transforms import (
    compose_matrix,
    decompose_matrix,
    rotation_matrix_to_quaternion,
    slerp,
)


def load_trajectory(camera_dir: Path) -> tuple[list[np.ndarray], dict]:
    """Load camera trajectory from pipeline format.

    Args:
        camera_dir: Directory containing extrinsics.json and intrinsics.json

    Returns:
        Tuple of (list of 4x4 camera-to-world matrices, intrinsics dict)

    Raises:
        FileNotFoundError: If required files are missing
    """
    extrinsics_path = camera_dir / "extrinsics.json"
    intrinsics_path = camera_dir / "intrinsics.json"

    if not extrinsics_path.exists():
        raise FileNotFoundError(f"No extrinsics.json in {camera_dir}")

    with open(extrinsics_path, encoding="utf-8") as f:
        extrinsics_data = json.load(f)

    matrices = []
    for matrix_data in extrinsics_data:
        matrices.append(np.array(matrix_data, dtype=np.float64).reshape(4, 4))

    intrinsics: dict = {}
    if intrinsics_path.exists():
        with open(intrinsics_path, encoding="utf-8") as f:
            intrinsics = json.load(f)

    return matrices, intrinsics


def extract_positions(matrices: list[np.ndarray]) -> np.ndarray:
    """Extract camera positions from 4x4 camera-to-world matrices.

    Args:
        matrices: List of 4x4 camera-to-world matrices

    Returns:
        (N, 3) array of camera world positions
    """
    return np.array([m[:3, 3] for m in matrices])


def extract_rotations(matrices: list[np.ndarray]) -> list[np.ndarray]:
    """Extract 3x3 rotation matrices from 4x4 camera-to-world matrices.

    Args:
        matrices: List of 4x4 camera-to-world matrices

    Returns:
        List of 3x3 rotation matrices
    """
    return [m[:3, :3].copy() for m in matrices]


def umeyama_alignment(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute optimal rigid+scale alignment (Umeyama method).

    Finds rotation R, translation t, and scale s that minimizes:
        sum || target_i - (s * R @ source_i + t) ||^2

    Args:
        source: (N, 3) source points (e.g. GVHMR camera positions)
        target: (N, 3) target points (e.g. COLMAP camera positions)

    Returns:
        Tuple of (R 3x3 rotation, t 3-vector translation, s scalar scale)

    Raises:
        ValueError: If inputs have different lengths or fewer than 3 points
    """
    if source.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: source {source.shape} vs target {target.shape}"
        )
    n = source.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 points, got {n}")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)

    source_centered = source - mu_source
    target_centered = target - mu_target

    var_source = np.sum(source_centered ** 2) / n

    # Time: O(n*d^2) for covariance, O(d^3) for SVD — trivial for d=3
    cov = (target_centered.T @ source_centered) / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_source if var_source > 1e-10 else 1.0
    t = mu_target - s * R @ mu_source

    return R, t, s


def apply_transform_to_trajectory(
    matrices: list[np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> list[np.ndarray]:
    """Apply rigid+scale transform to a camera trajectory.

    Transforms each camera-to-world matrix so the camera positions
    and orientations are mapped into the target coordinate space.

    For camera-to-world matrix M with position p and rotation R_cam:
        p' = s * R @ p + t
        R_cam' = R @ R_cam

    Args:
        matrices: List of 4x4 camera-to-world matrices
        R: 3x3 alignment rotation
        t: 3-vector alignment translation
        s: Scale factor

    Returns:
        List of transformed 4x4 camera-to-world matrices
    """
    aligned = []
    for m in matrices:
        cam_pos = m[:3, 3]
        cam_rot = m[:3, :3]

        new_pos = s * R @ cam_pos + t
        new_rot = R @ cam_rot

        new_m = np.eye(4)
        new_m[:3, :3] = new_rot
        new_m[:3, 3] = new_pos
        aligned.append(new_m)

    return aligned


def rotation_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic distance between two rotation matrices in degrees.

    Args:
        R1: 3x3 rotation matrix
        R2: 3x3 rotation matrix

    Returns:
        Angular distance in degrees
    """
    R_diff = R1.T @ R2
    cos_angle = (np.trace(R_diff) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compare_trajectories(
    source: list[np.ndarray],
    target: list[np.ndarray],
) -> dict:
    """Compare two camera trajectories and report alignment metrics.

    Args:
        source: List of 4x4 camera-to-world matrices (e.g. GVHMR)
        target: List of 4x4 camera-to-world matrices (e.g. COLMAP)

    Returns:
        Dict with comparison metrics:
            n_frames: frame count
            position_errors: per-frame position distance
            rotation_errors: per-frame rotation distance (degrees)
            mean_position_error: mean position error
            max_position_error: max position error
            mean_rotation_error: mean rotation error (degrees)
            max_rotation_error: max rotation error (degrees)
            source_extent: total distance traveled by source camera
            target_extent: total distance traveled by target camera
            scale_ratio: ratio of target extent to source extent
    """
    n = min(len(source), len(target))

    src_pos = extract_positions(source[:n])
    tgt_pos = extract_positions(target[:n])

    position_errors = np.linalg.norm(src_pos - tgt_pos, axis=1)

    rotation_errors = []
    for i in range(n):
        r_src = source[i][:3, :3]
        r_tgt = target[i][:3, :3]
        rotation_errors.append(rotation_geodesic_distance(r_src, r_tgt))
    rotation_errors_arr = np.array(rotation_errors)

    src_diffs = np.diff(src_pos, axis=0)
    tgt_diffs = np.diff(tgt_pos, axis=0)
    source_extent = float(np.sum(np.linalg.norm(src_diffs, axis=1))) if n > 1 else 0.0
    target_extent = float(np.sum(np.linalg.norm(tgt_diffs, axis=1))) if n > 1 else 0.0

    scale_ratio = target_extent / source_extent if source_extent > 1e-10 else float("inf")

    return {
        "n_frames": n,
        "position_errors": position_errors.tolist(),
        "rotation_errors": rotation_errors,
        "mean_position_error": float(np.mean(position_errors)),
        "max_position_error": float(np.max(position_errors)),
        "mean_rotation_error": float(np.mean(rotation_errors_arr)),
        "max_rotation_error": float(np.max(rotation_errors_arr)),
        "source_extent": source_extent,
        "target_extent": target_extent,
        "scale_ratio": scale_ratio,
    }


def print_comparison(metrics: dict, label: str = "") -> None:
    """Print human-readable trajectory comparison.

    Args:
        metrics: Output from compare_trajectories
        label: Optional label for the comparison
    """
    header = f"Camera Trajectory Comparison{f' ({label})' if label else ''}"
    print(f"\n{'=' * 60}")
    print(header)
    print("=" * 60)
    print(f"  Frames compared:      {metrics['n_frames']}")
    print(f"  Position error (mean): {metrics['mean_position_error']:.4f}")
    print(f"  Position error (max):  {metrics['max_position_error']:.4f}")
    print(f"  Rotation error (mean): {metrics['mean_rotation_error']:.2f} deg")
    print(f"  Rotation error (max):  {metrics['max_rotation_error']:.2f} deg")
    print(f"  Source path length:    {metrics['source_extent']:.4f}")
    print(f"  Target path length:    {metrics['target_extent']:.4f}")
    print(f"  Scale ratio (tgt/src): {metrics['scale_ratio']:.4f}")
    print()


def save_trajectory(
    matrices: list[np.ndarray],
    intrinsics: dict,
    output_dir: Path,
    metadata: Optional[dict] = None,
) -> None:
    """Save camera trajectory in pipeline format.

    Args:
        matrices: List of 4x4 camera-to-world matrices
        intrinsics: Camera intrinsics dict
        output_dir: Output directory
        metadata: Optional metadata to save alongside
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    extrinsics_data = [m.tolist() for m in matrices]
    with open(output_dir / "extrinsics.json", "w", encoding="utf-8") as f:
        json.dump(extrinsics_data, f, indent=2)

    with open(output_dir / "intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, indent=2)

    if metadata:
        with open(output_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def save_alignment_transform(
    R: np.ndarray,
    t: np.ndarray,
    s: float,
    output_path: Path,
    pre_metrics: Optional[dict] = None,
    post_metrics: Optional[dict] = None,
) -> None:
    """Save alignment transform to JSON.

    Args:
        R: 3x3 rotation matrix
        t: 3-vector translation
        s: Scale factor
        output_path: Output file path
        pre_metrics: Comparison metrics before alignment
        post_metrics: Comparison metrics after alignment
    """
    data: dict = {
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "scale": s,
    }
    if pre_metrics:
        data["pre_alignment"] = {
            k: v for k, v in pre_metrics.items()
            if k not in ("position_errors", "rotation_errors")
        }
    if post_metrics:
        data["post_alignment"] = {
            k: v for k, v in post_metrics.items()
            if k not in ("position_errors", "rotation_errors")
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_alignment_transform(
    transform_path: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load alignment transform from JSON.

    Args:
        transform_path: Path to calibration transform JSON

    Returns:
        Tuple of (R 3x3 rotation, t 3-vector translation, s scalar scale)
    """
    with open(transform_path, encoding="utf-8") as f:
        data = json.load(f)
    R = np.array(data["rotation"], dtype=np.float64)
    t = np.array(data["translation"], dtype=np.float64)
    s = float(data["scale"])
    return R, t, s


def run_calibration(
    project_dir: Path,
    mmcam_dir: Optional[Path] = None,
    mocap_dir: Optional[Path] = None,
    apply: bool = False,
    output_dir: Optional[Path] = None,
) -> bool:
    """Run camera calibration between matchmove and mocap trajectories.

    Args:
        project_dir: Project directory
        mmcam_dir: Matchmove camera directory (default: project_dir/camera/)
        mocap_dir: Mocap camera directory (default: project_dir/mocap/camera/)
        apply: If True, write aligned trajectory to output_dir
        output_dir: Output directory for aligned camera (default: mocap/camera_aligned/)

    Returns:
        True if calibration succeeded
    """
    mmcam_dir = mmcam_dir or project_dir / "camera"
    mocap_dir = mocap_dir or project_dir / "mocap" / "camera"
    output_dir = output_dir or project_dir / "mocap" / "camera_aligned"

    print(f"\n{'=' * 60}")
    print("Camera Calibration")
    print("=" * 60)
    print(f"  Matchmove: {mmcam_dir}")
    print(f"  Mocap:     {mocap_dir}")

    mmcam_matrices, mmcam_intrinsics = load_trajectory(mmcam_dir)
    mocap_matrices, mocap_intrinsics = load_trajectory(mocap_dir)

    print(f"  Matchmove frames: {len(mmcam_matrices)}")
    print(f"  Mocap frames:     {len(mocap_matrices)}")

    n = min(len(mmcam_matrices), len(mocap_matrices))
    if len(mmcam_matrices) != len(mocap_matrices):
        print(f"  Warning: Frame count mismatch, using first {n} frames")

    mmcam_trimmed = mmcam_matrices[:n]
    mocap_trimmed = mocap_matrices[:n]

    pre_metrics = compare_trajectories(mocap_trimmed, mmcam_trimmed)
    print_comparison(pre_metrics, "before alignment")

    mocap_pos = extract_positions(mocap_trimmed)
    mmcam_pos = extract_positions(mmcam_trimmed)

    R, t, s = umeyama_alignment(mocap_pos, mmcam_pos)

    print(f"Alignment transform (mocap → matchmove):")
    print(f"  Scale:       {s:.6f}")
    print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

    aligned = apply_transform_to_trajectory(mocap_trimmed, R, t, s)

    post_metrics = compare_trajectories(aligned, mmcam_trimmed)
    print_comparison(post_metrics, "after alignment")

    improvement = pre_metrics["mean_position_error"] - post_metrics["mean_position_error"]
    if pre_metrics["mean_position_error"] > 1e-10:
        pct = improvement / pre_metrics["mean_position_error"] * 100
        print(f"Position error reduction: {pct:.1f}%")

    transform_path = project_dir / "mocap" / "alignment_transform.json"
    save_alignment_transform(R, t, s, transform_path, pre_metrics, post_metrics)
    print(f"Transform saved: {transform_path}")

    if apply:
        save_trajectory(
            aligned,
            mocap_intrinsics,
            output_dir,
            metadata={
                "source": "calibrate.py",
                "aligned_from": str(mocap_dir),
                "aligned_to": str(mmcam_dir),
                "scale": s,
                "mean_position_error": post_metrics["mean_position_error"],
                "mean_rotation_error": post_metrics["mean_rotation_error"],
            },
        )
        print(f"Aligned trajectory saved: {output_dir}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align mocap and matchmove camera trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory",
    )
    parser.add_argument(
        "--mmcam-dir",
        type=Path,
        default=None,
        help="Matchmove camera directory (default: <project>/camera/)",
    )
    parser.add_argument(
        "--mocap-dir",
        type=Path,
        default=None,
        help="Mocap camera directory (default: <project>/mocap/camera/)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write aligned trajectory to output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for aligned camera (default: <project>/mocap/camera_aligned/)",
    )

    args = parser.parse_args()
    project_dir = args.project_dir.resolve()

    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_calibration(
        project_dir=project_dir,
        mmcam_dir=args.mmcam_dir.resolve() if args.mmcam_dir else None,
        mocap_dir=args.mocap_dir.resolve() if args.mocap_dir else None,
        apply=args.apply,
        output_dir=args.output.resolve() if args.output else None,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
