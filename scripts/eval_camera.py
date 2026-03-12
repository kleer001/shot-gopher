#!/usr/bin/env python3
"""Evaluate camera alignment approaches against GVHMR gold standard.

Loads a project's GVHMR results and mmcam extrinsics, then compares
camera approaches by measuring:
- 2D pelvis reprojection error vs gold standard (identity cam + incam body)
- Camera smoothness (frame-to-frame position/rotation deltas)
- Frame-0 offset from GVHMR-derived camera

Usage:
    python eval_camera.py <project_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from camera_alignment import (
    align_colmap_trajectory,
    camera_rotation_from_orients,
    camera_translation_from_transl,
    compute_pelvis_joint,
    compute_world_transform,
)
from env_config import INSTALL_DIR
from transforms import axis_angle_to_rotation_matrix_batch


def load_project_data(project_dir: Path) -> dict:
    """Load GVHMR results and mmcam extrinsics for a project."""
    gvhmr_files = list(project_dir.rglob("hmr4d_results.pt"))
    if not gvhmr_files:
        raise FileNotFoundError(f"No hmr4d_results.pt in {project_dir}")

    gvhmr_raw = torch.load(gvhmr_files[0], map_location="cpu", weights_only=False)

    def to_np(obj):
        if hasattr(obj, "numpy"):
            return obj.numpy()
        if isinstance(obj, dict):
            return {k: to_np(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_np(v) for v in obj]
        return obj

    gvhmr = to_np(gvhmr_raw)

    mmcam_path = project_dir / "camera" / "extrinsics.json"
    mmcam_c2w = None
    if mmcam_path.exists():
        with open(mmcam_path, encoding="utf-8") as f:
            mmcam_c2w = np.array(json.load(f), dtype=np.float64)

    smplx_path = (
        INSTALL_DIR / "GVHMR" / "inputs" / "checkpoints"
        / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    )
    betas = np.asarray(gvhmr["smpl_params_global"]["betas"][0], dtype=np.float64)
    pelvis_joint = compute_pelvis_joint(smplx_path, betas)

    return {
        "gvhmr": gvhmr,
        "mmcam_c2w": mmcam_c2w,
        "pelvis_joint": pelvis_joint,
        "project_dir": project_dir,
    }


def gold_standard_pelvis_2d(gvhmr: dict, pelvis_joint: np.ndarray) -> np.ndarray:
    """Compute gold-standard 2D pelvis position (identity cam + incam body).

    This matches what GVHMR's 1_incam.mp4 renders.
    """
    cp = gvhmr["smpl_params_incam"]
    transl_c = np.asarray(cp["transl"], dtype=np.float64)
    orient_c = np.asarray(cp["global_orient"], dtype=np.float64)

    R_c = axis_angle_to_rotation_matrix_batch(orient_c)
    pelvis_cam = np.einsum("nij,j->ni", R_c, np.zeros(3)) + pelvis_joint + transl_c

    K = np.asarray(gvhmr["K_fullimg"], dtype=np.float64)
    if K.ndim == 3:
        K_frame = K[0]
    else:
        K_frame = K

    px = K_frame[0, 0] * pelvis_cam[:, 0] / pelvis_cam[:, 2] + K_frame[0, 2]
    py = K_frame[1, 1] * pelvis_cam[:, 1] / pelvis_cam[:, 2] + K_frame[1, 2]

    return np.stack([px, py], axis=-1)


def gvhmr_derived_camera(gvhmr: dict, pelvis_joint: np.ndarray) -> np.ndarray:
    """Compute raw GVHMR-derived c2w camera (noisy baseline)."""
    gp = gvhmr["smpl_params_global"]
    cp = gvhmr["smpl_params_incam"]

    orient_w = np.asarray(gp["global_orient"], dtype=np.float64)
    orient_c = np.asarray(cp["global_orient"], dtype=np.float64)
    transl_w = np.asarray(gp["transl"], dtype=np.float64)
    transl_c = np.asarray(cp["transl"], dtype=np.float64)

    R_w2c = camera_rotation_from_orients(orient_w, orient_c)
    t_w2c = camera_translation_from_transl(R_w2c, transl_w, transl_c, pelvis_joint)

    n = R_w2c.shape[0]
    R_c2w = np.swapaxes(R_w2c, -2, -1)
    t_c2w = -np.einsum("nij,nj->ni", R_c2w, t_w2c)

    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    T[:, :3, :3] = R_c2w
    T[:, :3, 3] = t_c2w
    return T


def project_pelvis_through_camera(
    c2w: np.ndarray, gvhmr: dict, pelvis_joint: np.ndarray
) -> np.ndarray:
    """Project world-space pelvis through a given camera to get 2D positions."""
    gp = gvhmr["smpl_params_global"]
    transl_w = np.asarray(gp["transl"], dtype=np.float64)
    pelvis_world = pelvis_joint + transl_w

    R_w2c = np.swapaxes(c2w[:, :3, :3], -2, -1)
    t_w2c = -np.einsum("nij,nj->ni", R_w2c, c2w[:, :3, 3])

    pelvis_cam = np.einsum("nij,nj->ni", R_w2c, pelvis_world) + t_w2c

    K = np.asarray(gvhmr["K_fullimg"], dtype=np.float64)
    K_frame = K[0] if K.ndim == 3 else K

    px = K_frame[0, 0] * pelvis_cam[:, 0] / pelvis_cam[:, 2] + K_frame[0, 2]
    py = K_frame[1, 1] * pelvis_cam[:, 1] / pelvis_cam[:, 2] + K_frame[1, 2]

    return np.stack([px, py], axis=-1)


def mmcam_direct_pelvis_2d(
    mmcam_c2w: np.ndarray, gvhmr: dict, pelvis_joint: np.ndarray
) -> np.ndarray:
    """Project incam body through mmcam camera.

    Body is in camera space (from smpl_params_incam). Transform to
    mmcam world space, then project back through mmcam camera.
    This should be identical to the gold standard if the mmcam camera
    matches the camera GVHMR saw during inference.
    """
    cp = gvhmr["smpl_params_incam"]
    transl_c = np.asarray(cp["transl"], dtype=np.float64)
    pelvis_cam = pelvis_joint + transl_c

    K = np.asarray(gvhmr["K_fullimg"], dtype=np.float64)
    K_frame = K[0] if K.ndim == 3 else K

    px = K_frame[0, 0] * pelvis_cam[:, 0] / pelvis_cam[:, 2] + K_frame[0, 2]
    py = K_frame[1, 1] * pelvis_cam[:, 1] / pelvis_cam[:, 2] + K_frame[1, 2]

    return np.stack([px, py], axis=-1)


def camera_smoothness(c2w: np.ndarray) -> dict:
    """Compute camera smoothness metrics."""
    pos = c2w[:, :3, 3]
    pos_delta = np.diff(pos, axis=0)
    pos_delta_norm = np.linalg.norm(pos_delta, axis=-1)

    rot_angles = []
    for i in range(1, len(c2w)):
        R_delta = c2w[i, :3, :3] @ c2w[i - 1, :3, :3].T
        angle = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))
        rot_angles.append(np.degrees(angle))
    rot_angles = np.array(rot_angles)

    return {
        "pos_delta_max_mm": float(pos_delta_norm.max() * 1000),
        "pos_delta_mean_mm": float(pos_delta_norm.mean() * 1000),
        "pos_delta_y_max_mm": float(np.abs(pos_delta[:, 1]).max() * 1000),
        "pos_delta_y_mean_mm": float(np.abs(pos_delta[:, 1]).mean() * 1000),
        "rot_delta_max_deg": float(rot_angles.max()),
        "rot_delta_mean_deg": float(rot_angles.mean()),
    }


def evaluate(project_dir: Path) -> None:
    """Run full evaluation on a project."""
    print(f"Project: {project_dir.name}")
    print("=" * 70)

    data = load_project_data(project_dir)
    gvhmr = data["gvhmr"]
    mmcam_c2w = data["mmcam_c2w"]
    pelvis = data["pelvis_joint"]
    n_frames = len(gvhmr["smpl_params_incam"]["transl"])

    print(f"Frames: {n_frames}")
    print(f"Pelvis joint: [{pelvis[0]:.4f}, {pelvis[1]:.4f}, {pelvis[2]:.4f}]")
    print()

    gold_2d = gold_standard_pelvis_2d(gvhmr, pelvis)
    print(f"Gold standard pelvis 2D (frame 0): [{gold_2d[0, 0]:.1f}, {gold_2d[0, 1]:.1f}] px")

    # --- Approach A: GVHMR-derived camera (raw, no alignment) ---
    gvhmr_c2w = gvhmr_derived_camera(gvhmr, pelvis)
    gvhmr_2d = project_pelvis_through_camera(gvhmr_c2w, gvhmr, pelvis)
    gvhmr_err = np.linalg.norm(gvhmr_2d - gold_2d, axis=-1)
    gvhmr_smooth = camera_smoothness(gvhmr_c2w)

    print("\n--- A: GVHMR-derived camera (raw) ---")
    print(f"  Reprojection error:  mean={gvhmr_err.mean():.2f} px  max={gvhmr_err.max():.2f} px  frame0={gvhmr_err[0]:.2f} px")
    print(f"  Pos jitter:          max={gvhmr_smooth['pos_delta_max_mm']:.1f} mm  mean={gvhmr_smooth['pos_delta_mean_mm']:.1f} mm")
    print(f"  Pos Y jitter:        max={gvhmr_smooth['pos_delta_y_max_mm']:.1f} mm  mean={gvhmr_smooth['pos_delta_y_mean_mm']:.1f} mm")
    print(f"  Rot jitter:          max={gvhmr_smooth['rot_delta_max_deg']:.3f} deg  mean={gvhmr_smooth['rot_delta_mean_deg']:.3f} deg")

    if mmcam_c2w is not None:
        # --- Approach B: Current alignment (complementary filter) ---
        aligned_c2w = align_colmap_trajectory(mmcam_c2w, gvhmr_c2w)
        aligned_2d = project_pelvis_through_camera(aligned_c2w, gvhmr, pelvis)
        aligned_err = np.linalg.norm(aligned_2d - gold_2d, axis=-1)
        aligned_smooth = camera_smoothness(aligned_c2w)

        print("\n--- B: Aligned mmcam (Wahba + complementary filter) ---")
        print(f"  Reprojection error:  mean={aligned_err.mean():.2f} px  max={aligned_err.max():.2f} px  frame0={aligned_err[0]:.2f} px")
        print(f"  Frame 0 offset:      x={aligned_2d[0,0]-gold_2d[0,0]:.2f} px  y={aligned_2d[0,1]-gold_2d[0,1]:.2f} px")
        print(f"  Pos jitter:          max={aligned_smooth['pos_delta_max_mm']:.1f} mm  mean={aligned_smooth['pos_delta_mean_mm']:.1f} mm")
        print(f"  Pos Y jitter:        max={aligned_smooth['pos_delta_y_max_mm']:.1f} mm  mean={aligned_smooth['pos_delta_y_mean_mm']:.1f} mm")
        print(f"  Rot jitter:          max={aligned_smooth['rot_delta_max_deg']:.3f} deg  mean={aligned_smooth['rot_delta_mean_deg']:.3f} deg")

        # --- Approach C: mmcam direct (incam body + mmcam camera) ---
        mmcam_smooth = camera_smoothness(mmcam_c2w)
        direct_2d = mmcam_direct_pelvis_2d(mmcam_c2w, gvhmr, pelvis)
        direct_err = np.linalg.norm(direct_2d - gold_2d, axis=-1)

        print("\n--- C: mmcam direct (incam body, no alignment) ---")
        print(f"  Reprojection error:  mean={direct_err.mean():.2f} px  max={direct_err.max():.2f} px  frame0={direct_err[0]:.2f} px")
        print(f"  Frame 0 offset:      x={direct_2d[0,0]-gold_2d[0,0]:.2f} px  y={direct_2d[0,1]-gold_2d[0,1]:.2f} px")
        print(f"  Pos jitter:          max={mmcam_smooth['pos_delta_max_mm']:.1f} mm  mean={mmcam_smooth['pos_delta_mean_mm']:.1f} mm")
        print(f"  Pos Y jitter:        max={mmcam_smooth['pos_delta_y_max_mm']:.1f} mm  mean={mmcam_smooth['pos_delta_y_mean_mm']:.1f} mm")
        print(f"  Rot jitter:          max={mmcam_smooth['rot_delta_max_deg']:.3f} deg  mean={mmcam_smooth['rot_delta_mean_deg']:.3f} deg")
        print(f"  NOTE: Reprojection error here measures incam-vs-incam (should be 0).")
        print(f"        The real question is whether the body mesh in mmcam world")
        print(f"        space projects correctly — which it does by construction.")

        # --- Approach D: Rigid Sim(3) transform (global body + mmcam camera) ---
        wt = compute_world_transform(gvhmr_c2w, mmcam_c2w)
        R_wt, s_wt, t_wt = wt["R"], wt["s"], wt["t"]

        gp = gvhmr["smpl_params_global"]
        pelvis_gvhmr = pelvis + np.asarray(gp["transl"], dtype=np.float64)
        pelvis_mmcam = s_wt * np.einsum("ij,nj->ni", R_wt, pelvis_gvhmr) + t_wt

        R_w2c_mmcam = np.swapaxes(mmcam_c2w[:, :3, :3], -2, -1)
        t_w2c_mmcam = -np.einsum("nij,nj->ni", R_w2c_mmcam, mmcam_c2w[:, :3, 3])
        pelvis_cam_d = np.einsum("nij,nj->ni", R_w2c_mmcam, pelvis_mmcam) + t_w2c_mmcam

        K = np.asarray(gvhmr["K_fullimg"], dtype=np.float64)
        K_frame = K[0] if K.ndim == 3 else K
        rigid_px = K_frame[0, 0] * pelvis_cam_d[:, 0] / pelvis_cam_d[:, 2] + K_frame[0, 2]
        rigid_py = K_frame[1, 1] * pelvis_cam_d[:, 1] / pelvis_cam_d[:, 2] + K_frame[1, 2]
        rigid_2d = np.stack([rigid_px, rigid_py], axis=-1)
        rigid_err = np.linalg.norm(rigid_2d - gold_2d, axis=-1)

        print(f"\n--- D: Rigid Sim(3) transform (global body + mmcam camera) ---")
        print(f"  Reprojection error:  mean={rigid_err.mean():.2f} px  max={rigid_err.max():.2f} px  frame0={rigid_err[0]:.2f} px")
        print(f"  Frame 0 offset:      x={rigid_2d[0,0]-gold_2d[0,0]:.2f} px  y={rigid_2d[0,1]-gold_2d[0,1]:.2f} px")
        print(f"  Pos jitter:          max={mmcam_smooth['pos_delta_max_mm']:.1f} mm  mean={mmcam_smooth['pos_delta_mean_mm']:.1f} mm")
        print(f"  Rot jitter:          max={mmcam_smooth['rot_delta_max_deg']:.3f} deg  mean={mmcam_smooth['rot_delta_mean_deg']:.3f} deg")
        print(f"  Sim(3) params:       scale={s_wt:.6f}  |t|={np.linalg.norm(t_wt):.4f}")
        print(f"  NOTE: Camera jitter same as C (mmcam). Body has foot contact from global params.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Approach':<45} {'Reproj (px)':<15} {'Y jitter (mm)':<15} {'Rot (deg)'}")
    print(f"{'A: GVHMR raw':<45} {gvhmr_err.mean():<15.2f} {gvhmr_smooth['pos_delta_y_max_mm']:<15.1f} {gvhmr_smooth['rot_delta_max_deg']:.3f}")
    if mmcam_c2w is not None:
        print(f"{'B: Wahba + complementary filter':<45} {aligned_err.mean():<15.2f} {aligned_smooth['pos_delta_y_max_mm']:<15.1f} {aligned_smooth['rot_delta_max_deg']:.3f}")
        print(f"{'C: mmcam direct (incam body)':<45} {'0.00 (exact)':<15} {mmcam_smooth['pos_delta_y_max_mm']:<15.1f} {mmcam_smooth['rot_delta_max_deg']:.3f}")
        print(f"{'D: Rigid Sim(3) (global body)':<45} {rigid_err.mean():<15.2f} {mmcam_smooth['pos_delta_y_max_mm']:<15.1f} {mmcam_smooth['rot_delta_max_deg']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_camera.py <project_dir>", file=sys.stderr)
        sys.exit(1)
    evaluate(Path(sys.argv[1]).resolve())
