"""Derive camera extrinsics from GVHMR SMPL orient params.

Computes per-frame camera-to-world matrices in GVHMR's global
coordinate frame using the algebraic relationship between
smpl_params_global and smpl_params_incam orient/transl fields.

No pytorch3d or torch dependency — numpy only, reusing transforms.py.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.signal import savgol_filter

from transforms import (
    axis_angle_to_rotation_matrix_batch,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def camera_rotation_from_orients(
    global_orient_world: np.ndarray,
    global_orient_incam: np.ndarray,
) -> np.ndarray:
    """Compute per-frame world-to-camera rotation from SMPL orient params.

    R_w2c[t] = R_c[t] @ R_w[t]^T

    Args:
        global_orient_world: (N, 3) axis-angle from smpl_params_global
        global_orient_incam: (N, 3) axis-angle from smpl_params_incam

    Returns:
        (N, 3, 3) world-to-camera rotation matrices
    """
    R_w = axis_angle_to_rotation_matrix_batch(global_orient_world)
    R_c = axis_angle_to_rotation_matrix_batch(global_orient_incam)
    R_w2c = np.einsum("nij,nkj->nik", R_c, R_w)
    return R_w2c


def compute_pelvis_joint(
    smplx_model_path: Path,
    betas: np.ndarray,
) -> np.ndarray:
    """Compute SMPL-X pelvis joint position from shape parameters.

    Loads J_regressor and shape blend shapes from the model file
    to compute the pelvis position in rest pose. Pure numpy, no
    SMPL-X package dependency.

    Args:
        smplx_model_path: Path to SMPLX_NEUTRAL.npz model file
        betas: (n_betas,) shape coefficients

    Returns:
        (3,) pelvis joint position in rest pose
    """
    model_data = np.load(str(smplx_model_path), allow_pickle=True)
    v_template = model_data["v_template"]
    shapedirs = model_data["shapedirs"]
    j_regressor = model_data["J_regressor"]

    n_betas = min(betas.shape[0], shapedirs.shape[2])
    v_shaped = v_template + np.einsum("vcd,d->vc", shapedirs[:, :, :n_betas], betas[:n_betas])
    return j_regressor[0] @ v_shaped


def camera_translation_from_transl(
    R_w2c: np.ndarray,
    transl_global: np.ndarray,
    transl_incam: np.ndarray,
    pelvis_joint: np.ndarray,
) -> np.ndarray:
    """Compute per-frame world-to-camera translation from SMPL transl params.

    Accounts for SMPL's pelvis-centered rotation:
        t_w2c = (I - R_w2c) @ J_pelvis + transl_cam - R_w2c @ transl_global

    GVHMR post-processing (foot grounding, smoothing) modifies transl_w
    independently of transl_c. The pelvis correction compensates for
    the fact that SMPL rotates around J_pelvis, not the origin.

    Args:
        R_w2c: (N, 3, 3) world-to-camera rotation
        transl_global: (N, 3) translation from smpl_params_global
        transl_incam: (N, 3) translation from smpl_params_incam
        pelvis_joint: (3,) pelvis joint position in rest pose

    Returns:
        (N, 3) world-to-camera translation vectors
    """
    pelvis_correction = pelvis_joint - np.einsum("nij,j->ni", R_w2c, pelvis_joint)
    return pelvis_correction + transl_incam - np.einsum("nij,nj->ni", R_w2c, transl_global)


def compute_chordal_mean_rotation(
    rotations: np.ndarray,
) -> np.ndarray:
    """Compute the chordal L2 mean of rotation matrices.

    Solves the Wahba problem: find R* minimising
    sum_t ||R* - R[t]||_F^2, via SVD projection onto SO(3).

    Args:
        rotations: (N, 3, 3) rotation matrices

    Returns:
        (3, 3) mean rotation matrix in SO(3)
    """
    M = rotations.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    S = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return U @ S @ Vt


def compute_world_transform(
    gvhmr_c2w: np.ndarray,
    mmcam_c2w: np.ndarray,
) -> dict[str, Any]:
    """Compute Sim(3) transform from GVHMR-world to mmcam-world.

    Uses camera position correspondences to find a single (R, s, t)
    that maps GVHMR's coordinate system to mmcam's. Applied once
    to all frames, preserving relative body motion (foot contact, etc).

    Stage 1: Wahba rotation (chordal L2 mean of per-frame R_rel)
    Stage 2: Least-squares scale and translation offset

    Args:
        gvhmr_c2w: (N, 4, 4) GVHMR-derived camera-to-world matrices
        mmcam_c2w: (N, 4, 4) mmcam camera-to-world matrices

    Returns:
        Dict with R (3,3), s (float), t (3,) such that:
        x_mmcam = s * R @ x_gvhmr + t
    """
    R_rel = np.einsum(
        "nij,nkj->nik",
        mmcam_c2w[:, :3, :3],
        gvhmr_c2w[:, :3, :3],
    )
    R = compute_chordal_mean_rotation(R_rel)

    P_src = gvhmr_c2w[:, :3, 3]
    P_tgt = mmcam_c2w[:, :3, 3]
    P_rotated = np.einsum("ij,nj->ni", R, P_src)

    mu_src = P_rotated.mean(axis=0)
    mu_tgt = P_tgt.mean(axis=0)
    src_centered = P_rotated - mu_src
    tgt_centered = P_tgt - mu_tgt

    var_src = np.sum(src_centered ** 2)
    s = float(np.sum(tgt_centered * src_centered) / var_src) if var_src > 1e-12 else 1.0
    t = mu_tgt - s * mu_src

    return {"R": R, "s": s, "t": t}


def align_colmap_trajectory(
    colmap_c2w: np.ndarray,
    gvhmr_c2w: np.ndarray,
    residual_window: int = 31,
    residual_poly: int = 2,
) -> np.ndarray:
    """Align COLMAP trajectory into GVHMR's world frame.

    Combines COLMAP's smooth camera motion with GVHMR's body-tracking
    accuracy using a three-stage approach:

    1. Robust rotation alignment — chordal L2 mean (Wahba problem)
       across all frames, avoiding single-frame noise sensitivity.
    2. Scale + translation offset — least-squares fit on rotated
       positions to account for COLMAP's arbitrary scale.
    3. Complementary filter — low-pass filters the per-frame position
       residual between GVHMR and aligned COLMAP, then adds it back.
       High-frequency motion from COLMAP (feature-tracked stability),
       low-frequency body tracking from GVHMR.

    Args:
        colmap_c2w: (N, 4, 4) camera-to-world matrices from COLMAP/VGGSfM
        gvhmr_c2w: (N, 4, 4) camera-to-world matrices from GVHMR (noisy)
        residual_window: Savitzky-Golay window for residual smoothing
            (must be odd, default 31 ~= 1.3s at 24fps)
        residual_poly: Polynomial order for residual smoothing

    Returns:
        (N, 4, 4) aligned camera-to-world matrices
    """
    n_frames = colmap_c2w.shape[0]

    R_rel = np.einsum(
        "nij,nkj->nik",
        gvhmr_c2w[:, :3, :3],
        colmap_c2w[:, :3, :3],
    )
    R_align = compute_chordal_mean_rotation(R_rel)

    R_rotated = np.einsum("ij,njk->nik", R_align, colmap_c2w[:, :3, :3])
    P_rotated = np.einsum("ij,nj->ni", R_align, colmap_c2w[:, :3, 3])

    tgt = gvhmr_c2w[:, :3, 3]
    mu_src = P_rotated.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    src_centered = P_rotated - mu_src
    tgt_centered = tgt - mu_tgt

    var_src = np.sum(src_centered ** 2)
    s = np.sum(tgt_centered * src_centered) / var_src if var_src > 1e-12 else 1.0
    P_aligned = s * P_rotated + (mu_tgt - s * mu_src)

    residual = tgt - P_aligned
    window = residual_window
    if n_frames < window:
        window = n_frames if n_frames % 2 == 1 else n_frames - 1
    if window > residual_poly:
        residual_smooth = np.empty_like(residual)
        for axis in range(3):
            residual_smooth[:, axis] = savgol_filter(
                residual[:, axis], window, residual_poly,
            )
    else:
        residual_smooth = residual.copy()

    P_final = P_aligned + residual_smooth

    result = np.broadcast_to(np.eye(4), (n_frames, 4, 4)).copy()
    result[:, :3, :3] = R_rotated
    result[:, :3, 3] = P_final

    return result


def smooth_camera_trajectory(
    c2w_matrices: np.ndarray,
    window: int = 11,
    poly_order: int = 3,
    translation_only: bool = False,
) -> np.ndarray:
    """Smooth camera trajectory using Savitzky-Golay filtering.

    Smooths translation components directly. Smooths rotation by
    converting to quaternions (hemisphere-consistent), filtering
    each component, and renormalising.

    Args:
        c2w_matrices: (N, 4, 4) camera-to-world matrices
        window: Savitzky-Golay window length (must be odd)
        poly_order: Polynomial order for Savitzky-Golay filter
        translation_only: If True, smooth only translation, keep rotation

    Returns:
        (N, 4, 4) smoothed camera-to-world matrices
    """
    n_frames = c2w_matrices.shape[0]
    if n_frames < window:
        window = n_frames if n_frames % 2 == 1 else n_frames - 1
        if window <= poly_order:
            return c2w_matrices.copy()

    result = c2w_matrices.copy()

    trans = c2w_matrices[:, :3, 3].copy()
    for axis in range(3):
        trans[:, axis] = savgol_filter(trans[:, axis], window, poly_order)
    result[:, :3, 3] = trans

    if translation_only:
        return result

    quats = np.zeros((n_frames, 4))
    for i in range(n_frames):
        quats[i] = rotation_matrix_to_quaternion(c2w_matrices[i, :3, :3])

    for i in range(1, n_frames):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    for c in range(4):
        quats[:, c] = savgol_filter(quats[:, c], window, poly_order)

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats /= norms

    for i in range(n_frames):
        result[i, :3, :3] = quaternion_to_rotation_matrix(quats[i])

    return result


def compute_aligned_camera(
    gvhmr_data: dict[str, Any],
    image_width: int,
    image_height: int,
    pelvis_joint: np.ndarray,
    colmap_extrinsics: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute body-aligned camera extrinsics from GVHMR output.

    Uses SMPL orient params to derive camera rotation and translation
    that are algebraically consistent with the body mesh in
    GVHMR-global coordinates.

    When colmap_extrinsics are provided, aligns COLMAP's smooth
    trajectory into GVHMR's world frame using robust rotation
    alignment (Wahba problem) and a complementary filter that
    fuses COLMAP's stability with GVHMR's body tracking. Falls
    back to Savitzky-Golay smoothing when no COLMAP data is
    available.

    Args:
        gvhmr_data: Dict from hmr4d_results.pt (must contain
            smpl_params_global, smpl_params_incam, K_fullimg)
        image_width: Source image width in pixels
        image_height: Source image height in pixels
        pelvis_joint: (3,) SMPL pelvis joint in rest pose from
            compute_pelvis_joint()
        colmap_extrinsics: Optional (N, 4, 4) c2w matrices from COLMAP

    Returns:
        Dict with:
            extrinsics: list of (4, 4) c2w matrices (numpy)
            intrinsics: dict with fx, fy, cx, cy, width, height, model, params
            metadata: dict with source info
    """
    gp = gvhmr_data["smpl_params_global"]
    cp = gvhmr_data["smpl_params_incam"]

    orient_w = np.asarray(gp["global_orient"], dtype=np.float64)
    orient_c = np.asarray(cp["global_orient"], dtype=np.float64)
    transl_w = np.asarray(gp["transl"], dtype=np.float64)
    transl_c = np.asarray(cp["transl"], dtype=np.float64)

    R_w2c = camera_rotation_from_orients(orient_w, orient_c)
    t_w2c = camera_translation_from_transl(R_w2c, transl_w, transl_c, pelvis_joint)

    n_frames = R_w2c.shape[0]
    R_c2w = np.swapaxes(R_w2c, -2, -1)
    t_c2w = -np.einsum("nij,nj->ni", R_c2w, t_w2c)
    T_c2w_gvhmr = np.broadcast_to(np.eye(4), (n_frames, 4, 4)).copy()
    T_c2w_gvhmr[:, :3, :3] = R_c2w
    T_c2w_gvhmr[:, :3, 3] = t_c2w

    if colmap_extrinsics is not None:
        T_c2w = np.asarray(colmap_extrinsics, dtype=np.float64).copy()
        rotation_source = "mmcam_direct"
    else:
        T_c2w = smooth_camera_trajectory(T_c2w_gvhmr)
        rotation_source = "smpl_orient"

    K_fullimg = np.asarray(gvhmr_data["K_fullimg"])
    if K_fullimg.ndim == 3:
        K_fullimg = K_fullimg[0]
    fx = float(K_fullimg[0, 0])
    fy = float(K_fullimg[1, 1])
    cx = float(K_fullimg[0, 2])
    cy = float(K_fullimg[1, 2])

    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": image_width,
        "height": image_height,
        "model": "GVHMR_ESTIMATED",
        "params": [fx, fy, cx, cy],
    }

    translation_source = rotation_source if rotation_source != "smpl_orient" else "smpl_transl"
    metadata = {
        "source": "gvhmr_aligned",
        "rotation_source": rotation_source,
        "translation_source": translation_source,
        "total_frames": n_frames,
        "K_fullimg": K_fullimg.tolist(),
    }

    result = {
        "extrinsics": list(T_c2w),
        "intrinsics": intrinsics,
        "metadata": metadata,
    }

    if colmap_extrinsics is not None:
        result["world_transform"] = compute_world_transform(T_c2w_gvhmr, T_c2w)

    return result
