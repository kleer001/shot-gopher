"""Derive camera extrinsics from GVHMR SMPL orient params.

Computes per-frame camera-to-world matrices in GVHMR's global
coordinate frame using the algebraic relationship between
smpl_params_global and smpl_params_incam orient/transl fields.

No pytorch3d or torch dependency — numpy only, reusing transforms.py.
"""

from pathlib import Path
from typing import Any

import numpy as np

from transforms import axis_angle_to_rotation_matrix_batch


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


def compute_aligned_camera(
    gvhmr_data: dict[str, Any],
    image_width: int,
    image_height: int,
    pelvis_joint: np.ndarray,
) -> dict[str, Any]:
    """Compute body-aligned camera extrinsics from GVHMR output.

    Uses SMPL orient params (Source A) to derive camera rotation and
    translation that are algebraically consistent with the body mesh
    in GVHMR-global coordinates.

    Args:
        gvhmr_data: Dict from hmr4d_results.pt (must contain
            smpl_params_global, smpl_params_incam, K_fullimg)
        image_width: Source image width in pixels
        image_height: Source image height in pixels
        pelvis_joint: (3,) SMPL pelvis joint in rest pose from
            compute_pelvis_joint()

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
    T_w2c = np.broadcast_to(np.eye(4), (n_frames, 4, 4)).copy()
    T_w2c[:, :3, :3] = R_w2c
    T_w2c[:, :3, 3] = t_w2c

    R_c2w = np.swapaxes(R_w2c, -2, -1)
    t_c2w = -np.einsum("nij,nj->ni", R_c2w, t_w2c)
    T_c2w = np.broadcast_to(np.eye(4), (n_frames, 4, 4)).copy()
    T_c2w[:, :3, :3] = R_c2w
    T_c2w[:, :3, 3] = t_c2w

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

    metadata = {
        "source": "gvhmr_aligned",
        "rotation_source": "smpl_orient",
        "translation_source": "smpl_transl",
        "total_frames": n_frames,
        "K_fullimg": K_fullimg.tolist(),
    }

    return {
        "extrinsics": list(T_c2w),
        "intrinsics": intrinsics,
        "metadata": metadata,
    }
