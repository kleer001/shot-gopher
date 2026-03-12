#!/usr/bin/env python3
"""Convert SMPL-H parameters to SMPLX format.

SLAHMR outputs SMPL-H (body only, 63 pose params). This module fits
SMPLX betas to match SMPL-H T-pose joints, then transfers pose and
translation with pelvis offset correction.

Validated approach: <5cm mean joint error on real sequences.

Environment:
    Requires 'gvhmr' conda environment (has smplx, chumpy, torch).

Usage:
    conda run -n gvhmr python convert_smplh_to_smplx.py \\
        --smplh-npz /path/to/slahmr_results.npz \\
        --output /path/to/motion.pkl \\
        --gender neutral
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from env_config import INSTALL_DIR


def get_smplh_model_path(gender: str = "neutral") -> Path:
    """Get path to SMPL-H body model.

    Args:
        gender: neutral, male, or female

    Returns:
        Path to SMPL-H pkl/npz file.

    Raises:
        FileNotFoundError: If model file not found.
    """
    smplh_dir = INSTALL_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smpl"
    gender_map = {
        "neutral": "SMPL_NEUTRAL.pkl",
        "male": "SMPL_MALE.pkl",
        "female": "SMPL_FEMALE.pkl",
    }
    model_path = smplh_dir / gender_map[gender]
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL-H model not found: {model_path}")
    return model_path


def get_smplx_model_path(gender: str = "neutral") -> Path:
    """Get path to SMPLX body model.

    Args:
        gender: neutral, male, or female

    Returns:
        Path to SMPLX npz file.

    Raises:
        FileNotFoundError: If model file not found.
    """
    smplx_dir = INSTALL_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx"
    gender_map = {
        "neutral": "SMPLX_NEUTRAL.npz",
        "male": "SMPLX_MALE.npz",
        "female": "SMPLX_FEMALE.npz",
    }
    model_path = smplx_dir / gender_map[gender]
    if not model_path.exists():
        raise FileNotFoundError(f"SMPLX model not found: {model_path}")
    return model_path


def compute_smplh_joints(
    model_path: Path,
    betas: np.ndarray,
) -> np.ndarray:
    """Compute SMPL-H T-pose joints for given betas.

    Args:
        model_path: Path to SMPL-H model file.
        betas: Shape parameters (10,).

    Returns:
        Joint positions (J, 3) in T-pose.
    """
    import smplx

    model = smplx.create(
        model_path=str(model_path.parent.parent),
        model_type="smpl",
        gender=model_path.stem.split("_")[1].lower(),
        num_betas=10,
        batch_size=1,
    )

    betas_t = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)
    if betas_t.shape[1] < 10:
        betas_t = torch.nn.functional.pad(betas_t, (0, 10 - betas_t.shape[1]))

    with torch.no_grad():
        output = model(betas=betas_t)

    return output.joints[0].cpu().numpy()


def compute_smplx_joints(
    model_path: Path,
    betas: np.ndarray,
) -> np.ndarray:
    """Compute SMPLX T-pose joints for given betas.

    Args:
        model_path: Path to SMPLX model file.
        betas: Shape parameters (10,).

    Returns:
        Joint positions (J, 3) in T-pose.
    """
    import smplx

    model = smplx.create(
        model_path=str(model_path.parent.parent),
        model_type="smplx",
        gender=model_path.stem.split("_")[1].lower(),
        num_betas=10,
        batch_size=1,
        flat_hand_mean=True,
    )

    betas_t = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)
    if betas_t.shape[1] < 10:
        betas_t = torch.nn.functional.pad(betas_t, (0, 10 - betas_t.shape[1]))

    with torch.no_grad():
        output = model(betas=betas_t)

    return output.joints[0].cpu().numpy()


def fit_smplx_betas(
    smplh_model_path: Path,
    smplx_model_path: Path,
    smplh_betas: np.ndarray,
    n_iters: int = 500,
    lr: float = 0.01,
) -> np.ndarray:
    """Optimize SMPLX betas to match SMPL-H T-pose joint positions.

    Uses Adam optimizer to minimize L2 distance between pelvis-relative
    positions of the first 22 body joints. Pelvis-relative comparison
    is necessary because SMPL and SMPLX have a systematic ~13cm
    vertical offset that betas cannot correct.

    Args:
        smplh_model_path: Path to SMPL-H model.
        smplx_model_path: Path to SMPLX model.
        smplh_betas: SMPL-H shape parameters (up to 16 dims).
        n_iters: Optimization iterations.
        lr: Learning rate.

    Returns:
        Fitted SMPLX betas (10,).
    """
    import smplx

    smplh_gender = smplh_model_path.stem.split("_")[1].lower()
    smplx_gender = smplx_model_path.stem.split("_")[1].lower()

    smplh_model = smplx.create(
        model_path=str(smplh_model_path.parent.parent),
        model_type="smpl",
        gender=smplh_gender,
        num_betas=10,
        batch_size=1,
    )

    smplx_model = smplx.create(
        model_path=str(smplx_model_path.parent.parent),
        model_type="smplx",
        gender=smplx_gender,
        num_betas=10,
        batch_size=1,
        flat_hand_mean=True,
    )

    smplh_betas_10 = np.zeros(10, dtype=np.float32)
    n_copy = min(len(smplh_betas), 10)
    smplh_betas_10[:n_copy] = smplh_betas[:n_copy]

    smplh_betas_t = torch.tensor(smplh_betas_10, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        smplh_output = smplh_model(betas=smplh_betas_t)
    target_joints = smplh_output.joints[0, :22].detach()
    target_rel = target_joints - target_joints[0:1]

    smplx_betas = torch.zeros(1, 10, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([smplx_betas], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()
        smplx_output = smplx_model(betas=smplx_betas)
        pred_joints = smplx_output.joints[0, :22]
        pred_rel = pred_joints - pred_joints[0:1]
        loss = torch.nn.functional.mse_loss(pred_rel, target_rel)
        loss.backward()
        optimizer.step()

    return smplx_betas.detach().squeeze(0).numpy()


def convert_smplh_to_smplx(
    smplh_params: dict,
    gender: str = "neutral",
    n_betas_iters: int = 500,
) -> dict:
    """Convert SMPL-H motion parameters to SMPLX format.

    Steps:
    1. Fit SMPLX betas to match SMPL-H T-pose joints.
    2. Compute pelvis offset between models.
    3. Transfer body_pose and global_orient directly.
    4. Correct translation with pelvis offset.
    5. Zero hand poses (WiLoR fills later).

    Args:
        smplh_params: Dict with keys:
            - body_pose: (N, 63) or (N, 69) body pose axis-angle
            - global_orient: (N, 3) root orientation
            - trans: (N, 3) translation
            - betas: (B,) shape parameters
        gender: Body model gender.
        n_betas_iters: Iterations for betas fitting.

    Returns:
        Dict with SMPLX-compatible parameters:
            - poses: (N, 66) = global_orient(3) + body_pose(63)
            - trans: (N, 3) corrected translation
            - betas: (10,) fitted SMPLX betas
            - gender: str
            - left_hand_pose: (N, 45) zeros
            - right_hand_pose: (N, 45) zeros
    """
    smplh_model_path = get_smplh_model_path(gender)
    smplx_model_path = get_smplx_model_path(gender)

    body_pose = np.asarray(smplh_params["body_pose"], dtype=np.float32)
    global_orient = np.asarray(smplh_params["global_orient"], dtype=np.float32)
    trans = np.asarray(smplh_params["trans"], dtype=np.float32)
    smplh_betas = np.asarray(smplh_params["betas"], dtype=np.float32)

    if smplh_betas.ndim > 1:
        smplh_betas = smplh_betas[0]

    n_frames = len(body_pose)

    print(f"  → Fitting SMPLX betas ({n_betas_iters} iterations)...")
    smplx_betas = fit_smplx_betas(
        smplh_model_path, smplx_model_path,
        smplh_betas, n_iters=n_betas_iters,
    )

    smplh_joints = compute_smplh_joints(smplh_model_path, smplh_betas)
    smplx_joints = compute_smplx_joints(smplx_model_path, smplx_betas)
    pelvis_offset = smplh_joints[0] - smplx_joints[0]
    print(f"    Pelvis offset: [{pelvis_offset[0]:.4f}, {pelvis_offset[1]:.4f}, {pelvis_offset[2]:.4f}]")

    corrected_trans = trans + pelvis_offset[np.newaxis, :]

    body_pose_63 = body_pose[:, :63]
    poses = np.concatenate([global_orient, body_pose_63], axis=1)

    return {
        "poses": poses,
        "trans": corrected_trans,
        "betas": smplx_betas,
        "gender": gender,
        "left_hand_pose": np.zeros((n_frames, 45), dtype=np.float32),
        "right_hand_pose": np.zeros((n_frames, 45), dtype=np.float32),
    }


def convert_slahmr_npz_to_smplx(
    npz_path: Path,
    output_path: Path,
    gender: str = "neutral",
    person_idx: int = 0,
) -> bool:
    """Convert a stitched SLAHMR NPZ file to SMPLX motion.pkl.

    Args:
        npz_path: Path to stitched SLAHMR NPZ.
        output_path: Output motion.pkl path.
        gender: Body model gender.
        person_idx: Person index in multi-person output.

    Returns:
        True if conversion succeeded.
    """
    data = np.load(npz_path, allow_pickle=True)

    trans = np.asarray(data["trans"])
    root_orient = np.asarray(data["root_orient"])
    pose_body = np.asarray(data["pose_body"])
    betas = np.asarray(data["betas"])

    if trans.ndim == 3:
        trans = trans[person_idx]
    if root_orient.ndim == 3:
        root_orient = root_orient[person_idx]
    if pose_body.ndim == 3:
        pose_body = pose_body[person_idx]
    if betas.ndim == 2:
        betas = betas[person_idx]

    smplh_params = {
        "body_pose": pose_body,
        "global_orient": root_orient,
        "trans": trans,
        "betas": betas,
    }

    print(f"  → Converting SMPL-H → SMPLX ({len(trans)} frames, {gender})...")
    smplx_params = convert_smplh_to_smplx(smplh_params, gender=gender)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(smplx_params, f)

    print(f"  OK Wrote {output_path}")
    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert SMPL-H parameters to SMPLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--smplh-npz",
        type=Path,
        required=True,
        help="Path to SLAHMR stitched NPZ file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output motion.pkl path",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Body model gender (default: neutral)",
    )
    parser.add_argument(
        "--person-idx",
        type=int,
        default=0,
        help="Person index for multi-person output (default: 0)",
    )
    args = parser.parse_args()

    from env_config import require_conda_env
    require_conda_env("gvhmr")

    if not args.smplh_npz.exists():
        print(f"Error: Input file not found: {args.smplh_npz}", file=sys.stderr)
        sys.exit(1)

    success = convert_slahmr_npz_to_smplx(
        npz_path=args.smplh_npz,
        output_path=args.output,
        gender=args.gender,
        person_idx=args.person_idx,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
