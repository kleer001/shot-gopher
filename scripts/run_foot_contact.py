#!/usr/bin/env python3
"""Foot contact detection and footskate reduction.

Detects per-frame foot contact labels using UnderPressure (InterDigital,
SCA 2022), then applies translation + ankle rotation IK to reduce foot
sliding artifacts in the SMPLX mocap output.

Environment:
    Requires 'gvhmr' conda environment (has PyTorch, smplx, scipy).

Usage:
    conda run -n gvhmr python run_foot_contact.py <project_dir> [options]

Example:
    conda run -n gvhmr python run_foot_contact.py /path/to/project --fps 30
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from env_config import require_conda_env, UNDERPRESSURE_INSTALL_DIR, INSTALL_DIR

REQUIRED_ENV = "gvhmr"

SMPLX_TO_AMASS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 30, 39, 45, 54,
]

UP_JOINT_TO_SMPLX = [
    0, 3, 6, 9, 9, 12, 15, 14, 17, 19, 21, 13, 16, 18, 20,
    2, 5, 8, 11, 1, 4, 7, 10,
]

UP_PARENT_IDX = [
    -1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13,
    0, 15, 16, 17, 0, 19, 20, 21,
]

SMPLX_LEFT_FOOT = 10
SMPLX_RIGHT_FOOT = 11
LEFT_ANKLE_BODY_IDX = 7
RIGHT_ANKLE_BODY_IDX = 8
MIN_SEGMENT_FRAMES = 3
BLEND_FRAMES = 5
DEFAULT_CONTACT_RATIO = 0.5
DEFAULT_VELOCITY_THRESHOLD = 0.02
BODY_MODEL_DIR = INSTALL_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models"


def build_skeleton_from_smplx(joints_rest: torch.Tensor) -> torch.Tensor:
    """Build UnderPressure skeleton tensor from SMPLX rest-pose joints.

    UnderPressure's FK expects bone offsets (parent-to-child vectors) for
    its 23-joint Xsens topology. We compute these from SMPLX rest-pose
    joint positions.

    The spine_4 joint has no SMPLX equivalent and maps to spine3 (SMPLX 9),
    producing a zero-length bone. The retargeting optimizer's scale parameter
    absorbs any proportion differences.

    Args:
        joints_rest: SMPLX rest-pose joint positions, shape (J, 3).

    Returns:
        Bone offset tensor, shape (23, 3).
    """
    up_positions = joints_rest[UP_JOINT_TO_SMPLX]
    skeleton = torch.zeros(23, 3)
    for i in range(23):
        if UP_PARENT_IDX[i] == -1:
            skeleton[i] = up_positions[i]
        else:
            skeleton[i] = up_positions[i] - up_positions[UP_PARENT_IDX[i]]
    return skeleton


def smplx_fk(
    model: "smplx.SMPLXLayer",
    poses: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Run SMPLX forward kinematics in batches.

    Args:
        model: SMPLX model instance.
        poses: Body poses, shape (N, 66).
        trans: Root translation, shape (N, 3).
        betas: Shape parameters, shape (B,).
        device: Torch device.
        batch_size: Frames per FK batch.

    Returns:
        Joint positions, shape (N, J, 3).
    """
    n_frames = len(poses)
    betas_10 = betas[:10]
    all_joints = []

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        bs = end - start
        b_poses = torch.tensor(poses[start:end], dtype=torch.float32, device=device)
        b_trans = torch.tensor(trans[start:end], dtype=torch.float32, device=device)
        b_betas = torch.tensor(
            betas_10, dtype=torch.float32, device=device,
        ).unsqueeze(0).expand(bs, -1)

        zeros3 = torch.zeros(bs, 3, dtype=torch.float32, device=device)
        zeros45 = torch.zeros(bs, 45, dtype=torch.float32, device=device)
        zeros10 = torch.zeros(bs, 10, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(
                global_orient=b_poses[:, :3],
                body_pose=b_poses[:, 3:66],
                betas=b_betas,
                transl=b_trans,
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                left_hand_pose=zeros45,
                right_hand_pose=zeros45,
                expression=zeros10,
            )
            all_joints.append(output.joints.cpu().numpy())

    return np.concatenate(all_joints, axis=0)


AMASS_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine_1", "left_knee",
    "right_knee", "spine_2", "left_ankle", "right_ankle", "neck",
    "left_foot", "right_foot", "head", "left_clavicle", "right_clavicle",
    "head_top", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_finger_middle_3",
    "left_finger_thumb_3", "right_finger_middle_3", "right_finger_thumb_3",
]


def detect_contacts(
    amass_joints: torch.Tensor,
    skeleton: torch.Tensor,
    fps: float,
    device: torch.device,
    contact_ratio: float = DEFAULT_CONTACT_RATIO,
) -> np.ndarray:
    """Run UnderPressure contact detection on AMASS joint positions.

    Inlines the retargeting and detection logic from UnderPressure's demo.py
    to avoid importing the visualization module (requires panda3d).

    Args:
        amass_joints: Joint positions, shape (N, 26, 3).
        skeleton: UnderPressure bone offsets, shape (23, 3).
        fps: Source frame rate (resampled to 100fps internally).
        device: Torch device.
        contact_ratio: Fraction of pressure cells that must be active
            to declare foot contact (0.0-1.0). Higher = more conservative.

    Returns:
        Per-foot contact flags, shape (N, 2) boolean [left, right].
    """
    sys.path.insert(0, str(UNDERPRESSURE_INSTALL_DIR))
    import anim
    import models as up_models
    import util
    from data import FRAMERATE, TOPOLOGY

    checkpoint = torch.load(
        str(UNDERPRESSURE_INSTALL_DIR / "pretrained.tar"),
        map_location=device,
        weights_only=False,
    )
    up_model = up_models.DeepNetwork(
        state_dict=checkpoint["model"],
    ).to(device).eval()

    target = amass_joints.to(device)
    joints = [j for j in TOPOLOGY if j in AMASS_JOINT_NAMES]
    jidxs = list(map(TOPOLOGY.index, joints))
    target_jidxs = list(map(AMASS_JOINT_NAMES.index, joints))
    target_pos = target[..., target_jidxs, :]
    nframes = target_pos.shape[-3]

    angles = torch.nn.Parameter(
        util.SU2.identity(nframes, len(TOPOLOGY)).to(target_pos),
    )
    trajectory = torch.nn.Parameter(target_pos[..., [0], :].clone())
    translate = torch.nn.Parameter(torch.zeros(1, 1, 3).to(target_pos))
    scale = torch.nn.Parameter(torch.full([1, 1, 1], 1.0).to(target_pos))
    optimiser = torch.optim.Adam(
        [angles, trajectory, translate, scale], lr=1e-1,
    )
    skel_dev = skeleton.to(target_pos)
    p_weight = 1 / (
        skel_dev[..., 2].amax(dim=-1) - skel_dev[..., 2].amin(dim=-1)
    ).mean().square()
    q_weight = 1e-3

    for _ in range(150):
        positions = anim.FK(
            util.SU2.normalize(angles), skel_dev, None, TOPOLOGY,
        )[:, jidxs]
        positions = scale * positions + trajectory + translate
        p_error = (target_pos - positions).norm(p=2, dim=-1).square().mean()
        q_error = (angles.norm(p=2, dim=-1) - 1).square().mean()
        loss = p_weight * p_error + q_weight * q_error
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    angles_opt = util.SU2.normalize(angles.data)
    trajectory_opt = (trajectory.data + translate.data) / scale.data

    out_nframes = round(nframes / fps * FRAMERATE)
    angles_rs = util.resample(
        angles_opt, out_nframes, dim=-3, interpolation_fn=util.SU2.slerp,
    )
    trajectory_rs = util.resample(trajectory_opt, out_nframes)

    positions_rs = anim.FK(angles_rs, skel_dev, trajectory_rs, TOPOLOGY)
    contacts_raw = up_model.contacts(
        positions_rs.unsqueeze(0),
    ).squeeze(0).detach()
    contacts_raw = util.resample(
        contacts_raw.float(), nframes,
    ) >= 0.5

    if contacts_raw.dim() == 3:
        n_cells = contacts_raw.shape[-1]
        min_cells = max(1, int(n_cells * contact_ratio))
        left = (contacts_raw[:, 0, :].sum(dim=-1) >= min_cells).cpu().numpy()
        right = (contacts_raw[:, 1, :].sum(dim=-1) >= min_cells).cpu().numpy()
    else:
        left = contacts_raw[:, 0].cpu().numpy().astype(bool)
        right = contacts_raw[:, 1].cpu().numpy().astype(bool)

    return np.stack([left, right], axis=-1)


def find_contact_segments(
    contact: np.ndarray, min_length: int = MIN_SEGMENT_FRAMES,
) -> list[tuple[int, int]]:
    """Find contiguous frame segments where a foot is in contact.

    Args:
        contact: Per-frame contact flags, shape (N,) boolean.
        min_length: Minimum segment length (shorter segments are noise).

    Returns:
        List of (start, end) tuples, end exclusive.
    """
    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i in range(len(contact)):
        if contact[i] and not in_seg:
            start = i
            in_seg = True
        elif not contact[i] and in_seg:
            if i - start >= min_length:
                segments.append((start, i))
            in_seg = False
    if in_seg and len(contact) - start >= min_length:
        segments.append((start, len(contact)))
    return segments


def apply_foot_planting(
    poses: np.ndarray,
    trans: np.ndarray,
    joints: np.ndarray,
    contacts: np.ndarray,
    blend_frames: int = BLEND_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply foot planting via ankle rotation only (hips-down).

    For each contact segment, adjusts ankle pitch to bring the foot
    toward the ground plane. Does NOT modify root translation, so
    the upper body is unaffected.

    Args:
        poses: Body poses, shape (N, 66), modified in place.
        trans: Root translation, shape (N, 3), returned unmodified.
        joints: FK joint positions, shape (N, J, 3).
        contacts: Per-foot contact, shape (N, 2) boolean [left, right].
        blend_frames: Ramp length for boundary smoothing.

    Returns:
        Modified (poses, trans) arrays.
    """
    from scipy.spatial.transform import Rotation

    foot_indices = [SMPLX_LEFT_FOOT, SMPLX_RIGHT_FOOT]
    ankle_body_indices = [LEFT_ANKLE_BODY_IDX, RIGHT_ANKLE_BODY_IDX]

    for foot_idx in range(2):
        foot_pos = joints[:, foot_indices[foot_idx], :]
        ankle_pos = joints[:, ankle_body_indices[foot_idx], :]
        bone_len = np.linalg.norm(foot_pos - ankle_pos, axis=-1).mean()
        if bone_len < 0.01:
            continue

        ankle_pose_start = ankle_body_indices[foot_idx] * 3
        segments = find_contact_segments(contacts[:, foot_idx])

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            seg_foot = foot_pos[seg_start:seg_end]
            target_y = np.min(seg_foot[:, 1])
            delta_y = target_y - seg_foot[:, 1]

            blend = np.ones(seg_len, dtype=np.float64)
            ramp = min(blend_frames, seg_len // 2)
            if ramp > 0:
                blend[:ramp] = np.linspace(0, 1, ramp)
                blend[-ramp:] = np.linspace(1, 0, ramp)

            delta_pitch = delta_y / bone_len
            for fi in range(seg_start, seg_end):
                bp = blend[fi - seg_start] * delta_pitch[fi - seg_start]
                if abs(bp) < 1e-6:
                    continue
                aa = poses[fi, ankle_pose_start:ankle_pose_start + 3].copy()
                r_cur = Rotation.from_rotvec(aa)
                r_fix = Rotation.from_rotvec([bp, 0, 0])
                poses[fi, ankle_pose_start:ankle_pose_start + 3] = (
                    r_fix * r_cur
                ).as_rotvec()

    return poses, trans


def run_foot_contact(
    project_dir: Path,
    mocap_person: str = "person",
    fps: float = 30.0,
    apply_ik: bool = True,
    contact_ratio: float = DEFAULT_CONTACT_RATIO,
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
) -> bool:
    """Detect foot contacts and optionally apply foot planting IK.

    Args:
        project_dir: Project directory.
        mocap_person: Person folder name under mocap/.
        fps: Source frame rate.
        apply_ik: Apply foot planting to reduce footskate.
        contact_ratio: Fraction of pressure cells required for contact (0-1).
        velocity_threshold: Max foot speed (m/frame) to accept contact.
            Set to 0 to disable velocity filtering.

    Returns:
        True if successful.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    mocap_person_dir = project_dir / "mocap" / mocap_person
    motion_pkl = mocap_person_dir / "motion.pkl"
    contacts_path = mocap_person_dir / "foot_contacts.npz"

    if not motion_pkl.exists():
        print(f"Error: motion.pkl not found: {motion_pkl}", file=sys.stderr)
        return False

    with open(motion_pkl, "rb") as f:
        motion = pickle.load(f)

    poses = np.asarray(motion["poses"], dtype=np.float32)
    trans = np.asarray(motion["trans"], dtype=np.float32)
    betas = np.asarray(motion["betas"])
    gender = str(motion.get("gender", "neutral"))
    n_frames = len(poses)

    import smplx as smplx_lib

    smplx_model = smplx_lib.create(
        str(BODY_MODEL_DIR), model_type="smplx", gender=gender,
        batch_size=1, use_pca=False, flat_hand_mean=True,
    )
    smplx_model.to(device)

    with torch.no_grad():
        rest_out = smplx_model(
            betas=torch.tensor(
                betas[:10], dtype=torch.float32, device=device,
            ).unsqueeze(0),
        )
        joints_rest = rest_out.joints[0].cpu()

    skeleton = build_skeleton_from_smplx(joints_rest)

    print(f"  Running SMPLX forward kinematics ({n_frames} frames)...")
    joints = smplx_fk(smplx_model, poses, trans, betas, device)
    amass_joints = torch.tensor(
        joints[:, SMPLX_TO_AMASS, :], dtype=torch.float32,
    )

    print("  [1/2] Detecting foot contacts (UnderPressure)...")
    print(f"    Contact ratio: {contact_ratio:.0%} of pressure cells")
    contacts = detect_contacts(
        amass_joints, skeleton, fps, device, contact_ratio=contact_ratio,
    )

    if velocity_threshold > 0:
        foot_joints = joints[:, [SMPLX_LEFT_FOOT, SMPLX_RIGHT_FOOT], :]
        foot_vel = np.linalg.norm(np.diff(foot_joints, axis=0), axis=-1)
        foot_vel = np.concatenate([foot_vel[:1], foot_vel], axis=0)
        vel_mask = foot_vel < velocity_threshold
        pre_left = int(contacts[:, 0].sum())
        pre_right = int(contacts[:, 1].sum())
        contacts[:, 0] &= vel_mask[:, 0]
        contacts[:, 1] &= vel_mask[:, 1]
        print(f"    Velocity filter (<{velocity_threshold:.3f} m/frame): "
              f"L {pre_left}→{int(contacts[:, 0].sum())}, "
              f"R {pre_right}→{int(contacts[:, 1].sum())}")

    n_left = int(contacts[:, 0].sum())
    n_right = int(contacts[:, 1].sum())

    np.savez(
        contacts_path,
        left_heel=contacts[:, 0],
        left_toe=contacts[:, 0],
        right_heel=contacts[:, 1],
        right_toe=contacts[:, 1],
    )

    print(f"  OK Saved {contacts_path.name}")
    print(f"    Left foot:  {n_left}/{n_frames} frames grounded")
    print(f"    Right foot: {n_right}/{n_frames} frames grounded")

    if apply_ik:
        print("  [2/2] Applying foot planting IK...")
        poses_f64 = poses.astype(np.float64)
        trans_f64 = trans.astype(np.float64)
        poses_f64, trans_f64 = apply_foot_planting(
            poses_f64, trans_f64, joints, contacts,
        )
        motion["poses"] = poses_f64.astype(np.float32)
        motion["trans"] = trans_f64.astype(np.float32)
        with open(motion_pkl, "wb") as f:
            pickle.dump(motion, f)
        print("  OK Modified motion.pkl with foot corrections")
    else:
        print("  [2/2] Skipping foot planting (--no-ik)")

    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect foot contacts and reduce footskate (UnderPressure)",
    )
    parser.add_argument("project_dir", type=Path, help="Project directory")
    parser.add_argument(
        "--mocap-person",
        default="person",
        help="Person folder name under mocap/ (default: person)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Source frame rate (default: 30)",
    )
    parser.add_argument(
        "--no-ik",
        action="store_true",
        help="Skip foot planting IK (only detect contacts)",
    )
    parser.add_argument(
        "--contact-ratio",
        type=float,
        default=DEFAULT_CONTACT_RATIO,
        help=f"Fraction of pressure cells required for contact, 0.0-1.0 "
             f"(default: {DEFAULT_CONTACT_RATIO}). Higher = more conservative.",
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=DEFAULT_VELOCITY_THRESHOLD,
        help=f"Max foot speed (m/frame) to accept contact "
             f"(default: {DEFAULT_VELOCITY_THRESHOLD}). Set to 0 to disable.",
    )

    args = parser.parse_args()

    require_conda_env(REQUIRED_ENV)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Foot Contact Detection & Footskate Reduction (UnderPressure)")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print(f"Person:  {args.mocap_person}")
    print(f"FPS:     {args.fps}")
    print()

    success = run_foot_contact(
        project_dir,
        mocap_person=args.mocap_person,
        fps=args.fps,
        apply_ik=not args.no_ik,
        contact_ratio=args.contact_ratio,
        velocity_threshold=args.velocity_threshold,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
