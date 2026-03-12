#!/usr/bin/env python3
"""Export motion capture data to Alembic and USD formats.

Converts motion.pkl (SMPL parameters) to animated mesh files:
- Alembic (.abc) - Industry standard for mesh animation
- USD (.usd) - Universal Scene Description

Environment:
    Requires 'gvhmr' conda environment (has smplx, chumpy, torch).
    This script handles SMPL body model processing which shares
    dependencies with GVHMR motion capture.

Usage:
    conda run -p <prefix> python export_mocap.py <project_dir> [options]

Example:
    conda run -p <prefix> python export_mocap.py /path/to/project --format abc
    conda run -p <prefix> python export_mocap.py /path/to/project --format usd --fps 30
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional, List, Tuple

from env_config import require_conda_env, INSTALL_DIR, GVHMR_CONDA_PREFIX

REQUIRED_ENV = GVHMR_CONDA_PREFIX


def convert_slahmr_to_motion(
    slahmr_dir: Path,
    output_path: Path,
    gender: str = "neutral",
) -> bool:
    """Convert SLAHMR output to motion.pkl format via SMPL-H → SMPLX.

    Looks for slahmr_stitched.npz in the slahmr directory and converts
    to SMPLX format using betas fitting.

    Args:
        slahmr_dir: Directory containing SLAHMR output (e.g., mocap/person/slahmr).
        output_path: Path for output motion.pkl.
        gender: Body model gender.

    Returns:
        True if conversion successful.
    """
    stitched_npz = slahmr_dir / "slahmr_stitched.npz"
    if not stitched_npz.exists():
        npz_files = sorted(slahmr_dir.glob("*_world_results.npz"))
        if npz_files:
            stitched_npz = npz_files[0]
        else:
            print(f"Error: No SLAHMR output found in {slahmr_dir}", file=sys.stderr)
            return False

    from convert_smplh_to_smplx import convert_slahmr_npz_to_smplx

    if not convert_slahmr_npz_to_smplx(
        npz_path=stitched_npz,
        output_path=output_path,
        gender=gender,
    ):
        return False

    hand_poses_path = slahmr_dir.parent / "hand_poses.npz"
    if hand_poses_path.exists():
        import numpy as np

        with open(output_path, "rb") as f:
            motion = pickle.load(f)
        hand_data = np.load(hand_poses_path)
        motion["left_hand_pose"] = hand_data["left_hand_pose"]
        motion["right_hand_pose"] = hand_data["right_hand_pose"]
        with open(output_path, "wb") as f:
            pickle.dump(motion, f)
        print(f"  → Loaded hand poses from {hand_poses_path.name}")

    return True


def load_motion_data(motion_path: Path) -> Optional[dict]:
    """Load motion data from pickle file.

    Args:
        motion_path: Path to motion.pkl

    Returns:
        Dictionary with poses, trans, betas, gender or None if failed
    """
    if not motion_path.exists():
        print(f"Error: Motion file not found: {motion_path}", file=sys.stderr)
        return None

    try:
        with open(motion_path, 'rb') as f:
            data = pickle.load(f)

        required_keys = ['poses', 'trans', 'betas']
        for key in required_keys:
            if key not in data:
                print(f"Error: Missing '{key}' in motion data", file=sys.stderr)
                return None

        if 'gender' not in data:
            data['gender'] = 'neutral'

        return data

    except Exception as e:
        print(f"Error loading motion data: {e}", file=sys.stderr)
        return None


def convert_gvhmr_to_motion(
    gvhmr_dir: Path,
    output_path: Path,
    gender: str = "neutral",
    use_incam: bool = False,
) -> bool:
    """Convert GVHMR output (hmr4d_results.pt) to motion.pkl format.

    By default uses smpl_params_global (GVHMR world coordinates).
    When use_incam=True, uses smpl_params_incam (camera coordinates)
    for the mmcam-direct approach where the body mesh will be
    transformed to world space using mmcam extrinsics.

    Args:
        gvhmr_dir: Directory containing GVHMR output (e.g., mocap/person/gvhmr)
        output_path: Path for output motion.pkl
        gender: Body model gender
        use_incam: Use smpl_params_incam instead of smpl_params_global

    Returns:
        True if conversion successful
    """
    import numpy as np
    import torch

    gvhmr_files = list(gvhmr_dir.rglob("hmr4d*.pt"))
    if not gvhmr_files:
        print(f"Error: No GVHMR output (hmr4d*.pt) found in {gvhmr_dir}", file=sys.stderr)
        return False

    gvhmr_file = gvhmr_files[0]
    param_source = "smpl_params_incam" if use_incam else "smpl_params_global"
    print(f"  → Converting {gvhmr_file.name} to motion.pkl ({param_source})...")

    def to_numpy(obj):
        if hasattr(obj, 'numpy'):
            return obj.numpy()
        elif isinstance(obj, dict):
            return {k: to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_numpy(v) for v in obj]
        return obj

    gvhmr_data = torch.load(gvhmr_file, map_location='cpu', weights_only=False)
    gvhmr_data = to_numpy(gvhmr_data)

    param_key = "smpl_params_incam" if use_incam else "smpl_params_global"
    if param_key in gvhmr_data:
        params = gvhmr_data[param_key]
    elif 'global_orient' in gvhmr_data:
        params = gvhmr_data
    else:
        params = gvhmr_data

    body_pose = params.get('body_pose', params.get('poses'))
    global_orient = params.get('global_orient')
    transl = params.get('transl', params.get('trans'))
    betas = params.get('betas')

    if body_pose is None:
        print(f"Error: Could not find body_pose in {gvhmr_file.name}", file=sys.stderr)
        return False

    body_pose = np.array(body_pose)
    if body_pose.ndim < 2 or body_pose.shape[0] == 0:
        print(f"Error: Invalid body_pose shape in {gvhmr_file.name}", file=sys.stderr)
        return False

    n_frames = len(body_pose)

    if global_orient is not None:
        global_orient = np.array(global_orient)
        if global_orient.ndim == 1:
            global_orient = global_orient.reshape(1, -1)
        if len(global_orient) == 1 and n_frames > 1:
            global_orient = np.tile(global_orient, (n_frames, 1))
    else:
        global_orient = np.zeros((n_frames, 3))

    if transl is not None:
        transl = np.array(transl)
    else:
        transl = np.zeros((n_frames, 3))

    if betas is not None:
        betas = np.array(betas)
        if betas.ndim > 1:
            betas = betas[0]
    else:
        betas = np.zeros(10)

    if body_pose.shape[1] >= 63:
        poses = np.concatenate([
            global_orient,
            body_pose[:, :63],
        ], axis=1)
    else:
        padding_needed = 63 - body_pose.shape[1]
        poses = np.concatenate([
            global_orient,
            body_pose,
            np.zeros((n_frames, padding_needed))
        ], axis=1)

    motion_format = {
        'poses': poses,
        'trans': transl,
        'betas': betas,
        'gender': gender
    }

    hand_poses_path = gvhmr_dir.parent / "hand_poses.npz"
    if hand_poses_path.exists():
        hand_data = np.load(hand_poses_path)
        motion_format['left_hand_pose'] = hand_data['left_hand_pose']
        motion_format['right_hand_pose'] = hand_data['right_hand_pose']
        print(f"  → Loaded hand poses from {hand_poses_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(motion_format, f)

    print(f"  OK Converted {n_frames} frames to {output_path.name}")
    return True


def get_body_model_path(gender: str) -> Optional[Path]:
    """Get path to SMPL-X body model for specified gender.

    Args:
        gender: neutral, male, or female

    Returns:
        Path to SMPL-X npz file or None if not found
    """
    body_models_dir = INSTALL_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models"
    smplx_dir = body_models_dir / "smplx"

    gender_map = {
        'neutral': 'SMPLX_NEUTRAL.npz',
        'male': 'SMPLX_MALE.npz',
        'female': 'SMPLX_FEMALE.npz',
    }

    if gender not in gender_map:
        print(f"Error: Invalid gender '{gender}'", file=sys.stderr)
        return None

    model_path = smplx_dir / gender_map[gender]
    if not model_path.exists():
        print(f"Error: SMPL-X model not found: {model_path}", file=sys.stderr)
        print("Run the installation wizard to download SMPL-X models", file=sys.stderr)
        return None

    return model_path


def generate_meshes(
    motion_data: dict,
    model_path: Path,
    device: str = "cpu"
) -> Optional[List[Tuple]]:
    """Generate mesh vertices and faces for each frame.

    Args:
        motion_data: Dictionary with poses, trans, betas
        model_path: Path to SMPL model
        device: torch device (cpu or cuda)

    Returns:
        List of (vertices, faces) tuples per frame, or None if failed
    """
    try:
        import numpy as np
        import torch
        import smplx
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install numpy torch smplx", file=sys.stderr)
        return None

    try:
        poses = np.array(motion_data['poses'])
        trans = np.array(motion_data['trans'])
        betas = np.array(motion_data['betas'])
        gender = motion_data.get('gender', 'neutral')

        n_frames = len(poses)
        if n_frames == 0:
            print("Error: Motion data contains no frames", file=sys.stderr)
            return None

        print(f"  Generating {n_frames} frames with {gender} body model...")

        model = smplx.create(
            model_path=str(model_path.parent.parent),
            model_type='smplx',
            gender=gender,
            num_betas=10,
            batch_size=1,
            flat_hand_mean=True,
            use_pca=False,
        ).to(device)

        faces = model.faces

        if betas.ndim == 1:
            betas = betas.reshape(1, -1)
        if betas.shape[1] < 10:
            betas = np.pad(betas, ((0, 0), (0, 10 - betas.shape[1])))

        left_hand = motion_data.get('left_hand_pose')
        right_hand = motion_data.get('right_hand_pose')

        meshes = []

        for i in range(n_frames):
            if i % 100 == 0:
                print(f"    Frame {i}/{n_frames}...")

            pose = poses[i]
            global_orient = pose[:3] if pose.shape[0] >= 3 else np.zeros(3)
            body_pose = pose[3:66] if pose.shape[0] >= 66 else np.zeros(63)
            if body_pose.shape[0] < 63:
                body_pose = np.pad(body_pose, (0, 63 - body_pose.shape[0]))

            global_orient_t = torch.tensor(global_orient, dtype=torch.float32, device=device).unsqueeze(0)
            body_pose_t = torch.tensor(body_pose, dtype=torch.float32, device=device).unsqueeze(0)
            betas_t = torch.tensor(betas[0], dtype=torch.float32, device=device).unsqueeze(0)
            transl_t = torch.tensor(trans[i], dtype=torch.float32, device=device).unsqueeze(0)

            kwargs = {
                'global_orient': global_orient_t,
                'body_pose': body_pose_t,
                'betas': betas_t,
                'transl': transl_t,
            }
            if left_hand is not None:
                kwargs['left_hand_pose'] = torch.tensor(
                    left_hand[i], dtype=torch.float32, device=device
                ).unsqueeze(0)
            if right_hand is not None:
                kwargs['right_hand_pose'] = torch.tensor(
                    right_hand[i], dtype=torch.float32, device=device
                ).unsqueeze(0)

            with torch.no_grad():
                output = model(**kwargs)

            vertices = output.vertices[0].cpu().numpy()
            meshes.append((vertices, faces))

        print(f"  OK Generated {n_frames} mesh frames")
        return meshes

    except Exception as e:
        print(f"Error generating meshes: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def transform_meshes_to_world(
    meshes: List[Tuple],
    c2w_matrices: "np.ndarray",
) -> List[Tuple]:
    """Transform mesh vertices from camera space to world space.

    Applies per-frame camera-to-world transforms: v_world = R @ v_cam + t

    Args:
        meshes: List of (vertices, faces) tuples in camera space
        c2w_matrices: (N, 4, 4) camera-to-world matrices

    Returns:
        List of (vertices, faces) tuples in world space
    """
    import numpy as np

    if len(meshes) != len(c2w_matrices):
        raise ValueError(
            f"Frame count mismatch: {len(meshes)} meshes vs {len(c2w_matrices)} c2w matrices"
        )

    transformed = []
    for i, (vertices, faces) in enumerate(meshes):
        R = c2w_matrices[i, :3, :3]
        t = c2w_matrices[i, :3, 3]
        v_world = vertices @ R.T + t
        transformed.append((v_world, faces))
    return transformed


def export_alembic(
    meshes: List[Tuple],
    output_path: Path,
    fps: int = 24
) -> bool:
    """Export meshes to Alembic format using Blender.

    Writes OBJ sequence to temp directory, then uses Blender headless
    to convert to Alembic.

    Args:
        meshes: List of (vertices, faces) tuples
        output_path: Output .abc file path
        fps: Frames per second

    Returns:
        True if successful
    """
    import tempfile
    import shutil

    try:
        from blender import export_mesh_sequence_to_alembic, check_blender_available
    except ImportError:
        print("Error: Blender integration module not found", file=sys.stderr)
        return False

    available, message = check_blender_available()
    if not available:
        print(f"Error: {message}", file=sys.stderr)
        return False

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="mocap_export_"))
        obj_dir = temp_dir / "obj_sequence"
        obj_dir.mkdir()

        print(f"  Writing {len(meshes)} OBJ frames for Alembic conversion...")

        for frame_idx, (vertices, faces) in enumerate(meshes):
            if frame_idx % 100 == 0:
                print(f"    Frame {frame_idx}/{len(meshes)}...")

            obj_path = obj_dir / f"frame_{frame_idx:05d}.obj"
            with open(obj_path, 'w', encoding='utf-8') as f:
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"  Converting to Alembic via Blender...")
        export_mesh_sequence_to_alembic(
            input_dir=obj_dir,
            output_path=output_path,
            fps=fps,
            start_frame=1,
        )

        if output_path.exists():
            print(f"  OK Exported to {output_path}")
            return True
        else:
            print("Error: Alembic export completed but file not created", file=sys.stderr)
            return False

    except Exception as e:
        print(f"Error exporting Alembic: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def export_usd(
    meshes: List[Tuple],
    output_path: Path,
    fps: int = 24
) -> bool:
    """Export meshes to USD format using Blender.

    Writes OBJ sequence to temp directory, then uses Blender headless
    to convert to USD.

    Args:
        meshes: List of (vertices, faces) tuples
        output_path: Output .usd/.usda/.usdc file path
        fps: Frames per second

    Returns:
        True if successful
    """
    import tempfile
    import shutil

    try:
        from blender import export_mesh_sequence_to_usd, check_blender_available
    except ImportError:
        print("Error: Blender integration module not found", file=sys.stderr)
        return False

    available, message = check_blender_available()
    if not available:
        print(f"Error: {message}", file=sys.stderr)
        return False

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="mocap_export_"))
        obj_dir = temp_dir / "obj_sequence"
        obj_dir.mkdir()

        print(f"  Writing {len(meshes)} OBJ frames for USD conversion...")

        for frame_idx, (vertices, faces) in enumerate(meshes):
            if frame_idx % 100 == 0:
                print(f"    Frame {frame_idx}/{len(meshes)}...")

            obj_path = obj_dir / f"frame_{frame_idx:05d}.obj"
            with open(obj_path, 'w', encoding='utf-8') as f:
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"  Converting to USD via Blender...")
        export_mesh_sequence_to_usd(
            input_dir=obj_dir,
            output_path=output_path,
            fps=fps,
            start_frame=1,
        )

        if output_path.exists():
            print(f"  OK Exported to {output_path}")
            return True
        else:
            print("Error: USD export completed but file not created", file=sys.stderr)
            return False

    except Exception as e:
        print(f"Error exporting USD: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def export_obj_sequence(
    meshes: List[Tuple],
    output_dir: Path,
    prefix: str = "frame"
) -> bool:
    """Export meshes as OBJ sequence (fallback if Alembic/USD unavailable).

    Args:
        meshes: List of (vertices, faces) tuples
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        True if successful
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Exporting {len(meshes)} frames as OBJ sequence...")

        for frame_idx, (vertices, faces) in enumerate(meshes):
            if frame_idx % 100 == 0:
                print(f"    Frame {frame_idx}/{len(meshes)}...")

            obj_path = output_dir / f"{prefix}_{frame_idx:05d}.obj"

            with open(obj_path, 'w', encoding='utf-8') as f:
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"  OK Exported {len(meshes)} OBJ files to {output_dir}")
        return True

    except Exception as e:
        print(f"Error exporting OBJ sequence: {e}", file=sys.stderr)
        return False


def detect_gpu() -> bool:
    """Detect if CUDA GPU is available.

    Returns:
        True if CUDA is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def export_tpose(
    model_path: Path,
    output_path: Path,
    betas: Optional[list] = None,
    gender: str = "neutral",
    device: str = "cpu"
) -> bool:
    """Export T-pose reference mesh as OBJ.

    Generates the SMPL body model in its default T-pose (zero pose)
    with optional shape parameters (betas) from motion data.

    Args:
        model_path: Path to SMPL model
        output_path: Output .obj file path
        betas: Shape parameters (10 values), None for default shape
        gender: Body model gender
        device: torch device (cpu or cuda)

    Returns:
        True if successful
    """
    try:
        import numpy as np
        import torch
        import smplx
    except ImportError as e:
        print(f"Error: Missing dependency for T-pose export: {e}", file=sys.stderr)
        return False

    try:
        print(f"  → Generating T-pose ({gender})...")

        model = smplx.create(
            model_path=str(model_path.parent.parent),
            model_type='smplx',
            gender=gender,
            num_betas=10,
            batch_size=1,
        ).to(device)

        if betas is not None:
            betas_array = np.array(betas)
            if betas_array.ndim == 1:
                betas_array = betas_array.reshape(1, -1)
            if betas_array.shape[1] < 10:
                betas_array = np.pad(betas_array, ((0, 0), (0, 10 - betas_array.shape[1])))
            betas_t = torch.tensor(betas_array[0], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            betas_t = torch.zeros(1, 10, dtype=torch.float32, device=device)

        global_orient_t = torch.zeros(1, 3, dtype=torch.float32, device=device)
        body_pose_t = torch.zeros(1, 63, dtype=torch.float32, device=device)
        transl_t = torch.zeros(1, 3, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(
                global_orient=global_orient_t,
                body_pose=body_pose_t,
                betas=betas_t,
                transl=transl_t,
            )

        vertices = output.vertices[0].cpu().numpy()
        faces = model.faces

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# SMPL T-pose reference mesh\n")
            f.write(f"# Gender: {gender}\n")
            f.write(f"# Betas: {'custom' if betas is not None else 'default'}\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"  OK Exported T-pose: {output_path.name}")
        return True

    except Exception as e:
        print(f"Error exporting T-pose: {e}", file=sys.stderr)
        return False


def run_export(
    project_dir: Path,
    formats: List[str] = None,
    fps: int = 24,
    motion_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    mocap_person: Optional[str] = None,
    camera_extrinsics_path: Optional[Path] = None,
    world_transform_path: Optional[Path] = None,
    no_convert: bool = False,
) -> bool:
    """Run motion export pipeline.

    Two approaches for mmcam camera alignment:

    --camera-extrinsics: mmcam-direct. Body from smpl_params_incam (camera
        space), transformed per-frame via c2w. Pixel-perfect alignment,
        no foot grounding.

    --world-transform: Rigid transform. Body from smpl_params_global (world
        space with foot contact), transformed once via Sim(3). Preserves
        foot grounding, static offset from pixel-perfect.

    Args:
        project_dir: Project directory
        formats: Export formats (abc, usd, obj) - defaults to [abc, usd]
        fps: Frames per second
        motion_file: Override motion.pkl path
        output_dir: Override output directory
        mocap_person: Person folder name (e.g., 'person_00'), defaults to 'person'
        camera_extrinsics_path: Path to extrinsics.json with c2w matrices.
            Uses incam body + per-frame transform.
        world_transform_path: Path to world_transform.json with R, s, t.
            Uses global body + single rigid transform.

    Returns:
        True if all exports successful
    """
    if formats is None:
        formats = ["abc", "usd"]

    if fps <= 0:
        print("Error: FPS must be greater than 0", file=sys.stderr)
        return False

    person_folder = mocap_person or "person"
    mocap_person_dir = project_dir / "mocap" / person_folder
    motion_path = motion_file or mocap_person_dir / "motion.pkl"
    gvhmr_dir = mocap_person_dir / "gvhmr"
    export_dir = output_dir or mocap_person_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    slahmr_dir = mocap_person_dir / "slahmr"
    use_incam = camera_extrinsics_path is not None and world_transform_path is None

    if not no_convert:
        if slahmr_dir.exists() and list(slahmr_dir.glob("*.npz")):
            print("  → Converting SLAHMR output to motion.pkl...")
            gender = "neutral"
            if motion_path.exists():
                existing = load_motion_data(motion_path)
                if existing:
                    gender = existing.get("gender", "neutral")
            if not convert_slahmr_to_motion(slahmr_dir, motion_path, gender=gender):
                return False
        elif gvhmr_dir.exists():
            gvhmr_files = list(gvhmr_dir.rglob("hmr4d*.pt"))
            if gvhmr_files:
                print("  → Converting GVHMR output to motion.pkl...")
                if not convert_gvhmr_to_motion(gvhmr_dir, motion_path, use_incam=use_incam):
                    return False

    use_gpu = detect_gpu()
    device = "cuda" if use_gpu else "cpu"

    print(f"\n{'=' * 60}")
    print("Motion Capture Export")
    print("=" * 60)
    print(f"Motion file: {motion_path}")
    print(f"Formats: {', '.join(formats).upper()}")
    print(f"FPS: {fps}")
    print(f"Device: {device.upper()}")
    print(f"Output dir: {export_dir}")
    print()

    motion_data = load_motion_data(motion_path)
    if motion_data is None:
        return False

    gender = motion_data.get('gender', 'neutral')
    print(f"Gender: {gender}")
    print(f"Frames: {len(motion_data['poses'])}")

    model_path = get_body_model_path(gender)
    if model_path is None:
        return False

    tpose_path = export_dir / "tpose.obj"
    betas = motion_data.get('betas')
    if not export_tpose(model_path, tpose_path, betas=betas, gender=gender, device=device):
        print("Warning: T-pose export failed, continuing with animation export", file=sys.stderr)

    meshes = generate_meshes(motion_data, model_path, device)
    if meshes is None:
        return False

    if not meshes:
        print("Error: No mesh frames generated", file=sys.stderr)
        return False

    if world_transform_path is not None:
        import json
        import numpy as np

        with open(world_transform_path, encoding="utf-8") as f:
            wt = json.load(f)
        R = np.array(wt["R"], dtype=np.float64)
        s = float(wt["s"])
        t = np.array(wt["t"], dtype=np.float64)
        print(f"  Applying rigid Sim(3) transform (scale={s:.4f})...")
        transformed = []
        for vertices, faces in meshes:
            v_world = s * (vertices @ R.T) + t
            transformed.append((v_world, faces))
        meshes = transformed
    elif camera_extrinsics_path is not None:
        import json
        import numpy as np

        with open(camera_extrinsics_path, encoding="utf-8") as f:
            c2w = np.array(json.load(f), dtype=np.float64)
        print(f"  Transforming meshes to world space ({len(c2w)} c2w matrices)...")
        meshes = transform_meshes_to_world(meshes, c2w)

    all_success = True
    exported_paths = []

    if tpose_path.exists():
        exported_paths.append(tpose_path)

    for fmt in formats:
        if fmt == "abc":
            output_path = export_dir / "body_motion.abc"
            success = export_alembic(meshes, output_path, fps)
        elif fmt == "usd":
            output_path = export_dir / "body_motion.usd"
            success = export_usd(meshes, output_path, fps)
        elif fmt == "obj":
            output_path = export_dir / "obj_sequence"
            success = export_obj_sequence(meshes, output_path)
        else:
            print(f"Warning: Unknown format '{fmt}', skipping", file=sys.stderr)
            continue

        if success:
            exported_paths.append(output_path)
        else:
            all_success = False

    if exported_paths:
        print(f"\n{'=' * 60}")
        print("Export Complete")
        print("=" * 60)
        for p in exported_paths:
            print(f"  {p}")
        print()

    return all_success


def parse_formats(format_str: str) -> List[str]:
    """Parse comma-separated format string.

    Args:
        format_str: Format string like "abc,usd" or "all"

    Returns:
        List of format strings
    """
    if format_str.lower() == "all":
        return ["abc", "usd"]

    formats = [f.strip().lower() for f in format_str.split(",")]
    valid_formats = {"abc", "usd", "obj"}

    for fmt in formats:
        if fmt not in valid_formats:
            print(f"Warning: Unknown format '{fmt}'", file=sys.stderr)

    return [f for f in formats if f in valid_formats]


def main():
    parser = argparse.ArgumentParser(
        description="Export motion capture data to Alembic/USD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing mocap/motion.pkl"
    )
    parser.add_argument(
        "--format", "-f",
        default="all",
        help="Export format(s): abc, usd, obj, or comma-separated (default: all = abc,usd)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Integer FPS (default: 24)"
    )
    parser.add_argument(
        "--motion-file", "-m",
        type=Path,
        help="Override motion.pkl path"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Override output directory"
    )
    parser.add_argument(
        "--mocap-person",
        type=str,
        default=None,
        help="Person folder name (e.g., 'person_00'). Defaults to 'person'."
    )
    parser.add_argument(
        "--camera-extrinsics",
        type=Path,
        default=None,
        help="Path to extrinsics.json (c2w matrices). Uses incam body + per-frame transform."
    )
    parser.add_argument(
        "--world-transform",
        type=Path,
        default=None,
        help="Path to world_transform.json (R, s, t). Uses global body + rigid Sim(3) transform."
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip SLAHMR/GVHMR → motion.pkl conversion (use existing motion.pkl as-is)"
    )

    args = parser.parse_args()

    require_conda_env(REQUIRED_ENV)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    formats = parse_formats(args.format)
    if not formats:
        print("Error: No valid formats specified", file=sys.stderr)
        sys.exit(1)

    success = run_export(
        project_dir=project_dir,
        formats=formats,
        fps=args.fps,
        motion_file=args.motion_file,
        output_dir=args.output_dir,
        mocap_person=args.mocap_person,
        camera_extrinsics_path=args.camera_extrinsics,
        world_transform_path=args.world_transform,
        no_convert=args.no_convert,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
