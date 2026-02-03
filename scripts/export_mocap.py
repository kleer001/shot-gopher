#!/usr/bin/env python3
"""Export motion capture data to Alembic and USD formats.

Converts motion.pkl (SMPL parameters) to animated mesh files:
- Alembic (.abc) - Industry standard for mesh animation
- USD (.usd) - Universal Scene Description

Usage:
    python export_mocap.py <project_dir> [options]

Example:
    python export_mocap.py /path/to/project --format abc
    python export_mocap.py /path/to/project --format usd --fps 30
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional, List, Tuple

from env_config import is_conda_env_active, get_activation_instructions, INSTALL_DIR


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


def get_smpl_model_path(gender: str) -> Optional[Path]:
    """Get path to SMPL model for specified gender.

    Args:
        gender: neutral, male, or female

    Returns:
        Path to SMPL pkl file or None if not found
    """
    body_models_dir = INSTALL_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models"
    smpl_dir = body_models_dir / "smpl"

    gender_map = {
        'neutral': 'SMPL_NEUTRAL.pkl',
        'male': 'SMPL_MALE.pkl',
        'female': 'SMPL_FEMALE.pkl',
    }

    if gender not in gender_map:
        print(f"Error: Invalid gender '{gender}'", file=sys.stderr)
        return None

    model_path = smpl_dir / gender_map[gender]
    if not model_path.exists():
        print(f"Error: SMPL model not found: {model_path}", file=sys.stderr)
        print("Run the installation wizard to download SMPL models", file=sys.stderr)
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
            model_type='smpl',
            gender=gender,
            num_betas=10,
            batch_size=1,
        ).to(device)

        faces = model.faces

        if betas.ndim == 1:
            betas = betas.reshape(1, -1)
        if betas.shape[1] < 10:
            betas = np.pad(betas, ((0, 0), (0, 10 - betas.shape[1])))

        meshes = []

        for i in range(n_frames):
            if i % 100 == 0:
                print(f"    Frame {i}/{n_frames}...")

            pose = poses[i]
            if pose.shape[0] >= 72:
                global_orient = pose[:3]
                body_pose = pose[3:72]
            else:
                global_orient = pose[:3] if pose.shape[0] >= 3 else np.zeros(3)
                body_pose = pose[3:] if pose.shape[0] > 3 else np.zeros(69)
                if body_pose.shape[0] < 69:
                    body_pose = np.pad(body_pose, (0, 69 - body_pose.shape[0]))

            global_orient_t = torch.tensor(global_orient, dtype=torch.float32, device=device).unsqueeze(0)
            body_pose_t = torch.tensor(body_pose, dtype=torch.float32, device=device).unsqueeze(0)
            betas_t = torch.tensor(betas[0], dtype=torch.float32, device=device).unsqueeze(0)
            transl_t = torch.tensor(trans[i], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                output = model(
                    global_orient=global_orient_t,
                    body_pose=body_pose_t,
                    betas=betas_t,
                    transl=transl_t,
                )

            vertices = output.vertices[0].cpu().numpy()
            meshes.append((vertices, faces))

        print(f"  OK Generated {n_frames} mesh frames")
        return meshes

    except Exception as e:
        print(f"Error generating meshes: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def export_alembic(
    meshes: List[Tuple],
    output_path: Path,
    fps: float = 24.0
) -> bool:
    """Export meshes to Alembic format.

    Args:
        meshes: List of (vertices, faces) tuples
        output_path: Output .abc file path
        fps: Frames per second

    Returns:
        True if successful
    """
    try:
        import alembic
        from alembic import Abc, AbcGeom
        import imath
    except ImportError:
        print("Error: alembic package not installed", file=sys.stderr)
        print("Install with: pip install alembic", file=sys.stderr)
        return False

    try:
        import numpy as np

        output_path.parent.mkdir(parents=True, exist_ok=True)

        archive = Abc.OArchive(str(output_path))
        top = archive.getTop()

        time_per_frame = 1.0 / fps
        time_sampling = AbcGeom.TimeSampling(time_per_frame, 0.0)
        time_sampling_index = archive.addTimeSampling(time_sampling)

        mesh_obj = AbcGeom.OPolyMesh(top, "body_mesh", time_sampling_index)
        mesh_schema = mesh_obj.getSchema()

        faces = meshes[0][1]
        face_counts = imath.IntArray(len(faces))
        face_indices = imath.IntArray(len(faces) * 3)

        for i, face in enumerate(faces):
            face_counts[i] = 3
            face_indices[i * 3] = int(face[0])
            face_indices[i * 3 + 1] = int(face[1])
            face_indices[i * 3 + 2] = int(face[2])

        print(f"  Exporting {len(meshes)} frames to Alembic...")

        for frame_idx, (vertices, _) in enumerate(meshes):
            if frame_idx % 100 == 0:
                print(f"    Frame {frame_idx}/{len(meshes)}...")

            positions = imath.V3fArray(len(vertices))
            for i, v in enumerate(vertices):
                positions[i] = imath.V3f(float(v[0]), float(v[1]), float(v[2]))

            sample = AbcGeom.OPolyMeshSchemaSample(
                positions,
                face_indices,
                face_counts
            )
            mesh_schema.set(sample)

        del archive

        print(f"  OK Exported to {output_path}")
        return True

    except Exception as e:
        print(f"Error exporting Alembic: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def export_usd(
    meshes: List[Tuple],
    output_path: Path,
    fps: float = 24.0
) -> bool:
    """Export meshes to USD format.

    Args:
        meshes: List of (vertices, faces) tuples
        output_path: Output .usd/.usda/.usdc file path
        fps: Frames per second

    Returns:
        True if successful
    """
    try:
        from pxr import Usd, UsdGeom, Vt, Gf
    except ImportError:
        print("Error: pxr (OpenUSD) package not installed", file=sys.stderr)
        print("Install USD from: https://developer.nvidia.com/usd", file=sys.stderr)
        print("Or: pip install usd-core", file=sys.stderr)
        return False

    try:
        import numpy as np

        output_path.parent.mkdir(parents=True, exist_ok=True)

        stage = Usd.Stage.CreateNew(str(output_path))
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(len(meshes) - 1)
        stage.SetFramesPerSecond(fps)

        mesh_prim = UsdGeom.Mesh.Define(stage, "/World/body_mesh")

        faces = meshes[0][1]
        face_vertex_counts = [3] * len(faces)
        face_vertex_indices = []
        for face in faces:
            face_vertex_indices.extend([int(face[0]), int(face[1]), int(face[2])])

        mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

        print(f"  Exporting {len(meshes)} frames to USD...")

        points_attr = mesh_prim.GetPointsAttr()

        for frame_idx, (vertices, _) in enumerate(meshes):
            if frame_idx % 100 == 0:
                print(f"    Frame {frame_idx}/{len(meshes)}...")

            points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
            points_attr.Set(Vt.Vec3fArray(points), frame_idx)

        stage.Save()

        print(f"  OK Exported to {output_path}")
        return True

    except Exception as e:
        print(f"Error exporting USD: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


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

            with open(obj_path, 'w') as f:
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
            model_type='smpl',
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
        body_pose_t = torch.zeros(1, 69, dtype=torch.float32, device=device)
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
        with open(output_path, 'w') as f:
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
    fps: float = 24.0,
    motion_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    mocap_person: Optional[str] = None,
) -> bool:
    """Run motion export pipeline.

    Args:
        project_dir: Project directory
        formats: Export formats (abc, usd, obj) - defaults to [abc, usd]
        fps: Frames per second
        motion_file: Override motion.pkl path
        output_dir: Override output directory
        mocap_person: Person folder name (e.g., 'person_00'), defaults to 'person'

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
    export_dir = output_dir or mocap_person_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

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

    model_path = get_smpl_model_path(gender)
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
        type=float,
        default=24.0,
        help="Frames per second (default: 24)"
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

    args = parser.parse_args()

    if not (is_conda_env_active('vfx-pipeline') or is_conda_env_active('gvhmr')):
        print(get_activation_instructions(), file=sys.stderr)
        sys.exit(1)

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
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
