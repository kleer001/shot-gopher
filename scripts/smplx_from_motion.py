#!/usr/bin/env python3
"""Generate SMPL-X mesh sequence from WHAM motion data.

Converts WHAM motion.pkl (pose parameters) into animated SMPL-X OBJ meshes.
This is required for the mesh_deform.py script which needs animated SMPL-X
as the deformation driver.

Usage:
    python smplx_from_motion.py <project_dir> [options]

Example:
    # Generate animated SMPL-X from WHAM motion
    python smplx_from_motion.py /path/to/project \\
        --motion mocap/wham/motion.pkl \\
        --output mocap/smplx_animated/

    # Also export rest pose (frame 0)
    python smplx_from_motion.py /path/to/project \\
        --motion mocap/wham/motion.pkl \\
        --output mocap/smplx_animated/ \\
        --rest-pose mocap/smplx_rest.obj
"""

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from env_config import INSTALL_DIR


def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_0001.png'.

    Args:
        filename: Filename to parse

    Returns:
        Frame number as integer, or -1 if not found
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def detect_source_start_frame(project_dir: Path) -> int:
    """Detect the starting frame number from source/frames/ directory.

    Scans the source frames directory to find the minimum frame number,
    which serves as the base offset for output mesh numbering.

    Args:
        project_dir: Project root directory

    Returns:
        Starting frame number (e.g., 1 for frame_0001.png, 1001 for frame_1001.png)
        Defaults to 1 if detection fails
    """
    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        return 1

    frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
    if not frame_files:
        return 1

    frame_numbers = []
    for f in frame_files:
        num = extract_frame_number(f.stem)
        if num >= 0:
            frame_numbers.append(num)

    if not frame_numbers:
        return 1

    return min(frame_numbers)


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    required = {
        "numpy": "numpy",
        "torch": "torch",
        "smplx": "smplx",
        "trimesh": "trimesh",
    }

    missing = []
    for name, import_path in required.items():
        try:
            __import__(import_path)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print("\nInstall with:", file=sys.stderr)
        print(f"  pip install {' '.join(missing)}", file=sys.stderr)
        return False

    return True


def find_smplx_models() -> Optional[Path]:
    """Find SMPL-X model directory.

    Searches in common locations for SMPL-X body model files.

    Returns:
        Path to SMPL-X model directory, or None if not found
    """
    search_paths = [
        INSTALL_DIR / "smplx_models",
        Path.home() / ".smplx_models",
        Path("/data/smplx_models"),
        Path("smplx_models"),
    ]

    for path in search_paths:
        if path.exists():
            # Check for model files
            if (path / "smplx").exists() or list(path.glob("*.pkl")):
                return path

    return None


def load_motion_data(motion_path: Path) -> Dict[str, Any]:
    """Load WHAM motion data from pickle file.

    Args:
        motion_path: Path to motion.pkl

    Returns:
        Dictionary with motion data:
        - poses: [N, 72] SMPL-X pose parameters
        - trans: [N, 3] root translation
        - betas: [10] or [N, 10] shape parameters
    """
    with open(motion_path, "rb") as f:
        data = pickle.load(f)

    # Validate required keys
    if "poses" not in data:
        raise ValueError("motion.pkl missing 'poses' key")
    if "trans" not in data:
        raise ValueError("motion.pkl missing 'trans' key")

    poses = np.array(data["poses"])
    trans = np.array(data["trans"])

    # Handle betas - may be per-frame or constant
    if "betas" in data:
        betas = np.array(data["betas"])
        # If single set of betas, expand to per-frame
        if betas.ndim == 1:
            betas = np.tile(betas[np.newaxis, :], (len(poses), 1))
    else:
        # Default neutral shape
        betas = np.zeros((len(poses), 10))

    print(f"  Loaded motion: {len(poses)} frames")
    print(f"    Poses shape: {poses.shape}")
    print(f"    Trans shape: {trans.shape}")
    print(f"    Betas shape: {betas.shape}")

    return {
        "poses": poses,
        "trans": trans,
        "betas": betas
    }


def generate_smplx_meshes(
    motion_data: Dict[str, Any],
    model_path: Path,
    output_dir: Path,
    rest_pose_path: Optional[Path] = None,
    gender: str = "neutral",
    device: str = "cuda",
    start_frame: int = 1
) -> bool:
    """Generate SMPL-X mesh sequence from motion data.

    Args:
        motion_data: Motion dictionary with poses, trans, betas
        model_path: Path to SMPL-X models directory
        output_dir: Output directory for OBJ files
        rest_pose_path: Optional path to save rest pose mesh
        gender: Body model gender ("neutral", "male", "female")
        device: Torch device ("cuda" or "cpu")
        start_frame: Starting frame number for output filenames (from source sequence)

    Returns:
        True if successful
    """
    import torch
    import smplx
    import trimesh

    print(f"\n  Initializing SMPL-X model ({gender})...")

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("  Warning: CUDA not available, using CPU")
        device = "cpu"

    # Initialize SMPL-X model
    try:
        model = smplx.create(
            model_path=str(model_path),
            model_type="smplx",
            gender=gender,
            use_face_contour=False,
            use_pca=False,
            num_betas=10,
            num_expression_coeffs=10
        ).to(device)
    except Exception as e:
        print(f"Error initializing SMPL-X: {e}", file=sys.stderr)
        print("\nMake sure SMPL-X models are downloaded:", file=sys.stderr)
        print("  1. Register at https://smpl-x.is.tue.mpg.de/", file=sys.stderr)
        print(f"  2. Download models to {INSTALL_DIR}/smplx_models/", file=sys.stderr)
        return False

    # Get model properties
    faces = model.faces.astype(np.int32)

    # Extract UV coordinates if available
    # SMPL-X has standard UV layout
    try:
        # Try to load UV data from model
        uv_path = model_path / "smplx" / "smplx_uv.obj"
        if uv_path.exists():
            uv_mesh = trimesh.load(uv_path, process=False)
            if hasattr(uv_mesh.visual, 'uv') and uv_mesh.visual.uv is not None:
                uvs = np.array(uv_mesh.visual.uv)
            else:
                uvs = None
        else:
            uvs = None
    except Exception:
        uvs = None

    if uvs is None:
        print("  Warning: UV coordinates not found, meshes will have no UVs")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    poses = motion_data["poses"]
    trans = motion_data["trans"]
    betas = motion_data["betas"]
    n_frames = len(poses)

    print(f"\n  Generating {n_frames} SMPL-X meshes...")

    # Parse SMPL-X pose parameters
    # WHAM outputs 72 values: 3 (global) + 23*3 (body joints) = 72 (SMPL format)
    # SMPL-X uses 21 body joints (63 values), so we take pose[3:66]

    for frame_idx in range(n_frames):
        # Extract pose components
        pose = poses[frame_idx]
        translation = trans[frame_idx]
        beta = betas[frame_idx] if betas.ndim > 1 else betas

        # Convert to torch tensors
        global_orient = torch.tensor(pose[:3], dtype=torch.float32, device=device).unsqueeze(0)
        body_pose = torch.tensor(pose[3:66], dtype=torch.float32, device=device).unsqueeze(0)
        betas_t = torch.tensor(beta[:10], dtype=torch.float32, device=device).unsqueeze(0)
        transl = torch.tensor(translation, dtype=torch.float32, device=device).unsqueeze(0)

        # Handle extra pose parameters (jaw, eyes) if present
        if len(pose) > 66:
            jaw_pose = torch.tensor(pose[66:69], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            jaw_pose = torch.zeros(1, 3, dtype=torch.float32, device=device)

        # Forward pass
        with torch.no_grad():
            output = model(
                betas=betas_t,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                jaw_pose=jaw_pose,
                return_verts=True
            )

        vertices = output.vertices.cpu().numpy()[0]

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Add UVs if available
        if uvs is not None:
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

        output_frame_num = start_frame + frame_idx
        output_path = output_dir / f"frame_{output_frame_num:04d}.obj"
        mesh.export(output_path)

        # Save rest pose (frame 0) if requested
        if frame_idx == 0 and rest_pose_path:
            mesh.export(rest_pose_path)
            print(f"    Rest pose saved: {rest_pose_path}")

        # Progress
        if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1:
            print(f"    [{frame_idx + 1}/{n_frames}] frames generated")

    print(f"\n  Mesh sequence saved to: {output_dir}")
    print(f"    {n_frames} OBJ files")
    print(f"    Vertices per mesh: {len(vertices)}")

    # Clear GPU memory after model inference
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  → Cleared GPU memory")

    return True


def run_generation_pipeline(
    project_dir: Path,
    motion_path: Path,
    output_dir: Path,
    rest_pose_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    gender: str = "neutral",
    device: str = "cuda"
) -> bool:
    """Run SMPL-X mesh generation pipeline.

    Args:
        project_dir: Project root directory
        motion_path: Path to WHAM motion.pkl
        output_dir: Output directory for mesh sequence
        rest_pose_path: Optional path to save rest pose
        model_path: Path to SMPL-X models (auto-detected if None)
        gender: Body model gender
        device: Torch device

    Returns:
        True if successful
    """
    print(f"\n{'=' * 60}")
    print("SMPL-X Mesh Generation")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print(f"Motion: {motion_path}")
    print(f"Output: {output_dir}")
    print()

    try:
        # Find SMPL-X models
        if model_path is None:
            model_path = find_smplx_models()

        if model_path is None:
            print("Error: SMPL-X models not found", file=sys.stderr)
            print("\nDownload SMPL-X models:", file=sys.stderr)
            print("  1. Register at https://smpl-x.is.tue.mpg.de/", file=sys.stderr)
            print(f"  2. Download and extract to {INSTALL_DIR}/smplx_models/", file=sys.stderr)
            return False

        print(f"  SMPL-X models: {model_path}")

        start_frame = detect_source_start_frame(project_dir)
        print(f"  Source start frame: {start_frame}")

        # Load motion data
        print("\nLoading motion data...")
        motion_data = load_motion_data(motion_path)

        # Generate meshes
        success = generate_smplx_meshes(
            motion_data=motion_data,
            model_path=model_path,
            output_dir=output_dir,
            rest_pose_path=rest_pose_path,
            gender=gender,
            device=device,
            start_frame=start_frame
        )

        if success:
            print("\n  SMPL-X mesh generation complete")

        return success

    except Exception as e:
        print(f"Error in SMPL-X generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate SMPL-X mesh sequence from WHAM motion data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory"
    )
    parser.add_argument(
        "--motion",
        type=Path,
        required=True,
        help="Path to WHAM motion.pkl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for mesh sequence"
    )
    parser.add_argument(
        "--rest-pose",
        type=Path,
        default=None,
        help="Optional: save rest pose (frame 0) to this path"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to SMPL-X models (auto-detected if not specified)"
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Body model gender (default: neutral)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Torch device (default: cuda)"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve paths relative to project
    def resolve_path(p: Path) -> Path:
        if p and not p.is_absolute():
            return project_dir / p
        return p

    motion_path = resolve_path(args.motion)
    if not motion_path.exists():
        print(f"Error: Motion file not found: {motion_path}", file=sys.stderr)
        sys.exit(1)

    success = run_generation_pipeline(
        project_dir=project_dir,
        motion_path=motion_path,
        output_dir=resolve_path(args.output),
        rest_pose_path=resolve_path(args.rest_pose) if args.rest_pose else None,
        model_path=resolve_path(args.model_path) if args.model_path else None,
        gender=args.gender,
        device=args.device
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
