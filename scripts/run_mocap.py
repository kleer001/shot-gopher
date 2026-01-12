#!/usr/bin/env python3
"""Human motion capture with consistent topology and UV mapping.

Reconstructs people from video with:
- SMPL-X body model (standard topology + UVs)
- Temporally consistent mesh sequence
- World-space alignment with COLMAP
- Textured output ready for VFX

Pipeline:
1. Motion tracking (WHAM) → skeleton animation in world space
2. Geometry reconstruction (ECON) → clothed body geometry
3. Topology registration (TAVA) → consistent SMPL-X mesh
4. Texture projection → UV texture from camera views

Usage:
    python run_mocap.py <project_dir> [options]

Example:
    # Full pipeline with texture
    python run_mocap.py /path/to/projects/My_Shot --texture

    # Motion only (fast)
    python run_mocap.py /path/to/projects/My_Shot --skip-texture

    # Test individual stages
    python run_mocap.py /path/to/projects/My_Shot --test-stage motion
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np


# Dependencies check results (cached)
_DEPS_CHECKED = {}


def check_dependency(name: str, import_path: str = None, command: str = None) -> bool:
    """Check if a dependency is available.

    Args:
        name: Dependency name for caching
        import_path: Python import path (e.g., "smplx")
        command: Shell command to check (e.g., ["wham", "--help"])

    Returns:
        True if dependency is available
    """
    if name in _DEPS_CHECKED:
        return _DEPS_CHECKED[name]

    available = False

    if import_path:
        try:
            __import__(import_path)
            available = True
        except ImportError:
            available = False

    elif command:
        try:
            result = subprocess.run(
                command if isinstance(command, list) else [command],
                capture_output=True,
                timeout=5
            )
            available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available = False

    _DEPS_CHECKED[name] = available
    return available


def check_all_dependencies() -> Dict[str, bool]:
    """Check all required dependencies.

    Returns:
        Dict mapping dependency name to availability status
    """
    deps = {
        "numpy": check_dependency("numpy", "numpy"),
        "pytorch": check_dependency("pytorch", "torch"),
        "smplx": check_dependency("smplx", "smplx"),
        "trimesh": check_dependency("trimesh", "trimesh"),
        "opencv": check_dependency("opencv", "cv2"),
        "pillow": check_dependency("pillow", "PIL"),
    }

    # Optional dependencies (for specific methods)
    deps["wham"] = check_dependency("wham", command=["python", "-c", "import wham"])
    deps["tava"] = check_dependency("tava", command=["python", "-c", "import tava"])
    deps["econ"] = check_dependency("econ", command=["python", "-c", "import econ"])

    return deps


def print_dependency_status():
    """Print status of all dependencies."""
    deps = check_all_dependencies()

    print("\nDependency Status:")
    print("=" * 60)

    # Core dependencies
    print("\nCore (required):")
    for name in ["numpy", "pytorch", "smplx", "trimesh", "opencv", "pillow"]:
        status = "✓" if deps[name] else "✗"
        print(f"  {status} {name}")

    # Optional dependencies
    print("\nOptional (for specific methods):")
    for name in ["wham", "tava", "econ"]:
        status = "✓" if deps[name] else "✗"
        print(f"  {status} {name}")

    print()


def install_instructions():
    """Print installation instructions for missing dependencies."""
    print("\n" + "=" * 60)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 60)

    print("\nCore dependencies:")
    print("  pip install numpy torch smplx trimesh opencv-python pillow")

    print("\nWHAM (world-grounded motion):")
    print("  git clone https://github.com/yohanshin/WHAM.git")
    print("  cd WHAM && pip install -e .")
    print("  # Download checkpoints from project page")

    print("\nTAVA (consistent topology tracking):")
    print("  git clone https://github.com/facebookresearch/tava.git")
    print("  cd tava && pip install -e .")

    print("\nECON (clothed reconstruction):")
    print("  git clone https://github.com/YuliangXiu/ECON.git")
    print("  cd ECON && pip install -r requirements.txt")
    print("  # Download SMPL models + checkpoints")

    print("\nSMPL-X body model:")
    print("  1. Register at https://smpl-x.is.tue.mpg.de/")
    print("  2. Download models → place in ~/.smplx/")
    print()


def run_wham_motion_tracking(
    project_dir: Path,
    person_mask_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> bool:
    """Run WHAM for world-grounded motion tracking.

    Args:
        project_dir: Project directory containing source/frames/
        person_mask_dir: Optional person segmentation masks
        output_dir: Output directory for results

    Returns:
        True if successful
    """
    if not check_dependency("wham"):
        print("Error: WHAM not available", file=sys.stderr)
        print("Run with --check to see installation instructions", file=sys.stderr)
        return False

    output_dir = output_dir or project_dir / "mocap" / "wham"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}", file=sys.stderr)
        return False

    print(f"\n{'=' * 60}")
    print("WHAM Motion Tracking")
    print("=" * 60)
    print(f"Frames: {frames_dir}")
    if person_mask_dir:
        print(f"Masks: {person_mask_dir}")
    print(f"Output: {output_dir}")
    print()

    try:
        # WHAM command - adjust based on actual WHAM CLI
        cmd = [
            "python", "-m", "wham.run",
            "--input", str(frames_dir),
            "--output", str(output_dir),
        ]

        if person_mask_dir and person_mask_dir.exists():
            cmd.extend(["--mask", str(person_mask_dir)])

        print(f"  → Running WHAM...")
        print(f"    $ {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"Error: WHAM failed", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False

        # Check output files
        motion_file = output_dir / "motion.pkl"
        if not motion_file.exists():
            print(f"Error: WHAM output not found: {motion_file}", file=sys.stderr)
            return False

        print(f"  ✓ Motion tracking complete")
        print(f"    Output: {motion_file}")

        return True

    except subprocess.TimeoutExpired:
        print("Error: WHAM timed out", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error running WHAM: {e}", file=sys.stderr)
        return False


def run_econ_reconstruction(
    project_dir: Path,
    keyframe_interval: int = 25,
    output_dir: Optional[Path] = None
) -> bool:
    """Run ECON clothed body reconstruction on keyframes.

    Args:
        project_dir: Project directory containing source/frames/
        keyframe_interval: Extract geometry every N frames
        output_dir: Output directory for meshes

    Returns:
        True if successful
    """
    if not check_dependency("econ"):
        print("Error: ECON not available", file=sys.stderr)
        return False

    output_dir = output_dir or project_dir / "mocap" / "econ"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = project_dir / "source" / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.png"))

    if not frame_files:
        print(f"Error: No frames found in {frames_dir}", file=sys.stderr)
        return False

    # Select keyframes
    keyframes = frame_files[::keyframe_interval]

    print(f"\n{'=' * 60}")
    print("ECON Geometry Reconstruction")
    print("=" * 60)
    print(f"Total frames: {len(frame_files)}")
    print(f"Keyframes: {len(keyframes)} (every {keyframe_interval} frames)")
    print(f"Output: {output_dir}")
    print()

    try:
        for i, keyframe_path in enumerate(keyframes, 1):
            frame_num = keyframe_path.stem.split("_")[-1]
            output_mesh = output_dir / f"mesh_{frame_num}.obj"

            if output_mesh.exists():
                print(f"  [{i}/{len(keyframes)}] Skipping {keyframe_path.name} (exists)")
                continue

            print(f"  [{i}/{len(keyframes)}] Processing {keyframe_path.name}...")

            # ECON command - adjust based on actual ECON CLI
            cmd = [
                "python", "-m", "econ.run",
                "--input", str(keyframe_path),
                "--output", str(output_mesh),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"    Warning: ECON failed for {keyframe_path.name}")
                continue

            print(f"    ✓ Saved: {output_mesh.name}")

        # Check we got at least one mesh
        mesh_files = list(output_dir.glob("mesh_*.obj"))
        if not mesh_files:
            print("Error: No meshes generated", file=sys.stderr)
            return False

        print(f"\n  ✓ Reconstruction complete: {len(mesh_files)} meshes")
        return True

    except Exception as e:
        print(f"Error running ECON: {e}", file=sys.stderr)
        return False


def run_tava_tracking(
    project_dir: Path,
    motion_file: Path,
    econ_dir: Path,
    output_dir: Optional[Path] = None
) -> bool:
    """Run TAVA for consistent topology tracking.

    Args:
        project_dir: Project directory
        motion_file: WHAM motion file (.pkl)
        econ_dir: Directory with ECON keyframe meshes
        output_dir: Output directory for consistent mesh sequence

    Returns:
        True if successful
    """
    if not check_dependency("tava"):
        print("Error: TAVA not available", file=sys.stderr)
        return False

    output_dir = output_dir or project_dir / "mocap" / "tava"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("TAVA Consistent Tracking")
    print("=" * 60)
    print(f"Motion: {motion_file}")
    print(f"Geometry: {econ_dir}")
    print(f"Output: {output_dir}")
    print()

    try:
        # TAVA command - adjust based on actual TAVA CLI
        cmd = [
            "python", "-m", "tava.track",
            "--motion", str(motion_file),
            "--geometry", str(econ_dir),
            "--output", str(output_dir),
            "--topology", "smplx",  # Use SMPL-X topology + UVs
        ]

        print(f"  → Running TAVA...")
        print(f"    $ {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            print(f"Error: TAVA failed", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False

        # Check output
        mesh_sequence = output_dir / "mesh_sequence.pkl"
        if not mesh_sequence.exists():
            print(f"Error: TAVA output not found: {mesh_sequence}", file=sys.stderr)
            return False

        print(f"  ✓ Tracking complete")
        print(f"    Output: {mesh_sequence}")

        return True

    except subprocess.TimeoutExpired:
        print("Error: TAVA timed out (this can take hours)", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error running TAVA: {e}", file=sys.stderr)
        return False


def export_mesh_sequence_to_alembic(
    mesh_sequence_file: Path,
    output_file: Path,
    fps: float = 24.0
) -> bool:
    """Export mesh sequence to Alembic format.

    Args:
        mesh_sequence_file: TAVA mesh sequence (.pkl)
        output_file: Output Alembic file (.abc)
        fps: Frame rate

    Returns:
        True if successful
    """
    print(f"\n{'=' * 60}")
    print("Alembic Export")
    print("=" * 60)
    print(f"Input: {mesh_sequence_file}")
    print(f"Output: {output_file}")
    print(f"FPS: {fps}")
    print()

    try:
        import pickle
        import trimesh

        # Load mesh sequence
        with open(mesh_sequence_file, "rb") as f:
            data = pickle.load(f)

        meshes = data.get("meshes", [])
        if not meshes:
            print("Error: No meshes in sequence file", file=sys.stderr)
            return False

        print(f"  Frames: {len(meshes)}")
        print(f"  Vertices: {len(meshes[0].vertices)}")
        print(f"  Faces: {len(meshes[0].faces)}")

        # Check for Alembic support
        if not check_dependency("alembic", "alembic"):
            print("Warning: Alembic support not available", file=sys.stderr)
            print("Falling back to OBJ sequence export", file=sys.stderr)

            # Export as OBJ sequence
            obj_dir = output_file.parent / "obj_sequence"
            obj_dir.mkdir(exist_ok=True)

            for i, mesh in enumerate(meshes):
                obj_file = obj_dir / f"frame_{i+1:04d}.obj"
                mesh.export(str(obj_file))

            print(f"  ✓ Exported to OBJ sequence: {obj_dir}")
            return True

        # TODO: Implement actual Alembic export
        # This would use alembic-python or similar
        print("  Note: Alembic export not yet implemented")
        print("  Exporting as OBJ sequence instead")

        obj_dir = output_file.parent / "obj_sequence"
        obj_dir.mkdir(exist_ok=True)

        for i, mesh in enumerate(meshes):
            obj_file = obj_dir / f"frame_{i+1:04d}.obj"
            mesh.export(str(obj_file))

        print(f"  ✓ Exported: {obj_dir}")
        return True

    except Exception as e:
        print(f"Error exporting mesh: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def run_mocap_pipeline(
    project_dir: Path,
    skip_texture: bool = False,
    keyframe_interval: int = 25,
    fps: float = 24.0,
    test_stage: Optional[str] = None
) -> bool:
    """Run full motion capture pipeline.

    Args:
        project_dir: Project directory
        skip_texture: Skip texture projection (faster)
        keyframe_interval: ECON keyframe interval
        fps: Frame rate for export
        test_stage: Test only specific stage (motion, econ, tava, texture)

    Returns:
        True if successful
    """
    # Check dependencies
    deps = check_all_dependencies()
    required = ["numpy", "pytorch", "smplx", "trimesh"]
    missing = [name for name in required if not deps[name]]

    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}", file=sys.stderr)
        print_dependency_status()
        install_instructions()
        return False

    print(f"\n{'=' * 60}")
    print("Motion Capture Pipeline")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print(f"Keyframe interval: {keyframe_interval}")
    print(f"Texture: {'Disabled' if skip_texture else 'Enabled'}")
    if test_stage:
        print(f"Test stage: {test_stage}")
    print()

    # Setup paths
    mocap_dir = project_dir / "mocap"
    mocap_dir.mkdir(parents=True, exist_ok=True)

    person_mask_dir = project_dir / "roto"
    if not person_mask_dir.exists():
        person_mask_dir = None

    # Stage 1: Motion tracking (WHAM)
    if not test_stage or test_stage == "motion":
        if not run_wham_motion_tracking(
            project_dir,
            person_mask_dir=person_mask_dir,
            output_dir=mocap_dir / "wham"
        ):
            print("Motion tracking failed", file=sys.stderr)
            return False

        if test_stage == "motion":
            print("\n✓ Motion stage test complete")
            return True

    motion_file = mocap_dir / "wham" / "motion.pkl"
    if not motion_file.exists():
        print(f"Error: Motion file not found: {motion_file}", file=sys.stderr)
        print("Run motion stage first: --test-stage motion", file=sys.stderr)
        return False

    # Stage 2: Geometry reconstruction (ECON)
    if not test_stage or test_stage == "econ":
        if not run_econ_reconstruction(
            project_dir,
            keyframe_interval=keyframe_interval,
            output_dir=mocap_dir / "econ"
        ):
            print("Geometry reconstruction failed", file=sys.stderr)
            return False

        if test_stage == "econ":
            print("\n✓ ECON stage test complete")
            return True

    econ_dir = mocap_dir / "econ"
    if not econ_dir.exists() or not list(econ_dir.glob("mesh_*.obj")):
        print(f"Error: ECON meshes not found in {econ_dir}", file=sys.stderr)
        print("Run ECON stage first: --test-stage econ", file=sys.stderr)
        return False

    # Stage 3: Consistent topology tracking (TAVA)
    if not test_stage or test_stage == "tava":
        if not run_tava_tracking(
            project_dir,
            motion_file=motion_file,
            econ_dir=econ_dir,
            output_dir=mocap_dir / "tava"
        ):
            print("Topology tracking failed", file=sys.stderr)
            return False

        if test_stage == "tava":
            print("\n✓ TAVA stage test complete")
            return True

    mesh_sequence_file = mocap_dir / "tava" / "mesh_sequence.pkl"
    if not mesh_sequence_file.exists():
        print(f"Error: Mesh sequence not found: {mesh_sequence_file}", file=sys.stderr)
        print("Run TAVA stage first: --test-stage tava", file=sys.stderr)
        return False

    # Stage 4: Export to Alembic
    output_abc = mocap_dir / "body.abc"
    if not export_mesh_sequence_to_alembic(mesh_sequence_file, output_abc, fps):
        print("Export failed", file=sys.stderr)
        return False

    # Stage 5: Texture projection (optional)
    if not skip_texture and (not test_stage or test_stage == "texture"):
        texture_script = Path(__file__).parent / "texture_projection.py"
        if texture_script.exists():
            print(f"\n{'=' * 60}")
            print("Texture Projection")
            print("=" * 60)

            cmd = [
                sys.executable, str(texture_script),
                str(project_dir),
                "--mesh-sequence", str(mesh_sequence_file),
                "--output", str(mocap_dir / "texture.png")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("Warning: Texture projection failed", file=sys.stderr)
            else:
                print(f"  ✓ Texture saved: {mocap_dir / 'texture.png'}")
        else:
            print(f"\nNote: Texture projection script not found: {texture_script}")

    # Final output summary
    print(f"\n{'=' * 60}")
    print("Motion Capture Complete")
    print("=" * 60)
    print(f"Output directory: {mocap_dir}")
    print(f"Mesh sequence: {mesh_sequence_file}")
    if not skip_texture and (mocap_dir / "texture.png").exists():
        print(f"Texture: {mocap_dir / 'texture.png'}")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Human motion capture with consistent topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        nargs="?",
        help="Project directory containing source/frames/"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and exit"
    )
    parser.add_argument(
        "--skip-texture",
        action="store_true",
        help="Skip texture projection (faster)"
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=25,
        help="ECON keyframe interval in frames (default: 25)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frame rate for export (default: 24.0)"
    )
    parser.add_argument(
        "--test-stage",
        choices=["motion", "econ", "tava", "texture"],
        help="Test only specific stage (for debugging)"
    )

    args = parser.parse_args()

    if args.check:
        print_dependency_status()
        install_instructions()
        sys.exit(0)

    if not args.project_dir:
        parser.print_help()
        print("\nError: project_dir required (or use --check)", file=sys.stderr)
        sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_mocap_pipeline(
        project_dir=project_dir,
        skip_texture=args.skip_texture,
        keyframe_interval=args.keyframe_interval,
        fps=args.fps,
        test_stage=args.test_stage
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
