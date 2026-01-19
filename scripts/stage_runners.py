"""Stage runner functions.

Thin wrappers that invoke external pipeline scripts for specific stages.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from pipeline_utils import run_command

__all__ = [
    "run_export_camera",
    "run_colmap_reconstruction",
    "export_camera_to_vfx_formats",
    "run_mocap",
    "run_gsir_materials",
    "setup_project",
]


def run_export_camera(project_dir: Path, fps: float = 24.0) -> bool:
    """Run camera export script.

    Args:
        project_dir: Project directory with camera data
        fps: Frames per second

    Returns:
        True if export succeeded
    """
    script_path = Path(__file__).parent / "export_camera.py"

    if not script_path.exists():
        print("    Error: export_camera.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--fps", str(fps),
        "--format", "both"
    ]

    try:
        run_command(cmd, "Exporting camera data")
        return True
    except subprocess.CalledProcessError:
        return False


def run_colmap_reconstruction(
    project_dir: Path,
    quality: str = "medium",
    run_dense: bool = False,
    run_mesh: bool = False,
    use_masks: bool = True
) -> bool:
    """Run COLMAP Structure-from-Motion reconstruction.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high')
        run_dense: Whether to run dense reconstruction
        run_mesh: Whether to generate mesh
        use_masks: If True, use segmentation masks from roto/ (if available)

    Returns:
        True if reconstruction succeeded
    """
    script_path = Path(__file__).parent / "run_colmap.py"

    if not script_path.exists():
        print("    Error: run_colmap.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--quality", quality,
    ]

    if run_dense:
        cmd.append("--dense")
    if run_mesh:
        cmd.append("--mesh")
    if not use_masks:
        cmd.append("--no-masks")

    try:
        run_command(cmd, "Running COLMAP reconstruction")
        return True
    except subprocess.CalledProcessError:
        return False


def export_camera_to_vfx_formats(
    project_dir: Path,
    start_frame: int = 1,
    fps: float = 24.0,
) -> bool:
    """Export camera data to VFX-friendly formats.

    Exports to .chan (Nuke), .csv, .clip (Houdini), .camera.json.
    Also exports .abc if PyAlembic is available.

    Args:
        project_dir: Project directory with camera/ subfolder
        start_frame: Starting frame number
        fps: Frames per second

    Returns:
        True if export succeeded
    """
    script_path = Path(__file__).parent / "export_camera.py"

    if not script_path.exists():
        print("    Error: export_camera.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--format", "all",
        "--start-frame", str(start_frame),
        "--fps", str(fps),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if 'Exported' in line or 'formats:' in line or '.chan' in line or '.clip' in line:
                    print(f"    {line}")
            return True
        else:
            print(f"    Camera export failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"    Camera export failed: {e}", file=sys.stderr)
        return False


def run_mocap(
    project_dir: Path,
    skip_texture: bool = False,
) -> bool:
    """Run human motion capture with WHAM.

    Args:
        project_dir: Project directory with frames and camera data
        skip_texture: Skip texture projection (faster)

    Returns:
        True if mocap succeeded
    """
    script_path = Path(__file__).parent / "run_mocap.py"

    if not script_path.exists():
        print("    Error: run_mocap.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
    ]

    if skip_texture:
        cmd.append("--skip-texture")

    try:
        run_command(cmd, "Running motion capture")
        return True
    except subprocess.CalledProcessError:
        return False


def run_gsir_materials(
    project_dir: Path,
    iterations_stage1: int = 30000,
    iterations_stage2: int = 35000,
    gsir_path: Optional[str] = None
) -> bool:
    """Run GS-IR material decomposition.

    Args:
        project_dir: Project directory with COLMAP output
        iterations_stage1: Training iterations for stage 1
        iterations_stage2: Total training iterations
        gsir_path: Path to GS-IR installation

    Returns:
        True if material decomposition succeeded
    """
    script_path = Path(__file__).parent / "run_gsir.py"

    if not script_path.exists():
        print("    Error: run_gsir.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--iterations-stage1", str(iterations_stage1),
        "--iterations-stage2", str(iterations_stage2),
    ]

    if gsir_path:
        cmd.extend(["--gsir-path", gsir_path])

    try:
        run_command(cmd, "Running GS-IR material decomposition")
        return True
    except subprocess.CalledProcessError:
        return False


def setup_project(
    project_dir: Path,
    workflows_dir: Path
) -> bool:
    """Set up project structure and populate workflows.

    Args:
        project_dir: Project directory to set up
        workflows_dir: Directory containing workflow templates

    Returns:
        True if setup succeeded
    """
    script_path = Path(__file__).parent / "setup_project.py"

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--workflows-dir", str(workflows_dir)
    ]

    try:
        run_command(cmd, "Setting up project structure")
        return True
    except subprocess.CalledProcessError:
        return False
