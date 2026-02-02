"""Blender integration utilities for the VFX pipeline.

This module provides functions to run Blender headless for various
export operations, particularly Alembic mesh sequence export.
"""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from typing import Optional, Generator

SCRIPTS_DIR = Path(__file__).parent


@contextmanager
def _scripts_path() -> Generator[None, None, None]:
    """Temporarily add scripts directory to sys.path."""
    scripts_dir = str(Path(__file__).parent.parent)
    already_present = scripts_dir in sys.path
    if not already_present:
        sys.path.insert(0, scripts_dir)
    try:
        yield
    finally:
        if not already_present and scripts_dir in sys.path:
            sys.path.remove(scripts_dir)


def find_blender() -> Optional[Path]:
    """Find Blender executable.

    Searches in order:
    1. Repo-local tools directory (.vfx_pipeline/tools/blender/)
    2. System PATH
    3. Platform-specific standard locations

    Returns:
        Path to Blender executable, or None if not found
    """
    with _scripts_path():
        from install_wizard.platform import PlatformManager
        return PlatformManager.find_tool("blender")


def install_blender() -> Optional[Path]:
    """Install Blender to the repo-local tools directory.

    Downloads Blender 4.2 LTS and installs to .vfx_pipeline/tools/blender/.

    Returns:
        Path to installed Blender executable, or None if installation failed
    """
    with _scripts_path():
        from install_wizard.platform import PlatformManager
        return PlatformManager.install_tool("blender")


def export_mesh_sequence_to_alembic(
    input_dir: Path,
    output_path: Path,
    fps: float = 24.0,
    start_frame: int = 1,
    blender_path: Optional[Path] = None,
    timeout: int = 3600,
) -> bool:
    """Export OBJ mesh sequence to Alembic using Blender.

    Args:
        input_dir: Directory containing OBJ files
        output_path: Output .abc file path
        fps: Frames per second
        start_frame: Starting frame number
        blender_path: Optional path to Blender executable
        timeout: Maximum time in seconds (default: 1 hour)

    Returns:
        True if export succeeded

    Raises:
        FileNotFoundError: If Blender is not installed or input_dir not found
        ValueError: If input_dir is not a directory or parameters invalid
        RuntimeError: If export fails
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if start_frame < 0:
        raise ValueError(f"Start frame must be non-negative, got {start_frame}")

    if blender_path is None:
        blender_path = find_blender()

    if blender_path is None:
        print("Blender not found. Attempting to install...")
        blender_path = install_blender()

    if blender_path is None:
        raise FileNotFoundError(
            "Blender not found and installation failed. "
            "Please install Blender manually or run the installation wizard."
        )

    script_path = SCRIPTS_DIR / "export_mesh_alembic.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Export script not found: {script_path}")

    cmd = [
        str(blender_path),
        "-b",
        "--python", str(script_path),
        "--",
        "--input", str(input_dir),
        "--output", str(output_path),
        "--fps", str(fps),
        "--start-frame", str(start_frame),
    ]

    print(f"Running Blender headless export...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}")
    print(f"  Start frame: {start_frame}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)

        if process.returncode != 0:
            print(f"Blender export failed with code {process.returncode}")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            raise RuntimeError(f"Blender export failed: {stderr}")

        if output_path.exists():
            print(f"Successfully exported: {output_path}")
            return True
        else:
            raise RuntimeError("Export completed but output file not created")

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError(f"Blender export timed out after {timeout} seconds")


def export_camera_to_alembic(
    camera_dir: Path,
    output_path: Path,
    fps: float = 24.0,
    start_frame: int = 1,
    camera_name: Optional[str] = None,
    blender_path: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Export camera animation to Alembic using Blender.

    Args:
        camera_dir: Directory containing extrinsics.json and intrinsics.json
        output_path: Output .abc file path
        fps: Frames per second
        start_frame: Starting frame number
        camera_name: Optional camera name (default: based on source)
        blender_path: Optional path to Blender executable
        timeout: Maximum time in seconds (default: 5 minutes)

    Returns:
        True if export succeeded

    Raises:
        FileNotFoundError: If Blender is not installed or camera_dir not found
        ValueError: If camera_dir is not a directory or parameters invalid
        RuntimeError: If export fails
    """
    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
    if not camera_dir.is_dir():
        raise ValueError(f"Camera path is not a directory: {camera_dir}")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if start_frame < 0:
        raise ValueError(f"Start frame must be non-negative, got {start_frame}")

    extrinsics_path = camera_dir / "extrinsics.json"
    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {extrinsics_path}")

    if blender_path is None:
        blender_path = find_blender()

    if blender_path is None:
        print("Blender not found. Attempting to install...")
        blender_path = install_blender()

    if blender_path is None:
        raise FileNotFoundError(
            "Blender not found and installation failed. "
            "Please install Blender manually or run the installation wizard."
        )

    script_path = SCRIPTS_DIR / "export_camera_alembic.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Export script not found: {script_path}")

    cmd = [
        str(blender_path),
        "-b",
        "--python", str(script_path),
        "--",
        "--input", str(camera_dir),
        "--output", str(output_path),
        "--fps", str(fps),
        "--start-frame", str(start_frame),
    ]

    if camera_name:
        cmd.extend(["--camera-name", camera_name])

    print(f"Running Blender headless camera export...")
    print(f"  Input: {camera_dir}")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}")
    print(f"  Start frame: {start_frame}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)

        if process.returncode != 0:
            print(f"Blender export failed with code {process.returncode}")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            raise RuntimeError(f"Blender camera export failed: {stderr}")

        if output_path.exists():
            print(f"Successfully exported: {output_path}")
            return True
        else:
            raise RuntimeError("Export completed but output file not created")

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError(f"Blender export timed out after {timeout} seconds")


def check_blender_available() -> tuple[bool, str]:
    """Check if Blender is available for use.

    Returns:
        Tuple of (is_available, message)
    """
    blender_path = find_blender()

    if blender_path is None:
        return False, "Blender not found. Run install wizard or install manually."

    try:
        result = subprocess.run(
            [str(blender_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            version_line = result.stdout.strip().splitlines()[0]
            return True, f"Blender available: {version_line}"
        else:
            return False, f"Blender found but version check failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Blender found but version check timed out"
    except Exception as e:
        return False, f"Blender check failed: {e}"
