"""Blender integration utilities for the VFX pipeline.

This module provides functions to run Blender headless for various
export operations, particularly Alembic mesh sequence export.
"""

from pathlib import Path
import subprocess
import sys
from typing import Optional

SCRIPTS_DIR = Path(__file__).parent


def find_blender() -> Optional[Path]:
    """Find Blender executable.

    Searches in order:
    1. Repo-local tools directory (.vfx_pipeline/tools/blender/)
    2. System PATH
    3. Platform-specific standard locations

    Returns:
        Path to Blender executable, or None if not found
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from install_wizard.platform import PlatformManager
        return PlatformManager.find_tool("blender")
    finally:
        sys.path.pop(0)


def install_blender() -> Optional[Path]:
    """Install Blender to the repo-local tools directory.

    Downloads Blender 4.2 LTS and installs to .vfx_pipeline/tools/blender/.

    Returns:
        Path to installed Blender executable, or None if installation failed
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from install_wizard.platform import PlatformManager
        return PlatformManager.install_tool("blender")
    finally:
        sys.path.pop(0)


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
        FileNotFoundError: If Blender is not installed
        RuntimeError: If export fails
    """
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            print(f"Blender export failed with code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Blender export failed: {result.stderr}")

        if output_path.exists():
            print(f"Successfully exported: {output_path}")
            return True
        else:
            raise RuntimeError("Export completed but output file not created")

    except subprocess.TimeoutExpired:
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
            version_line = result.stdout.strip().split("\n")[0]
            return True, f"Blender available: {version_line}"
        else:
            return False, f"Blender found but version check failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Blender found but version check timed out"
    except Exception as e:
        return False, f"Blender check failed: {e}"
