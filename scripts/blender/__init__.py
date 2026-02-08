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


def _ensure_blender(blender_path: Optional[Path] = None) -> Path:
    """Ensure Blender is available, installing if necessary.

    Args:
        blender_path: Optional explicit path to Blender

    Returns:
        Path to Blender executable

    Raises:
        FileNotFoundError: If Blender cannot be found or installed
    """
    if blender_path is not None:
        return blender_path

    blender_path = find_blender()

    if blender_path is None:
        print("Blender not found. Attempting to install...")
        blender_path = install_blender()

    if blender_path is None:
        raise FileNotFoundError(
            "Blender not found and installation failed. "
            "Please install Blender manually or run the installation wizard."
        )

    return blender_path


def _run_blender_script(
    cmd: list[str],
    output_path: Path,
    timeout: int,
    format_name: str,
) -> bool:
    """Run a Blender script via subprocess with standard error handling.

    Args:
        cmd: Command list to execute
        output_path: Expected output file path
        timeout: Maximum time in seconds
        format_name: Format name for error messages

    Returns:
        True if export succeeded

    Raises:
        RuntimeError: If export fails or times out
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)

        if stdout:
            print(stdout)
        if stderr:
            print(f"Blender stderr: {stderr}")

        if process.returncode != 0:
            print(f"Blender export failed with code {process.returncode}")
            raise RuntimeError(f"Blender {format_name} export failed: {stderr}")

        if output_path.exists():
            print(f"Successfully exported: {output_path}")
            return True
        else:
            raise RuntimeError("Export completed but output file not created")

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError(f"Blender export timed out after {timeout} seconds")


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

    blender_path = _ensure_blender(blender_path)

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

    return _run_blender_script(cmd, output_path, timeout, "mesh")


def _export_camera(
    camera_dir: Path,
    output_path: Path,
    script_name: str,
    format_name: str,
    fps: float = 24.0,
    start_frame: int = 1,
    camera_name: Optional[str] = None,
    blender_path: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Export camera animation using Blender (shared implementation).

    Args:
        camera_dir: Directory containing extrinsics.json and intrinsics.json
        output_path: Output file path
        script_name: Name of the export script (e.g., "export_camera_alembic.py")
        format_name: Format name for messages (e.g., "Alembic", "USD")
        fps: Frames per second
        start_frame: Starting frame number
        camera_name: Optional camera name (default: based on source)
        blender_path: Optional path to Blender executable
        timeout: Maximum time in seconds

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

    blender_path = _ensure_blender(blender_path)

    script_path = SCRIPTS_DIR / script_name
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

    print(f"Running Blender headless {format_name} camera export...")
    print(f"  Input: {camera_dir}")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}")
    print(f"  Start frame: {start_frame}")

    return _run_blender_script(cmd, output_path, timeout, format_name)


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
    return _export_camera(
        camera_dir=camera_dir,
        output_path=output_path,
        script_name="export_camera_alembic.py",
        format_name="Alembic",
        fps=fps,
        start_frame=start_frame,
        camera_name=camera_name,
        blender_path=blender_path,
        timeout=timeout,
    )


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


def export_camera_to_usd(
    camera_dir: Path,
    output_path: Path,
    fps: float = 24.0,
    start_frame: int = 1,
    camera_name: Optional[str] = None,
    blender_path: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Export camera animation to USD using Blender.

    Args:
        camera_dir: Directory containing extrinsics.json and intrinsics.json
        output_path: Output .usd/.usda/.usdc file path
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
    return _export_camera(
        camera_dir=camera_dir,
        output_path=output_path,
        script_name="export_camera_usd.py",
        format_name="USD",
        fps=fps,
        start_frame=start_frame,
        camera_name=camera_name,
        blender_path=blender_path,
        timeout=timeout,
    )


def export_mesh_sequence_to_usd(
    input_dir: Path,
    output_path: Path,
    fps: float = 24.0,
    start_frame: int = 1,
    blender_path: Optional[Path] = None,
    timeout: int = 3600,
) -> bool:
    """Export OBJ mesh sequence to USD using Blender.

    Args:
        input_dir: Directory containing OBJ files
        output_path: Output .usd file path
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

    blender_path = _ensure_blender(blender_path)

    script_path = SCRIPTS_DIR / "export_mesh_usd.py"
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

    print(f"Running Blender headless USD export...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}")
    print(f"  Start frame: {start_frame}")

    return _run_blender_script(cmd, output_path, timeout, "USD mesh")


def export_gsir_materials_to_usd(
    camera_dir: Path,
    output_path: Path,
    material_name: str = "gsir_material",
    create_geometry: bool = False,
    export_textures: bool = True,
    blender_path: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Export GS-IR materials to USD using Blender.

    Creates a USD file with PBR materials from GS-IR output including:
    - Albedo/base color texture
    - Roughness texture
    - Metallic texture
    - Normal map
    - Environment map (as emission material)

    Args:
        camera_dir: Directory containing GS-IR outputs (materials/, normals/, etc.)
        output_path: Output .usd/.usda/.usdc file path
        material_name: Name for the PBR material
        create_geometry: Create geometry (card for materials, dome for environment)
        export_textures: Copy textures alongside USD file
        blender_path: Optional path to Blender executable
        timeout: Maximum time in seconds (default: 5 minutes)

    Returns:
        True if export succeeded

    Raises:
        FileNotFoundError: If Blender is not installed or camera_dir not found
        ValueError: If camera_dir is not a directory
        RuntimeError: If export fails
    """
    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
    if not camera_dir.is_dir():
        raise ValueError(f"Camera path is not a directory: {camera_dir}")

    materials_dir = camera_dir / "materials"
    if not materials_dir.exists():
        raise FileNotFoundError(f"Materials directory not found: {materials_dir}")

    blender_path = _ensure_blender(blender_path)

    script_path = SCRIPTS_DIR / "export_gsir_usd.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Export script not found: {script_path}")

    cmd = [
        str(blender_path),
        "-b",
        "--python", str(script_path),
        "--",
        "--input", str(camera_dir),
        "--output", str(output_path),
        "--material-name", material_name,
    ]

    if create_geometry:
        cmd.append("--create-geometry")

    if not export_textures:
        cmd.append("--no-textures")

    print(f"Running Blender headless USD export...")
    print(f"  Input: {camera_dir}")
    print(f"  Output: {output_path}")
    print(f"  Material: {material_name}")

    return _run_blender_script(cmd, output_path, timeout, "USD material")


def export_ply_to_alembic(
    input_path: Path,
    output_path: Path,
    fps: float = 24.0,
    blender_path: Optional[Path] = None,
    timeout: int = 600,
) -> bool:
    """Export a PLY file to Alembic using Blender.

    Args:
        input_path: Input PLY file path
        output_path: Output .abc file path
        fps: Frames per second
        blender_path: Optional path to Blender executable
        timeout: Maximum time in seconds (default: 10 minutes)

    Returns:
        True if export succeeded

    Raises:
        FileNotFoundError: If Blender is not installed or input file not found
        RuntimeError: If export fails
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")

    blender_path = _ensure_blender(blender_path)

    script_path = SCRIPTS_DIR / "export_ply_alembic.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Export script not found: {script_path}")

    cmd = [
        str(blender_path),
        "-b",
        "--python", str(script_path),
        "--",
        "--input", str(input_path),
        "--output", str(output_path),
        "--fps", str(fps),
    ]

    print(f"Running Blender headless PLY → Alembic export...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    return _run_blender_script(cmd, output_path, timeout, "PLY Alembic")


def export_ply_to_usd(
    input_path: Path,
    output_path: Path,
    fps: float = 24.0,
    blender_path: Optional[Path] = None,
    timeout: int = 600,
) -> bool:
    """Export a PLY file to USD using Blender.

    Args:
        input_path: Input PLY file path
        output_path: Output .usd file path
        fps: Frames per second
        blender_path: Optional path to Blender executable
        timeout: Maximum time in seconds (default: 10 minutes)

    Returns:
        True if export succeeded

    Raises:
        FileNotFoundError: If Blender is not installed or input file not found
        RuntimeError: If export fails
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")

    blender_path = _ensure_blender(blender_path)

    script_path = SCRIPTS_DIR / "export_ply_usd.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Export script not found: {script_path}")

    cmd = [
        str(blender_path),
        "-b",
        "--python", str(script_path),
        "--",
        "--input", str(input_path),
        "--output", str(output_path),
        "--fps", str(fps),
    ]

    print(f"Running Blender headless PLY → USD export...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    return _run_blender_script(cmd, output_path, timeout, "PLY USD")
