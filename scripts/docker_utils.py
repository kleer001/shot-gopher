"""Docker mode utilities.

Handles Docker container detection, image checks, and container execution.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig

CONTAINER_NAME = "vfx-pipeline-run"
REPO_ROOT = Path(__file__).resolve().parent.parent


def check_docker_available() -> bool:
    """Check if Docker is available and running.

    Returns:
        True if Docker is accessible
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_docker_image_exists(image_name: str = "vfx-ingest:latest") -> bool:
    """Check if a Docker image exists.

    Args:
        image_name: Image name with tag

    Returns:
        True if image exists
    """
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build_container_args(config: "PipelineConfig", container_input: str) -> list[str]:
    """Build command line arguments for container execution.

    Args:
        config: Pipeline configuration
        container_input: Path to input file inside container

    Returns:
        List of command line arguments
    """
    args = [container_input]

    if config.project_name:
        args.extend(["--name", config.project_name])

    args.extend(["--projects-dir", "/workspace/projects"])

    if config.stages:
        args.extend(["--stages", ",".join(config.stages)])

    if config.fps:
        args.extend(["--fps", str(config.fps)])

    if config.skip_existing:
        args.append("--skip-existing")

    if not config.overwrite:
        args.append("--no-overwrite")

    if config.auto_movie:
        args.append("--auto-movie")

    if not config.auto_start_comfyui:
        args.append("--no-auto-comfyui")

    args.extend(["--colmap-quality", config.colmap_quality])

    if config.colmap_dense:
        args.append("--colmap-dense")

    if config.colmap_mesh:
        args.append("--colmap-mesh")

    if not config.colmap_use_masks:
        args.append("--colmap-no-masks")

    if config.colmap_max_size > 0:
        args.extend(["--colmap-max-size", str(config.colmap_max_size)])

    args.extend(["--gsir-iterations", str(config.gsir_iterations)])

    if config.gsir_path:
        args.extend(["--gsir-path", config.gsir_path])

    args.extend(["--mocap-method", config.mocap_method])

    if config.roto_prompt:
        args.extend(["--prompt", config.roto_prompt])

    if config.roto_start_frame:
        args.extend(["--start-frame", str(config.roto_start_frame)])

    if config.separate_instances:
        args.append("--separate-instances")
    else:
        args.append("--no-separate-instances")

    args.append("--local")

    return args


def run_docker_mode(
    config: "PipelineConfig",
    models_dir: Path,
) -> int:
    """Run the pipeline in Docker mode.

    Args:
        config: Pipeline configuration
        models_dir: Path to models directory

    Returns:
        Exit code
    """
    if not config.input_path:
        print("Error: Docker mode requires an input file", file=sys.stderr)
        return 1

    input_path = config.input_path.resolve()
    projects_dir = config.projects_dir.resolve()
    models_dir = models_dir.resolve()

    project_name = config.project_name or input_path.stem.replace(" ", "_")

    print(f"\n{'='*60}")
    print("VFX Pipeline (Docker Mode)")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Project: {project_name}")
    print(f"Projects dir: {projects_dir}")
    print(f"Models: {models_dir}")

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}", file=sys.stderr)
        print("Run model download or specify --models-dir", file=sys.stderr)
        return 1

    if not check_docker_available():
        print("Error: Docker is not available or not running", file=sys.stderr)
        return 1

    if not check_docker_image_exists():
        print("Error: Docker image 'vfx-ingest:latest' not found", file=sys.stderr)
        print("Build it first: docker compose build", file=sys.stderr)
        return 1

    projects_dir.mkdir(parents=True, exist_ok=True)

    container_input = f"/workspace/input/{input_path.name}"
    forward_args = build_container_args(config, container_input)

    print(f"\nStarting Docker container...")

    def signal_handler(sig, frame):
        print("\n\nInterrupted! Cleaning up...")
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        uid = os.getuid()
        gid = os.getgid()
        cmd = [
            "docker", "run",
            "--rm",
            "--name", CONTAINER_NAME,
            "--gpus", "all",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", "START_COMFYUI=false",
            "-e", f"HOST_UID={uid}",
            "-e", f"HOST_GID={gid}",
            "-v", f"{input_path.parent}:/workspace/input:ro",
            "-v", f"{projects_dir}:/workspace/projects",
            "-v", f"{models_dir}:/models:ro",
            "-v", f"{REPO_ROOT / 'scripts'}:/app/scripts:ro",
            "-v", f"{REPO_ROOT / 'workflow_templates'}:/app/workflow_templates:ro",
            "vfx-ingest:latest",
        ] + forward_args

        print(f"Running: docker run ... {' '.join(forward_args)}")
        print()

        result = subprocess.run(cmd, cwd=REPO_ROOT)
        return result.returncode

    except Exception as e:
        print(f"Error running Docker container: {e}", file=sys.stderr)
        return 1

    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
