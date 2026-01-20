#!/usr/bin/env python3
"""Launch interactive segmentation workflow in ComfyUI.

Prepares the project with the interactive segmentation workflow and optionally
opens ComfyUI in the browser for manual point selection.

This workflow is for complex shots where automated text prompts don't provide
sufficient control (e.g., segmenting individual legs, specific body parts).

The script auto-detects your environment:
- If local ComfyUI + SAM3 is installed → uses local mode
- If Docker image exists → uses Docker mode (starts container, opens browser, cleanup)

Usage:
    # Auto-detect mode (recommended)
    python launch_interactive_segmentation.py /path/to/projects/My_Shot

    # Force Docker mode
    python launch_interactive_segmentation.py /path/to/projects/My_Shot --docker

    # Force local mode
    python launch_interactive_segmentation.py /path/to/projects/My_Shot --local --open

Requirements:
    - Docker mode: Docker with nvidia-container-toolkit, image built
    - Local mode: ComfyUI-SAM3 extension installed, ComfyUI running
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from env_config import check_conda_env_or_warn, is_in_container, INSTALL_DIR
from workflow_utils import WORKFLOW_TEMPLATES_DIR


TEMPLATE_NAME = "05_interactive_segmentation.json"
DEFAULT_COMFYUI_URL = "http://localhost:8188"
COMFYUI_DIR = INSTALL_DIR / "ComfyUI"
CUSTOM_NODES_DIR = COMFYUI_DIR / "custom_nodes"
CONTAINER_NAME = "vfx-ingest-interactive"
REPO_ROOT = Path(__file__).resolve().parent.parent


def find_default_models_dir() -> Path:
    """Find the default models directory."""
    env_models = os.environ.get("VFX_MODELS_DIR")
    if env_models:
        return Path(env_models)
    default_path = INSTALL_DIR / "models"
    if default_path.exists():
        return default_path
    return REPO_ROOT / ".vfx_pipeline" / "models"


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_docker_image_exists() -> bool:
    """Check if the vfx-ingest Docker image exists."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "vfx-ingest:latest"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_docker_container(
    project_dir: Path,
    models_dir: Path
) -> bool:
    """Start the Docker container with appropriate volume mounts.

    Args:
        project_dir: Host path to project directory
        models_dir: Host path to models directory

    Returns:
        True if container started successfully
    """
    project_name = project_dir.name
    container_project_path = f"/workspace/projects/{project_name}"

    print(f"Starting Docker container...")
    print(f"  Project mount: {project_dir} -> {container_project_path}")
    print(f"  Models mount: {models_dir} -> /models")

    env = os.environ.copy()
    env["VFX_PROJECTS_DIR"] = str(project_dir.parent)
    env["VFX_MODELS_DIR"] = str(models_dir)

    cmd = [
        "docker", "compose",
        "-f", str(REPO_ROOT / "docker-compose.yml"),
        "run",
        "--rm",
        "--name", CONTAINER_NAME,
        "-d",
        "-p", "8188:8188",
        "-e", "START_COMFYUI=true",
        "-v", f"{project_dir.parent}:/workspace/projects",
        "-v", f"{models_dir}:/models:ro",
        "vfx-ingest",
        "sleep", "infinity"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"Error starting container: {result.stderr}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Timeout starting container", file=sys.stderr)
        return False


def wait_for_comfyui(url: str, timeout: int = 120) -> bool:
    """Wait for ComfyUI to be ready.

    Args:
        url: ComfyUI URL to check
        timeout: Maximum seconds to wait

    Returns:
        True if ComfyUI is ready
    """
    import urllib.request
    import urllib.error

    print(f"Waiting for ComfyUI to start (timeout: {timeout}s)...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"{url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                print("ComfyUI is ready!")
                return True
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
            pass
        time.sleep(2)
        elapsed = int(time.time() - start_time)
        print(f"  ...waiting ({elapsed}s)")

    print("Timeout waiting for ComfyUI", file=sys.stderr)
    return False


def prepare_workflow_in_container(project_name: str) -> bool:
    """Prepare the workflow inside the Docker container.

    Args:
        project_name: Name of the project directory

    Returns:
        True if workflow prepared successfully
    """
    container_project_path = f"/workspace/projects/{project_name}"

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python3", "/app/scripts/launch_interactive_segmentation.py",
        container_project_path,
        "--internal-prepare-only"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print(f"Error preparing workflow: {result.stderr}", file=sys.stderr)
            return False
        print(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        print("Timeout preparing workflow", file=sys.stderr)
        return False


def stop_docker_container() -> None:
    """Stop and remove the Docker container."""
    print("\nStopping Docker container...")
    try:
        subprocess.run(
            ["docker", "stop", CONTAINER_NAME],
            capture_output=True,
            timeout=30
        )
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True,
            timeout=10
        )
        print("Container stopped.")
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


def run_docker_mode(
    project_dir: Path,
    models_dir: Path,
    url: str
) -> int:
    """Run interactive segmentation in Docker mode.

    Args:
        project_dir: Path to project directory
        models_dir: Path to models directory
        url: ComfyUI URL

    Returns:
        Exit code
    """
    project_dir = project_dir.resolve()
    models_dir = models_dir.resolve()
    project_name = project_dir.name

    print(f"\n{'='*60}")
    print("Interactive Segmentation (Docker Mode)")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")
    print(f"Models: {models_dir}")

    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}", file=sys.stderr)
        print("Run the model download script first or specify --models-dir", file=sys.stderr)
        return 1

    if not check_docker_available():
        print("Error: Docker is not available or not running", file=sys.stderr)
        return 1

    if not check_docker_image_exists():
        print("Error: Docker image 'vfx-ingest:latest' not found", file=sys.stderr)
        print("Build it first: docker compose build", file=sys.stderr)
        return 1

    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
        print(f"\nWarning: No source frames found in {frames_dir}")
        print("Run the ingest stage first to extract frames from your video.")
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            return 1

    def signal_handler(sig, frame):
        print("\n\nInterrupted! Cleaning up...")
        stop_docker_container()
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if not start_docker_container(project_dir, models_dir):
            return 1

        if not wait_for_comfyui(url, timeout=120):
            stop_docker_container()
            return 1

        if not prepare_workflow_in_container(project_name):
            stop_docker_container()
            return 1

        container_workflow = f"/workspace/projects/{project_name}/workflows/{TEMPLATE_NAME}"

        print(f"\n{'='*60}")
        print("Ready for Interactive Segmentation")
        print(f"{'='*60}")
        print(f"""
ComfyUI is running at: {url}

Opening browser...

Instructions:
1. In ComfyUI, click Menu > Load
2. Navigate to: {container_workflow}
3. Click points on the image in the 'Interactive Selector' node
   - Left-click = include in mask
   - Right-click = exclude from mask
4. Click 'Queue Prompt' to run segmentation
5. Masks will be saved to: {project_dir}/roto/custom/
""")

        webbrowser.open(url)

        print("="*60)
        print("Press ENTER when you're done to stop the container...")
        print("="*60)
        input()

    finally:
        stop_docker_container()

    print(f"\nDone! Check your masks at: {project_dir / 'roto' / 'custom'}")
    return 0


def populate_workflow(workflow_data: dict, project_dir: Path) -> dict:
    """Replace placeholder paths in workflow with actual project paths."""

    def replace_in_value(value):
        if isinstance(value, str):
            relative_patterns = [
                ("source/frames/", str(project_dir / "source/frames") + "/"),
                ("source/frames", str(project_dir / "source/frames")),
                ("roto/custom/", str(project_dir / "roto/custom") + "/"),
                ("roto/custom", str(project_dir / "roto/custom")),
                ("roto/", str(project_dir / "roto") + "/"),
                ("roto", str(project_dir / "roto")),
            ]
            for pattern, replacement in relative_patterns:
                if value == pattern:
                    value = replacement
                elif value.startswith(pattern + "/"):
                    value = replacement + "/" + value[len(pattern) + 1:]
            return value
        elif isinstance(value, list):
            return [replace_in_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: replace_in_value(v) for k, v in value.items()}
        else:
            return value

    return replace_in_value(workflow_data)


def prepare_workflow(project_dir: Path) -> Path:
    """Copy and populate the interactive segmentation workflow to project.

    Args:
        project_dir: Project directory

    Returns:
        Path to the prepared workflow file
    """
    template_path = WORKFLOW_TEMPLATES_DIR / TEMPLATE_NAME
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    workflows_dir = project_dir / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    output_path = workflows_dir / TEMPLATE_NAME

    with open(template_path) as f:
        workflow_data = json.load(f)

    populated = populate_workflow(workflow_data, project_dir)

    with open(output_path, 'w') as f:
        json.dump(populated, f, indent=2)

    return output_path


def check_source_frames(project_dir: Path) -> tuple[bool, int]:
    """Check if source frames exist and count them."""
    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        return False, 0

    frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
    return len(frame_files) > 0, len(frame_files)


def check_comfyui_running(url: str) -> bool:
    """Check if ComfyUI is running at the given URL."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{url}/system_stats", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def check_sam3_installed() -> tuple[bool, Path | None]:
    """Check if ComfyUI-SAM3 extension is installed."""
    if not CUSTOM_NODES_DIR.exists():
        return False, None

    possible_names = ["ComfyUI-SAM3", "comfyui-sam3"]
    for name in possible_names:
        ext_path = CUSTOM_NODES_DIR / name
        if ext_path.exists() and ext_path.is_dir():
            return True, ext_path

    return False, None


def check_local_installation() -> bool:
    """Check if local ComfyUI installation with SAM3 is available."""
    if not COMFYUI_DIR.exists():
        return False
    sam3_installed, _ = check_sam3_installed()
    return sam3_installed


def detect_execution_mode() -> str:
    """Auto-detect the best execution mode based on available installations.

    Returns:
        'local' if local ComfyUI+SAM3 is installed
        'docker' if Docker is available with the vfx-ingest image
        'none' if neither is available
    """
    if check_local_installation():
        return "local"

    if check_docker_available() and check_docker_image_exists():
        return "docker"

    return "none"


def create_output_dirs(project_dir: Path) -> None:
    """Create output directories for custom segmentation."""
    custom_roto_dir = project_dir / "roto" / "custom"
    custom_roto_dir.mkdir(parents=True, exist_ok=True)


def run_local_mode(args) -> int:
    """Run in local mode (non-Docker)."""
    check_conda_env_or_warn()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print("Interactive Segmentation Setup")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")

    frames_exist, frame_count = check_source_frames(project_dir)
    if not frames_exist:
        print(f"\nWarning: No source frames found in {project_dir / 'source' / 'frames'}")
        print("  Run ingest step first or copy frames manually.")
        print()
    else:
        print(f"Frames: {frame_count} found")

    try:
        workflow_path = prepare_workflow(project_dir)
        print(f"\nWorkflow prepared: {workflow_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    create_output_dirs(project_dir)
    print(f"Output directory: {project_dir / 'roto' / 'custom'}")

    print(f"\n{'='*60}")
    print("Instructions")
    print(f"{'='*60}")
    print("""
1. Open ComfyUI and load the workflow:
   {workflow}

2. In the 'Interactive Selector' node:
   - Left-click to add POSITIVE points (include in mask)
   - Right-click to add NEGATIVE points (exclude from mask)
   - Each unique object needs a different object ID

3. For leg segmentation:
   - Click once on each leg with different IDs
   - Add negative points between legs if they merge
   - Add negative points on shorts/clothing to exclude

4. Click 'Queue Prompt' to run segmentation

5. Masks will be saved to:
   {output}
""".format(
        workflow=workflow_path,
        output=project_dir / "roto" / "custom"
    ))

    sam3_installed, sam3_path = check_sam3_installed()

    print(f"\n{'='*60}")
    print("Extension Status")
    print(f"{'='*60}")

    if sam3_installed:
        print(f"ComfyUI-SAM3: INSTALLED at {sam3_path}")
    else:
        print("ComfyUI-SAM3: NOT FOUND")
        print("""
Re-run the install wizard to ensure all custom nodes are installed:
  python scripts/install_wizard.py

Then restart ComfyUI.
""")

    if args.open:
        if not check_comfyui_running(args.url):
            print(f"\nWarning: ComfyUI not running at {args.url}")
            print("  Start ComfyUI first: python main.py --listen")
            print()
        else:
            print(f"\nOpening ComfyUI: {args.url}")
            webbrowser.open(args.url)
            print("\nLoad the workflow manually via: Menu > Load")
            print(f"  {workflow_path}")
    else:
        print(f"\nTo open ComfyUI automatically, run with --open flag")
        print(f"Or manually navigate to: {args.url}")

    return 0


def run_internal_prepare(project_dir: Path) -> int:
    """Internal mode: just prepare workflow (called from within container)."""
    project_dir = Path(project_dir).resolve()

    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    try:
        workflow_path = prepare_workflow(project_dir)
        print(f"Workflow prepared: {workflow_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    create_output_dirs(project_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive segmentation workflow in ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory (must have source/frames/)"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--docker", "-d",
        action="store_true",
        help="Force Docker mode (auto-detected if not specified)"
    )
    mode_group.add_argument(
        "--local", "-l",
        action="store_true",
        help="Force local mode (auto-detected if not specified)"
    )

    parser.add_argument(
        "--models-dir", "-m",
        type=Path,
        default=None,
        help="Path to models directory (Docker mode, auto-detected if not specified)"
    )
    parser.add_argument(
        "--open", "-o",
        action="store_true",
        help="Open ComfyUI in browser (local mode only, Docker always opens browser)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"ComfyUI URL (default: {DEFAULT_COMFYUI_URL})"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing workflow even if newer"
    )
    parser.add_argument(
        "--internal-prepare-only",
        action="store_true",
        help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    if args.internal_prepare_only:
        return run_internal_prepare(args.project_dir)

    if args.docker:
        mode = "docker"
    elif args.local:
        mode = "local"
    else:
        mode = detect_execution_mode()
        if mode == "none":
            print("Error: No valid execution environment detected.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Options:", file=sys.stderr)
            print("  1. Install locally: python scripts/install_wizard.py", file=sys.stderr)
            print("  2. Build Docker image: docker compose build", file=sys.stderr)
            print("", file=sys.stderr)
            print("Or force a mode with --docker or --local", file=sys.stderr)
            return 1
        print(f"Auto-detected mode: {mode}")

    if mode == "docker":
        models_dir = args.models_dir or find_default_models_dir()
        return run_docker_mode(args.project_dir, models_dir, args.url)

    return run_local_mode(args)


if __name__ == "__main__":
    sys.exit(main())
