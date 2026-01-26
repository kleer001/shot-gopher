#!/usr/bin/env python3
"""Launch interactive segmentation workflow in ComfyUI.

One command to run interactive segmentation - handles everything automatically:
starts ComfyUI (local or Docker), opens browser, waits for you, cleans up.

This workflow is for complex shots where automated text prompts don't provide
sufficient control (e.g., segmenting individual legs, specific body parts).

The script auto-detects your environment:
- If local ComfyUI is installed → starts ComfyUI, opens browser, cleanup when done
- If Docker image exists → starts container, opens browser, cleanup when done

Usage:
    # Just run it - auto-detects your setup
    python launch_interactive_segmentation.py /path/to/projects/My_Shot

    # Force a specific mode
    python launch_interactive_segmentation.py /path/to/projects/My_Shot --docker
    python launch_interactive_segmentation.py /path/to/projects/My_Shot --local

Requirements:
    - Docker mode: Docker with nvidia-container-toolkit, image built
    - Local mode: ComfyUI installed (SAM3 can be installed via ComfyUI Manager)
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
from comfyui_manager import ensure_comfyui, stop_comfyui, get_comfyui_path


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
        "-v", f"{project_dir.parent}:/workspace/projects",
        "-v", f"{models_dir}:/models:ro",
        "vfx-ingest",
        "interactive"
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
        if not check_container_running():
            print("Container stopped unexpectedly!", file=sys.stderr)
            logs = get_container_logs()
            if logs:
                print("Container logs:", file=sys.stderr)
                print(logs, file=sys.stderr)
            return False

        try:
            req = urllib.request.Request(f"{url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                print("ComfyUI is ready!")
                return True
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError,
                ConnectionResetError, OSError):
            pass
        time.sleep(2)
        elapsed = int(time.time() - start_time)
        print(f"  ...waiting ({elapsed}s)")

    print("Timeout waiting for ComfyUI", file=sys.stderr)
    logs = get_container_logs()
    if logs:
        print("Container logs:", file=sys.stderr)
        print(logs, file=sys.stderr)
    return False


def copy_workflow_to_comfyui_output(project_name: str) -> bool:
    """Copy the prepared workflow to ComfyUI's output directory for auto-loading.

    The workflow is copied as 'auto_load_workflow.json' which is automatically
    loaded by the vfx_autoload extension when ComfyUI starts. We use the output
    directory because ComfyUI serves it via HTTP at /view?filename=...&type=output.

    Args:
        project_name: Name of the project directory

    Returns:
        True if successful
    """
    container_workflow = f"/workspace/projects/{project_name}/workflows/{TEMPLATE_NAME}"
    comfyui_output = "/workspace"
    dest_file = f"{comfyui_output}/auto_load_workflow.json"

    print(f"Copying workflow: {container_workflow} -> {dest_file}")

    check_source_cmd = [
        "docker", "exec", CONTAINER_NAME,
        "grep", "-o", "source/frames[^\"]*", container_workflow
    ]
    check_result = subprocess.run(
        check_source_cmd,
        capture_output=True,
        text=True,
        timeout=10
    )
    print(f"  Source file path: {check_result.stdout.strip() or '(not found)'}")

    copy_cmd = [
        "docker", "exec", CONTAINER_NAME,
        "cp", container_workflow, dest_file
    ]

    try:
        result = subprocess.run(
            copy_cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"Failed to copy workflow: {result.stderr}", file=sys.stderr)
            return False

        print(f"  Copy successful")

        verify_cmd = [
            "docker", "exec", CONTAINER_NAME,
            "grep", "-o", "source/frames[^\"]*", dest_file
        ]
        verify_result = subprocess.run(
            verify_cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        if verify_result.stdout.strip():
            print(f"  Verified path in copied file: {verify_result.stdout.strip()}")
        else:
            print("  WARNING: Path not found in copied file!", file=sys.stderr)

        return True
    except subprocess.TimeoutExpired:
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


def check_container_running() -> bool:
    """Check if the container is still running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_container_logs(tail: int = 50) -> str:
    """Get recent container logs for debugging."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


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

        if not copy_workflow_to_comfyui_output(project_name):
            print("Warning: Could not copy workflow to ComfyUI output dir", file=sys.stderr)

        print(f"\n{'='*60}")
        print("Ready for Interactive Segmentation")
        print(f"{'='*60}")
        print(f"""
ComfyUI is running at: {url}

Opening browser - workflow will load automatically...

USAGE:
1. The workflow will load automatically (wait a moment)
2. Click points on the image in the 'Interactive Selector' node
   - Left-click = include in mask  (positive)
   - Right-click = exclude from mask (negative)
3. Click 'Queue Prompt' to run segmentation
4. Masks will be saved to: {project_dir}/roto/custom/

If the workflow doesn't load automatically:
  Menu > Load > Look in output folder for auto_load_workflow.json
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


def get_comfyui_output_dir() -> Path:
    """Get the output directory that ComfyUI uses for SaveImage nodes."""
    if is_in_container():
        return Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/workspace"))
    return INSTALL_DIR.parent.parent


def populate_workflow(workflow_data: dict, project_dir: Path) -> dict:
    """Replace placeholder paths in workflow with actual project paths.

    Input nodes (VHS_LoadImagesPath): Use absolute paths
    Output nodes (SaveImage): Use paths relative to ComfyUI output directory
    """
    comfyui_output = get_comfyui_output_dir()
    project_dir_str = str(project_dir)

    print(f"  Populating workflow paths:")
    print(f"    Project dir: {project_dir}")
    print(f"    ComfyUI output dir: {comfyui_output}")

    nodes = workflow_data.get("nodes", [])
    for node in nodes:
        node_type = node.get("type")
        node_title = node.get("title", "")
        widgets = node.get("widgets_values")

        if widgets is None or len(widgets) == 0:
            continue

        if node_type == "VHS_LoadImagesPath":
            old_path = widgets[0]
            if isinstance(old_path, str) and "source/frames" in old_path:
                new_path = str(project_dir / "source" / "frames")
                widgets[0] = new_path
                print(f"    VHS_LoadImagesPath: '{old_path}' -> '{new_path}'")

        elif node_type == "SaveImage":
            old_prefix = widgets[0]
            if isinstance(old_prefix, str) and "roto/custom" in old_prefix:
                full_path = project_dir / "roto" / "custom"
                try:
                    relative_path = full_path.relative_to(comfyui_output)
                    new_prefix = str(relative_path / "mask")
                except ValueError:
                    new_prefix = str(full_path / "mask")
                widgets[0] = new_prefix
                print(f"    SaveImage: '{old_prefix}' -> '{new_prefix}'")

    return workflow_data


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

    print(f"  Template: {template_path}")

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


def check_local_comfyui_installed() -> bool:
    """Check if local ComfyUI installation exists."""
    return get_comfyui_path() is not None


def detect_execution_mode() -> str:
    """Auto-detect the best execution mode based on available installations.

    Returns:
        'local' if local ComfyUI is installed
        'docker' if Docker is available with the vfx-ingest image
        'none' if neither is available
    """
    if check_local_comfyui_installed():
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
    print("Interactive Segmentation (Local Mode)")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")

    frames_exist, frame_count = check_source_frames(project_dir)
    if not frames_exist:
        print(f"\nWarning: No source frames found in {project_dir / 'source' / 'frames'}")
        print("Run the ingest stage first to extract frames from your video.")
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            return 1
    else:
        print(f"Frames: {frame_count} found")

    try:
        workflow_path = prepare_workflow(project_dir)
        print(f"Workflow prepared: {workflow_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    create_output_dirs(project_dir)

    sam3_installed, sam3_path = check_sam3_installed()
    if not sam3_installed:
        print(f"\n{'='*60}")
        print("WARNING: ComfyUI-SAM3 Extension Not Found")
        print(f"{'='*60}")
        print("""
The SAM3 extension is required for interactive segmentation.
You can install it via ComfyUI Manager in the browser, or run:
  python scripts/install_wizard.py
""")

    def signal_handler(sig, frame):
        print("\n\nInterrupted! Cleaning up...")
        stop_comfyui()
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    comfyui_was_started = False
    try:
        if not check_comfyui_running(args.url):
            print(f"\nStarting ComfyUI...")
            if not ensure_comfyui(url=args.url, timeout=120):
                print("Error: Failed to start ComfyUI", file=sys.stderr)
                return 1
            comfyui_was_started = True
        else:
            print(f"ComfyUI already running at {args.url}")

        print(f"\n{'='*60}")
        print("Ready for Interactive Segmentation")
        print(f"{'='*60}")
        print(f"""
ComfyUI is running at: {args.url}

Opening browser...

Instructions:
1. In ComfyUI, click Menu > Load
2. Navigate to: {workflow_path}
3. Click points on the image in the 'Interactive Selector' node
   - Left-click = include in mask
   - Right-click = exclude from mask
4. Click 'Queue Prompt' to run segmentation
5. Masks will be saved to: {project_dir}/roto/custom/
""")

        webbrowser.open(args.url)

        print("="*60)
        print("Press ENTER when you're done...")
        print("="*60)
        input()

    finally:
        if comfyui_was_started:
            print("\nStopping ComfyUI...")
            stop_comfyui()

    print(f"\nDone! Check your masks at: {project_dir / 'roto' / 'custom'}")
    return 0


def run_internal_prepare(project_dir: Path) -> int:
    """Internal mode: just prepare workflow (called from within container)."""
    project_dir = Path(project_dir).resolve()

    print(f"Preparing interactive segmentation workflow...")
    print(f"  Project: {project_dir}")
    print(f"  Container: {is_in_container()}")

    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    source_frames = project_dir / "source" / "frames"
    if source_frames.exists():
        frame_count = len(list(source_frames.glob("*.png"))) + len(list(source_frames.glob("*.jpg")))
        print(f"  Source frames: {frame_count} found in {source_frames}")
    else:
        print(f"  Warning: Source frames directory not found: {source_frames}")

    try:
        workflow_path = prepare_workflow(project_dir)
        print(f"  Output: {workflow_path}")

        with open(workflow_path) as f:
            workflow = json.load(f)

        print(f"\nVerifying populated paths:")
        for node in workflow.get("nodes", []):
            node_type = node.get("type")
            node_title = node.get("title", "")
            widgets = node.get("widgets_values", [])
            if node_type == "VHS_LoadImagesPath" and widgets:
                print(f"  ✓ Load Source Frames: {widgets[0]}")
            elif node_type == "SaveImage" and widgets:
                print(f"  ✓ Save Masks: {widgets[0]}")

        print(f"\nWorkflow ready!")

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
