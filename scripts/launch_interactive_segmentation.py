#!/usr/bin/env python3
"""Launch interactive segmentation workflow in ComfyUI.

One command to run interactive segmentation - handles everything automatically:
starts ComfyUI, opens browser, waits for you, cleans up.

This workflow is for complex shots where automated text prompts don't provide
sufficient control (e.g., segmenting individual legs, specific body parts).

Usage:
    python launch_interactive_segmentation.py /path/to/projects/My_Shot

Requirements:
    - ComfyUI installed (SAM3 can be installed via ComfyUI Manager)
"""

import argparse
import json
import signal
import sys
import webbrowser
from pathlib import Path

from env_config import check_conda_env_or_warn, INSTALL_DIR
from workflow_utils import WORKFLOW_TEMPLATES_DIR
from comfyui_manager import ensure_comfyui, stop_comfyui


TEMPLATE_NAME = "05_interactive_segmentation.json"
DEFAULT_COMFYUI_URL = "http://localhost:8188"
COMFYUI_DIR = INSTALL_DIR / "ComfyUI"
CUSTOM_NODES_DIR = COMFYUI_DIR / "custom_nodes"
REPO_ROOT = Path(__file__).resolve().parent.parent


def check_comfyui_installed() -> bool:
    """Check if local ComfyUI installation exists."""
    return COMFYUI_DIR.exists() and (COMFYUI_DIR / "main.py").exists()


def get_comfyui_output_dir() -> Path:
    """Get the output directory that ComfyUI uses for SaveImage nodes."""
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
                new_prefix = f"projects/{project_dir.name}/roto/custom/mask"
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


def create_output_dirs(project_dir: Path) -> None:
    """Create output directories for custom segmentation."""
    custom_roto_dir = project_dir / "roto" / "custom"
    custom_roto_dir.mkdir(parents=True, exist_ok=True)


def run_local_mode(args) -> int:
    """Run interactive segmentation in local ComfyUI."""
    check_conda_env_or_warn()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print("Interactive Segmentation")
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
    parser.add_argument(
        "--url", "-u",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"ComfyUI URL (default: {DEFAULT_COMFYUI_URL})"
    )

    args = parser.parse_args()

    if not check_comfyui_installed():
        print("Error: ComfyUI not installed.", file=sys.stderr)
        print("Install with: python scripts/install_wizard.py", file=sys.stderr)
        return 1

    return run_local_mode(args)


if __name__ == "__main__":
    sys.exit(main())
