#!/usr/bin/env python3
"""Launch interactive segmentation workflow in ComfyUI.

Prepares the project with the interactive segmentation workflow and optionally
opens ComfyUI in the browser for manual point selection.

This workflow is for complex shots where automated text prompts don't provide
sufficient control (e.g., segmenting individual legs, specific body parts).

Usage:
    python launch_interactive_segmentation.py <project_dir> [options]

Example:
    # Prepare workflow and print instructions
    python launch_interactive_segmentation.py /path/to/projects/My_Shot

    # Prepare and open ComfyUI in browser
    python launch_interactive_segmentation.py /path/to/projects/My_Shot --open

    # Custom ComfyUI URL
    python launch_interactive_segmentation.py /path/to/projects/My_Shot --open --url http://localhost:8188

Requirements:
    - ComfyUI-SAM3 extension installed in ComfyUI (included in standard install)
    - ComfyUI running (for --open flag)
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from env_config import check_conda_env_or_warn, is_in_container, INSTALL_DIR
from workflow_utils import WORKFLOW_TEMPLATES_DIR


TEMPLATE_NAME = "05_interactive_segmentation.json"
DEFAULT_COMFYUI_URL = "http://localhost:8188"
COMFYUI_DIR = INSTALL_DIR / "ComfyUI"
CUSTOM_NODES_DIR = COMFYUI_DIR / "custom_nodes"


def populate_workflow(workflow_data: dict, project_dir: Path) -> dict:
    """Replace placeholder paths in workflow with actual project paths."""
    project_str = str(project_dir)

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
    """Check if source frames exist and count them.

    Returns:
        Tuple of (frames_exist, frame_count)
    """
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
    """Check if ComfyUI-SAM3 extension is installed.

    Returns:
        Tuple of (is_installed, extension_path)
    """
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


def main():
    check_conda_env_or_warn()

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
        "--open", "-o",
        action="store_true",
        help="Open ComfyUI in browser after preparing workflow"
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

    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

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
        sys.exit(1)

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

    in_container = is_in_container()
    sam3_installed, sam3_path = check_sam3_installed()

    print(f"\n{'='*60}")
    print("Extension Status")
    print(f"{'='*60}")

    if sam3_installed:
        print(f"ComfyUI-SAM3: INSTALLED at {sam3_path}")
    elif in_container:
        print("ComfyUI-SAM3: NOT FOUND (unexpected in Docker)")
        print("""
The Docker image should include ComfyUI-SAM3 pre-installed.
If you're seeing this error, the Docker image may need rebuilding:
  docker compose build --no-cache
""")
    else:
        print("ComfyUI-SAM3: NOT FOUND")
        print("""
Re-run the install wizard to ensure all custom nodes are installed:
  python scripts/install_wizard.py

Then restart ComfyUI.
""")

    if in_container:
        print(f"\n{'='*60}")
        print("Docker Mode - Access from Host")
        print(f"{'='*60}")
        print(f"""
Running in Docker container. ComfyUI should be running automatically.

1. Open ComfyUI from your HOST machine's browser:
   http://localhost:8188

2. Load the workflow via: Menu > Load

3. Navigate to the workflow file:
   {workflow_path}

   Note: In Docker, the project is mounted at /workspace/projects/
   The workflow path above is the container path.
""")
    elif args.open:
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


if __name__ == "__main__":
    main()
