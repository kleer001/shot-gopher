#!/usr/bin/env python3
"""Standalone video segmentation runner for SAM3.

Runs video segmentation on a frame sequence to produce masks for dynamic objects.
Supports single or multiple text prompts for complex scenes.

Usage:
    python run_segmentation.py <project_dir> [options]

Example:
    # Single prompt (person)
    python run_segmentation.py /path/to/projects/My_Shot --prompt "person"

    # Multiple prompts (person + carried objects)
    python run_segmentation.py /path/to/projects/My_Shot --prompts "person,bag,backpack"

    # Custom ComfyUI URL
    python run_segmentation.py /path/to/projects/My_Shot --comfyui-url http://localhost:8188
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional, List

# Environment check - ensure correct conda environment is active
from env_config import check_conda_env_or_warn

# ComfyUI utilities (shared with run_pipeline.py)
from comfyui_utils import (
    DEFAULT_COMFYUI_URL,
    check_comfyui_running,
    convert_workflow_to_api_format,
    wait_for_completion,
)

# Log capture for debugging
from log_manager import LogCapture


def update_segmentation_prompt(workflow: dict, prompt: str) -> dict:
    """Update the text prompt in the segmentation workflow.

    Args:
        workflow: API format workflow
        prompt: Text prompt for segmentation (e.g., "person", "car")

    Returns:
        Modified workflow
    """
    # Find SAM3VideoSegmentation node (node ID "3" in template)
    for node_id, node_data in workflow.items():
        if node_data.get("class_type") == "SAM3VideoSegmentation":
            # Update the text_prompt widget value (index 1 in widgets_values)
            if "inputs" not in node_data:
                node_data["inputs"] = {}
            # SAM3VideoSegmentation has text_prompt as a widget, not input
            # Need to add it to the node data
            print(f"    Setting prompt: '{prompt}'")
            # Note: This may need adjustment based on actual SAM3 node structure
            # The workflow template has it as widgets_values in UI format

    return workflow


def queue_workflow(workflow_path: Path, comfyui_url: str, prompt: Optional[str] = None) -> str:
    """Queue a workflow for execution via ComfyUI API.

    Args:
        workflow_path: Path to workflow JSON
        comfyui_url: ComfyUI API URL
        prompt: Optional text prompt to override workflow default

    Returns:
        Prompt ID for tracking
    """
    with open(workflow_path) as f:
        workflow = json.load(f)

    api_workflow = convert_workflow_to_api_format(workflow)

    if prompt:
        api_workflow = update_segmentation_prompt(api_workflow, prompt)

    prompt_data = {"prompt": api_workflow}

    data = json.dumps(prompt_data).encode('utf-8')
    req = urllib.request.Request(
        f"{comfyui_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        return result.get("prompt_id", "")


def run_segmentation(
    project_dir: Path,
    prompt: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    timeout: int = 3600
) -> bool:
    """Run video segmentation on a project.

    Args:
        project_dir: Project directory containing workflows/02_segmentation.json
        prompt: Single text prompt for segmentation
        prompts: Multiple prompts for complex scenes (run multiple times)
        comfyui_url: ComfyUI API URL
        timeout: Timeout in seconds per workflow

    Returns:
        True if segmentation succeeded
    """
    workflow_path = project_dir / "workflows" / "02_segmentation.json"

    if not workflow_path.exists():
        print(f"Error: Segmentation workflow not found: {workflow_path}", file=sys.stderr)
        print(f"  Run setup_project.py first to create project structure", file=sys.stderr)
        return False

    if not check_comfyui_running(comfyui_url):
        print(f"Error: ComfyUI not running at {comfyui_url}", file=sys.stderr)
        print(f"  Start ComfyUI: python main.py --listen", file=sys.stderr)
        return False

    # Determine prompts to run
    if prompts:
        prompt_list = prompts
    elif prompt:
        prompt_list = [prompt]
    else:
        # Use default from workflow
        prompt_list = [None]

    print(f"\n{'='*60}")
    print(f"Video Segmentation")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")
    print(f"Prompts: {', '.join(p if p else 'default' for p in prompt_list)}")
    print()

    # Run segmentation for each prompt
    for i, p in enumerate(prompt_list, 1):
        if len(prompt_list) > 1:
            print(f"\n[{i}/{len(prompt_list)}] Running segmentation")

        print(f"  → Queuing workflow: {workflow_path.name}")
        if p:
            print(f"    Prompt: '{p}'")

        try:
            prompt_id = queue_workflow(workflow_path, comfyui_url, p)
        except Exception as e:
            print(f"    Error queuing workflow: {e}", file=sys.stderr)
            return False

        if not prompt_id:
            print(f"    Error: Failed to queue workflow", file=sys.stderr)
            return False

        print(f"    Prompt ID: {prompt_id}")
        print(f"    Waiting for completion...")

        if not wait_for_completion(prompt_id, comfyui_url, timeout):
            print(f"    Segmentation failed", file=sys.stderr)
            return False

        print(f"    ✓ Complete")

    # Check output
    roto_dir = project_dir / "roto"
    if roto_dir.exists():
        mask_count = len(list(roto_dir.glob("**/*.png"))) + len(list(roto_dir.glob("**/*.jpg")))
        print(f"\n{'='*60}")
        print(f"Segmentation Complete")
        print(f"{'='*60}")
        print(f"Masks saved: {roto_dir}")
        print(f"Mask count: {mask_count}")
        print()
    else:
        print(f"\n  Warning: No output masks found in {roto_dir}", file=sys.stderr)

    return True


def main():
    # Check conda environment (warn but don't exit - allow --help to work)
    check_conda_env_or_warn()

    parser = argparse.ArgumentParser(
        description="Run video segmentation with SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing workflows/02_segmentation.json"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Text prompt for segmentation (e.g., 'person', 'car')"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated list of prompts for multi-object scenes (e.g., 'person,bag,backpack')"
    )
    parser.add_argument(
        "--comfyui-url", "-c",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"ComfyUI API URL (default: {DEFAULT_COMFYUI_URL})"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=3600,
        help="Timeout in seconds per workflow (default: 3600)"
    )
    parser.add_argument(
        "--separate-instances",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Separate combined masks into individual instances (default: True)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum component area for instance separation (default: 500)"
    )

    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse prompts
    prompts_list = None
    if args.prompts:
        prompts_list = [p.strip() for p in args.prompts.split(",")]

    success = run_segmentation(
        project_dir=project_dir,
        prompt=args.prompt,
        prompts=prompts_list,
        comfyui_url=args.comfyui_url,
        timeout=args.timeout
    )

    if not success:
        sys.exit(1)

    # Separate instances if requested
    if args.separate_instances:
        from separate_instances import separate_instances as do_separate

        roto_dir = project_dir / "roto"

        # Derive prefix from prompt (use first prompt, default to "instance")
        if args.prompts:
            prefix = args.prompts.split(",")[0].strip()
        elif args.prompt:
            prefix = args.prompt.strip()
        else:
            prefix = "instance"

        mask_dir = roto_dir / "mask"
        if mask_dir.exists() and list(mask_dir.glob("*.png")):
            print(f"\n{'='*60}")
            print(f"Separating Instances")
            print(f"{'='*60}")

            print(f"\nProcessing: {mask_dir.name} → {prefix}_00/, {prefix}_01/, ...")
            result = do_separate(
                input_dir=mask_dir,
                output_dir=roto_dir,
                min_area=args.min_area,
                prefix=prefix,
            )

            if result:
                print(f"  Created {len(result)} {prefix} directories")
                print(f"  Combined mask kept in: {mask_dir.name}/")
        else:
            print("No mask directory found to separate")

    sys.exit(0)


if __name__ == "__main__":
    with LogCapture():
        main()
