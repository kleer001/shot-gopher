"""Shared utilities for interacting with ComfyUI API.

Provides functions for workflow queuing, monitoring, and execution.
Used by run_pipeline.py, run_segmentation.py, and other scripts.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"


def check_comfyui_running(url: str = DEFAULT_COMFYUI_URL) -> bool:
    """Check if ComfyUI server is accessible."""
    try:
        urllib.request.urlopen(f"{url}/system_stats", timeout=5)
        return True
    except (urllib.error.URLError, TimeoutError):
        return False


def convert_workflow_to_api_format(workflow: dict) -> dict:
    """Convert ComfyUI workflow format to API format if needed.

    Workflow format (saved from UI): {"nodes": [...], "links": [...]}
    API format (for /prompt): {"1": {"class_type": "...", "inputs": {...}}, ...}

    Note: Full conversion from workflow format requires node type definitions.
    Workflow templates should be saved in API format directly.

    Args:
        workflow: Workflow dict (either format)

    Returns:
        Workflow in API format for /prompt endpoint
    """
    # If already in API format (no "nodes" key), return as-is
    if "nodes" not in workflow:
        return workflow

    # Workflow format detected - attempt basic conversion
    print("  Warning: Workflow in UI format, attempting conversion...", file=sys.stderr)
    print("  For reliable execution, save workflows in API format", file=sys.stderr)

    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])

    # Build link lookup: link_id -> (source_node_id, source_slot)
    link_lookup = {}
    for link in links:
        # Link format: [link_id, source_node_id, source_slot, dest_node_id, dest_slot, type]
        if len(link) >= 5:
            link_id, src_node, src_slot, dst_node, dst_slot = link[:5]
            link_lookup[link_id] = (src_node, src_slot)

    api_workflow = {}

    for node in nodes:
        node_id = str(node.get("id"))
        node_type = node.get("type")

        # Skip special nodes that don't execute
        if node_type in ("Note", "Reroute"):
            continue

        inputs = {}

        # Process linked inputs only
        node_inputs = node.get("inputs", [])
        for inp in node_inputs:
            inp_name = inp.get("name")
            link_id = inp.get("link")

            if link_id is not None and link_id in link_lookup:
                src_node, src_slot = link_lookup[link_id]
                inputs[inp_name] = [str(src_node), src_slot]

        api_workflow[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }

        if node.get("title"):
            api_workflow[node_id]["_meta"] = {"title": node.get("title")}

    return api_workflow


def queue_workflow(workflow_path: Path, comfyui_url: str = DEFAULT_COMFYUI_URL) -> str:
    """Queue a workflow for execution via ComfyUI API.

    Args:
        workflow_path: Path to workflow JSON file
        comfyui_url: ComfyUI API URL

    Returns:
        Prompt ID for tracking
    """
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Convert workflow format to API format if needed
    api_workflow = convert_workflow_to_api_format(workflow)

    # Wrap workflow in prompt format
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


def wait_for_completion(
    prompt_id: str,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    timeout: int = 3600,
    output_dir: Optional[Path] = None,
    total_frames: int = 0,
    stage_name: str = "",
) -> bool:
    """Wait for a queued workflow to complete.

    Args:
        prompt_id: The prompt ID returned from queue_workflow
        comfyui_url: ComfyUI API URL
        timeout: Maximum wait time in seconds
        output_dir: If provided, monitor this directory for progress (file count)
        total_frames: Expected total frames for progress calculation
        stage_name: Stage name for progress output

    Returns:
        True if completed successfully, False otherwise
    """
    start_time = time.time()
    check_interval = 2  # seconds
    last_file_count = 0

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{comfyui_url}/history/{prompt_id}") as response:
                history = json.loads(response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False):
                        return True
                    if status.get("status_str") == "error":
                        print(f"    Workflow error: {status}", file=sys.stderr)
                        return False

        except urllib.error.URLError:
            pass

        # File-based progress monitoring
        if output_dir and output_dir.exists():
            current_files = len(list(output_dir.glob("*.png")))
            if current_files != last_file_count:
                last_file_count = current_files
                if total_frames > 0:
                    print(f"[ComfyUI] {stage_name} frame {current_files}/{total_frames}")
                else:
                    print(f"[ComfyUI] {stage_name} frame {current_files}")
                sys.stdout.flush()

        # Also print elapsed time periodically
        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 30 == 0:
            print(f"    Elapsed: {elapsed}s")
            sys.stdout.flush()

        time.sleep(check_interval)

    print("    Timeout waiting for workflow completion", file=sys.stderr)
    return False


def run_comfyui_workflow(
    workflow_path: Path,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    wait: bool = True,
    timeout: int = 3600,
    output_dir: Optional[Path] = None,
    total_frames: int = 0,
    stage_name: str = "",
) -> bool:
    """Run a ComfyUI workflow via API.

    Args:
        workflow_path: Path to workflow JSON file
        comfyui_url: ComfyUI API URL
        wait: Whether to wait for completion
        timeout: Maximum wait time in seconds
        output_dir: If provided, monitor this directory for progress (file count)
        total_frames: Expected total frames for progress calculation
        stage_name: Stage name for progress output

    Returns:
        True if successful
    """
    if not check_comfyui_running(comfyui_url):
        print(f"    Error: ComfyUI not running at {comfyui_url}", file=sys.stderr)
        print("    Start ComfyUI first: python main.py --listen", file=sys.stderr)
        return False

    print(f"  → Queuing workflow: {workflow_path.name}")
    prompt_id = queue_workflow(workflow_path, comfyui_url)

    if not prompt_id:
        print("    Error: Failed to queue workflow", file=sys.stderr)
        return False

    print(f"    Prompt ID: {prompt_id}")

    if wait:
        print("    Waiting for completion...")
        return wait_for_completion(
            prompt_id, comfyui_url, timeout,
            output_dir=output_dir,
            total_frames=total_frames,
            stage_name=stage_name,
        )

    return True
