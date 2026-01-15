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


def get_node_definitions(comfyui_url: str = DEFAULT_COMFYUI_URL, retries: int = 3) -> dict:
    """Get node type definitions from ComfyUI's object_info endpoint.

    Args:
        comfyui_url: ComfyUI API URL
        retries: Number of retry attempts with exponential backoff

    Returns:
        Node definitions dict, or empty dict on failure
    """
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{comfyui_url}/object_info", timeout=15) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}: {e.reason}"
            # Server error - might be temporary, retry
            if e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            return {}
        except Exception as e:
            last_error = str(e)
            time.sleep(2 ** attempt)

    if last_error:
        print(f"  Warning: Failed to fetch node definitions after {retries} attempts: {last_error}", file=sys.stderr)
    return {}


def convert_workflow_to_api_format(
    workflow: dict,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    require_node_defs: bool = False,
) -> Optional[dict]:
    """Convert ComfyUI workflow format to API format if needed.

    Workflow format (saved from UI): {"nodes": [...], "links": [...]}
    API format (for /prompt): {"1": {"class_type": "...", "inputs": {...}}, ...}

    Args:
        workflow: Workflow dict (either format)
        comfyui_url: ComfyUI API URL for fetching node definitions
        require_node_defs: If True, return None when node definitions unavailable

    Returns:
        Workflow in API format for /prompt endpoint, or None on failure
    """
    # If already in API format (no "nodes" key), return as-is
    if "nodes" not in workflow:
        return workflow

    # Workflow format detected - convert it
    print("  Converting workflow from UI format to API format...")

    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])

    # Get node definitions from ComfyUI to map widget values
    node_defs = get_node_definitions(comfyui_url)
    if not node_defs:
        print("  Warning: Could not fetch node definitions from ComfyUI", file=sys.stderr)
        print("  Widget values may not be mapped correctly", file=sys.stderr)
        if require_node_defs:
            return None

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
        if node_type in ("Note", "Reroute", "PrimitiveNode"):
            continue

        inputs = {}

        # Get node definition for widget names
        node_def = node_defs.get(node_type, {})
        input_def = node_def.get("input", {})
        required_inputs = input_def.get("required", {})
        optional_inputs = input_def.get("optional", {})

        # First, identify which inputs have connections (from the node's inputs array)
        connected_input_names = set()
        node_inputs = node.get("inputs", [])
        for inp in node_inputs:
            if inp.get("link") is not None:
                connected_input_names.add(inp.get("name"))

        # Collect widget names in order (required first, then optional)
        # Skip connection-type inputs - they're handled via links
        widget_names = []

        # Fallback widget names for common nodes when node_defs unavailable
        fallback_widgets = {
            "ImageToMask": ["channel"],
            "SaveImage": ["filename_prefix"],
            "PreviewImage": [],
            "VHS_LoadImagesPath": ["directory", "image_load_cap", "skip_first_images", "select_every_nth", "meta_batch"],
            "LoadVideoDepthAnythingModel": ["model"],
            "VideoDepthAnythingProcess": ["input_size", "max_res", "precision"],
            "VideoDepthAnythingOutput": ["colormap"],
            "ProPainterInpaint": ["width", "height", "mask_dilates", "flow_mask_dilates", "ref_stride",
                                  "neighbor_length", "subvideo_length", "raft_iter", "mode"],
        }

        if not node_def and node_type in fallback_widgets:
            widget_names = fallback_widgets[node_type]
        else:
            for name, spec in required_inputs.items():
                if isinstance(spec, list) and len(spec) > 0:
                    first = spec[0]
                    # If first element is a list, it's a dropdown (widget)
                    # If first element is a string that's ALL_CAPS, it's likely a connection type
                    if isinstance(first, list):
                        widget_names.append(name)  # Dropdown widget
                    elif isinstance(first, str):
                        # Connection types are typically ALL_CAPS custom types
                        # Widget types: INT, FLOAT, STRING, BOOLEAN, COMBO, or mixed-case
                        # COMBO is used for dropdown/combo box widgets
                        if first in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                            widget_names.append(name)
                        elif not first.isupper():
                            # Mixed case = widget type
                            widget_names.append(name)
                        # else: ALL_CAPS = connection type, skip

            for name, spec in optional_inputs.items():
                if isinstance(spec, list) and len(spec) > 0:
                    first = spec[0]
                    if isinstance(first, list):
                        widget_names.append(name)
                    elif isinstance(first, str):
                        if first in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                            widget_names.append(name)
                        elif not first.isupper():
                            widget_names.append(name)

        # Process linked inputs (connections)
        for inp in node_inputs:
            inp_name = inp.get("name")
            link_id = inp.get("link")

            if link_id is not None and link_id in link_lookup:
                src_node, src_slot = link_lookup[link_id]
                inputs[inp_name] = [str(src_node), src_slot]

        # Process widget values
        # Filter out widgets that have connections (they're not in widgets_values)
        widget_values = node.get("widgets_values", [])
        if widget_values:
            # Only map to widgets that don't have connections
            non_connected_widgets = [n for n in widget_names if n not in connected_input_names]

            for i, value in enumerate(widget_values):
                if i >= len(non_connected_widgets):
                    break

                # Skip None/null values and UI state indicators like "Disabled"
                # These consume a widget slot but shouldn't be passed to the API
                if value is None:
                    continue
                if isinstance(value, str) and value in ("Disabled", "disabled", "None", "none"):
                    continue

                inputs[non_connected_widgets[i]] = value

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
        Prompt ID for tracking, or empty string on failure
    """
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Convert workflow format to API format if needed
    api_workflow = convert_workflow_to_api_format(workflow, comfyui_url)
    if api_workflow is None:
        print("    Error: Workflow conversion failed", file=sys.stderr)
        return ""

    # Wrap workflow in prompt format
    prompt_data = {"prompt": api_workflow}

    data = json.dumps(prompt_data).encode('utf-8')
    req = urllib.request.Request(
        f"{comfyui_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("prompt_id", "")
    except urllib.error.HTTPError as e:
        # Read error response body for details
        error_body = ""
        try:
            error_body = e.read().decode()
            error_json = json.loads(error_body)
            if "error" in error_json:
                error_msg = error_json["error"].get("message", str(error_json["error"]))
                print(f"    ComfyUI error: {error_msg}", file=sys.stderr)
                if "node_errors" in error_json:
                    for node_id, node_err in error_json["node_errors"].items():
                        print(f"      Node {node_id}: {node_err}", file=sys.stderr)
        except Exception:
            if error_body:
                print(f"    ComfyUI response: {error_body[:500]}", file=sys.stderr)
        print(f"    HTTP {e.code}: {e.reason}", file=sys.stderr)
        return ""


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

    # Braille spinner characters for visual feedback
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spin_idx = 0

    while time.time() - start_time < timeout:
        elapsed = int(time.time() - start_time)

        try:
            with urllib.request.urlopen(f"{comfyui_url}/history/{prompt_id}") as response:
                history = json.loads(response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False):
                        print("\r    Done!                              ")
                        return True
                    if status.get("status_str") == "error":
                        print(f"\r    Workflow error: {status}", file=sys.stderr)
                        return False

        except urllib.error.URLError:
            pass

        # File-based progress monitoring
        file_info = ""
        if output_dir and output_dir.exists():
            current_files = len(list(output_dir.glob("*.png")))
            if current_files != last_file_count:
                last_file_count = current_files
            if current_files > 0:
                if total_frames > 0:
                    file_info = f" | {current_files}/{total_frames} frames"
                else:
                    file_info = f" | {current_files} frames"

        # Show spinner with elapsed time
        spin_char = spinner[spin_idx % len(spinner)]
        spin_idx += 1
        status_line = f"\r    {spin_char} Processing{file_info} [{elapsed}s]"
        print(status_line, end="", flush=True)

        time.sleep(check_interval)

    print(f"\r    Timeout waiting for workflow completion ({timeout}s)", file=sys.stderr)
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
