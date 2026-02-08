"""ComfyUI workflow manipulation utilities.

Functions for loading, modifying, and saving ComfyUI workflow JSON files.
"""

import json
import os
from pathlib import Path

from pipeline_constants import WORKFLOW_TEMPLATES_DIR
from pipeline_utils import get_image_dimensions

__all__ = [
    "get_comfyui_output_dir",
    "refresh_workflow_from_template",
    "update_segmentation_prompt",
    "update_workflow_resolution",
]


def _get_max_processing_dimensions() -> tuple[int, int]:
    """Get max processing dimensions from environment or defaults."""
    max_width = int(os.environ.get("CLEANPLATE_MAX_WIDTH", "1920"))
    max_height = int(os.environ.get("CLEANPLATE_MAX_HEIGHT", "1080"))
    return max_width, max_height


def _calculate_processing_resolution(
    source_width: int,
    source_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    """Calculate VRAM-safe processing resolution while maintaining aspect ratio.

    Args:
        source_width: Original width
        source_height: Original height
        max_width: Maximum allowed width
        max_height: Maximum allowed height

    Returns:
        Tuple of (width, height) aligned to 8-pixel boundaries
    """
    if source_height == 0 or source_width == 0:
        return 0, 0

    proc_width = min(source_width, max_width)
    proc_height = min(source_height, max_height)

    source_aspect = source_width / source_height
    proc_aspect = proc_width / proc_height

    if abs(source_aspect - proc_aspect) > 0.01:
        if source_aspect > proc_aspect:
            proc_height = int(proc_width / source_aspect)
        else:
            proc_width = int(proc_height * source_aspect)

    proc_width = (proc_width // 8) * 8
    proc_height = (proc_height // 8) * 8

    return proc_width, proc_height


def _align_to_8(value: int) -> int:
    """Align value to 8-pixel boundary."""
    return (value // 8) * 8


def get_comfyui_output_dir() -> Path:
    """Get the output directory that ComfyUI uses for SaveImage nodes.

    This must match the --output-directory argument passed to ComfyUI
    in comfyui_manager.py.

    Returns:
        Path to ComfyUI output directory
    """
    from env_config import INSTALL_DIR
    return INSTALL_DIR.parent.parent


def refresh_workflow_from_template(
    workflow_path: Path,
    template_name: str,
    project_dir: Path = None
) -> bool:
    """Refresh a project's workflow from the template if template is newer.

    Args:
        workflow_path: Path to project's workflow file
        template_name: Name of template file (e.g., "02_segmentation.json")
        project_dir: Project directory for path population (inferred if not provided)

    Returns:
        True if refreshed, False if no refresh needed
    """
    from setup_project import populate_workflow

    template_path = WORKFLOW_TEMPLATES_DIR / template_name
    if not template_path.exists():
        return False

    if project_dir is None:
        project_dir = workflow_path.parent.parent

    def copy_and_populate():
        with open(template_path) as f:
            workflow_data = json.load(f)
        populated = populate_workflow(workflow_data, project_dir)
        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(populated, f, indent=2)

    if not workflow_path.exists():
        copy_and_populate()
        print(f"  → Copied workflow from template: {template_name}")
        return True

    if template_path.stat().st_mtime > workflow_path.stat().st_mtime:
        copy_and_populate()
        print(f"  → Refreshed workflow from template: {template_name}")
        return True

    return False


def update_segmentation_prompt(
    workflow_path: Path,
    prompt: str,
    output_subdir: Path = None,
    project_dir: Path = None,
    start_frame: int = None
) -> None:
    """Update the text prompt and output path in segmentation workflow.

    Args:
        workflow_path: Path to workflow JSON
        prompt: Segmentation target text
        output_subdir: If provided, update SaveImage to write here
        project_dir: Project root for computing relative paths
        start_frame: If provided, set frame_idx and enable bidirectional propagation
    """
    print(f"  → Setting segmentation prompt: {prompt}")
    if start_frame is not None:
        print(f"  → Starting segmentation from frame {start_frame} (bidirectional)")

    with open(workflow_path) as f:
        workflow = json.load(f)

    for node in workflow.get("nodes", []):
        if node.get("type") == "SAM3Grounding":
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 2:
                widgets[1] = prompt
                node["widgets_values"] = widgets
        elif node.get("type") == "SAM3VideoSegmentation":
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 2:
                widgets[1] = prompt
                if start_frame is not None and len(widgets) >= 3:
                    widgets[2] = start_frame
                node["widgets_values"] = widgets

        elif node.get("type") == "SAM3Propagate":
            if start_frame is not None:
                widgets = node.get("widgets_values", [])
                if len(widgets) >= 3:
                    widgets[2] = "both"
                    node["widgets_values"] = widgets

        if output_subdir and node.get("type") == "SaveImage":
            widgets = node.get("widgets_values", [])
            if widgets:
                comfyui_output = get_comfyui_output_dir()
                try:
                    relative_path = output_subdir.relative_to(comfyui_output)
                except ValueError:
                    relative_path = output_subdir
                widgets[0] = str(relative_path / "mask")
                node["widgets_values"] = widgets

    with open(workflow_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)


def update_workflow_resolution(
    workflow_path: Path,
    width: int,
    height: int,
    update_loaders: bool = True,
    update_scales: bool = True,
    **kwargs,
) -> None:
    """Update resolution-dependent nodes in a ComfyUI workflow.

    Patches VHS_LoadImagesPath and ImageScale nodes to use the specified resolution.

    Args:
        workflow_path: Path to workflow JSON
        width: Source video width in pixels
        height: Source video height in pixels
        update_loaders: Update VHS_LoadImagesPath force_size values
        update_scales: Update ImageScale width/height values
    """
    if "update_propainter" in kwargs:  # BREADCRUMB: guard against old API
        raise RuntimeError(
            "BREADCRUMB: update_propainter was passed to update_workflow_resolution() "
            f"— caller still using old API. kwargs={kwargs}"
        )

    if width <= 0 or height <= 0:
        print("  → Warning: Invalid resolution, skipping workflow update")
        return

    width_8 = _align_to_8(width)
    height_8 = _align_to_8(height)

    with open(workflow_path) as f:
        workflow = json.load(f)

    nodes_updated = []

    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")
        widgets = node.get("widgets_values", [])

        if update_loaders and node_type == "VHS_LoadImagesPath":
            if len(widgets) >= 7:
                widgets[5] = width_8
                widgets[6] = height_8
                node["widgets_values"] = widgets
                nodes_updated.append(f"VHS_LoadImagesPath (force_size: {width_8}x{height_8})")

        elif update_scales and node_type == "ImageScale":
            if len(widgets) >= 3:
                widgets[1] = width_8
                widgets[2] = height_8
                node["widgets_values"] = widgets
                title = node.get("title", "ImageScale")
                nodes_updated.append(f"{title} ({width_8}x{height_8})")

    with open(workflow_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)

    if nodes_updated:
        print(f"  → Updated workflow resolution to {width}x{height}")
        for node_info in nodes_updated:
            print(f"    - {node_info}")
