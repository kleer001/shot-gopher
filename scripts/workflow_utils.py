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
    "update_cleanplate_resolution",
]


def get_comfyui_output_dir() -> Path:
    """Get the output directory that ComfyUI uses for SaveImage nodes.

    This must match the --output-directory argument passed to ComfyUI
    in comfyui_manager.py.

    Returns:
        Path to ComfyUI output directory
    """
    from env_config import INSTALL_DIR, is_in_container
    if is_in_container():
        return Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/workspace"))
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
        with open(workflow_path, 'w') as f:
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

    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)


def update_cleanplate_resolution(
    workflow_path: Path,
    source_frames_dir: Path,
    max_processing_width: int = None,
    max_processing_height: int = None
) -> None:
    """Update ProPainterInpaint internal processing resolution in cleanplate workflow.

    ProPainter's width/height control the INTERNAL optical flow processing resolution,
    not the output resolution. The output is automatically scaled to match input.
    Higher resolution = better quality but more VRAM. For 24GB VRAM with 180 frames,
    960x540 is typically safe. For 16GB or less, use 640x360.

    Args:
        workflow_path: Path to workflow JSON
        source_frames_dir: Directory containing source frames
        max_processing_width: Max internal width (default: from CLEANPLATE_MAX_WIDTH env or 1920)
        max_processing_height: Max internal height (default: from CLEANPLATE_MAX_HEIGHT env or 1080)
    """
    if max_processing_width is None:
        max_processing_width = int(os.environ.get("CLEANPLATE_MAX_WIDTH", "1920"))
    if max_processing_height is None:
        max_processing_height = int(os.environ.get("CLEANPLATE_MAX_HEIGHT", "1080"))

    frames = sorted(source_frames_dir.glob("*.png"))
    if not frames:
        frames = sorted(source_frames_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(source_frames_dir.glob("*.exr"))
    if not frames:
        print("  → Warning: No source frames found, using default resolution")
        return

    source_width, source_height = get_image_dimensions(frames[0])

    proc_width = min(source_width, max_processing_width)
    proc_height = min(source_height, max_processing_height)

    source_aspect = source_width / source_height
    proc_aspect = proc_width / proc_height

    if abs(source_aspect - proc_aspect) > 0.01:
        if source_aspect > proc_aspect:
            proc_height = int(proc_width / source_aspect)
        else:
            proc_width = int(proc_height * source_aspect)

    proc_width = (proc_width // 8) * 8
    proc_height = (proc_height // 8) * 8

    if proc_width < source_width or proc_height < source_height:
        print(f"  → Source resolution: {source_width}x{source_height}")
        print(f"  → ProPainter internal processing: {proc_width}x{proc_height} (capped for VRAM)")
    else:
        print(f"  → Setting cleanplate resolution to {proc_width}x{proc_height}")

    with open(workflow_path) as f:
        workflow = json.load(f)

    for node in workflow.get("nodes", []):
        if node.get("type") == "ProPainterInpaint":
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 2:
                widgets[0] = proc_width
                widgets[1] = proc_height
                node["widgets_values"] = widgets
            break

    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
