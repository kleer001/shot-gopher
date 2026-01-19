"""ComfyUI workflow manipulation utilities.

Functions for loading, modifying, and saving ComfyUI workflow JSON files.
"""

import json
from pathlib import Path

from pipeline_constants import WORKFLOW_TEMPLATES_DIR
from pipeline_utils import get_image_dimensions

__all__ = [
    "get_comfyui_output_dir",
    "refresh_workflow_from_template",
    "update_segmentation_prompt",
    "update_matanyone_input",
    "update_cleanplate_resolution",
]


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
    project_dir: Path = None
) -> None:
    """Update the text prompt and output path in segmentation workflow.

    Args:
        workflow_path: Path to workflow JSON
        prompt: Segmentation target text
        output_subdir: If provided, update SaveImage to write here
        project_dir: Project root for computing relative paths
    """
    print(f"  → Setting segmentation prompt: {prompt}")

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


def update_matanyone_input(
    workflow_path: Path,
    mask_dir: Path,
    output_dir: Path,
    project_dir: Path
) -> None:
    """Update the MatAnyone workflow to read masks from a specific directory.

    Args:
        workflow_path: Path to workflow JSON
        mask_dir: Directory containing person masks to refine
        output_dir: Directory to write refined mattes
        project_dir: Project root directory for computing relative paths
    """
    with open(workflow_path) as f:
        workflow = json.load(f)

    comfyui_output = get_comfyui_output_dir()

    for node in workflow.get("nodes", []):
        if node.get("type") == "VHS_LoadImagesPath" and "Person" in node.get("title", ""):
            widgets = node.get("widgets_values", [])
            if widgets:
                widgets[0] = str(mask_dir)
                node["widgets_values"] = widgets

        if node.get("type") == "SaveImage":
            widgets = node.get("widgets_values", [])
            if widgets:
                try:
                    relative_path = output_dir.relative_to(comfyui_output)
                except ValueError:
                    relative_path = output_dir
                widgets[0] = str(relative_path / "matte")
                node["widgets_values"] = widgets

    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)


def update_cleanplate_resolution(
    workflow_path: Path,
    source_frames_dir: Path
) -> None:
    """Update ProPainterInpaint resolution in cleanplate workflow to match source frames.

    Args:
        workflow_path: Path to workflow JSON
        source_frames_dir: Directory containing source frames
    """
    frames = sorted(source_frames_dir.glob("frame_*.png"))
    if not frames:
        print("  → Warning: No source frames found, using default resolution")
        return

    width, height = get_image_dimensions(frames[0])
    print(f"  → Setting cleanplate resolution to {width}x{height}")

    with open(workflow_path) as f:
        workflow = json.load(f)

    for node in workflow.get("nodes", []):
        if node.get("type") == "ProPainterInpaint":
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 2:
                widgets[0] = width
                widgets[1] = height
                node["widgets_values"] = widgets
            break

    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
