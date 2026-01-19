#!/usr/bin/env python3
"""Automated VFX pipeline runner.

Single command to process footage through the entire pipeline:
  Movie file → Frame extraction → ComfyUI workflows → Post-processing

Usage:
    python run_pipeline.py <input_movie> [options]

Example:
    python run_pipeline.py /path/to/footage.mp4 --name "My_Shot" --stages all
    python run_pipeline.py /path/to/footage.mp4 --stages depth,camera
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Environment check and configuration
from env_config import check_conda_env_or_warn, DEFAULT_PROJECTS_DIR, is_in_container

# ComfyUI utilities (shared with run_segmentation.py)
from comfyui_utils import (
    DEFAULT_COMFYUI_URL,
    check_comfyui_running,
    run_comfyui_workflow,
)
from comfyui_manager import ensure_comfyui, stop_comfyui, kill_all_comfyui_processes

# Workflow templates directory
WORKFLOW_TEMPLATES_DIR = Path(__file__).parent.parent / "workflow_templates"


def clear_gpu_memory() -> None:
    """Clear GPU VRAM to free memory after a stage completes.

    This helps prevent out-of-memory errors when running multiple
    GPU-intensive stages sequentially. Safe to call even if CUDA
    is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  → Cleared GPU memory")
    except ImportError:
        pass  # torch not installed, skip
    except Exception as e:
        print(f"  → Warning: Could not clear GPU memory: {e}")

START_FRAME = 1  # ComfyUI SaveImage outputs start at 1, so we match that
SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf", ".exr", ".dpx", ".jpg", ".png"}

# Stage definitions
STAGES = {
    "ingest": "Extract frames from movie",
    "depth": "Run depth analysis (01_analysis.json)",
    "roto": "Run segmentation (02_segmentation.json)",
    "matanyone": "Refine person mattes (04_matanyone.json)",
    "cleanplate": "Run clean plate generation (03_cleanplate.json)",
    "colmap": "Run COLMAP SfM reconstruction",
    "mocap": "Run human motion capture (WHAM)",
    "gsir": "Run GS-IR material decomposition",
    "camera": "Export camera to Alembic",
}

# Correct execution order for stages
STAGE_ORDER = ["ingest", "depth", "roto", "matanyone", "cleanplate", "colmap", "mocap", "gsir", "camera"]


STAGES_REQUIRING_FRAMES = {"depth", "roto", "matanyone", "cleanplate", "colmap", "mocap", "gsir", "camera"}


def sanitize_stages(stages: list[str]) -> list[str]:
    """Deduplicate, inject dependencies, and reorder stages.

    Automatically adds 'ingest' if any frame-dependent stage is requested,
    ensuring frames are extracted before processing.

    Args:
        stages: List of stage names (may have duplicates or wrong order)

    Returns:
        Deduplicated list in correct execution order with dependencies
    """
    requested = set(stages)

    if requested & STAGES_REQUIRING_FRAMES:
        requested.add("ingest")

    return [s for s in STAGE_ORDER if s in requested]


def run_command(cmd: list[str], description: str, check: bool = True, stream: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging and optional streaming output.

    Args:
        cmd: Command and arguments
        description: Human-readable description
        check: Raise exception on non-zero exit
        stream: If True, stream stdout/stderr in real-time (default: True)
    """
    print(f"  → {description}")
    print(f"    $ {' '.join(cmd)}")
    sys.stdout.flush()

    if stream:
        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stdout_lines = []
        for line in iter(process.stdout.readline, ''):
            stdout_lines.append(line)
            # Print output directly (child process handles its own formatting)
            print(line.rstrip())
            sys.stdout.flush()

        process.wait()

        stdout = ''.join(stdout_lines)
        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, "")

        # Return a CompletedProcess-like object
        class Result:
            def __init__(self):
                self.returncode = process.returncode
                self.stdout = stdout
                self.stderr = ""
        return Result()
    else:
        # Capture mode for quick commands
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"    Error: {result.stderr}", file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result


def get_frame_count(input_path: Path) -> int:
    """Get total frame count from video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except (ValueError, AttributeError):
        # Fallback: try duration-based estimate
        return 0


def extract_frames(
    input_path: Path,
    output_dir: Path,
    start_frame: int = START_FRAME,
    fps: Optional[float] = None
) -> int:
    """Extract frames from video file using ffmpeg.

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%04d.png"

    # Get total frame count for progress reporting
    print(f"  → Analyzing video for frame count...")
    total_frames = get_frame_count(input_path)
    if total_frames > 0:
        print(f"    Video contains {total_frames} frames")
    else:
        print(f"    Frame count unknown, progress will be estimated")

    cmd = ["ffmpeg", "-i", str(input_path)]

    if fps:
        cmd.extend(["-vf", f"fps={fps}"])

    cmd.extend([
        "-start_number", str(start_frame),
        "-q:v", "2",  # High quality
        "-progress", "pipe:1",  # Output progress to stdout in parseable format
        "-nostats",  # Disable default stderr stats (we use -progress instead)
        str(output_pattern),
        "-y"  # Overwrite
    ])

    print(f"  → Extracting frames to {output_dir}")
    print(f"    $ {' '.join(cmd)}")

    # Run FFmpeg with progress streaming
    # Using -progress pipe:1 outputs line-by-line "key=value" format to stdout
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Parse FFmpeg progress output (format: "frame=142\nfps=30.0\n...progress=continue\n")
    last_reported = 0
    report_interval = max(1, total_frames // 100) if total_frames > 0 else 10

    # Use iter(readline, '') to avoid Python's internal buffering on pipe iteration
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line.startswith("frame="):
            try:
                current_frame = int(line.split("=")[1])
                # Only report progress periodically to avoid flooding
                if current_frame - last_reported >= report_interval or current_frame == total_frames:
                    if total_frames > 0:
                        print(f"[FFmpeg] Extracting frame {current_frame}/{total_frames}")
                    else:
                        print(f"[FFmpeg] Extracting frame {current_frame}")
                    last_reported = current_frame
                    sys.stdout.flush()
            except (ValueError, IndexError):
                pass

    process.wait()

    if process.returncode != 0:
        # Read stderr for error message
        stderr_output = process.stderr.read() if process.stderr else ""
        print(f"    Error during extraction: {stderr_output}", file=sys.stderr)
        raise subprocess.CalledProcessError(process.returncode, cmd)

    # Count extracted frames
    frames = list(output_dir.glob("frame_*.png"))
    return len(frames)


def get_video_info(input_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def run_export_camera(project_dir: Path, fps: float = 24.0) -> bool:
    """Run camera export script."""
    script_path = Path(__file__).parent / "export_camera.py"

    if not script_path.exists():
        print("    Error: export_camera.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--fps", str(fps),
        "--format", "both"
    ]

    try:
        run_command(cmd, "Exporting camera data")
        return True
    except subprocess.CalledProcessError:
        return False


def run_colmap_reconstruction(
    project_dir: Path,
    quality: str = "medium",
    run_dense: bool = False,
    run_mesh: bool = False,
    use_masks: bool = True
) -> bool:
    """Run COLMAP Structure-from-Motion reconstruction.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high')
        run_dense: Whether to run dense reconstruction
        run_mesh: Whether to generate mesh
        use_masks: If True, use segmentation masks from roto/ (if available)

    Returns:
        True if reconstruction succeeded
    """
    script_path = Path(__file__).parent / "run_colmap.py"

    if not script_path.exists():
        print("    Error: run_colmap.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--quality", quality,
    ]

    if run_dense:
        cmd.append("--dense")
    if run_mesh:
        cmd.append("--mesh")
    if not use_masks:
        cmd.append("--no-masks")

    try:
        run_command(cmd, "Running COLMAP reconstruction")
        return True
    except subprocess.CalledProcessError:
        return False


def export_camera_to_vfx_formats(
    project_dir: Path,
    start_frame: int = 1,
    fps: float = 24.0,
) -> bool:
    """Export camera data to VFX-friendly formats.

    Exports to .chan (Nuke), .csv, .clip (Houdini), .camera.json.
    Also exports .abc if PyAlembic is available.

    Args:
        project_dir: Project directory with camera/ subfolder
        start_frame: Starting frame number
        fps: Frames per second

    Returns:
        True if export succeeded
    """
    script_path = Path(__file__).parent / "export_camera.py"

    if not script_path.exists():
        print("    Error: export_camera.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--format", "all",
        "--start-frame", str(start_frame),
        "--fps", str(fps),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Print export summary (filter to relevant lines)
            for line in result.stdout.strip().split('\n'):
                if 'Exported' in line or 'formats:' in line or '.chan' in line or '.clip' in line:
                    print(f"    {line}")
            return True
        else:
            print(f"    Camera export failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"    Camera export failed: {e}", file=sys.stderr)
        return False


def run_mocap(
    project_dir: Path,
    skip_texture: bool = False,
) -> bool:
    """Run human motion capture with WHAM.

    Args:
        project_dir: Project directory with frames and camera data
        skip_texture: Skip texture projection (faster)

    Returns:
        True if mocap succeeded
    """
    script_path = Path(__file__).parent / "run_mocap.py"

    if not script_path.exists():
        print("    Error: run_mocap.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
    ]

    if skip_texture:
        cmd.append("--skip-texture")

    try:
        run_command(cmd, "Running motion capture")
        return True
    except subprocess.CalledProcessError:
        return False


def generate_preview_movie(
    image_dir: Path,
    output_path: Path,
    fps: float = 24.0,
    pattern: str = "*.png",
    crf: int = 23,
) -> bool:
    """Generate a preview MP4 from an image sequence.

    Args:
        image_dir: Directory containing images
        output_path: Output MP4 path
        fps: Frame rate
        pattern: Glob pattern for images
        crf: Quality (lower = better, 18-28 typical)

    Returns:
        True if successful
    """
    images = sorted(image_dir.glob(pattern))
    if not images:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg with image sequence input
    # -pattern_type glob for flexible matching
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(image_dir / pattern),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    → Created: {output_path.name}")
            return True
        else:
            # Try alternative approach with numbered frames
            first_image = images[0]
            # Detect numbering pattern from filename
            match = re.search(r'(\d+)', first_image.stem)
            if match:
                num_digits = len(match.group(1))
                prefix = first_image.stem[:match.start()]
                suffix = first_image.suffix
                input_pattern = str(image_dir / f"{prefix}%0{num_digits}d{suffix}")
                start_num = int(match.group(1))

                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-start_number", str(start_num),
                    "-i", input_pattern,
                    "-c:v", "libx264",
                    "-crf", str(crf),
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"    → Created: {output_path.name}")
                    return True
            return False
    except Exception:
        return False


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get width and height from an image file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(image_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        width, height = result.stdout.strip().split("x")
        return int(width), int(height)
    except (ValueError, AttributeError):
        return 1920, 1080  # Default fallback


def get_comfyui_output_dir() -> Path:
    """Get the output directory that ComfyUI uses for SaveImage nodes.

    This must match the --output-directory argument passed to ComfyUI in comfyui_manager.py.
    """
    from env_config import INSTALL_DIR, is_in_container
    # Container-aware: use COMFYUI_OUTPUT_DIR environment variable if in container
    # (same logic as comfyui_manager.py start_comfyui())
    if is_in_container():
        return Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/workspace"))
    # ComfyUI is started with: --output-directory <COMFYUI_DIR.parent.parent.parent>
    # where COMFYUI_DIR = INSTALL_DIR / "ComfyUI"
    # See comfyui_manager.py start_comfyui()
    return INSTALL_DIR.parent.parent


def combine_mattes(
    input_dirs: list,
    output_dir: Path,
    output_prefix: str = "combined"
) -> bool:
    """Combine multiple matte directories into a single combined output.

    Takes the maximum (union) of all mattes at each frame.

    Args:
        input_dirs: List of directories containing matte images
        output_dir: Directory to write combined mattes
        output_prefix: Prefix for output filenames

    Returns:
        True if successful, False otherwise
    """
    import cv2
    import numpy as np

    if not input_dirs:
        print("  → No input directories to combine")
        return False

    # Get list of frame files from first directory
    first_dir = input_dirs[0]
    frame_files = sorted(first_dir.glob("*.png"))
    if not frame_files:
        print(f"  → No PNG files found in {first_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(frame_files)
    print(f"  → Combining {len(input_dirs)} matte directories ({num_frames} frames)")

    for frame_idx in range(num_frames):
        combined = None

        for input_dir in input_dirs:
            # Find corresponding frame file in this directory
            dir_files = sorted(input_dir.glob("*.png"))
            if frame_idx >= len(dir_files):
                continue

            frame_file = dir_files[frame_idx]
            matte = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
            if matte is None:
                continue

            if combined is None:
                combined = matte.astype(np.float32)
            else:
                # Take maximum (union) of mattes
                combined = np.maximum(combined, matte.astype(np.float32))

        if combined is not None:
            out_file = output_dir / f"{output_prefix}_{frame_idx:05d}_.png"
            cv2.imwrite(str(out_file), combined.astype(np.uint8))

        if (frame_idx + 1) % 50 == 0:
            print(f"    Combined {frame_idx + 1}/{num_frames} frames...")

    print(f"  → Combined mattes written to: {output_dir}")
    return True


def refresh_workflow_from_template(workflow_path: Path, template_name: str, project_dir: Path = None) -> bool:
    """Refresh a project's workflow from the template if template is newer.

    Args:
        workflow_path: Path to project's workflow file
        template_name: Name of template file (e.g., "02_segmentation.json")
        project_dir: Project directory for path population (inferred from workflow_path if not provided)

    Returns:
        True if refreshed, False if no refresh needed
    """
    from setup_project import populate_workflow

    template_path = WORKFLOW_TEMPLATES_DIR / template_name
    if not template_path.exists():
        return False

    # Infer project_dir from workflow_path if not provided
    # workflow_path is typically: project_dir/workflows/02_segmentation.json
    if project_dir is None:
        project_dir = workflow_path.parent.parent

    def copy_and_populate():
        """Copy template and populate with project paths."""
        with open(template_path) as f:
            workflow_data = json.load(f)
        populated = populate_workflow(workflow_data, project_dir)
        with open(workflow_path, 'w') as f:
            json.dump(populated, f, indent=2)

    # Always refresh if project workflow doesn't exist
    if not workflow_path.exists():
        copy_and_populate()
        print(f"  → Copied workflow from template: {template_name}")
        return True

    # Refresh if template is newer
    if template_path.stat().st_mtime > workflow_path.stat().st_mtime:
        copy_and_populate()
        print(f"  → Refreshed workflow from template: {template_name}")
        return True

    return False


def update_segmentation_prompt(workflow_path: Path, prompt: str, output_subdir: Path = None, project_dir: Path = None) -> None:
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
        # Update the segmentation prompt (supports both node types)
        if node.get("type") == "SAM3Grounding":
            widgets = node.get("widgets_values", [])
            # SAM3Grounding: [threshold, text_prompt, max_detections, bool]
            if len(widgets) >= 2:
                widgets[1] = prompt
                node["widgets_values"] = widgets
        elif node.get("type") == "SAM3VideoSegmentation":
            widgets = node.get("widgets_values", [])
            # SAM3VideoSegmentation: ["text", text_prompt, frame_idx, threshold]
            if len(widgets) >= 2:
                widgets[1] = prompt
                node["widgets_values"] = widgets

        # Update SaveImage output path if specified
        if output_subdir and node.get("type") == "SaveImage":
            widgets = node.get("widgets_values", [])
            if widgets:
                # ComfyUI SaveImage expects path relative to its output directory
                # (set via --output-directory when starting ComfyUI)
                comfyui_output = get_comfyui_output_dir()
                try:
                    relative_path = output_subdir.relative_to(comfyui_output)
                except ValueError:
                    # output_subdir is not under comfyui_output, use absolute path
                    relative_path = output_subdir
                # Add "mask" as the filename prefix
                widgets[0] = str(relative_path / "mask")
                node["widgets_values"] = widgets

    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)


def update_matanyone_input(workflow_path: Path, mask_dir: Path, output_dir: Path, project_dir: Path) -> None:
    """Update the MatAnyone workflow to read masks from a specific directory and write to output.

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
        # Update the mask input path (second VHS_LoadImagesPath node)
        if node.get("type") == "VHS_LoadImagesPath" and "Person" in node.get("title", ""):
            widgets = node.get("widgets_values", [])
            if widgets:
                # Use absolute path (consistent with setup_project.py)
                widgets[0] = str(mask_dir)
                node["widgets_values"] = widgets

        # Update SaveImage output path
        if node.get("type") == "SaveImage":
            widgets = node.get("widgets_values", [])
            if widgets:
                # ComfyUI SaveImage expects path relative to its output directory
                try:
                    relative_path = output_dir.relative_to(comfyui_output)
                except ValueError:
                    relative_path = output_dir
                widgets[0] = str(relative_path / "matte")
                node["widgets_values"] = widgets

    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)


def combine_mask_sequences(source_dirs: list[Path], output_dir: Path, prefix: str = "combined") -> int:
    """Combine multiple mask sequences by OR-ing them together.

    Args:
        source_dirs: List of directories containing mask sequences
        output_dir: Output directory for combined masks
        prefix: Filename prefix for combined masks

    Returns:
        Number of frames processed
    """
    from PIL import Image
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all frame files from first source
    if not source_dirs:
        return 0

    frame_files = sorted(source_dirs[0].glob("*.png"))
    if not frame_files:
        return 0

    count = 0
    for i, frame_file in enumerate(frame_files):
        # Load and combine masks from all sources
        combined = None
        for src_dir in source_dirs:
            # Find matching frame in this source (by index, not name)
            src_files = sorted(src_dir.glob("*.png"))
            if i < len(src_files):
                img = Image.open(src_files[i]).convert('L')
                arr = np.array(img)
                if combined is None:
                    combined = arr
                else:
                    # OR the masks together
                    combined = np.maximum(combined, arr)

        if combined is not None:
            # Use consistent naming: prefix_00001.png
            out_name = f"{prefix}_{i+1:05d}.png"
            result = Image.fromarray(combined)
            result.save(output_dir / out_name)
            count += 1

    return count


def update_cleanplate_resolution(workflow_path: Path, source_frames_dir: Path) -> None:
    """Update ProPainterInpaint resolution in cleanplate workflow to match source frames."""
    # Get resolution from first source frame
    frames = sorted(source_frames_dir.glob("frame_*.png"))
    if not frames:
        print("  → Warning: No source frames found, using default resolution")
        return

    width, height = get_image_dimensions(frames[0])
    print(f"  → Setting cleanplate resolution to {width}x{height}")

    # Load workflow and update ProPainterInpaint node
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


def run_gsir_materials(
    project_dir: Path,
    iterations_stage1: int = 30000,
    iterations_stage2: int = 35000,
    gsir_path: Optional[str] = None
) -> bool:
    """Run GS-IR material decomposition.

    Args:
        project_dir: Project directory with COLMAP output
        iterations_stage1: Training iterations for stage 1
        iterations_stage2: Total training iterations
        gsir_path: Path to GS-IR installation

    Returns:
        True if material decomposition succeeded
    """
    script_path = Path(__file__).parent / "run_gsir.py"

    if not script_path.exists():
        print("    Error: run_gsir.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--iterations-stage1", str(iterations_stage1),
        "--iterations-stage2", str(iterations_stage2),
    ]

    if gsir_path:
        cmd.extend(["--gsir-path", gsir_path])

    try:
        run_command(cmd, "Running GS-IR material decomposition")
        return True
    except subprocess.CalledProcessError:
        return False


def setup_project(
    project_dir: Path,
    workflows_dir: Path
) -> bool:
    """Set up project structure and populate workflows."""
    script_path = Path(__file__).parent / "setup_project.py"

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--workflows-dir", str(workflows_dir)
    ]

    try:
        run_command(cmd, "Setting up project structure")
        return True
    except subprocess.CalledProcessError:
        return False


def run_pipeline(
    input_path: Path,
    project_name: Optional[str] = None,
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
    stages: list[str] = None,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    fps: Optional[float] = None,
    skip_existing: bool = False,
    colmap_quality: str = "medium",
    colmap_dense: bool = False,
    colmap_mesh: bool = False,
    colmap_use_masks: bool = True,
    gsir_iterations: int = 35000,
    gsir_path: Optional[str] = None,
    auto_start_comfyui: bool = True,
    roto_prompt: Optional[str] = None,
    separate_instances: bool = False,
    auto_movie: bool = False,
) -> bool:
    """Run the full VFX pipeline.

    Args:
        input_path: Input movie file
        project_name: Project name (default: derived from filename)
        projects_dir: Parent directory for projects
        stages: Which stages to run (default: all)
        comfyui_url: ComfyUI API URL
        fps: Override frame rate
        skip_existing: Skip stages with existing output
        colmap_quality: COLMAP quality preset ('low', 'medium', 'high')
        colmap_dense: Run COLMAP dense reconstruction
        colmap_mesh: Generate mesh from COLMAP dense reconstruction
        colmap_use_masks: Use segmentation masks for COLMAP (if available)
        gsir_iterations: Total GS-IR training iterations
        gsir_path: Path to GS-IR installation
        auto_start_comfyui: Auto-start ComfyUI if not running (default: True)

    Returns:
        True if all stages successful
    """
    stages = stages or list(STAGES.keys())

    # Check if any stage requires ComfyUI
    comfyui_stages = {"depth", "roto", "matanyone", "cleanplate"}
    needs_comfyui = bool(comfyui_stages & set(stages))

    # Kill stale ComfyUI processes and clear GPU memory BEFORE starting fresh
    if needs_comfyui:
        print("\n[GPU Cleanup]")
        kill_all_comfyui_processes()
        clear_gpu_memory()

    # Auto-start ComfyUI if needed (after killing stale processes)
    comfyui_was_started = False
    if needs_comfyui and auto_start_comfyui:
        print("\n[ComfyUI] Starting ComfyUI...")
        if not ensure_comfyui(url=comfyui_url):
            print("Error: Failed to start ComfyUI", file=sys.stderr)
            print("Install ComfyUI with the install wizard or start it manually", file=sys.stderr)
            return False
        comfyui_was_started = True

    # Derive project name from input if not specified
    if not project_name:
        project_name = input_path.stem.replace(" ", "_")

    project_dir = projects_dir / project_name

    # Container-aware path validation
    if is_in_container():
        # In container, ensure project_dir is under /workspace for volume mount safety
        if not str(project_dir).startswith("/workspace"):
            print(f"Error: In container, project directory must be under /workspace", file=sys.stderr)
            print(f"  Got: {project_dir}", file=sys.stderr)
            print(f"  Mount your project directory to /workspace or set VFX_PROJECTS_DIR=/workspace/projects", file=sys.stderr)
            return False

    source_frames = project_dir / "source" / "frames"
    workflows_dir = Path(__file__).parent.parent / "workflow_templates"

    print(f"\n{'='*60}")
    print(f"VFX Pipeline: {project_name}")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Project: {project_dir}")
    print(f"Stages: {', '.join(stages)}")
    print()

    # Get video info for fps
    if not fps and input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"}:
        info = get_video_info(input_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "24/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_str)
                break
    fps = fps or 24.0
    print(f"Frame rate: {fps} fps")

    # Ensure project directory exists before writing metadata
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save project metadata for standalone scripts
    metadata_path = project_dir / "project.json"
    project_metadata = {
        "name": project_name,
        "fps": fps,
        "source": str(input_path),
        "start_frame": 1,
    }
    # Load existing and merge (preserve other fields)
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                existing = json.load(f)
            existing.update(project_metadata)
            project_metadata = existing
        except json.JSONDecodeError:
            pass
    with open(metadata_path, "w") as f:
        json.dump(project_metadata, f, indent=2)

    # Stage: Setup
    print("\n[Setup]")
    if not setup_project(project_dir, workflows_dir):
        print("Failed to set up project", file=sys.stderr)
        return False

    # Stage: Ingest
    if "ingest" in stages:
        print("\n=== Stage: ingest ===")
        if skip_existing and list(source_frames.glob("frame_*.png")):
            print("  → Skipping (frames exist)")
        else:
            frame_count = extract_frames(input_path, source_frames, START_FRAME, fps)
            print(f"  → Extracted {frame_count} frames")

        # Copy source video to preview folder for reference
        preview_dir = project_dir / "preview"
        preview_dir.mkdir(exist_ok=True)
        source_preview = preview_dir / f"source{input_path.suffix}"
        if not source_preview.exists():
            shutil.copy2(input_path, source_preview)
            print(f"  → Copied source to {source_preview.name}")

    # Count total frames for progress reporting
    total_frames = len(list(source_frames.glob("frame_*.png")))

    # Update metadata with frame count
    if total_frames > 0:
        project_metadata["frame_count"] = total_frames
        # Get resolution from first frame
        first_frame = sorted(source_frames.glob("frame_*.png"))[0]
        from PIL import Image
        with Image.open(first_frame) as img:
            project_metadata["width"] = img.width
            project_metadata["height"] = img.height
        with open(metadata_path, "w") as f:
            json.dump(project_metadata, f, indent=2)

    # Stage: Depth (Video Depth Anything handles long videos natively)
    if "depth" in stages:
        print("\n=== Stage: depth ===")
        workflow_path = project_dir / "workflows" / "01_analysis.json"
        depth_dir = project_dir / "depth"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list(depth_dir.glob("*.png")):
            print("  → Skipping (depth maps exist)")
        else:
            # Video Depth Anything processes long videos with sliding window
            # (32 frames per segment with temporal consistency)
            if not run_comfyui_workflow(
                workflow_path, comfyui_url,
                output_dir=depth_dir,
                total_frames=total_frames,
                stage_name="depth",
            ):
                print("  → Depth stage failed", file=sys.stderr)
                return False
        # Generate preview movie
        if auto_movie and list(depth_dir.glob("*.png")):
            generate_preview_movie(depth_dir, project_dir / "preview" / "depth.mp4", fps)

        # Clear GPU memory after depth stage
        clear_gpu_memory()

    # Stage: Roto
    if "roto" in stages:
        print("\n=== Stage: roto ===")
        workflow_path = project_dir / "workflows" / "02_segmentation.json"
        roto_dir = project_dir / "roto"

        # Refresh workflow from template if template is newer
        refresh_workflow_from_template(workflow_path, "02_segmentation.json")

        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and (list(roto_dir.glob("*.png")) or list(roto_dir.glob("*/*.png"))):
            print("  → Skipping (masks exist)")
        else:
            # Clean up any existing .png files in roto/ base directory
            for old_file in roto_dir.glob("*.png"):
                old_file.unlink()

            # Parse prompts - support comma-separated list for multiple objects
            prompts = [p.strip() for p in (roto_prompt or "person").split(",")]
            prompts = [p for p in prompts if p]  # Remove empty

            # Run segmentation for each prompt to its own subdirectory
            # Consolidation is handled by cleanplate stage
            print(f"  → Segmenting {len(prompts)} target(s): {', '.join(prompts)}")

            for i, prompt in enumerate(prompts):
                prompt_name = prompt.replace(" ", "_")
                output_subdir = roto_dir / prompt_name
                output_subdir.mkdir(parents=True, exist_ok=True)

                if len(prompts) > 1:
                    print(f"\n  [{i+1}/{len(prompts)}] Segmenting: {prompt}")
                update_segmentation_prompt(workflow_path, prompt, output_subdir, project_dir)
                if not run_comfyui_workflow(
                    workflow_path, comfyui_url,
                    output_dir=output_subdir,
                    total_frames=total_frames,
                    stage_name=f"roto ({prompt})" if len(prompts) > 1 else "roto",
                ):
                    print(f"  → Segmentation failed for '{prompt}'", file=sys.stderr)
                    if len(prompts) == 1:
                        return False
                    # Continue with other prompts for multi-prompt case

        # Generate preview movie from first subdirectory with masks
        if auto_movie:
            for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
                if subdir.is_dir() and list(subdir.glob("*.png")):
                    generate_preview_movie(subdir, project_dir / "preview" / "roto.mp4", fps)
                    break

        # Separate instances if requested (multi-person detection)
        if separate_instances:
            from separate_instances import separate_instances as do_separate

            # Find person-related directories to separate
            person_dirs = []
            for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
                if subdir.is_dir() and "person" in subdir.name.lower():
                    if list(subdir.glob("*.png")):
                        person_dirs.append(subdir)

            if person_dirs:
                print("\n  --- Separating instances ---")
                for person_dir in person_dirs:
                    print(f"  → Processing: {person_dir.name}")
                    result = do_separate(
                        input_dir=person_dir,
                        output_dir=roto_dir,
                        min_area=500,
                        prefix=person_dir.name,
                    )

                    if result and len(result) > 1:
                        # Found multiple instances, remove original combined directory
                        print(f"    Found {len(result)} instances, removing combined directory")
                        shutil.rmtree(person_dir)
                    elif result and len(result) == 1:
                        # Only one instance found, keep as-is but rename for consistency
                        print(f"    Only 1 instance found, keeping original")
                        # Remove the person_0 directory since it's redundant
                        for idx, out_dir in result.items():
                            if out_dir.exists() and out_dir != person_dir:
                                shutil.rmtree(out_dir)
            else:
                print("  → No person directories to separate")

        # Clear GPU memory after roto stage
        clear_gpu_memory()

    # Stage: MatAnyone (refine person mattes)
    if "matanyone" in stages:
        print("\n=== Stage: matanyone ===")
        workflow_path = project_dir / "workflows" / "04_matanyone.json"
        roto_dir = project_dir / "roto"
        combined_dir = roto_dir / "combined"
        matte_dir = project_dir / "matte"  # Temp storage for individual refined mattes

        # Refresh workflow from template if template is newer
        refresh_workflow_from_template(workflow_path, "04_matanyone.json")

        # Find ALL person-related mask directories (supports multi-instance)
        person_dirs = []
        for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
            if subdir.is_dir() and "person" in subdir.name.lower():
                if list(subdir.glob("*.png")):
                    person_dirs.append(subdir)

        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif not person_dirs:
            print("  → Skipping (no person masks found in roto/)")
        else:
            # Track output directories for combining
            output_dirs = []

            # Process each person directory
            for i, person_dir in enumerate(person_dirs):
                # Output individual refined mattes to matte/person_XX/
                out_dir = matte_dir / person_dir.name
                output_dirs.append(out_dir)

                if skip_existing and list(out_dir.glob("*.png")):
                    print(f"  → Skipping {person_dir.name} (mattes exist)")
                    continue

                if len(person_dirs) > 1:
                    print(f"\n  [{i+1}/{len(person_dirs)}] Refining: {person_dir.name}")
                else:
                    print(f"  → Refining person masks from: {person_dir.name}")

                out_dir.mkdir(parents=True, exist_ok=True)

                # Update workflow to read from this person mask directory and write to out_dir
                update_matanyone_input(workflow_path, person_dir, out_dir, project_dir)
                if not run_comfyui_workflow(
                    workflow_path, comfyui_url,
                    output_dir=out_dir,
                    total_frames=total_frames,
                    stage_name=f"matanyone ({person_dir.name})" if len(person_dirs) > 1 else "matanyone",
                ):
                    print(f"  → MatAnyone stage failed for {person_dir.name}", file=sys.stderr)
                    # Continue with other instances instead of failing completely
                    if len(person_dirs) == 1:
                        return False

            # Combine all refined mattes into roto/combined/
            valid_output_dirs = [d for d in output_dirs if d.exists() and list(d.glob("*.png"))]
            if valid_output_dirs:
                if skip_existing and list(combined_dir.glob("*.png")):
                    print(f"  → Skipping combine (roto/combined/ exists)")
                else:
                    print("\n  --- Combining mattes ---")
                    combine_mattes(valid_output_dirs, combined_dir, "combined")

            # Generate preview movie from combined directory
            if auto_movie and list(combined_dir.glob("*.png")):
                generate_preview_movie(combined_dir, project_dir / "preview" / "matte.mp4", fps)

        # Clear GPU memory after matanyone stage
        clear_gpu_memory()

    # Stage: Cleanplate
    if "cleanplate" in stages:
        print("\n=== Stage: cleanplate ===")
        workflow_path = project_dir / "workflows" / "03_cleanplate.json"
        cleanplate_dir = project_dir / "cleanplate"
        roto_dir = project_dir / "roto"
        combined_dir = roto_dir / "combined"

        # Refresh workflow from template if template is newer
        refresh_workflow_from_template(workflow_path, "03_cleanplate.json")

        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list(cleanplate_dir.glob("*.png")):
            print("  → Skipping (cleanplates exist)")
        else:
            # Check for combined MatAnyone mattes first (preferred)
            has_combined_mattes = combined_dir.exists() and list(combined_dir.glob("*.png"))

            # Collect all mask directories (excluding 'combined' which we handle separately)
            mask_dirs = []
            for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
                if subdir.is_dir() and subdir.name != "combined" and list(subdir.glob("*.png")):
                    mask_dirs.append(subdir)

            # Clear any existing combined masks in roto/ base
            for old_file in roto_dir.glob("*.png"):
                old_file.unlink()

            # Determine mask source for cleanplate
            if has_combined_mattes:
                # Use roto/combined/ directly (MatAnyone refined + combined)
                print(f"  → Using combined MatAnyone mattes from roto/combined/")
                # Copy combined mattes to roto/ for cleanplate workflow
                for i, mask_file in enumerate(sorted(combined_dir.glob("*.png"))):
                    out_name = f"mask_{i+1:05d}.png"
                    shutil.copy2(mask_file, roto_dir / out_name)
            elif len(mask_dirs) > 1:
                # Multiple sources without combined - consolidate them
                count = combine_mask_sequences(mask_dirs, roto_dir, prefix="combined")
                print(f"  → Consolidated {count} frames from {len(mask_dirs)} mask sources")
            elif len(mask_dirs) == 1:
                # Single source - copy to roto/ with consistent naming
                source_dir = mask_dirs[0]
                for i, mask_file in enumerate(sorted(source_dir.glob("*.png"))):
                    out_name = f"mask_{i+1:05d}.png"
                    shutil.copy2(mask_file, roto_dir / out_name)
                print(f"  → Using masks from {source_dir.name}/")
            else:
                print("  → Warning: No masks found in roto/ subdirectories")

            # Update ProPainterInpaint resolution to match source frames
            update_cleanplate_resolution(workflow_path, source_frames)
            if not run_comfyui_workflow(
                workflow_path, comfyui_url,
                output_dir=cleanplate_dir,
                total_frames=total_frames,
                stage_name="cleanplate",
            ):
                print("  → Cleanplate stage failed", file=sys.stderr)
                return False
        # Generate preview movie
        if auto_movie and list(cleanplate_dir.glob("*.png")):
            generate_preview_movie(cleanplate_dir, project_dir / "preview" / "cleanplate.mp4", fps)

        # Clear GPU memory after cleanplate stage
        clear_gpu_memory()

    # Stage: COLMAP reconstruction
    if "colmap" in stages:
        print("\n=== Stage: colmap ===")
        colmap_sparse = project_dir / "colmap" / "sparse" / "0"
        if skip_existing and colmap_sparse.exists():
            print("  → Skipping (COLMAP sparse model exists)")
        else:
            if not run_colmap_reconstruction(
                project_dir,
                quality=colmap_quality,
                run_dense=colmap_dense,
                run_mesh=colmap_mesh,
                use_masks=colmap_use_masks
            ):
                print("  → COLMAP reconstruction failed", file=sys.stderr)
                # Non-fatal for pipeline - continue to camera export
                # (may use DA3 camera data as fallback)

        # Export camera to VFX formats after COLMAP completes
        camera_dir = project_dir / "camera"
        if (camera_dir / "extrinsics.json").exists():
            print("\n  → Exporting camera to VFX formats...")
            export_camera_to_vfx_formats(
                project_dir,
                start_frame=1,
                fps=fps,
            )

        # Clear GPU memory after colmap stage
        clear_gpu_memory()

    # Stage: Motion capture
    if "mocap" in stages:
        print("\n=== Stage: mocap ===")
        mocap_output = project_dir / "mocap" / "wham"
        camera_dir = project_dir / "camera"
        if not camera_dir.exists() or not (camera_dir / "extrinsics.json").exists():
            print("  → Skipping (camera data required - run colmap stage first)")
        elif skip_existing and mocap_output.exists():
            print("  → Skipping (mocap data exists)")
        else:
            if not run_mocap(
                project_dir,
                skip_texture=False,  # Could add as pipeline option
            ):
                print("  → Motion capture failed", file=sys.stderr)
                # Non-fatal - continue to other stages

        # Clear GPU memory after mocap stage
        clear_gpu_memory()

    # Stage: GS-IR material decomposition
    if "gsir" in stages:
        print("\n=== Stage: gsir ===")
        colmap_sparse = project_dir / "colmap" / "sparse" / "0"
        gsir_checkpoint = project_dir / "gsir" / "model" / f"chkpnt{gsir_iterations}.pth"
        if not colmap_sparse.exists():
            print("  → Skipping (COLMAP reconstruction required first)")
        elif skip_existing and gsir_checkpoint.exists():
            print("  → Skipping (GS-IR checkpoint exists)")
        else:
            if not run_gsir_materials(
                project_dir,
                iterations_stage1=30000,
                iterations_stage2=gsir_iterations,
                gsir_path=gsir_path
            ):
                print("  → GS-IR material decomposition failed", file=sys.stderr)
                # Non-fatal - continue to camera export

        # Clear GPU memory after gsir stage
        clear_gpu_memory()

    # Stage: Camera export
    if "camera" in stages:
        print("\n=== Stage: camera ===")
        camera_dir = project_dir / "camera"
        if not (camera_dir / "extrinsics.json").exists():
            print("  → Skipping (no camera data - run depth or colmap stage first)")
        elif skip_existing and (camera_dir / "camera.abc").exists():
            print("  → Skipping (camera.abc exists)")
        else:
            if not run_export_camera(project_dir, fps):
                print("  → Camera export failed", file=sys.stderr)
                # Non-fatal - continue

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {project_dir}")
    print(f"{'='*60}\n")

    # Stop ComfyUI if we started it
    if comfyui_was_started:
        print("[ComfyUI] Stopping ComfyUI...")
        stop_comfyui()

    return True


def main():
    # Check conda environment (warn but don't exit - allow --help to work)
    check_conda_env_or_warn()

    parser = argparse.ArgumentParser(
        description="Automated VFX pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input movie file or image sequence"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Project name (default: derived from input filename)"
    )
    parser.add_argument(
        "--projects-dir", "-p",
        type=Path,
        default=DEFAULT_PROJECTS_DIR,
        help=f"Projects directory (default: {DEFAULT_PROJECTS_DIR})"
    )
    parser.add_argument(
        "--stages", "-s",
        type=str,
        default="all",
        help=f"Comma-separated stages to run: {','.join(STAGES.keys())} or 'all'"
    )
    parser.add_argument(
        "--comfyui-url", "-c",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"ComfyUI API URL (default: {DEFAULT_COMFYUI_URL})"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=None,
        help="Override frame rate (default: auto-detect)"
    )
    parser.add_argument(
        "--skip-existing", "-e",
        action="store_true",
        help="Skip stages that have existing output"
    )
    parser.add_argument(
        "--list-stages", "-l",
        action="store_true",
        help="List available stages and exit"
    )

    # COLMAP options
    parser.add_argument(
        "--colmap-quality", "-q",
        choices=["low", "medium", "high", "slow"],
        default="medium",
        help="COLMAP quality preset: low, medium, high, or 'slow' for minimal camera motion (default: medium)"
    )
    parser.add_argument(
        "--colmap-dense", "-d",
        action="store_true",
        help="Run COLMAP dense reconstruction (slower, produces point cloud)"
    )
    parser.add_argument(
        "--colmap-mesh", "-m",
        action="store_true",
        help="Generate mesh from COLMAP dense reconstruction (requires --colmap-dense)"
    )
    parser.add_argument(
        "--colmap-no-masks", "-M",
        action="store_true",
        help="Disable automatic use of segmentation masks for COLMAP (default: use masks if available)"
    )

    # GS-IR options
    parser.add_argument(
        "--gsir-iterations", "-i",
        type=int,
        default=35000,
        help="GS-IR total training iterations (default: 35000)"
    )
    parser.add_argument(
        "--gsir-path", "-g",
        type=str,
        default=None,
        help="Path to GS-IR installation (default: auto-detect)"
    )
    parser.add_argument(
        "--no-auto-comfyui",
        action="store_true",
        help="Don't auto-start ComfyUI (assume it's already running)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Segmentation prompt for roto stage (default: 'person'). Example: 'person, ball, backpack'"
    )
    parser.add_argument(
        "--separate-instances",
        action="store_true",
        help="Separate multi-person masks into individual instances (person_0/, person_1/, etc.)"
    )
    parser.add_argument(
        "--auto-movie",
        action="store_true",
        help="Generate preview MP4s from completed image sequences (depth, roto, cleanplate)"
    )

    args = parser.parse_args()

    if args.list_stages:
        print("Available stages:")
        for name, desc in STAGES.items():
            print(f"  {name}: {desc}")
        sys.exit(0)

    # Validate input
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse stages
    if args.stages.lower() == "all":
        stages = STAGE_ORDER.copy()
    else:
        stages = [s.strip() for s in args.stages.split(",")]
        invalid = set(stages) - set(STAGES.keys())
        if invalid:
            print(f"Error: Invalid stages: {invalid}", file=sys.stderr)
            print(f"Valid stages: {', '.join(STAGE_ORDER)}")
            sys.exit(1)
        # Sanitize: deduplicate and reorder
        stages = sanitize_stages(stages)

    print(f"Stages to run: {', '.join(stages)}")

    # Run pipeline
    success = run_pipeline(
        input_path=input_path,
        project_name=args.name,
        projects_dir=args.projects_dir,
        stages=stages,
        comfyui_url=args.comfyui_url,
        fps=args.fps,
        skip_existing=args.skip_existing,
        colmap_quality=args.colmap_quality,
        colmap_dense=args.colmap_dense,
        colmap_mesh=args.colmap_mesh,
        colmap_use_masks=not args.colmap_no_masks,
        gsir_iterations=args.gsir_iterations,
        gsir_path=args.gsir_path,
        auto_start_comfyui=not args.no_auto_comfyui,
        roto_prompt=args.prompt,
        separate_instances=args.separate_instances,
        auto_movie=args.auto_movie,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
