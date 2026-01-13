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
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Environment check and configuration
from env_config import check_conda_env_or_warn, DEFAULT_PROJECTS_DIR

# ComfyUI utilities (shared with run_segmentation.py)
from comfyui_utils import (
    DEFAULT_COMFYUI_URL,
    check_comfyui_running,
    run_comfyui_workflow,
)
from comfyui_manager import ensure_comfyui, stop_comfyui
START_FRAME = 1001
SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf", ".exr", ".dpx", ".jpg", ".png"}

# Stage definitions
STAGES = {
    "ingest": "Extract frames from movie",
    "depth": "Run depth analysis (01_analysis.json)",
    "roto": "Run segmentation (02_segmentation.json)",
    "cleanplate": "Run clean plate generation (03_cleanplate.json)",
    "colmap": "Run COLMAP SfM reconstruction",
    "mocap": "Run human motion capture (WHAM + ECON)",
    "gsir": "Run GS-IR material decomposition",
    "camera": "Export camera to Alembic",
}


def run_command(cmd: list[str], description: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    print(f"  → {description}")
    print(f"    $ {' '.join(cmd)}")
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


def calculate_depth_batch_size(
    frame_width: int,
    frame_height: int,
    gpu_memory_gb: float = None,
    model_type: str = "large",
) -> tuple[int, int]:
    """Calculate optimal batch size and overlap based on GPU memory and frame size.

    Args:
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        gpu_memory_gb: Available GPU memory in GB (auto-detected if None)
        model_type: Model variant ("small", "base", "large", "giant")

    Returns:
        (batch_size, overlap) tuple
    """
    # Auto-detect available GPU memory if not provided
    if gpu_memory_gb is None:
        try:
            import torch
            if torch.cuda.is_available():
                # Get available (free) GPU memory, not total
                free_memory, total_memory = torch.cuda.mem_get_info(0)
                gpu_memory_gb = free_memory / (1024**3)
                # No headroom needed for inference (no gradients stored)
            else:
                gpu_memory_gb = 8.0  # Conservative default for CPU
        except ImportError:
            gpu_memory_gb = 8.0  # Conservative default

    # Estimate memory per frame based on resolution and model
    # DinoV2 creates patches of 14x14 pixels
    num_patches = (frame_width // 14) * (frame_height // 14)

    # Base memory per frame (MB) - rough estimates
    model_memory_overhead = {
        "small": 500,   # ~500MB for small model
        "base": 800,    # ~800MB for base
        "large": 1400,  # ~1.4GB for large
        "giant": 2500,  # ~2.5GB for giant
    }
    base_overhead_mb = model_memory_overhead.get(model_type, 1400)

    # Memory per frame scales with patches and resolution
    # Cross-view attention scales O(n²) with total patches across frames
    # Simplified linear estimate: ~2-4MB per frame at 960x540, scales with resolution
    resolution_factor = (frame_width * frame_height) / (960 * 540)
    memory_per_frame_mb = 3 * resolution_factor  # ~3MB per frame at 960x540

    # Calculate max frames that fit
    available_mb = (gpu_memory_gb * 1024) - base_overhead_mb

    # Account for cross-view attention overhead (grows faster than linear)
    # Use sqrt scaling for attention memory
    if available_mb > 0:
        # Solve: frames * mem_per_frame + frames² * attention_factor < available
        # Simplified: max_frames ≈ sqrt(available / attention_factor)
        attention_factor = 0.1 * resolution_factor  # Attention memory per frame²
        max_frames = int((-memory_per_frame_mb +
                         (memory_per_frame_mb**2 + 4 * attention_factor * available_mb)**0.5)
                        / (2 * attention_factor))
        max_frames = max(8, min(max_frames, 128))  # Clamp to reasonable range
    else:
        max_frames = 16  # Minimum fallback

    # Set batch size (round down to nearest multiple of 8 for efficiency)
    # Max 32 frames due to MultiView node's internal tensor handling limitation
    batch_size = (max_frames // 8) * 8
    batch_size = max(8, min(batch_size, 32))

    # Overlap is 25% of batch size (minimum 4 frames)
    overlap = max(4, batch_size // 4)

    print(f"  GPU memory: {gpu_memory_gb:.1f}GB, Resolution: {frame_width}x{frame_height}")
    print(f"  Calculated batch_size={batch_size}, overlap={overlap}")

    return batch_size, overlap


def run_batched_depth_workflow(
    project_dir: Path,
    workflow_path: Path,
    source_frames_dir: Path,
    output_dir: Path,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    batch_size: int = 48,
    overlap: int = 12,
) -> bool:
    """Run depth workflow in batches with overlapping frames for temporal consistency.

    Processes frames in batches to fit in VRAM, with overlap regions that get
    blended together for seamless transitions.

    Args:
        project_dir: Project directory
        workflow_path: Path to depth workflow JSON
        source_frames_dir: Directory with source frames
        output_dir: Output directory for depth maps
        comfyui_url: ComfyUI API URL
        batch_size: Number of frames per batch (default 48 = 2 seconds at 24fps)
        overlap: Number of overlapping frames between batches (default 12)

    Returns:
        True if successful
    """
    import shutil
    import tempfile
    import numpy as np
    from PIL import Image

    # Get sorted list of source frames
    source_frames = sorted(source_frames_dir.glob("frame_*.png"))
    total_frames = len(source_frames)

    if total_frames == 0:
        print("    Error: No source frames found", file=sys.stderr)
        return False

    # If frames fit in one batch, run normally
    if total_frames <= batch_size:
        print(f"    Processing all {total_frames} frames in single batch")
        return run_comfyui_workflow(
            workflow_path, comfyui_url,
            output_dir=output_dir,
            total_frames=total_frames,
            stage_name="depth",
        )

    # Calculate batches with overlap
    batches = []
    start = 0
    while start < total_frames:
        end = min(start + batch_size, total_frames)
        batches.append((start, end))
        if end >= total_frames:
            break
        start = end - overlap  # Overlap with previous batch

    print(f"    Processing {total_frames} frames in {len(batches)} batches (batch_size={batch_size}, overlap={overlap})")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store batch outputs for blending
    batch_outputs = {}  # frame_index -> list of (batch_idx, image_array, weight)

    # Load original workflow
    with open(workflow_path) as f:
        original_workflow = json.load(f)

    # Process each batch
    for batch_idx, (start_idx, end_idx) in enumerate(batches):
        batch_frames = source_frames[start_idx:end_idx]
        batch_count = len(batch_frames)

        print(f"\n    Batch {batch_idx + 1}/{len(batches)}: frames {start_idx + 1}-{end_idx} ({batch_count} frames)")

        # Create temp directory with symlinks to batch frames
        with tempfile.TemporaryDirectory(prefix=f"depth_batch_{batch_idx}_") as temp_dir:
            temp_path = Path(temp_dir)
            batch_input_dir = temp_path / "frames"
            batch_output_dir = temp_path / "output"
            batch_input_dir.mkdir()
            batch_output_dir.mkdir()

            # Create symlinks to source frames (renumbered 0001, 0002, etc.)
            for i, frame_path in enumerate(batch_frames):
                link_path = batch_input_dir / f"frame_{i + 1:04d}.png"
                link_path.symlink_to(frame_path.resolve())

            # Modify workflow to use batch directories
            batch_workflow = json.loads(json.dumps(original_workflow))  # Deep copy

            # Update paths in workflow nodes
            for node in batch_workflow.get("nodes", []):
                widgets = node.get("widgets_values", [])
                for i, widget in enumerate(widgets):
                    if isinstance(widget, str):
                        # Replace source/frames path
                        if "source/frames" in widget or str(source_frames_dir) in widget:
                            widgets[i] = str(batch_input_dir)
                        # Replace depth output path
                        elif widget.startswith("depth/") or widget == "depth":
                            widgets[i] = str(batch_output_dir / "depth")

            # Write modified workflow
            batch_workflow_path = temp_path / "batch_workflow.json"
            with open(batch_workflow_path, "w") as f:
                json.dump(batch_workflow, f, indent=2)

            # Run workflow for this batch
            if not run_comfyui_workflow(
                batch_workflow_path, comfyui_url,
                output_dir=batch_output_dir,
                total_frames=batch_count,
                stage_name=f"depth batch {batch_idx + 1}",
            ):
                print(f"    Error: Batch {batch_idx + 1} failed", file=sys.stderr)
                return False

            # Collect batch outputs
            batch_output_files = sorted(batch_output_dir.glob("**/*.png"))
            if len(batch_output_files) != batch_count:
                print(f"    Warning: Expected {batch_count} outputs, got {len(batch_output_files)}")

            for local_idx, output_file in enumerate(batch_output_files):
                global_idx = start_idx + local_idx

                # Load image as numpy array
                img = np.array(Image.open(output_file)).astype(np.float32)

                # Calculate blend weight for this frame in this batch
                # Frames in overlap region get partial weight
                if batch_idx == 0:
                    # First batch: full weight except end overlap
                    if local_idx >= batch_count - overlap and batch_idx < len(batches) - 1:
                        # Ramp down at end
                        pos_in_overlap = local_idx - (batch_count - overlap)
                        weight = 1.0 - (pos_in_overlap / overlap)
                    else:
                        weight = 1.0
                elif batch_idx == len(batches) - 1:
                    # Last batch: ramp up at start
                    if local_idx < overlap:
                        weight = local_idx / overlap
                    else:
                        weight = 1.0
                else:
                    # Middle batch: ramp up at start, ramp down at end
                    if local_idx < overlap:
                        weight = local_idx / overlap
                    elif local_idx >= batch_count - overlap:
                        pos_in_overlap = local_idx - (batch_count - overlap)
                        weight = 1.0 - (pos_in_overlap / overlap)
                    else:
                        weight = 1.0

                if global_idx not in batch_outputs:
                    batch_outputs[global_idx] = []
                batch_outputs[global_idx].append((batch_idx, img, weight))

    # Blend and save final outputs
    print(f"\n    Blending {len(batch_outputs)} frames...")

    for global_idx in sorted(batch_outputs.keys()):
        contributions = batch_outputs[global_idx]

        if len(contributions) == 1:
            # No blending needed
            final_img = contributions[0][1]
        else:
            # Weighted blend
            total_weight = sum(c[2] for c in contributions)
            blended = np.zeros_like(contributions[0][1])
            for _, img, weight in contributions:
                blended += img * (weight / total_weight)
            final_img = blended

        # Convert back to image and save
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        output_path = output_dir / f"depth_{global_idx + 1:04d}.png"
        Image.fromarray(final_img).save(output_path)

    print(f"    Saved {len(batch_outputs)} depth maps to {output_dir}")
    return True


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


def run_mocap(
    project_dir: Path,
    skip_texture: bool = False,
    keyframe_interval: int = 25,
) -> bool:
    """Run human motion capture with WHAM + ECON.

    Args:
        project_dir: Project directory with frames and camera data
        skip_texture: Skip texture projection (faster)
        keyframe_interval: ECON keyframe interval

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
        "--keyframe-interval", str(keyframe_interval),
    ]

    if skip_texture:
        cmd.append("--skip-texture")

    try:
        run_command(cmd, "Running motion capture")
        return True
    except subprocess.CalledProcessError:
        return False


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
    comfyui_stages = {"depth", "roto", "cleanplate"}
    needs_comfyui = bool(comfyui_stages & set(stages))

    # Auto-start ComfyUI if needed
    comfyui_was_started = False
    if needs_comfyui and auto_start_comfyui:
        if not check_comfyui_running(comfyui_url):
            print("\n[ComfyUI] Starting ComfyUI automatically...")
            if not ensure_comfyui(url=comfyui_url):
                print("Error: Failed to start ComfyUI", file=sys.stderr)
                print("Install ComfyUI with the install wizard or start it manually", file=sys.stderr)
                return False
            comfyui_was_started = True

    # Derive project name from input if not specified
    if not project_name:
        project_name = input_path.stem.replace(" ", "_")

    project_dir = projects_dir / project_name
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

    # Count total frames for progress reporting
    total_frames = len(list(source_frames.glob("frame_*.png")))

    # Stage: Depth (uses batched processing for VRAM efficiency)
    if "depth" in stages:
        print("\n=== Stage: depth ===")
        workflow_path = project_dir / "workflows" / "01_analysis.json"
        depth_dir = project_dir / "depth"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list(depth_dir.glob("*.png")):
            print("  → Skipping (depth maps exist)")
        else:
            # Get frame dimensions for dynamic batch sizing
            sample_frames = sorted(source_frames.glob("frame_*.png"))
            if sample_frames:
                from PIL import Image
                with Image.open(sample_frames[0]) as img:
                    frame_width, frame_height = img.size
                batch_size, overlap = calculate_depth_batch_size(frame_width, frame_height)
            else:
                batch_size, overlap = 48, 12  # Fallback defaults

            if not run_batched_depth_workflow(
                project_dir=project_dir,
                workflow_path=workflow_path,
                source_frames_dir=source_frames,
                output_dir=depth_dir,
                comfyui_url=comfyui_url,
                batch_size=batch_size,
                overlap=overlap,
            ):
                print("  → Depth stage failed", file=sys.stderr)
                return False

    # Stage: Roto
    if "roto" in stages:
        print("\n=== Stage: roto ===")
        workflow_path = project_dir / "workflows" / "02_segmentation.json"
        roto_dir = project_dir / "roto"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list(roto_dir.glob("*.png")):
            print("  → Skipping (masks exist)")
        else:
            if not run_comfyui_workflow(
                workflow_path, comfyui_url,
                output_dir=roto_dir,
                total_frames=total_frames,
                stage_name="roto",
            ):
                print("  → Segmentation stage failed", file=sys.stderr)
                return False

    # Stage: Cleanplate
    if "cleanplate" in stages:
        print("\n=== Stage: cleanplate ===")
        workflow_path = project_dir / "workflows" / "03_cleanplate.json"
        cleanplate_dir = project_dir / "cleanplate"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list(cleanplate_dir.glob("*.png")):
            print("  → Skipping (cleanplates exist)")
        else:
            if not run_comfyui_workflow(
                workflow_path, comfyui_url,
                output_dir=cleanplate_dir,
                total_frames=total_frames,
                stage_name="cleanplate",
            ):
                print("  → Cleanplate stage failed", file=sys.stderr)
                return False

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

    # Stage: Motion capture
    if "mocap" in stages:
        print("\n=== Stage: mocap ===")
        mocap_output = project_dir / "mocap" / "econ" / "mesh_sequence"
        camera_dir = project_dir / "camera"
        if not camera_dir.exists() or not (camera_dir / "extrinsics.json").exists():
            print("  → Skipping (camera data required - run colmap stage first)")
        elif skip_existing and mocap_output.exists():
            print("  → Skipping (mocap data exists)")
        else:
            if not run_mocap(
                project_dir,
                skip_texture=False,  # Could add as pipeline option
                keyframe_interval=25,
            ):
                print("  → Motion capture failed", file=sys.stderr)
                # Non-fatal - continue to other stages

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
        choices=["low", "medium", "high"],
        default="medium",
        help="COLMAP quality preset (default: medium)"
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
        stages = list(STAGES.keys())
    else:
        stages = [s.strip() for s in args.stages.split(",")]
        invalid = set(stages) - set(STAGES.keys())
        if invalid:
            print(f"Error: Invalid stages: {invalid}", file=sys.stderr)
            print(f"Valid stages: {', '.join(STAGES.keys())}")
            sys.exit(1)

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
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
