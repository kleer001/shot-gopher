"""Stage runner functions.

Contains handlers for all pipeline stages. Each handler takes a StageContext
and PipelineConfig, returning True on success.
"""

import re
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

if sys.platform != "win32":
    import select
from typing import Any, Optional, Callable, TYPE_CHECKING

from comfyui_utils import run_comfyui_workflow
from cleanplate_median import run_cleanplate_median
from matte_utils import combine_mattes, prepare_roto_for_cleanplate
from pipeline_constants import START_FRAME
from pipeline_utils import (
    clear_gpu_memory,
    clear_output_directory,
    extract_frames,
    generate_preview_movie,
    run_command,
)
from workflow_utils import (
    refresh_workflow_from_template,
    update_segmentation_prompt,
    update_workflow_resolution,
)

if TYPE_CHECKING:
    from pipeline_config import StageContext, PipelineConfig

__all__ = [
    "run_export_camera",
    "run_matchmove_camera",
    "export_camera_to_vfx_formats",
    "run_mocap",
    "run_gsir_materials",
    "setup_project",
    "run_stage_ingest",
    "run_stage_interactive",
    "run_stage_depth",
    "run_stage_roto",
    "run_stage_mama",
    "run_stage_cleanplate",
    "run_stage_matchmove_camera",
    "run_stage_dense",
    "run_stage_mocap",
    "run_stage_gsir",
    "run_stage_camera",
    "STAGE_HANDLERS",
]


def run_export_camera(
    project_dir: Path,
    fps: float = 24.0,
    camera_dir: Optional[Path] = None,
) -> bool:
    """Run camera export script.

    Args:
        project_dir: Project directory with camera data
        fps: Frames per second
        camera_dir: Camera data directory (default: project_dir/camera/)

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
        "--fps", str(fps),
        "--format", "all"
    ]

    if camera_dir is not None:
        cmd.extend(["--camera-dir", str(camera_dir)])

    try:
        run_command(cmd, "Exporting camera data")
        return True
    except subprocess.CalledProcessError:
        return False


def run_matchmove_camera(
    project_dir: Path,
    quality: str = "medium",
    use_masks: bool = True,
    max_image_size: int = -1
) -> bool:
    """Run COLMAP Structure-from-Motion sparse reconstruction.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high', 'slow')
        use_masks: If True, use segmentation masks from roto/ (if available)
        max_image_size: Maximum image dimension (-1 for no limit)

    Returns:
        True if reconstruction succeeded
    """
    script_path = Path(__file__).parent / "run_matchmove_camera.py"

    if not script_path.exists():
        print("    Error: run_matchmove_camera.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--quality", quality,
    ]

    if not use_masks:
        cmd.append("--no-masks")
    if max_image_size > 0:
        cmd.extend(["--max-image-size", str(max_image_size)])

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

    Exports to .chan (Nuke), .csv, .clip (Houdini), .camera.json, .jsx (After Effects).
    Also exports .abc (Alembic) and .usd (USD) if Blender is available.

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
    use_colmap_intrinsics: bool = True,
    gender: str = "neutral",
    no_export: bool = False,
    fps: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    mocap_person: Optional[str] = None,
    export_camera: bool = True,
) -> bool:
    """Run human motion capture with GVHMR.

    Args:
        project_dir: Project directory with frames and camera data
        use_colmap_intrinsics: Use COLMAP focal length for GVHMR
        gender: Body model gender (neutral, male, female)
        no_export: Skip automatic Alembic/USD export
        fps: Frames per second for export
        start_frame: Start frame (1-indexed, inclusive)
        end_frame: End frame (1-indexed, inclusive)
        mocap_person: Roto person folder to isolate (e.g., 'person_00')
        export_camera: Export GVHMR camera estimate if COLMAP camera missing

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
        "--gender", gender,
    ]

    if not use_colmap_intrinsics:
        cmd.append("--no-colmap-intrinsics")

    if no_export:
        cmd.append("--no-export")

    if not export_camera:
        cmd.append("--no-camera-export")

    if fps is not None:
        cmd.extend(["--fps", str(fps)])

    if start_frame is not None:
        cmd.extend(["--start-frame", str(start_frame)])

    if end_frame is not None:
        cmd.extend(["--end-frame", str(end_frame)])

    if mocap_person is not None:
        cmd.extend(["--mocap-person", mocap_person])

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
    """Set up project structure and populate workflows.

    Args:
        project_dir: Project directory to set up
        workflows_dir: Directory containing workflow templates

    Returns:
        True if setup succeeded
    """
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


def run_stage_ingest(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run the ingest stage: extract frames from source video.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: ingest ===")

    if not config.input_path:
        print("  → Skipping (no input file, existing project)")
        _copy_source_preview(ctx, config)
        return True

    if ctx.skip_existing and list(ctx.source_frames.glob("*.png")):
        print("  → Skipping (frames exist)")
        _copy_source_preview(ctx, config)
        return True

    frame_count = extract_frames(config.input_path, ctx.source_frames, START_FRAME, ctx.fps)
    print(f"  → Extracted {frame_count} frames")

    _copy_source_preview(ctx, config)
    return True


def _copy_source_preview(ctx: "StageContext", config: "PipelineConfig") -> None:
    """Copy source file to preview directory."""
    preview_dir = ctx.project_dir / "preview"
    preview_dir.mkdir(exist_ok=True)

    if config.input_path:
        source_preview = preview_dir / f"source{config.input_path.suffix}"
        if not source_preview.exists():
            shutil.copy2(config.input_path, source_preview)
            print(f"  → Copied source to {source_preview.name}")


INTERACTIVE_SIGNAL_FILE = ".interactive_done"


def _check_stdin_ready(timeout: float) -> bool:
    """Check if stdin has input ready, cross-platform.

    Args:
        timeout: How long to wait for input (seconds)

    Returns:
        True if input is ready to be read
    """
    if sys.platform == "win32":
        import msvcrt
        start = time.time()
        while (time.time() - start) < timeout:
            if msvcrt.kbhit():
                return True
            time.sleep(0.05)
        return False
    else:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(ready)


def wait_for_interactive_signal(project_dir: Path, poll_interval: float = 0.5) -> None:
    """Wait for interactive segmentation completion signal.

    Checks for either:
    1. A signal file (.interactive_done) in the project directory
    2. User pressing Enter in the terminal (for backwards compatibility)

    Args:
        project_dir: Project directory to watch for signal file
        poll_interval: How often to check for signal file (seconds)

    Raises:
        FileNotFoundError: If project directory is deleted while waiting
    """
    signal_file = project_dir / INTERACTIVE_SIGNAL_FILE
    signal_file.unlink(missing_ok=True)

    print("  Waiting for interactive segmentation to complete...")
    print("    - Click 'Complete Interactive Roto' in the web UI, OR")
    print("    - Press Enter in this terminal")
    print()

    while True:
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory deleted: {project_dir}")

        try:
            if signal_file.exists():
                signal_file.unlink(missing_ok=True)
                print("  → Signal received from web UI")
                return
        except OSError as e:
            raise FileNotFoundError(f"Cannot access project directory: {e}") from e

        if sys.stdin.isatty():
            if _check_stdin_ready(poll_interval):
                sys.stdin.readline()
                print("  → Enter pressed")
                return
        else:
            time.sleep(poll_interval)


def run_stage_interactive(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run interactive segmentation stage.

    Opens ComfyUI for manual point-based segmentation.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: interactive ===")
    workflow_path = ctx.project_dir / "workflows" / "05_interactive_segmentation.json"
    roto_dir = ctx.project_dir / "roto"

    refresh_workflow_from_template(workflow_path, "05_interactive_segmentation.json")

    if not workflow_path.exists():
        print("  → Skipping (workflow not found)")
        return True

    if ctx.source_width > 0 and ctx.source_height > 0:
        update_workflow_resolution(
            workflow_path,
            ctx.source_width,
            ctx.source_height,
            update_loaders=True,
            update_scales=False,
        )

    print("  → Opening interactive segmentation in ComfyUI")
    print(f"    Workflow: {workflow_path}")
    print(f"    ComfyUI: {ctx.comfyui_url}")
    print()
    print("  Instructions:")
    print("    1. Open ComfyUI in your browser")
    print("    2. Load the workflow from: workflows/05_interactive_segmentation.json")
    print("    3. Click points on the first frame to define what to segment")
    print("    4. Run the workflow (Queue Prompt)")
    print("    5. Masks will be saved to: roto/")
    print()

    webbrowser.open(ctx.comfyui_url)
    wait_for_interactive_signal(ctx.project_dir)

    if list(roto_dir.glob("**/*.png")):
        print(f"  OK Masks found in {roto_dir}")
    else:
        print(f"  -> Warning: No masks found in {roto_dir}")

    return True


def run_stage_depth(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run depth analysis stage.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: depth ===")
    workflow_path = ctx.project_dir / "workflows" / "01_analysis.json"
    depth_dir = ctx.project_dir / "depth"

    if not workflow_path.exists():
        print("  → Skipping (workflow not found)")
        return True

    if ctx.skip_existing and list(depth_dir.glob("*.png")):
        print("  → Skipping (depth maps exist)")
        return True

    if ctx.overwrite:
        clear_output_directory(depth_dir)

    if ctx.source_width > 0 and ctx.source_height > 0:
        update_workflow_resolution(
            workflow_path,
            ctx.source_width,
            ctx.source_height,
            update_loaders=True,
            update_scales=False,
        )

    if not run_comfyui_workflow(
        workflow_path, ctx.comfyui_url,
        output_dir=depth_dir,
        total_frames=ctx.total_frames,
        stage_name="depth",
    ):
        print("  → Depth stage failed", file=sys.stderr)
        return False

    if ctx.auto_movie and list(depth_dir.glob("*.png")):
        generate_preview_movie(depth_dir, ctx.project_dir / "preview" / "depth.mp4", ctx.fps)

    clear_gpu_memory(ctx.comfyui_url)
    return True


def run_stage_roto(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run segmentation/roto stage.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: roto ===")
    workflow_path = ctx.project_dir / "workflows" / "02_segmentation.json"
    roto_dir = ctx.project_dir / "roto"

    prompts = [p.strip() for p in (config.roto_prompt or "person").split(",")]
    prompts = [p for p in prompts if p]

    refresh_workflow_from_template(workflow_path, "02_segmentation.json")

    if not workflow_path.exists():
        print("  → Skipping (workflow not found)")
        return True

    if ctx.source_width > 0 and ctx.source_height > 0:
        update_workflow_resolution(
            workflow_path,
            ctx.source_width,
            ctx.source_height,
            update_loaders=True,
            update_scales=False,
        )

    if ctx.skip_existing and (list(roto_dir.glob("*.png")) or list(roto_dir.glob("*/*.png"))):
        print("  → Skipping (masks exist)")
        return True

    if ctx.overwrite:
        clear_output_directory(roto_dir)

    print(f"  → Segmenting {len(prompts)} target(s): {', '.join(prompts)}")

    for i, prompt in enumerate(prompts):
        prompt_name = prompt.replace(" ", "_")
        output_subdir = roto_dir / prompt_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        if len(prompts) > 1:
            print(f"\n  [{i+1}/{len(prompts)}] Segmenting: {prompt}")

        update_segmentation_prompt(
            workflow_path, prompt, output_subdir, ctx.project_dir,
            start_frame=config.roto_start_frame
        )

        if not run_comfyui_workflow(
            workflow_path, ctx.comfyui_url,
            output_dir=output_subdir,
            total_frames=ctx.total_frames,
            stage_name=f"roto ({prompt})" if len(prompts) > 1 else "roto",
        ):
            print(f"  → Segmentation failed for '{prompt}'", file=sys.stderr)

    if ctx.auto_movie:
        for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
            if subdir.is_dir() and list(subdir.glob("*.png")):
                generate_preview_movie(subdir, ctx.project_dir / "preview" / "roto.mp4", ctx.fps)
                break

    if config.separate_instances:
        _separate_roto_instances(roto_dir, prompts)

    clear_gpu_memory(ctx.comfyui_url)
    return True


def _separate_roto_instances(roto_dir: Path, prompts: list[str]) -> None:
    """Separate multi-person masks into individual instances."""
    from separate_instances import separate_instances as do_separate

    print("\n  --- Separating instances ---")
    for prompt in prompts:
        prompt_name = prompt.replace(" ", "_")
        prompt_dir = roto_dir / prompt_name
        if prompt_dir.exists() and list(prompt_dir.glob("*.png")):
            print(f"  → Processing: {prompt_name} → {prompt_name}_00/, {prompt_name}_01/, ...")
            result = do_separate(
                input_dir=prompt_dir,
                output_dir=roto_dir,
                min_area=500,
                prefix=prompt_name,
            )

            if result:
                print(f"    Created {len(result)} {prompt_name} directories")
                print(f"    Combined mask kept in: {prompt_name}/")
        else:
            print(f"  → No {prompt_name} directory to separate")


def run_stage_mama(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run VideoMaMa matte refinement stage.

    Processes roto masks through VideoMaMa diffusion-based matting
    to produce high-quality alpha mattes.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: mama ===")
    roto_dir = ctx.project_dir / "roto"
    matte_dir = ctx.project_dir / "matte"
    combined_dir = roto_dir / "combined"

    numbered_pattern = re.compile(r"^.+_\d{2}$")
    skip_dirs = {"person", "combined", "mask"}

    roto_dirs = []
    for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
        if not subdir.is_dir():
            continue
        if subdir.name in skip_dirs:
            continue
        if numbered_pattern.match(subdir.name) and list(subdir.glob("*.png")):
            roto_dirs.append(subdir)

    if not roto_dirs:
        print("  → Skipping (no numbered roto directories found)")
        print("    Expected: roto/person_00/, roto/person_01/, roto/bag_00/, etc.")
        return True

    print(f"  → Found {len(roto_dirs)} roto directories to process")

    from video_mama import process_roto_directory, check_installation

    if not check_installation():
        print("  → VideoMaMa not installed. Run: python scripts/video_mama_install.py")
        return False

    if ctx.overwrite:
        clear_output_directory(matte_dir)
        if combined_dir.exists():
            clear_output_directory(combined_dir)

    output_dirs = []

    for i, roto_subdir in enumerate(roto_dirs):
        out_dir = matte_dir / roto_subdir.name
        output_dirs.append(out_dir)

        if ctx.skip_existing and out_dir.exists() and list(out_dir.glob("*.png")):
            print(f"  → Skipping {roto_subdir.name} (mattes exist)")
            continue

        if len(roto_dirs) > 1:
            print(f"\n  [{i+1}/{len(roto_dirs)}] Processing: {roto_subdir.name}")
        else:
            print(f"  → Processing: {roto_subdir.name}")

        mama_kwargs: dict[str, Any] = {
            "project_dir": ctx.project_dir,
            "roto_subdir": roto_subdir.name,
            "output_dir": out_dir,
        }
        if ctx.source_width > 0 and ctx.source_height > 0:
            mama_kwargs["width"] = ctx.source_width
            mama_kwargs["height"] = ctx.source_height

        success = process_roto_directory(**mama_kwargs)

        if not success:
            print(f"  → VideoMaMa failed for {roto_subdir.name}", file=sys.stderr)

    valid_output_dirs = [d for d in output_dirs if d.exists() and list(d.glob("*.png"))]
    if valid_output_dirs:
        if ctx.skip_existing and combined_dir.exists() and list(combined_dir.glob("*.png")):
            print(f"  → Skipping combine (roto/combined/ exists)")
        else:
            print("\n  --- Combining mattes ---")
            combine_mattes(valid_output_dirs, combined_dir, "combined")

    if ctx.auto_movie and combined_dir.exists() and list(combined_dir.glob("*.png")):
        generate_preview_movie(combined_dir, ctx.project_dir / "preview" / "matte.mp4", ctx.fps)

    return True


def run_stage_cleanplate(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run cleanplate generation stage (temporal median, static camera).

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: cleanplate (median-only) ===")
    cleanplate_dir = ctx.project_dir / "cleanplate"
    roto_dir = ctx.project_dir / "roto"

    roto_ready, roto_message = prepare_roto_for_cleanplate(roto_dir)

    if not roto_ready:
        print("")
        print("  " + "=" * 50)
        print("  !!!  NO ROTO DATA FOUND - SKIPPING CLEANPLATE  !!!")
        print("  " + "=" * 50)
        print("  Run the 'roto' stage first to generate masks.")
        print("")
        return True

    if ctx.skip_existing and list(cleanplate_dir.glob("*.png")):
        print("  → Skipping (cleanplates exist)")
        return True

    if ctx.overwrite:
        clear_output_directory(cleanplate_dir)

    print(f"  → {roto_message}")

    if not run_cleanplate_median(ctx.project_dir):
        print("  → Temporal median cleanplate failed", file=sys.stderr)
        return False

    if ctx.auto_movie and list(cleanplate_dir.glob("*.png")):
        generate_preview_movie(cleanplate_dir, ctx.project_dir / "preview" / "cleanplate.mp4", ctx.fps)

    return True


def run_stage_matchmove_camera(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run COLMAP reconstruction stage.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: matchmove_camera ===")
    mmcam_sparse = ctx.project_dir / "mmcam" / "sparse" / "0"
    assert "colmap" not in str(mmcam_sparse), f"BREADCRUMB: path still contains 'colmap': {mmcam_sparse}"

    if ctx.skip_existing and mmcam_sparse.exists():
        print("  → Skipping (COLMAP sparse model exists)")
    else:
        if not run_matchmove_camera(
            ctx.project_dir,
            quality=config.mmcam_quality,
            use_masks=config.mmcam_use_masks,
            max_image_size=config.mmcam_max_size
        ):
            print("  → COLMAP reconstruction failed", file=sys.stderr)

    clear_gpu_memory(ctx.comfyui_url)

    run_stage_camera(ctx, config)

    return True


def run_stage_dense(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run dense reconstruction stage.

    Produces geometry (point clouds, mesh), depth maps, and normal maps
    from a completed COLMAP sparse reconstruction. Exports to VFX-native
    formats (PLY, EXR, and optionally ABC/USD via Blender).

    Output directories (parallel to camera/):
        geometry/ — sparse/dense point clouds and mesh (PLY, ABC, USD)
        depth/    — per-frame 32-bit float EXR depth maps
        normals/  — per-frame 32-bit float EXR normal maps

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: dense ===")

    colmap_dir = ctx.project_dir / "colmap"
    sparse_model = colmap_dir / "sparse" / "0"
    dense_path = colmap_dir / "dense"
    geometry_dir = ctx.project_dir / "geometry"
    depth_dir = ctx.project_dir / "depth"
    normals_dir = ctx.project_dir / "normals"

    if not sparse_model.exists():
        print("  Error: No sparse model found. Run matchmove_camera first.", file=sys.stderr)
        return False

    from run_matchmove_camera import (
        export_sparse_ply,
        run_dense_reconstruction,
        run_mesh_reconstruction,
        convert_depth_maps_to_exr,
        convert_normal_maps_to_exr,
        QUALITY_PRESETS,
    )

    preset = QUALITY_PRESETS.get(config.mmcam_quality, QUALITY_PRESETS["medium"])
    geometry_dir.mkdir(parents=True, exist_ok=True)

    print("  [1/6] Exporting sparse point cloud")
    sparse_ply = geometry_dir / "sparse_pointcloud.ply"
    if ctx.skip_existing and sparse_ply.exists():
        print("    → Skipping (exists)")
    else:
        if export_sparse_ply(sparse_model, sparse_ply):
            print(f"    → {sparse_ply.name}")
        else:
            print("    → Sparse PLY export failed", file=sys.stderr)

    print("  [2/6] Dense reconstruction")
    fused_ply = dense_path / "fused.ply"
    if ctx.skip_existing and fused_ply.exists():
        print("    → Skipping (exists)")
    else:
        frames_dir = ctx.project_dir / "source" / "frames"
        if not run_dense_reconstruction(
            image_path=frames_dir,
            sparse_path=sparse_model,
            output_path=dense_path,
            max_image_size=preset["dense_max_size"],
        ):
            print("    Dense reconstruction failed", file=sys.stderr)
            return False

    print("  [3/6] Copying dense point cloud")
    dense_ply = geometry_dir / "dense_pointcloud.ply"
    if fused_ply.exists() and not (ctx.skip_existing and dense_ply.exists()):
        shutil.copy(fused_ply, dense_ply)
        print(f"    → {dense_ply.name}")

    print("  [4/6] Generating mesh")
    mesh_ply = geometry_dir / "mesh.ply"
    if ctx.skip_existing and mesh_ply.exists():
        print("    → Skipping (exists)")
    else:
        if run_mesh_reconstruction(dense_path, mesh_ply):
            print(f"    → {mesh_ply.name}")
        else:
            print("    → Mesh generation failed (continuing)", file=sys.stderr)

    print("  [5/6] Converting depth maps to EXR")
    if ctx.skip_existing and list(depth_dir.glob("*.exr")):
        print("    → Skipping (exists)")
    else:
        count = convert_depth_maps_to_exr(dense_path, depth_dir)
        if count > 0:
            print(f"    → {count} depth maps → {depth_dir.name}/")
        else:
            print("    → No depth maps to convert", file=sys.stderr)

    print("  [6/6] Converting normal maps to EXR")
    if ctx.skip_existing and list(normals_dir.glob("*.exr")):
        print("    → Skipping (exists)")
    else:
        count = convert_normal_maps_to_exr(dense_path, normals_dir)
        if count > 0:
            print(f"    → {count} normal maps → {normals_dir.name}/")
        else:
            print("    → No normal maps to convert", file=sys.stderr)

    _export_geometry_to_interchange(geometry_dir, ctx.fps)

    clear_gpu_memory(ctx.comfyui_url)

    return True


def _export_geometry_to_interchange(geometry_dir: Path, fps: float) -> None:
    """Export PLY geometry files to ABC and USD via Blender (if available).

    Args:
        geometry_dir: Directory containing PLY files
        fps: Frames per second for export
    """
    try:
        from blender import (
            export_ply_to_alembic,
            export_ply_to_usd,
            check_blender_available,
        )
    except ImportError:
        return

    if not check_blender_available():
        print("    Note: Blender not available, skipping ABC/USD geometry export")
        return

    ply_targets = [
        ("dense_pointcloud.ply", "pointcloud"),
        ("mesh.ply", "mesh"),
    ]

    for ply_name, output_stem in ply_targets:
        ply_path = geometry_dir / ply_name
        if not ply_path.exists():
            continue

        abc_path = geometry_dir / f"{output_stem}.abc"
        usd_path = geometry_dir / f"{output_stem}.usd"

        try:
            export_ply_to_alembic(
                input_path=ply_path,
                output_path=abc_path,
                fps=fps,
            )
            print(f"    → {abc_path.name}")
        except Exception as e:
            print(f"    → ABC export failed for {ply_name}: {e}", file=sys.stderr)

        try:
            export_ply_to_usd(
                input_path=ply_path,
                output_path=usd_path,
                fps=fps,
            )
            print(f"    → {usd_path.name}")
        except Exception as e:
            print(f"    → USD export failed for {ply_name}: {e}", file=sys.stderr)


def run_stage_mocap(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run motion capture stage.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: mocap ===")
    mocap_output = ctx.project_dir / "mocap" / "motion.pkl"
    camera_dir = ctx.project_dir / "camera"
    has_camera = camera_dir.exists() and (camera_dir / "extrinsics.json").exists()

    export_fps = config.mocap_fps if config.mocap_fps is not None else ctx.fps

    if ctx.skip_existing and mocap_output.exists():
        print("  → Skipping (mocap data exists)")
    else:
        if has_camera:
            print(f"  → Using matchmove camera data for improved accuracy")
        print(f"  → Gender: {config.mocap_gender}")
        if config.mocap_start_frame or config.mocap_end_frame:
            range_str = f"{config.mocap_start_frame or 1}-{config.mocap_end_frame or 'end'}"
            print(f"  → Frame range: {range_str}")
        if config.mocap_person is not None:
            print(f"  → Target person: {config.mocap_person}")
        if not config.mocap_no_export:
            print(f"  → Will export to Alembic/USD at {export_fps} fps")
        if not run_mocap(
            ctx.project_dir,
            use_colmap_intrinsics=has_camera,
            gender=config.mocap_gender,
            no_export=config.mocap_no_export,
            fps=export_fps,
            start_frame=config.mocap_start_frame,
            end_frame=config.mocap_end_frame,
            mocap_person=config.mocap_person,
        ):
            print("  → Motion capture failed", file=sys.stderr)

    mocap_camera_dir = ctx.project_dir / "mocap" / "camera"
    colmap_camera_dir = ctx.project_dir / "camera"
    if (mocap_camera_dir / "extrinsics.json").exists() and not (colmap_camera_dir / "extrinsics.json").exists():
        print("\n  → GVHMR camera exported, running camera stage...")
        run_stage_camera(ctx, config)

    clear_gpu_memory(ctx.comfyui_url)
    return True


def run_stage_gsir(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run GS-IR material decomposition stage.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: gsir ===")
    colmap_sparse = ctx.project_dir / "mmcam" / "sparse" / "0"
    gsir_checkpoint = ctx.project_dir / "gsir" / "model" / f"chkpnt{config.gsir_iterations}.pth"

    if not colmap_sparse.exists():
        print("  → Skipping (COLMAP reconstruction required first)")
        return True

    if ctx.skip_existing and gsir_checkpoint.exists():
        print("  → Skipping (GS-IR checkpoint exists)")
        return True

    if not run_gsir_materials(
        ctx.project_dir,
        iterations_stage1=30000,
        iterations_stage2=config.gsir_iterations,
        gsir_path=config.gsir_path
    ):
        print("  → GS-IR material decomposition failed", file=sys.stderr)

    clear_gpu_memory(ctx.comfyui_url)
    return True


def run_stage_camera(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    """Run camera export stage.

    Checks for camera data in order:
    1. camera/ (COLMAP)
    2. mocap/camera/ (GVHMR estimate)

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: camera ===")
    colmap_camera_dir = ctx.project_dir / "camera"
    mocap_camera_dir = ctx.project_dir / "mocap" / "camera"

    if (colmap_camera_dir / "extrinsics.json").exists():
        camera_dir = colmap_camera_dir
        print("  → Using matchmove camera data")
    elif (mocap_camera_dir / "extrinsics.json").exists():
        camera_dir = mocap_camera_dir
        print("  → Using GVHMR camera estimate")
    else:
        print("  → Skipping (no camera data - run matchmove_camera or mocap stage first)")
        return True

    if ctx.skip_existing and (camera_dir / "camera.abc").exists():
        print("  → Skipping (camera.abc exists)")
        return True

    if not run_export_camera(ctx.project_dir, ctx.fps, camera_dir=camera_dir):
        print("  → Camera export failed", file=sys.stderr)

    return True


STAGE_HANDLERS: dict[str, Callable[["StageContext", "PipelineConfig"], bool]] = {
    "ingest": run_stage_ingest,
    "interactive": run_stage_interactive,
    "depth": run_stage_depth,
    "roto": run_stage_roto,
    "mama": run_stage_mama,
    "cleanplate": run_stage_cleanplate,
    "matchmove_camera": run_stage_matchmove_camera,
    "dense": run_stage_dense,
    "mocap": run_stage_mocap,
    "gsir": run_stage_gsir,
    "camera": run_stage_camera,
}
