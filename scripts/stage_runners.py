"""Stage runner functions.

Contains handlers for all pipeline stages. Each handler takes a StageContext
and PipelineConfig, returning True on success.
"""

import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

if sys.platform != "win32":
    import select
from typing import Optional, Callable, TYPE_CHECKING

from comfyui_utils import run_comfyui_workflow
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
    update_cleanplate_resolution,
)

if TYPE_CHECKING:
    from pipeline_config import StageContext, PipelineConfig

__all__ = [
    "run_export_camera",
    "run_colmap_reconstruction",
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
    "run_stage_colmap",
    "run_stage_mocap",
    "run_stage_gsir",
    "run_stage_camera",
    "STAGE_HANDLERS",
]


def run_export_camera(project_dir: Path, fps: float = 24.0) -> bool:
    """Run camera export script.

    Args:
        project_dir: Project directory with camera data
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
        "--fps", str(fps),
        "--format", "all"
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
    use_masks: bool = True,
    max_image_size: int = -1
) -> bool:
    """Run COLMAP Structure-from-Motion reconstruction.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high')
        run_dense: Whether to run dense reconstruction
        run_mesh: Whether to generate mesh
        use_masks: If True, use segmentation masks from roto/ (if available)
        max_image_size: Maximum image dimension (-1 for no limit)

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
) -> bool:
    """Run human motion capture with GVHMR.

    Args:
        project_dir: Project directory with frames and camera data
        use_colmap_intrinsics: Use COLMAP focal length for GVHMR

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

    if not use_colmap_intrinsics:
        cmd.append("--no-colmap-intrinsics")

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
    """
    signal_file = project_dir / INTERACTIVE_SIGNAL_FILE
    signal_file.unlink(missing_ok=True)

    print("  Waiting for interactive segmentation to complete...")
    print("    - Click 'Complete Interactive Roto' in the web UI, OR")
    print("    - Press Enter in this terminal")
    print()

    while True:
        if signal_file.exists():
            signal_file.unlink(missing_ok=True)
            print("  → Signal received from web UI")
            return

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

        success = process_roto_directory(
            project_dir=ctx.project_dir,
            roto_subdir=roto_subdir.name,
            output_dir=out_dir,
        )

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
    """Run cleanplate generation stage.

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: cleanplate ===")
    workflow_path = ctx.project_dir / "workflows" / "03_cleanplate.json"
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

    refresh_workflow_from_template(workflow_path, "03_cleanplate.json")

    if not workflow_path.exists():
        print("  → Skipping (workflow not found)")
        return True

    if ctx.skip_existing and list(cleanplate_dir.glob("*.png")):
        print("  → Skipping (cleanplates exist)")
        return True

    if ctx.overwrite:
        clear_output_directory(cleanplate_dir)

    print(f"  → {roto_message}")

    update_cleanplate_resolution(workflow_path, ctx.source_frames)

    if not run_comfyui_workflow(
        workflow_path, ctx.comfyui_url,
        output_dir=cleanplate_dir,
        total_frames=ctx.total_frames,
        stage_name="cleanplate",
    ):
        print("  → Cleanplate stage failed", file=sys.stderr)

    if ctx.auto_movie and list(cleanplate_dir.glob("*.png")):
        generate_preview_movie(cleanplate_dir, ctx.project_dir / "preview" / "cleanplate.mp4", ctx.fps)

    clear_gpu_memory(ctx.comfyui_url)
    return True


def run_stage_colmap(
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
    print("\n=== Stage: colmap ===")
    colmap_sparse = ctx.project_dir / "colmap" / "sparse" / "0"

    if ctx.skip_existing and colmap_sparse.exists():
        print("  → Skipping (COLMAP sparse model exists)")
    else:
        if not run_colmap_reconstruction(
            ctx.project_dir,
            quality=config.colmap_quality,
            run_dense=config.colmap_dense,
            run_mesh=config.colmap_mesh,
            use_masks=config.colmap_use_masks,
            max_image_size=config.colmap_max_size
        ):
            print("  → COLMAP reconstruction failed", file=sys.stderr)

    camera_dir = ctx.project_dir / "camera"
    if (camera_dir / "extrinsics.json").exists():
        print("\n  → Exporting camera to VFX formats...")
        export_camera_to_vfx_formats(
            ctx.project_dir,
            start_frame=1,
            fps=ctx.fps,
        )

    clear_gpu_memory(ctx.comfyui_url)
    return True


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

    if ctx.skip_existing and mocap_output.exists():
        print("  → Skipping (mocap data exists)")
    else:
        if has_camera:
            print(f"  → Using COLMAP camera data for improved accuracy")
        if not run_mocap(
            ctx.project_dir,
            use_colmap_intrinsics=has_camera,
        ):
            print("  → Motion capture failed", file=sys.stderr)

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
    colmap_sparse = ctx.project_dir / "colmap" / "sparse" / "0"
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

    Args:
        ctx: Stage execution context
        config: Pipeline configuration

    Returns:
        True if successful
    """
    print("\n=== Stage: camera ===")
    camera_dir = ctx.project_dir / "camera"

    if not (camera_dir / "extrinsics.json").exists():
        print("  → Skipping (no camera data - run colmap stage first)")
        return True

    if ctx.skip_existing and (camera_dir / "camera.abc").exists():
        print("  → Skipping (camera.abc exists)")
        return True

    if not run_export_camera(ctx.project_dir, ctx.fps):
        print("  → Camera export failed", file=sys.stderr)

    return True


STAGE_HANDLERS: dict[str, Callable[["StageContext", "PipelineConfig"], bool]] = {
    "ingest": run_stage_ingest,
    "interactive": run_stage_interactive,
    "depth": run_stage_depth,
    "roto": run_stage_roto,
    "mama": run_stage_mama,
    "cleanplate": run_stage_cleanplate,
    "colmap": run_stage_colmap,
    "mocap": run_stage_mocap,
    "gsir": run_stage_gsir,
    "camera": run_stage_camera,
}
