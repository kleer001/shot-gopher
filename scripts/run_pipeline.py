#!/usr/bin/env python3
"""Automated VFX pipeline runner.

Single command to process footage through the entire pipeline:
  Movie file → Frame extraction → ComfyUI workflows → Post-processing

Auto-detects your environment:
- Local ComfyUI installed → runs locally
- Docker image exists → runs in container

Usage:
    python run_pipeline.py <input_movie> [options]

Example:
    python run_pipeline.py /path/to/footage.mp4 --name "My_Shot" --stages all
    python run_pipeline.py /path/to/footage.mp4 --stages depth,roto,cleanplate
    python run_pipeline.py /path/to/footage.mp4 --docker  # Force Docker mode
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from env_config import check_conda_env_or_warn, DEFAULT_PROJECTS_DIR, is_in_container, INSTALL_DIR
from comfyui_utils import DEFAULT_COMFYUI_URL, run_comfyui_workflow
from comfyui_manager import ensure_comfyui, stop_comfyui, kill_all_comfyui_processes

from pipeline_constants import (
    START_FRAME,
    STAGES,
    STAGE_ORDER,
    STAGES_REQUIRING_FRAMES,
)
from pipeline_utils import (
    clear_gpu_memory,
    clear_output_directory,
    extract_frames,
    get_video_info,
    generate_preview_movie,
)
from matte_utils import combine_mattes, combine_mask_sequences
from workflow_utils import (
    refresh_workflow_from_template,
    update_segmentation_prompt,
    update_matanyone_input,
    update_cleanplate_resolution,
)
from stage_runners import (
    run_export_camera,
    run_colmap_reconstruction,
    export_camera_to_vfx_formats,
    run_mocap,
    run_gsir_materials,
    setup_project,
)

# Log capture for debugging
from log_manager import LogCapture

# Docker configuration
CONTAINER_NAME = "vfx-pipeline-run"
REPO_ROOT = Path(__file__).resolve().parent.parent
COMFYUI_DIR = INSTALL_DIR / "ComfyUI"


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_docker_image_exists() -> bool:
    """Check if the vfx-ingest Docker image exists."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "vfx-ingest:latest"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_local_comfyui_installed() -> bool:
    """Check if local ComfyUI installation exists."""
    return COMFYUI_DIR.exists() and (COMFYUI_DIR / "main.py").exists()


def detect_execution_mode() -> str:
    """Auto-detect the best execution mode.

    Returns:
        'local' if local ComfyUI is installed
        'docker' if Docker is available with the vfx-ingest image
        'none' if neither is available
    """
    if check_local_comfyui_installed():
        return "local"

    if check_docker_available() and check_docker_image_exists():
        return "docker"

    return "none"


def find_default_models_dir() -> Path:
    """Find the default models directory."""
    env_models = os.environ.get("VFX_MODELS_DIR")
    if env_models:
        return Path(env_models)
    default_path = INSTALL_DIR / "models"
    if default_path.exists():
        return default_path
    return REPO_ROOT / ".vfx_pipeline" / "models"


def run_docker_mode(
    input_path: Path,
    project_name: Optional[str],
    projects_dir: Path,
    models_dir: Path,
    original_args: list[str],
) -> int:
    """Run the pipeline in Docker mode.

    Args:
        input_path: Path to input movie file
        project_name: Project name (or None to derive from input)
        projects_dir: Directory for projects
        models_dir: Path to models directory
        original_args: Original command line arguments (to forward to container)

    Returns:
        Exit code
    """
    input_path = input_path.resolve()
    projects_dir = projects_dir.resolve()
    models_dir = models_dir.resolve()

    if not project_name:
        project_name = input_path.stem.replace(" ", "_")

    print(f"\n{'='*60}")
    print("VFX Pipeline (Docker Mode)")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Project: {project_name}")
    print(f"Projects dir: {projects_dir}")
    print(f"Models: {models_dir}")

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}", file=sys.stderr)
        print("Run model download or specify --models-dir", file=sys.stderr)
        return 1

    if not check_docker_available():
        print("Error: Docker is not available or not running", file=sys.stderr)
        return 1

    if not check_docker_image_exists():
        print("Error: Docker image 'vfx-ingest:latest' not found", file=sys.stderr)
        print("Build it first: docker compose build", file=sys.stderr)
        return 1

    projects_dir.mkdir(parents=True, exist_ok=True)

    container_input = f"/workspace/input/{input_path.name}"
    container_projects = "/workspace/projects"

    forward_args = []
    skip_next = False
    for i, arg in enumerate(original_args[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--docker", "-D"):
            continue
        if arg in ("--local", "-L"):
            continue
        if arg in ("--models-dir",) and i + 1 < len(original_args) - 1:
            skip_next = True
            continue
        if arg.startswith("--models-dir="):
            continue
        if arg in ("--projects-dir", "-p") and i + 1 < len(original_args) - 1:
            skip_next = True
            continue
        if arg.startswith("--projects-dir="):
            continue
        if i == 0:
            forward_args.append(container_input)
        else:
            forward_args.append(arg)

    forward_args.extend(["--projects-dir", container_projects])
    forward_args.append("--local")

    print(f"\nStarting Docker container...")

    def signal_handler(sig, frame):
        print("\n\nInterrupted! Cleaning up...")
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        cmd = [
            "docker", "run",
            "--rm",
            "--name", CONTAINER_NAME,
            "--gpus", "all",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", "START_COMFYUI=false",
            "-v", f"{input_path.parent}:/workspace/input:ro",
            "-v", f"{projects_dir}:/workspace/projects",
            "-v", f"{models_dir}:/models:ro",
            "-v", f"{REPO_ROOT / 'scripts'}:/app/scripts:ro",
            "-v", f"{REPO_ROOT / 'workflow_templates'}:/app/workflow_templates:ro",
            "vfx-ingest:latest",
        ] + forward_args

        print(f"Running: docker run ... {' '.join(forward_args)}")
        print()

        result = subprocess.run(cmd, cwd=REPO_ROOT)
        return result.returncode

    except Exception as e:
        print(f"Error running Docker container: {e}", file=sys.stderr)
        return 1

    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


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
    overwrite: bool = True,
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
        roto_prompt: Segmentation prompt (default: 'person')
        separate_instances: Separate multi-person masks into individual instances
        auto_movie: Generate preview MP4s from completed image sequences
        overwrite: Clear existing output before running stages (default: True)

    Returns:
        True if all stages successful
    """
    stages = stages or list(STAGES.keys())

    comfyui_stages = {"depth", "roto", "matanyone", "cleanplate"}
    needs_comfyui = bool(comfyui_stages & set(stages))

    if needs_comfyui:
        print("\n[GPU Cleanup]")
        kill_all_comfyui_processes()
        clear_gpu_memory(comfyui_url)

    comfyui_was_started = False
    if needs_comfyui and auto_start_comfyui:
        print("\n[ComfyUI] Starting ComfyUI...")
        if not ensure_comfyui(url=comfyui_url):
            print("Error: Failed to start ComfyUI", file=sys.stderr)
            print("Install ComfyUI with the install wizard or start it manually", file=sys.stderr)
            return False
        comfyui_was_started = True

    if not project_name:
        project_name = input_path.stem.replace(" ", "_")

    project_dir = projects_dir / project_name

    if is_in_container():
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

    project_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = project_dir / "project.json"
    project_metadata = {
        "name": project_name,
        "fps": fps,
        "source": str(input_path),
        "start_frame": 1,
    }
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

    print("\n[Setup]")
    if not setup_project(project_dir, workflows_dir):
        print("Failed to set up project", file=sys.stderr)
        return False

    if "ingest" in stages:
        print("\n=== Stage: ingest ===")
        if skip_existing and list(source_frames.glob("frame_*.png")):
            print("  → Skipping (frames exist)")
        else:
            frame_count = extract_frames(input_path, source_frames, START_FRAME, fps)
            print(f"  → Extracted {frame_count} frames")

        preview_dir = project_dir / "preview"
        preview_dir.mkdir(exist_ok=True)
        source_preview = preview_dir / f"source{input_path.suffix}"
        if not source_preview.exists():
            shutil.copy2(input_path, source_preview)
            print(f"  → Copied source to {source_preview.name}")

    total_frames = len(list(source_frames.glob("frame_*.png")))

    if total_frames > 0:
        project_metadata["frame_count"] = total_frames
        first_frame = sorted(source_frames.glob("frame_*.png"))[0]
        from PIL import Image
        with Image.open(first_frame) as img:
            project_metadata["width"] = img.width
            project_metadata["height"] = img.height
        with open(metadata_path, "w") as f:
            json.dump(project_metadata, f, indent=2)

    if "interactive" in stages:
        print("\n=== Stage: interactive ===")
        workflow_path = project_dir / "workflows" / "05_interactive_segmentation.json"
        roto_dir = project_dir / "roto"

        refresh_workflow_from_template(workflow_path, "05_interactive_segmentation.json")

        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        else:
            print("  → Opening interactive segmentation in ComfyUI")
            print(f"    Workflow: {workflow_path}")
            print(f"    ComfyUI: {comfyui_url}")
            print()
            print("  Instructions:")
            print("    1. Open ComfyUI in your browser")
            print("    2. Load the workflow from: workflows/05_interactive_segmentation.json")
            print("    3. Click points on the first frame to define what to segment")
            print("    4. Run the workflow (Queue Prompt)")
            print("    5. Masks will be saved to: roto/")
            print()

            import webbrowser
            webbrowser.open(comfyui_url)

            input("  Press Enter when done with interactive segmentation...")

            if list(roto_dir.glob("**/*.png")):
                print(f"  ✓ Masks found in {roto_dir}")
            else:
                print(f"  → Warning: No masks found in {roto_dir}")

    if "depth" in stages:
        print("\n=== Stage: depth ===")
        workflow_path = project_dir / "workflows" / "01_analysis.json"
        depth_dir = project_dir / "depth"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list(depth_dir.glob("*.png")):
            print("  → Skipping (depth maps exist)")
        else:
            if overwrite:
                clear_output_directory(depth_dir)
            if not run_comfyui_workflow(
                workflow_path, comfyui_url,
                output_dir=depth_dir,
                total_frames=total_frames,
                stage_name="depth",
            ):
                print("  → Depth stage failed", file=sys.stderr)
                return False
        if auto_movie and list(depth_dir.glob("*.png")):
            generate_preview_movie(depth_dir, project_dir / "preview" / "depth.mp4", fps)

        clear_gpu_memory(comfyui_url)

    if "roto" in stages:
        print("\n=== Stage: roto ===")
        workflow_path = project_dir / "workflows" / "02_segmentation.json"
        roto_dir = project_dir / "roto"

        refresh_workflow_from_template(workflow_path, "02_segmentation.json")

        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and (list(roto_dir.glob("*.png")) or list(roto_dir.glob("*/*.png"))):
            print("  → Skipping (masks exist)")
        else:
            if overwrite:
                clear_output_directory(roto_dir)

            prompts = [p.strip() for p in (roto_prompt or "person").split(",")]
            prompts = [p for p in prompts if p]

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

        if auto_movie:
            for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
                if subdir.is_dir() and list(subdir.glob("*.png")):
                    generate_preview_movie(subdir, project_dir / "preview" / "roto.mp4", fps)
                    break

        if separate_instances:
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

        clear_gpu_memory(comfyui_url)

    if "matanyone" in stages:
        print("\n=== Stage: matanyone ===")
        workflow_path = project_dir / "workflows" / "04_matanyone.json"
        roto_dir = project_dir / "roto"
        combined_dir = roto_dir / "combined"
        matte_dir = project_dir / "matte"

        refresh_workflow_from_template(workflow_path, "04_matanyone.json")

        person_pattern = re.compile(r"^person_\d{2}$")
        person_dirs = []
        for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
            if not subdir.is_dir():
                continue
            if person_pattern.match(subdir.name) and list(subdir.glob("*.png")):
                person_dirs.append(subdir)

        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif not person_dirs:
            print("  → Skipping (no person masks found in roto/)")
        else:
            if overwrite:
                clear_output_directory(matte_dir)
                if combined_dir.exists():
                    clear_output_directory(combined_dir)

            output_dirs = []

            for i, person_dir in enumerate(person_dirs):
                out_dir = matte_dir / person_dir.name
                output_dirs.append(out_dir)

                if skip_existing and list(out_dir.glob("*.png")):
                    print(f"  → Skipping {person_dir.name} (mattes exist)")
                    continue

                if len(person_dirs) > 1:
                    print(f"\n  [{i+1}/{len(person_dirs)}] Refining: {person_dir.name}")
                else:
                    print(f"  → Refining masks from: {person_dir.name}")

                out_dir.mkdir(parents=True, exist_ok=True)

                update_matanyone_input(workflow_path, person_dir, out_dir, project_dir)
                if not run_comfyui_workflow(
                    workflow_path, comfyui_url,
                    output_dir=out_dir,
                    total_frames=total_frames,
                    stage_name=f"matanyone ({person_dir.name})" if len(person_dirs) > 1 else "matanyone",
                ):
                    print(f"  → MatAnyone stage failed for {person_dir.name}", file=sys.stderr)
                    print(f"    (cleanplate will use raw roto masks instead)", file=sys.stderr)

            valid_output_dirs = [d for d in output_dirs if d.exists() and list(d.glob("*.png"))]
            if valid_output_dirs:
                if skip_existing and list(combined_dir.glob("*.png")):
                    print(f"  → Skipping combine (roto/combined/ exists)")
                else:
                    print("\n  --- Combining mattes ---")
                    combine_mattes(valid_output_dirs, combined_dir, "combined")

            if auto_movie and list(combined_dir.glob("*.png")):
                generate_preview_movie(combined_dir, project_dir / "preview" / "matte.mp4", fps)

        clear_gpu_memory(comfyui_url)

    if "cleanplate" in stages:
        print("\n=== Stage: cleanplate ===")
        workflow_path = project_dir / "workflows" / "03_cleanplate.json"
        cleanplate_dir = project_dir / "cleanplate"
        roto_dir = project_dir / "roto"
        combined_dir = roto_dir / "combined"

        has_combined_mattes = combined_dir.exists() and list(combined_dir.glob("*.png"))
        mask_dirs = []
        for subdir in sorted(roto_dir.iterdir()) if roto_dir.exists() else []:
            if subdir.is_dir() and subdir.name != "combined" and list(subdir.glob("*.png")):
                mask_dirs.append(subdir)
        has_any_roto = has_combined_mattes or mask_dirs

        if not has_any_roto:
            print("")
            print("  " + "=" * 50)
            print("  !!!  NO ROTO DATA FOUND - SKIPPING CLEANPLATE  !!!")
            print("  " + "=" * 50)
            print("  Run the 'roto' stage first to generate masks.")
            print("")
        else:
            refresh_workflow_from_template(workflow_path, "03_cleanplate.json")

            if not workflow_path.exists():
                print("  → Skipping (workflow not found)")
            elif skip_existing and list(cleanplate_dir.glob("*.png")):
                print("  → Skipping (cleanplates exist)")
            else:
                if overwrite:
                    clear_output_directory(cleanplate_dir)

                for old_file in roto_dir.glob("*.png"):
                    old_file.unlink()

                if has_combined_mattes:
                    print(f"  → Using combined MatAnyone mattes from roto/combined/")
                    for i, mask_file in enumerate(sorted(combined_dir.glob("*.png"))):
                        out_name = f"mask_{i+1:05d}.png"
                        shutil.copy2(mask_file, roto_dir / out_name)
                elif len(mask_dirs) > 1:
                    count = combine_mask_sequences(mask_dirs, roto_dir, prefix="combined")
                    print(f"  → Consolidated {count} frames from {len(mask_dirs)} mask sources")
                elif len(mask_dirs) == 1:
                    source_dir = mask_dirs[0]
                    for i, mask_file in enumerate(sorted(source_dir.glob("*.png"))):
                        out_name = f"mask_{i+1:05d}.png"
                        shutil.copy2(mask_file, roto_dir / out_name)
                    print(f"  → Using masks from {source_dir.name}/")

                update_cleanplate_resolution(workflow_path, source_frames)
                if not run_comfyui_workflow(
                    workflow_path, comfyui_url,
                    output_dir=cleanplate_dir,
                    total_frames=total_frames,
                    stage_name="cleanplate",
                ):
                    print("  → Cleanplate stage failed", file=sys.stderr)
        if auto_movie and list(cleanplate_dir.glob("*.png")):
            generate_preview_movie(cleanplate_dir, project_dir / "preview" / "cleanplate.mp4", fps)

        clear_gpu_memory(comfyui_url)

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

        camera_dir = project_dir / "camera"
        if (camera_dir / "extrinsics.json").exists():
            print("\n  → Exporting camera to VFX formats...")
            export_camera_to_vfx_formats(
                project_dir,
                start_frame=1,
                fps=fps,
            )

        clear_gpu_memory(comfyui_url)

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
                skip_texture=False,
            ):
                print("  → Motion capture failed", file=sys.stderr)

        clear_gpu_memory(comfyui_url)

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

        clear_gpu_memory(comfyui_url)

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

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {project_dir}")
    print(f"{'='*60}\n")

    if comfyui_was_started:
        print("[ComfyUI] Stopping ComfyUI...")
        stop_comfyui()

    return True


def main():
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
        "--list-stages",
        action="store_true",
        help="List available stages and exit"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--docker", "-D",
        action="store_true",
        help="Force Docker mode (auto-detected if not specified)"
    )
    mode_group.add_argument(
        "--local", "-L",
        action="store_true",
        help="Force local mode (auto-detected if not specified)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Path to models directory (Docker mode, auto-detected if not specified)"
    )

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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Separate multi-person masks into individual instances (default: True)"
    )
    parser.add_argument(
        "--auto-movie",
        action="store_true",
        help="Generate preview MP4s from completed image sequences (depth, roto, cleanplate)"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing output files instead of clearing them before running stages"
    )

    args = parser.parse_args()

    if args.list_stages:
        print("Available stages:")
        for name, desc in STAGES.items():
            print(f"  {name}: {desc}")
        sys.exit(0)

    if args.docker:
        mode = "docker"
    elif args.local:
        mode = "local"
    else:
        mode = detect_execution_mode()
        if mode == "none":
            print("Error: No valid execution environment detected.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Options:", file=sys.stderr)
            print("  1. Install locally: python scripts/install_wizard.py", file=sys.stderr)
            print("  2. Build Docker image: docker compose build", file=sys.stderr)
            print("", file=sys.stderr)
            print("Or force a mode with --docker or --local", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected mode: {mode}")

    if mode == "docker":
        models_dir = args.models_dir or find_default_models_dir()
        exit_code = run_docker_mode(
            input_path=args.input,
            project_name=args.name,
            projects_dir=args.projects_dir,
            models_dir=models_dir,
            original_args=sys.argv,
        )
        sys.exit(exit_code)

    check_conda_env_or_warn()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.stages.lower() == "all":
        stages = STAGE_ORDER.copy()
    else:
        stages = [s.strip() for s in args.stages.split(",")]
        invalid = set(stages) - set(STAGES.keys())
        if invalid:
            print(f"Error: Invalid stages: {invalid}", file=sys.stderr)
            print(f"Valid stages: {', '.join(STAGE_ORDER)}")
            sys.exit(1)
        stages = sanitize_stages(stages)

    print(f"Stages to run: {', '.join(stages)}")

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
        overwrite=not args.no_overwrite,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
