#!/usr/bin/env python3
"""Automated VFX pipeline runner.

Single command to process footage through the entire pipeline:
  Movie file → Frame extraction → ComfyUI workflows → Post-processing

Usage:
    python run_pipeline.py <input_movie> [options]

Example:
    python run_pipeline.py /path/to/footage.mp4 --name "My_Shot" --stages all
    python run_pipeline.py /path/to/footage.mp4 --stages depth,roto,cleanplate
"""

import argparse
import os
import sys
from pathlib import Path

from comfyui_manager import stop_comfyui, prepare_comfyui_for_processing
from comfyui_utils import DEFAULT_COMFYUI_URL
from env_config import require_conda_env, DEFAULT_PROJECTS_DIR, INSTALL_DIR
from log_manager import LogCapture
from pipeline_config import PipelineConfig, StageContext
from pipeline_constants import STAGES, STAGE_ORDER, STAGES_REQUIRING_FRAMES
from pipeline_utils import get_image_dimensions, get_video_info
from project_utils import ProjectMetadata, save_last_project, get_last_project
from stage_runners import setup_project, STAGE_HANDLERS


COMFYUI_DIR = INSTALL_DIR / "ComfyUI"


def check_comfyui_installed() -> bool:
    """Check if ComfyUI installation exists."""
    return COMFYUI_DIR.exists() and (COMFYUI_DIR / "main.py").exists()


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


def run_pipeline(config: PipelineConfig) -> bool:
    """Run the full VFX pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        True if all stages successful
    """
    comfyui_stages = {"depth", "roto", "cleanplate"}
    needs_comfyui = bool(comfyui_stages & set(config.stages))

    comfyui_was_started = False
    if needs_comfyui:
        if not prepare_comfyui_for_processing(url=config.comfyui_url, auto_start=config.auto_start_comfyui):
            return False
        comfyui_was_started = config.auto_start_comfyui

    project_name = config.project_name
    if not project_name:
        if config.input_path:
            project_name = config.input_path.stem.replace(" ", "_")
        else:
            print("Error: project_name required when input_path is None", file=sys.stderr)
            return False

    project_dir = config.projects_dir / project_name
    save_last_project(project_dir)

    source_frames = project_dir / "source" / "frames"
    workflows_dir = Path(__file__).parent.parent / "workflow_templates"

    print(f"\n{'='*60}")
    print(f"VFX Pipeline: {project_name}")
    print(f"{'='*60}")
    if config.input_path:
        print(f"Input: {config.input_path}")
    print(f"Project: {project_dir}")
    print(f"Stages: {', '.join(config.stages)}")
    print()

    metadata = ProjectMetadata(project_dir)
    fps = config.fps or metadata.get_fps()

    if config.input_path and not fps and config.input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"}:
        info = get_video_info(config.input_path)
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
    metadata.initialize(project_name, fps, config.input_path)

    print("\n[Setup]")
    if not setup_project(project_dir, workflows_dir):
        print("Failed to set up project", file=sys.stderr)
        return False

    total_frames = len(list(source_frames.glob("*.png")))
    width, height = 0, 0

    if "ingest" in config.stages:
        ctx = StageContext.from_config(config, project_dir, total_frames, fps)
        handler = STAGE_HANDLERS["ingest"]
        if not handler(ctx, config):
            return False
        total_frames = len(list(source_frames.glob("*.png")))

    if total_frames > 0:
        first_frame = sorted(source_frames.glob("*.png"))[0]
        width, height = get_image_dimensions(first_frame)
        if width > 0 and height > 0:
            metadata.set_frame_info(total_frames, width, height)
            print(f"Source resolution: {width}x{height}")
        else:
            print("Warning: Could not determine source resolution")

    ctx = StageContext.from_config(config, project_dir, total_frames, fps, width, height)

    for stage in config.stages:
        if stage == "ingest":
            continue

        handler = STAGE_HANDLERS.get(stage)
        if handler:
            if not handler(ctx, config):
                if stage == "depth":
                    return False

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {project_dir}")
    print(f"{'='*60}\n")

    if comfyui_was_started:
        print("[ComfyUI] Stopping ComfyUI...")
        stop_comfyui()

    return True


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated VFX pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input movie file, image sequence, or existing project dir (default: last used project)"
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
        "--colmap-max-size",
        type=int,
        default=-1,
        help="Max image dimension for COLMAP (downscales larger images). Use 1000-2000 for faster processing."
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
        "--mocap-gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Body model gender for motion capture (default: neutral)"
    )
    parser.add_argument(
        "--mocap-no-export",
        action="store_true",
        help="Skip automatic Alembic/USD export after motion capture"
    )
    parser.add_argument(
        "--mocap-fps",
        type=float,
        default=None,
        help="Frames per second for mocap export (default: use project fps)"
    )
    parser.add_argument(
        "--mocap-start-frame",
        type=int,
        default=None,
        help="Start frame for motion capture (1-indexed, inclusive)"
    )
    parser.add_argument(
        "--mocap-end-frame",
        type=int,
        default=None,
        help="End frame for motion capture (1-indexed, inclusive)"
    )
    parser.add_argument(
        "--mocap-person",
        type=int,
        default=None,
        help="Person index for multi-person shots (0, 1, 2...). Uses person_N as primary output."
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
        "--start-frame",
        type=int,
        default=None,
        help="Frame to start segmentation from (enables bidirectional propagation). Use when subject isn't visible on first frame."
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

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.list_stages:
        print("Available stages:")
        for name, desc in STAGES.items():
            print(f"  {name}: {desc}")
        sys.exit(0)

    if not check_comfyui_installed():
        print("Error: ComfyUI not installed.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Install with: python scripts/install_wizard.py", file=sys.stderr)
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

    input_path = None
    project_dir = None

    if args.input is None:
        last_project = get_last_project()
        if last_project:
            project_dir = last_project
            print(f"Using last project: {project_dir.name}")
        else:
            print("Error: No input specified and no previous project found", file=sys.stderr)
            print("Usage: run_pipeline.py <movie_file_or_project_dir>", file=sys.stderr)
            sys.exit(1)
    else:
        input_path = args.input.resolve()
        if not input_path.exists():
            print(f"Error: Input not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        if input_path.is_dir() and (input_path / "source").exists():
            project_dir = input_path
            print(f"Using existing project: {project_dir.name}")

    config = PipelineConfig(
        input_path=input_path if not project_dir else None,
        project_name=project_dir.name if project_dir else args.name,
        projects_dir=project_dir.parent if project_dir else args.projects_dir,
        stages=stages,
        comfyui_url=args.comfyui_url,
        fps=args.fps,
        skip_existing=args.skip_existing,
        overwrite=not args.no_overwrite,
        auto_movie=args.auto_movie,
        auto_start_comfyui=not args.no_auto_comfyui,
        colmap_quality=args.colmap_quality,
        colmap_dense=args.colmap_dense,
        colmap_mesh=args.colmap_mesh,
        colmap_use_masks=not args.colmap_no_masks,
        colmap_max_size=args.colmap_max_size,
        gsir_iterations=args.gsir_iterations,
        gsir_path=args.gsir_path,
        mocap_gender=args.mocap_gender,
        mocap_no_export=args.mocap_no_export,
        mocap_fps=args.mocap_fps,
        roto_prompt=args.prompt,
        roto_start_frame=args.start_frame,
        separate_instances=args.separate_instances,
    )

    print(f"Stages to run: {', '.join(config.stages)}")

    require_conda_env()

    if project_dir:
        save_last_project(project_dir)

    success = run_pipeline(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
