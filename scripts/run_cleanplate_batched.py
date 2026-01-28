#!/usr/bin/env python3
"""Run batched cleanplate workflow for low-VRAM GPUs.

Processes large frame sequences in chunks with overlap, then blends the
overlapping frames for seamless results. Designed for GPUs that can't
process entire sequences at once.

Usage:
    python run_cleanplate_batched.py /path/to/project
    python run_cleanplate_batched.py /path/to/project --batch-size 10 --overlap 2
    python run_cleanplate_batched.py /path/to/project --dry-run
    python run_cleanplate_batched.py /path/to/project --resume
    python run_cleanplate_batched.py /path/to/project --docker  # Force Docker mode

Requirements:
    - ComfyUI (via Docker container or local installation)
    - Source frames in project/source/frames/
    - Roto masks in project/roto/
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from env_config import is_in_container
from comfyui_utils import (
    DEFAULT_COMFYUI_URL,
    free_comfyui_memory,
    queue_workflow,
    wait_for_completion,
)
from comfyui_manager import prepare_comfyui_for_processing
from workflow_utils import WORKFLOW_TEMPLATES_DIR
from matte_utils import prepare_roto_for_cleanplate


REPO_ROOT = Path(__file__).parent.parent
TEMPLATE_FILE = "03_cleanplate_chunk_template.json"


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
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
    except Exception:
        return False


def detect_execution_mode() -> str:
    """Detect whether to run in local or Docker mode.

    Returns:
        'container' if already in container
        'docker' if Docker is available with the vfx-ingest image
        'local' if local ComfyUI exists
        'none' if no execution environment available
    """
    if is_in_container():
        return "container"

    if check_docker_available() and check_docker_image_exists():
        return "docker"

    comfyui_path = REPO_ROOT / ".vfx_pipeline" / "ComfyUI"
    if comfyui_path.exists() and (comfyui_path / "main.py").exists():
        return "local"

    return "none"


def run_docker_mode(
    project_dir: Path,
    projects_dir: Path,
    models_dir: Path,
    original_args: list[str],
) -> int:
    """Run batched cleanplate in Docker mode.

    Args:
        project_dir: Path to project directory
        projects_dir: Directory containing projects
        models_dir: Path to models directory
        original_args: Original command line arguments

    Returns:
        Exit code
    """
    project_dir = project_dir.resolve()
    projects_dir = projects_dir.resolve()
    models_dir = models_dir.resolve()

    print(f"\n{'='*60}")
    print("Batched Cleanplate (Docker Mode)")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")
    print(f"Projects dir: {projects_dir}")
    print(f"Models: {models_dir}")

    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}", file=sys.stderr)
        return 1

    if not check_docker_available():
        print("Error: Docker is not available or not running", file=sys.stderr)
        return 1

    if not check_docker_image_exists():
        print("Error: Docker image 'vfx-ingest:latest' not found", file=sys.stderr)
        print("Build it first: docker compose build", file=sys.stderr)
        return 1

    container_project = f"/workspace/projects/{project_dir.name}"

    forward_args = []
    skip_next = False
    project_arg_replaced = False
    for i, arg in enumerate(original_args[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--docker", "-D", "--local", "-L"):
            continue
        if arg in ("--models-dir",) and i + 1 < len(original_args) - 1:
            skip_next = True
            continue
        if arg.startswith("--models-dir="):
            continue
        if not arg.startswith("-") and not project_arg_replaced:
            try:
                arg_resolved = Path(arg).resolve()
                if arg_resolved == project_dir:
                    forward_args.append(container_project)
                    project_arg_replaced = True
                    continue
            except Exception:
                pass
        forward_args.append(arg)

    if not project_arg_replaced:
        forward_args.insert(0, container_project)

    env = os.environ.copy()
    env["VFX_MODELS_DIR"] = str(models_dir)
    env["VFX_PROJECTS_DIR"] = str(projects_dir)
    env["HOST_UID"] = str(os.getuid()) if hasattr(os, 'getuid') else "0"
    env["HOST_GID"] = str(os.getgid()) if hasattr(os, 'getgid') else "0"

    docker_cmd = [
        "docker", "compose", "run", "--rm",
        "vfx-ingest",
        "cleanplate-batched",
    ] + forward_args

    print(f"\nStarting Docker container...")
    print(f"Running: docker compose run ... cleanplate-batched {' '.join(forward_args)}")

    try:
        result = subprocess.run(
            docker_cmd,
            cwd=str(REPO_ROOT),
            env=env,
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}", file=sys.stderr)
        return 1


STATE_FILENAME = "cleanplate_batch_state.json"


@dataclass
class ChunkInfo:
    """Information about a processing chunk."""
    index: int
    start_frame: int
    end_frame: int
    frame_count: int

    @property
    def name(self) -> str:
        return f"{self.start_frame:03d}_{self.end_frame:03d}"

    @property
    def workflow_name(self) -> str:
        return f"chunk_{self.name}.json"


def calculate_chunks(
    total_frames: int,
    batch_size: int,
    overlap: int
) -> List[ChunkInfo]:
    """Calculate chunk boundaries with overlap.

    Args:
        total_frames: Total number of frames to process
        batch_size: Frames per chunk
        overlap: Frames to overlap between chunks

    Returns:
        List of ChunkInfo objects
    """
    if total_frames <= batch_size:
        return [ChunkInfo(
            index=0,
            start_frame=0,
            end_frame=total_frames - 1,
            frame_count=total_frames
        )]

    chunks = []
    step = batch_size - overlap
    start = 0
    index = 0

    while start < total_frames:
        end = min(start + batch_size - 1, total_frames - 1)
        frame_count = end - start + 1

        chunks.append(ChunkInfo(
            index=index,
            start_frame=start,
            end_frame=end,
            frame_count=frame_count
        ))

        if end >= total_frames - 1:
            break

        start += step
        index += 1

    return chunks


def count_source_frames(project_dir: Path) -> int:
    """Count frames in project source/frames directory."""
    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        return 0

    frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
    return len(sorted(frame_files))


def load_template_workflow() -> dict:
    """Load the chunk template workflow."""
    template_path = WORKFLOW_TEMPLATES_DIR / TEMPLATE_FILE
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path) as f:
        return json.load(f)


def generate_chunk_workflow(
    template: dict,
    chunk: ChunkInfo,
    output_prefix: str
) -> dict:
    """Generate a workflow for a specific chunk.

    Args:
        template: Base workflow template
        chunk: Chunk information
        output_prefix: Output path prefix for SaveImage

    Returns:
        Modified workflow dict
    """
    workflow = json.loads(json.dumps(template))

    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")
        widgets = node.get("widgets_values", [])

        if node_type == "VHS_LoadImagesPath" and widgets and len(widgets) >= 3:
            widgets[1] = chunk.frame_count
            widgets[2] = chunk.start_frame

        elif node_type == "SaveImage" and widgets:
            widgets[0] = output_prefix

    return workflow


def save_chunk_workflow(
    workflow: dict,
    project_dir: Path,
    chunk: ChunkInfo
) -> Path:
    """Save chunk workflow to project workflows directory.

    Returns:
        Path to saved workflow file
    """
    workflows_dir = project_dir / "workflows" / "cleanplate"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    output_path = workflows_dir / chunk.workflow_name
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)

    return output_path


def load_batch_state(project_dir: Path) -> dict:
    """Load batch processing state."""
    state_file = project_dir / "cleanplate" / STATE_FILENAME
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"completed_chunks": [], "settings": {}}


def save_batch_state(project_dir: Path, state: dict) -> None:
    """Save batch processing state."""
    state_dir = project_dir / "cleanplate"
    state_dir.mkdir(parents=True, exist_ok=True)

    state_file = state_dir / STATE_FILENAME
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def verify_chunk_output(
    project_dir: Path,
    chunk: ChunkInfo
) -> bool:
    """Verify chunk output files exist.

    Returns:
        True if all expected output files exist
    """
    output_dir = project_dir / "output" / "cleanplate" / "chunks" / chunk.name
    if not output_dir.exists():
        comfyui_output = Path("/tmp/comfyui_output")
        if comfyui_output.exists():
            output_dir = comfyui_output

    if not output_dir.exists():
        return False

    output_files = list(output_dir.glob("*.png"))
    return len(output_files) >= chunk.frame_count


def process_chunk(
    project_dir: Path,
    chunk: ChunkInfo,
    template: dict,
    comfyui_url: str,
    timeout: int = 1800,
) -> bool:
    """Process a single chunk through ComfyUI.

    Returns:
        True if successful
    """
    output_prefix = f"cleanplate/chunks/{chunk.name}/clean"

    workflow = generate_chunk_workflow(template, chunk, output_prefix)

    workflow_path = save_chunk_workflow(workflow, project_dir, chunk)

    print(f"  → Queuing chunk {chunk.index}: frames {chunk.start_frame}-{chunk.end_frame}")

    prompt_id = queue_workflow(workflow_path, comfyui_url)
    if not prompt_id:
        print(f"    Error: Failed to queue chunk {chunk.index}", file=sys.stderr)
        return False

    print(f"    Prompt ID: {prompt_id}")
    print(f"    Waiting for completion...")

    success = wait_for_completion(
        prompt_id,
        comfyui_url,
        timeout=timeout,
        total_frames=chunk.frame_count,
        stage_name=f"Chunk {chunk.index}"
    )

    if success:
        print(f"    ✓ Chunk {chunk.index} complete")
    else:
        print(f"    ✗ Chunk {chunk.index} failed", file=sys.stderr)

    return success


def blend_images(
    img_a: Image.Image,
    img_b: Image.Image,
    weight_b: float
) -> Image.Image:
    """Blend two images with given weight.

    Args:
        img_a: First image
        img_b: Second image
        weight_b: Weight for img_b (0.0 = all A, 1.0 = all B)

    Returns:
        Blended image
    """
    return Image.blend(img_a, img_b, weight_b)


def find_chunk_outputs(
    project_dir: Path,
    chunk: ChunkInfo
) -> List[Path]:
    """Find output files for a chunk.

    ComfyUI saves to output/ directory with prefix-based naming.
    """
    possible_dirs = [
        project_dir / "output" / "cleanplate" / "chunks" / chunk.name,
        project_dir / "cleanplate" / "chunks" / chunk.name,
    ]

    for output_dir in possible_dirs:
        if output_dir.exists():
            files = sorted(output_dir.glob("clean*.png"))
            if files:
                return files

    return []


def blend_overlaps(
    project_dir: Path,
    chunks: List[ChunkInfo],
    overlap: int
) -> bool:
    """Blend overlapping frames between chunks.

    Args:
        project_dir: Project directory
        chunks: List of processed chunks
        overlap: Number of overlap frames

    Returns:
        True if successful
    """
    final_dir = project_dir / "cleanplate" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if len(chunks) <= 1:
        print("  Single chunk - copying to final directory...")
        chunk_files = find_chunk_outputs(project_dir, chunks[0])
        if not chunk_files:
            print("    Error: No output files found for single chunk", file=sys.stderr)
            return False
        for i, filepath in enumerate(chunk_files):
            output_path = final_dir / f"clean_{i + 1:05d}.png"
            shutil.copy2(filepath, output_path)
        print(f"    Copied {len(chunk_files)} frames to {final_dir}")
        return True

    print(f"\n  Blending {len(chunks)} chunks with {overlap}-frame overlap...")

    all_output_frames = {}

    for chunk in chunks:
        chunk_files = find_chunk_outputs(project_dir, chunk)
        if not chunk_files:
            print(f"    Warning: No output files found for chunk {chunk.name}")
            continue

        for i, filepath in enumerate(chunk_files):
            global_frame = chunk.start_frame + i

            if global_frame not in all_output_frames:
                all_output_frames[global_frame] = []
            all_output_frames[global_frame].append((chunk.index, filepath))

    if not all_output_frames:
        print("    Error: No output frames found for any chunk", file=sys.stderr)
        return False

    total_frames = max(all_output_frames.keys()) + 1
    print(f"    Processing {total_frames} frames...")

    for frame_idx in range(total_frames):
        sources = all_output_frames.get(frame_idx, [])

        if not sources:
            print(f"    Warning: No source for frame {frame_idx}")
            continue

        output_path = final_dir / f"clean_{frame_idx + 1:05d}.png"

        if len(sources) == 1:
            shutil.copy2(sources[0][1], output_path)
        else:
            sources.sort(key=lambda x: x[0])
            chunk_a_idx, path_a = sources[0]
            _, path_b = sources[1]

            chunk_a = next(c for c in chunks if c.index == chunk_a_idx)
            position_in_overlap = frame_idx - chunk_a.end_frame + overlap - 1
            weight_b = (position_in_overlap + 1) / (overlap + 1)

            img_a = Image.open(path_a)
            img_b = Image.open(path_b)

            blended = blend_images(img_a, img_b, weight_b)
            blended.save(output_path)

            img_a.close()
            img_b.close()

    output_files = list(final_dir.glob("clean_*.png"))
    print(f"    ✓ Blended {len(output_files)} frames to {final_dir}")

    return True


def run_batched_cleanplate(
    project_dir: Path,
    batch_size: int = 10,
    overlap: int = 2,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    dry_run: bool = False,
    resume: bool = False,
    no_blend: bool = False,
    timeout: int = 1800,
    auto_start_comfyui: bool = True,
) -> bool:
    """Run batched cleanplate processing.

    Args:
        project_dir: Path to project directory
        batch_size: Frames per chunk
        overlap: Frames to overlap between chunks
        comfyui_url: ComfyUI API URL
        dry_run: If True, only show what would be done
        resume: If True, skip completed chunks
        no_blend: If True, skip blending step
        timeout: Timeout per chunk in seconds
        auto_start_comfyui: Auto-start ComfyUI if not running (default: True)

    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print("Batched Cleanplate Processing")
    print(f"{'='*60}")

    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return False

    total_frames = count_source_frames(project_dir)
    if total_frames == 0:
        print(f"Error: No frames found in {project_dir}/source/frames/", file=sys.stderr)
        return False

    print(f"\nProject: {project_dir.name}")
    print(f"Total frames: {total_frames}")
    print(f"Batch size: {batch_size}")
    print(f"Overlap: {overlap}")

    roto_dir = project_dir / "roto"
    roto_ready, roto_message = prepare_roto_for_cleanplate(roto_dir)
    if not roto_ready:
        print(f"\nError: {roto_message}", file=sys.stderr)
        print("Run the 'roto' stage first to generate masks.", file=sys.stderr)
        return False
    print(f"Roto: {roto_message}")

    chunks = calculate_chunks(total_frames, batch_size, overlap)
    print(f"Chunks: {len(chunks)}")

    print(f"\nChunk breakdown:")
    for chunk in chunks:
        print(f"  [{chunk.index}] frames {chunk.start_frame:3d}-{chunk.end_frame:3d} ({chunk.frame_count} frames)")

    if dry_run:
        print("\n[DRY RUN] Would process the above chunks")
        return True

    if not prepare_comfyui_for_processing(url=comfyui_url, auto_start=auto_start_comfyui):
        return False

    template = load_template_workflow()

    state = load_batch_state(project_dir) if resume else {"completed_chunks": [], "settings": {}}
    state["settings"] = {
        "batch_size": batch_size,
        "overlap": overlap,
        "total_frames": total_frames
    }

    completed = set(state.get("completed_chunks", []))

    print(f"\n{'='*60}")
    print("Processing Chunks")
    print(f"{'='*60}")

    for chunk in chunks:
        if chunk.name in completed:
            print(f"\n  Chunk {chunk.index} ({chunk.name}) - already complete, skipping")
            continue

        print(f"\n  Processing chunk {chunk.index}/{len(chunks)-1}...")

        success = process_chunk(
            project_dir,
            chunk,
            template,
            comfyui_url,
            timeout=timeout
        )

        if success:
            completed.add(chunk.name)
            state["completed_chunks"] = list(completed)
            save_batch_state(project_dir, state)
        else:
            print(f"\n  Error: Chunk {chunk.index} failed. Use --resume to continue later.")
            return False

        print("    Clearing GPU memory...")
        free_comfyui_memory(comfyui_url)
        time.sleep(2)

    if no_blend:
        print("\n[SKIP] Blending disabled")
        return True

    print(f"\n{'='*60}")
    print("Blending Overlaps")
    print(f"{'='*60}")

    if not blend_overlaps(project_dir, chunks, overlap):
        print("\nError: Blending failed", file=sys.stderr)
        return False

    print(f"\n{'='*60}")
    print("✓ Batched cleanplate complete!")
    print(f"{'='*60}")
    print(f"\nOutput: {project_dir}/cleanplate/final/")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run batched cleanplate processing for low-VRAM GPUs"
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Path to project directory"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Frames per chunk (default: 10)"
    )
    parser.add_argument(
        "--overlap", "-o",
        type=int,
        default=2,
        help="Frames to overlap between chunks (default: 2)"
    )
    parser.add_argument(
        "--comfyui-url",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"ComfyUI API URL (default: {DEFAULT_COMFYUI_URL})"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last completed chunk"
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Skip blending step (for debugging)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=1800,
        help="Timeout per chunk in seconds (default: 1800)"
    )
    parser.add_argument(
        "--no-auto-comfyui",
        action="store_true",
        help="Don't auto-start ComfyUI (requires manual start)"
    )
    parser.add_argument(
        "--docker", "-D",
        action="store_true",
        help="Force Docker mode"
    )
    parser.add_argument(
        "--local", "-L",
        action="store_true",
        help="Force local mode (skip Docker)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Models directory (Docker mode only)"
    )

    args = parser.parse_args()

    if args.overlap >= args.batch_size:
        print(f"Error: Overlap ({args.overlap}) must be less than batch size ({args.batch_size})")
        sys.exit(1)

    if args.docker and args.local:
        print("Error: Cannot specify both --docker and --local", file=sys.stderr)
        sys.exit(1)

    execution_mode = detect_execution_mode()

    if args.docker:
        execution_mode = "docker"
    elif args.local:
        execution_mode = "local"

    if execution_mode == "docker":
        project_dir = args.project_dir.resolve()
        projects_dir = project_dir.parent
        models_dir = args.models_dir or REPO_ROOT / ".vfx_pipeline" / "models"

        exit_code = run_docker_mode(
            project_dir=project_dir,
            projects_dir=projects_dir,
            models_dir=models_dir,
            original_args=sys.argv,
        )
        sys.exit(exit_code)

    if execution_mode == "none":
        print("Error: No execution environment available", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  1. Build Docker image: docker compose build", file=sys.stderr)
        print("  2. Install local ComfyUI: ./scripts/install_comfyui.sh", file=sys.stderr)
        sys.exit(1)

    success = run_batched_cleanplate(
        project_dir=args.project_dir.resolve(),
        batch_size=args.batch_size,
        overlap=args.overlap,
        comfyui_url=args.comfyui_url,
        dry_run=args.dry_run,
        resume=args.resume,
        no_blend=args.no_blend,
        timeout=args.timeout,
        auto_start_comfyui=not args.no_auto_comfyui,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
