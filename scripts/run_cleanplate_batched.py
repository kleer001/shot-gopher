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

Requirements:
    - ComfyUI installed
    - Source frames in project/source/frames/
    - Roto masks in project/roto/
"""

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

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


def check_comfyui_installed() -> bool:
    """Check if ComfyUI is installed."""
    comfyui_path = REPO_ROOT / ".vfx_pipeline" / "ComfyUI"
    return comfyui_path.exists() and (comfyui_path / "main.py").exists()


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

    with open(template_path, encoding='utf-8') as f:
        return json.load(f)


def generate_chunk_workflow(
    template: dict,
    chunk: ChunkInfo,
    project_dir: Path,
    output_prefix: str
) -> dict:
    """Generate a workflow for a specific chunk.

    Args:
        template: Base workflow template
        chunk: Chunk information
        project_dir: Absolute path to project directory
        output_prefix: Output path prefix for SaveImage

    Returns:
        Modified workflow dict
    """
    workflow = json.loads(json.dumps(template))

    source_frames_dir = str(project_dir / "source" / "frames")
    roto_dir = str(project_dir / "roto")

    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")
        widgets = node.get("widgets_values", [])
        title = node.get("title", "")

        if node_type == "VHS_LoadImagesPath" and widgets and len(widgets) >= 3:
            if "Source" in title or widgets[0] == "source/frames":
                widgets[0] = source_frames_dir
            elif "Roto" in title or widgets[0] == "roto":
                widgets[0] = roto_dir
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
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)

    return output_path


def load_batch_state(project_dir: Path) -> dict:
    """Load batch processing state."""
    state_file = project_dir / "cleanplate" / STATE_FILENAME
    if state_file.exists():
        with open(state_file, encoding='utf-8') as f:
            return json.load(f)
    return {"completed_chunks": [], "settings": {}}


def save_batch_state(project_dir: Path, state: dict) -> None:
    """Save batch processing state."""
    state_dir = project_dir / "cleanplate"
    state_dir.mkdir(parents=True, exist_ok=True)

    state_file = state_dir / STATE_FILENAME
    with open(state_file, "w", encoding='utf-8') as f:
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
    output_prefix = f"projects/{project_dir.name}/cleanplate/chunks/{chunk.name}/clean"

    workflow = generate_chunk_workflow(template, chunk, project_dir, output_prefix)

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
        print(f"    OK Chunk {chunk.index} complete")
    else:
        print(f"    X Chunk {chunk.index} failed", file=sys.stderr)

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
    print(f"    OK Blended {len(output_files)} frames to {final_dir}")

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
    start_time = time.time()

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

    elapsed_seconds = int(time.time() - start_time)
    elapsed_hours = elapsed_seconds // 3600
    elapsed_minutes = (elapsed_seconds % 3600) // 60

    print(f"\n{'='*60}")
    print("OK Batched cleanplate complete!")
    print(f"{'='*60}")
    print(f"\nOutput: {project_dir}/cleanplate/final/")
    print(f"\nClean Plate Batch Time : {elapsed_hours:02d}:{elapsed_minutes:02d}")

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

    args = parser.parse_args()

    if args.overlap >= args.batch_size:
        print(f"Error: Overlap ({args.overlap}) must be less than batch size ({args.batch_size})")
        sys.exit(1)

    if not check_comfyui_installed():
        print("Error: ComfyUI not installed", file=sys.stderr)
        print("Install with: python scripts/install_wizard.py", file=sys.stderr)
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
