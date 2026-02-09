#!/usr/bin/env python3
"""VideoMaMa video matting processing script.

Refines coarse segmentation masks (from SAM3) into high-quality alpha mattes
using VideoMaMa's diffusion-based approach.

Pipeline integration:
    from video_mama import process_roto_directory, check_installation
    if check_installation():
        process_roto_directory(project_dir, "person_00", output_dir)

Standalone usage:
    python scripts/video_mama.py <project_dir>
    python scripts/video_mama.py <project_dir> --roto person_00
    python scripts/video_mama.py <project_dir> --chunk-size 14

Requirements:
    - VideoMaMa installed via: python scripts/video_mama_install.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from env_config import INSTALL_DIR
from pipeline_utils import get_gpu_vram_gb, get_image_dimensions

VIDEOMAMA_ENV_NAME = "videomama"
VIDEOMAMA_TOOLS_DIR = INSTALL_DIR / "tools" / "VideoMaMa"
VIDEOMAMA_MODELS_DIR = INSTALL_DIR / "models" / "VideoMaMa"
SVD_MODEL_DIR = VIDEOMAMA_MODELS_DIR / "stable-video-diffusion-img2vid-xt"
VIDEOMAMA_CHECKPOINT_DIR = VIDEOMAMA_MODELS_DIR / "checkpoints" / "VideoMaMa"


def print_info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def print_success(msg: str) -> None:
    print(f"  [OK] {msg}")


def print_error(msg: str) -> None:
    print(f"  [ERROR] {msg}", file=sys.stderr)


def print_warning(msg: str) -> None:
    print(f"  [WARN] {msg}")


def find_conda() -> str | None:
    """Find conda executable."""
    for cmd in ["conda", "mamba"]:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    return None


def check_installation() -> bool:
    """Verify VideoMaMa is properly installed."""
    if not VIDEOMAMA_TOOLS_DIR.exists():
        print_error("VideoMaMa not installed. Run: python scripts/video_mama_install.py")
        return False

    if not SVD_MODEL_DIR.exists():
        print_error("SVD model not downloaded. Run: python scripts/video_mama_install.py")
        return False

    if not VIDEOMAMA_CHECKPOINT_DIR.exists():
        print_error("VideoMaMa checkpoint not downloaded. Run: python scripts/video_mama_install.py")
        return False

    return True


REFERENCE_MEGAPIXELS = (1024 * 576) / 1_000_000
FALLBACK_WIDTH = 1024
FALLBACK_HEIGHT = 576


def detect_source_resolution(source_frames_dir: Path) -> tuple[int, int]:
    """Detect resolution from the first source frame.

    Args:
        source_frames_dir: Directory containing source frame images

    Returns:
        (width, height) from first frame, or fallback (1024, 576) if undetectable
    """
    frames = sorted(source_frames_dir.glob("*.png"))
    if not frames:
        frames = sorted(source_frames_dir.glob("*.jpg"))
    if frames:
        w, h = get_image_dimensions(frames[0])
        if w > 0 and h > 0:
            return w, h
    print_warning(
        f"Could not detect source resolution, falling back to {FALLBACK_WIDTH}x{FALLBACK_HEIGHT}"
    )
    return FALLBACK_WIDTH, FALLBACK_HEIGHT


def get_optimal_chunk_size(
    vram_gb: float | None,
    resolution: tuple[int, int] = (1024, 576),
) -> int:
    """Get optimal chunk size based on GPU VRAM and processing resolution.

    VideoMaMa uses Stable Video Diffusion which has significant VRAM requirements.
    These are conservative estimates - the diffusion model needs substantial headroom.
    Assumes ~10% VRAM is used by system/drivers.

    VRAM thresholds were calibrated at 1024x576 (~0.59 megapixels). For higher
    resolutions, effective VRAM is scaled down proportionally since each frame
    requires more memory.

    Args:
        vram_gb: GPU VRAM in gigabytes, or None if unknown
        resolution: Processing resolution as (width, height)

    Returns:
        Recommended chunk size (number of frames per batch)
    """
    if vram_gb is None:
        return 8

    actual_mpx = (resolution[0] * resolution[1]) / 1_000_000
    scale = REFERENCE_MEGAPIXELS / actual_mpx if actual_mpx > 0 else 1.0
    available_vram = vram_gb * 0.9 * scale

    if available_vram >= 43:
        return 20
    elif available_vram >= 21:
        return 14
    elif available_vram >= 14:
        return 10
    elif available_vram >= 10:
        return 8
    elif available_vram >= 7:
        return 6
    else:
        return 4


def prepare_chunk_structure(
    source_files: list[Path],
    mask_files: list[Path],
    work_dir: Path,
    chunk_idx: int,
) -> tuple[Path, Path]:
    """Prepare input structure for a single chunk."""
    video_name = "video"
    image_root = work_dir / "images"
    mask_root = work_dir / "masks"

    image_video_dir = image_root / video_name
    mask_video_dir = mask_root / video_name

    if image_video_dir.exists():
        shutil.rmtree(image_video_dir)
    if mask_video_dir.exists():
        shutil.rmtree(mask_video_dir)

    image_video_dir.mkdir(parents=True, exist_ok=True)
    mask_video_dir.mkdir(parents=True, exist_ok=True)

    for i, (src, msk) in enumerate(zip(source_files, mask_files)):
        dst_img = image_video_dir / f"{i:05d}.png"
        dst_msk = mask_video_dir / f"{i:05d}.png"
        os.symlink(src.resolve(), dst_img)
        os.symlink(msk.resolve(), dst_msk)

    return image_root, mask_root


def clear_cuda_memory(conda_exe: str) -> None:
    """Clear CUDA memory cache between chunks."""
    clear_script = """
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
"""
    try:
        subprocess.run(
            [conda_exe, "run", "-n", VIDEOMAMA_ENV_NAME, "python", "-c", clear_script],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass


def run_videomama_chunk(
    conda_exe: str,
    image_root: Path,
    mask_root: Path,
    output_dir: Path,
    num_frames: int,
    width: int,
    height: int,
) -> tuple[bool, bool]:
    """Run VideoMaMa inference on a single chunk.

    Returns:
        Tuple of (success, was_oom) - success indicates if inference worked,
        was_oom indicates if failure was due to CUDA OOM.
    """
    inference_script = VIDEOMAMA_TOOLS_DIR / "inference_onestep_folder.py"

    if not inference_script.exists():
        print_error(f"Inference script not found: {inference_script}")
        return False, False

    cmd = [
        conda_exe, "run", "-n", VIDEOMAMA_ENV_NAME,
        "--cwd", str(VIDEOMAMA_TOOLS_DIR),
        "python", str(inference_script),
        "--base_model_path", str(SVD_MODEL_DIR),
        "--unet_checkpoint_path", str(VIDEOMAMA_CHECKPOINT_DIR),
        "--image_root_path", str(image_root),
        "--mask_root_path", str(mask_root),
        "--output_dir", str(output_dir),
        "--num_frames", str(num_frames),
        "--width", str(width),
        "--height", str(height),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=VIDEOMAMA_TOOLS_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines: list[str] = []
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)

        process.wait()
        output = "".join(output_lines)
        was_oom = "CUDA out of memory" in output or "OutOfMemoryError" in output

        if process.returncode != 0:
            if was_oom:
                print_warning("CUDA out of memory detected")
            return False, was_oom
        return True, False
    except subprocess.CalledProcessError as e:
        print_error(f"VideoMaMa inference failed with code {e.returncode}")
        return False, False


def collect_chunk_results(work_dir: Path) -> list[Path]:
    """Collect result files from a chunk."""
    results_dir = work_dir / "results" / "video"

    if not results_dir.exists():
        results_dir = work_dir / "results"
        video_dirs = [d for d in results_dir.iterdir() if d.is_dir()] if results_dir.exists() else []
        if video_dirs:
            results_dir = video_dirs[0]

    if not results_dir.exists():
        return []

    return sorted(results_dir.glob("*.png"))


def process_project(
    project_dir: Path,
    chunk_size: int | None = None,
    overlap: int = 2,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    """Process a project's roto/person masks with VideoMaMa.

    Args:
        project_dir: Project directory containing source/frames and roto/person
        chunk_size: Number of frames per chunk (None = auto-detect from VRAM)
        overlap: Frame overlap between chunks for smoother transitions
        width: Processing width (None = auto-detect from source frames)
        height: Processing height (None = auto-detect from source frames)

    Returns:
        True if successful
    """
    print(f"\n=== VideoMaMa Processing ===")
    print(f"  Project: {project_dir}")

    source_frames = project_dir / "source" / "frames"
    mask_dir = project_dir / "roto" / "person"
    output_dir = project_dir / "roto" / "person_mama"

    if width is None or height is None:
        detected_w, detected_h = detect_source_resolution(source_frames)
        width = width or detected_w
        height = height or detected_h

    vram_gb = get_gpu_vram_gb()
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(vram_gb, (width, height))
        if vram_gb:
            print_info(f"Detected GPU VRAM: {vram_gb:.1f}GB @ {width}x{height} → chunk size: {chunk_size}")
        else:
            print_info(f"Could not detect VRAM, using default chunk size: {chunk_size}")

    if not source_frames.exists():
        print_error(f"Source frames not found: {source_frames}")
        return False

    if not mask_dir.exists():
        print_error(f"Person masks not found: {mask_dir}")
        print_info("Run the roto stage first: python run_pipeline.py <video> -s roto")
        return False

    source_files = sorted(source_frames.glob("*.png"))
    if not source_files:
        source_files = sorted(source_frames.glob("*.jpg"))

    mask_files = sorted(mask_dir.glob("*.png"))

    if not source_files:
        print_error(f"No source frames in {source_frames}")
        return False

    if not mask_files:
        print_error(f"No mask PNG files in {mask_dir}")
        return False

    total_frames = min(len(source_files), len(mask_files))
    print_info(f"Found {total_frames} frames to process")
    print_info(f"Chunk size: {chunk_size}, overlap: {overlap}")

    if not check_installation():
        return False

    conda_exe = find_conda()
    if not conda_exe:
        print_error("Conda not found")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[Path] = []
    chunk_idx = 0
    frame_idx = 0
    current_chunk_size = chunk_size
    min_chunk_size = 4

    with tempfile.TemporaryDirectory(prefix="videomama_") as work_dir:
        work_path = Path(work_dir)

        while frame_idx < total_frames:
            step = current_chunk_size - overlap
            end_idx = min(frame_idx + current_chunk_size, total_frames)
            chunk_sources = source_files[frame_idx:end_idx]
            chunk_masks = mask_files[frame_idx:end_idx]

            actual_chunk_size = len(chunk_sources)
            print_info(f"Processing chunk {chunk_idx + 1}: frames {frame_idx + 1}-{end_idx} ({actual_chunk_size} frames)")

            image_root, mask_root = prepare_chunk_structure(
                chunk_sources, chunk_masks, work_path, chunk_idx
            )

            success, was_oom = run_videomama_chunk(
                conda_exe,
                image_root,
                mask_root,
                work_path,
                num_frames=actual_chunk_size,
                width=width,
                height=height,
            )

            if not success:
                if was_oom and current_chunk_size > min_chunk_size:
                    new_chunk_size = max(min_chunk_size, current_chunk_size - 2)
                    print_info(f"Reducing chunk size: {current_chunk_size} → {new_chunk_size}")
                    current_chunk_size = new_chunk_size
                    print_info("Clearing CUDA memory before retry...")
                    clear_cuda_memory(conda_exe)
                    continue
                else:
                    print_warning(f"Chunk {chunk_idx + 1} failed, skipping...")
                    frame_idx += step
                    chunk_idx += 1
                    clear_cuda_memory(conda_exe)
                    continue

            chunk_results = collect_chunk_results(work_path)

            if chunk_idx == 0:
                results_to_copy = chunk_results
            else:
                results_to_copy = chunk_results[overlap:] if overlap > 0 else chunk_results

            for i, src in enumerate(results_to_copy):
                global_frame_idx = len(all_results)
                dst = output_dir / f"matte_{global_frame_idx + 1:05d}.png"
                shutil.copy2(src, dst)
                all_results.append(dst)

            results_subdir = work_path / "results"
            if results_subdir.exists():
                shutil.rmtree(results_subdir)

            frame_idx += step
            chunk_idx += 1

            print_info("Clearing CUDA memory...")
            clear_cuda_memory(conda_exe)

            if frame_idx >= total_frames:
                break

    if not all_results:
        print_error("No output frames generated")
        return False

    print_success(f"Generated {len(all_results)} refined matte frames")
    print_info(f"Output: {output_dir}")

    return True


def process_roto_directory(
    project_dir: Path,
    roto_subdir: str,
    output_dir: Path,
    chunk_size: int | None = None,
    overlap: int = 2,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    """Process a specific roto subdirectory with VideoMaMa.

    This is the main entry point for pipeline integration.

    Args:
        project_dir: Project directory containing source/frames and roto/
        roto_subdir: Name of roto subdirectory (e.g., "person_00", "bag_01")
        output_dir: Output directory for refined mattes (e.g., matte/person_00/)
        chunk_size: Frames per chunk (None = auto-detect from VRAM)
        overlap: Frame overlap between chunks
        width: Processing width (None = auto-detect from source frames)
        height: Processing height (None = auto-detect from source frames)

    Returns:
        True if successful
    """
    print(f"\n=== VideoMaMa: {roto_subdir} ===")

    source_frames = project_dir / "source" / "frames"
    mask_dir = project_dir / "roto" / roto_subdir

    if width is None or height is None:
        detected_w, detected_h = detect_source_resolution(source_frames)
        width = width or detected_w
        height = height or detected_h

    vram_gb = get_gpu_vram_gb()
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(vram_gb, (width, height))
        if vram_gb:
            print_info(f"Detected GPU VRAM: {vram_gb:.1f}GB @ {width}x{height} → chunk size: {chunk_size}")
        else:
            print_info(f"Could not detect VRAM, using default chunk size: {chunk_size}")

    if not source_frames.exists():
        print_error(f"Source frames not found: {source_frames}")
        return False

    if not mask_dir.exists():
        print_error(f"Roto directory not found: {mask_dir}")
        return False

    source_files = sorted(source_frames.glob("*.png"))
    if not source_files:
        source_files = sorted(source_frames.glob("*.jpg"))

    mask_files = sorted(mask_dir.glob("*.png"))

    if not source_files:
        print_error(f"No source frames in {source_frames}")
        return False

    if not mask_files:
        print_error(f"No mask PNG files in {mask_dir}")
        return False

    total_frames = min(len(source_files), len(mask_files))
    print_info(f"Found {total_frames} frames to process")
    print_info(f"Chunk size: {chunk_size}, overlap: {overlap}")

    if not check_installation():
        return False

    conda_exe = find_conda()
    if not conda_exe:
        print_error("Conda not found")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[Path] = []
    chunk_idx = 0
    frame_idx = 0
    current_chunk_size = chunk_size
    min_chunk_size = 4

    with tempfile.TemporaryDirectory(prefix="videomama_") as work_dir:
        work_path = Path(work_dir)

        while frame_idx < total_frames:
            step = current_chunk_size - overlap
            end_idx = min(frame_idx + current_chunk_size, total_frames)
            chunk_sources = source_files[frame_idx:end_idx]
            chunk_masks = mask_files[frame_idx:end_idx]

            actual_chunk_size = len(chunk_sources)
            print_info(f"Processing chunk {chunk_idx + 1}: frames {frame_idx + 1}-{end_idx} ({actual_chunk_size} frames)")

            image_root, mask_root = prepare_chunk_structure(
                chunk_sources, chunk_masks, work_path, chunk_idx
            )

            success, was_oom = run_videomama_chunk(
                conda_exe,
                image_root,
                mask_root,
                work_path,
                num_frames=actual_chunk_size,
                width=width,
                height=height,
            )

            if not success:
                if was_oom and current_chunk_size > min_chunk_size:
                    new_chunk_size = max(min_chunk_size, current_chunk_size - 2)
                    print_info(f"Reducing chunk size: {current_chunk_size} → {new_chunk_size}")
                    current_chunk_size = new_chunk_size
                    print_info("Clearing CUDA memory before retry...")
                    clear_cuda_memory(conda_exe)
                    continue
                else:
                    print_warning(f"Chunk {chunk_idx + 1} failed, skipping...")
                    frame_idx += step
                    chunk_idx += 1
                    clear_cuda_memory(conda_exe)
                    continue

            chunk_results = collect_chunk_results(work_path)

            if chunk_idx == 0:
                results_to_copy = chunk_results
            else:
                results_to_copy = chunk_results[overlap:] if overlap > 0 else chunk_results

            for i, src in enumerate(results_to_copy):
                global_frame_idx = len(all_results)
                dst = output_dir / f"matte_{global_frame_idx + 1:05d}.png"
                shutil.copy2(src, dst)
                all_results.append(dst)

            results_subdir = work_path / "results"
            if results_subdir.exists():
                shutil.rmtree(results_subdir)

            frame_idx += step
            chunk_idx += 1

            print_info("Clearing CUDA memory...")
            clear_cuda_memory(conda_exe)

            if frame_idx >= total_frames:
                break

    if not all_results:
        print_error("No output frames generated")
        return False

    print_success(f"Generated {len(all_results)} refined matte frames")
    print_info(f"Output: {output_dir}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine SAM3 masks with VideoMaMa diffusion matting"
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing source/frames and roto/"
    )
    parser.add_argument(
        "--roto",
        type=str,
        default=None,
        help="Specific roto subdirectory to process (e.g., 'person_00'). Default: process 'person'"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: matte/<roto_name>/"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Frames per chunk (default: auto-detect from GPU VRAM)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Frame overlap between chunks (default: 2)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Processing width (default: auto-detect from source frames)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Processing height (default: auto-detect from source frames)"
    )

    args = parser.parse_args()

    if not args.project_dir.exists():
        print_error(f"Project directory not found: {args.project_dir}")
        return 1

    project_dir = args.project_dir.resolve()

    if args.roto:
        roto_subdir = args.roto
        output_dir = args.output or (project_dir / "matte" / roto_subdir)
        success = process_roto_directory(
            project_dir,
            roto_subdir,
            output_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            width=args.width,
            height=args.height,
        )
    else:
        success = process_project(
            project_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            width=args.width,
            height=args.height,
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
