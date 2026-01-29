#!/usr/bin/env python3
"""VideoMaMa video matting processing script.

Refines coarse segmentation masks (from SAM3) into high-quality alpha mattes
using VideoMaMa's diffusion-based approach.

Input: roto/person/*.png (binary masks from SAM3)
Output: roto/person_mama/*.png (refined alpha mattes)

Usage:
    python scripts/video_mama.py <project_dir>
    python scripts/video_mama.py <project_dir> --num-frames 25

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


def prepare_input_structure(
    source_frames_dir: Path,
    mask_dir: Path,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Prepare input structure for VideoMaMa.

    VideoMaMa expects:
    - image_root_path/video_name/*.png (source frames)
    - mask_root_path/video_name/*.png (masks)

    Returns:
        Tuple of (image_root, mask_root) paths
    """
    video_name = "video"
    image_root = work_dir / "images"
    mask_root = work_dir / "masks"

    image_video_dir = image_root / video_name
    mask_video_dir = mask_root / video_name

    image_video_dir.mkdir(parents=True, exist_ok=True)
    mask_video_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_frames_dir.glob("*.png"))
    if not source_files:
        source_files = sorted(source_frames_dir.glob("*.jpg"))

    mask_files = sorted(mask_dir.glob("*.png"))

    if not source_files:
        raise ValueError(f"No source frames found in {source_frames_dir}")
    if not mask_files:
        raise ValueError(f"No mask files found in {mask_dir}")

    frame_count = min(len(source_files), len(mask_files))
    print_info(f"Processing {frame_count} frames")

    for i, (src, msk) in enumerate(zip(source_files[:frame_count], mask_files[:frame_count])):
        dst_img = image_video_dir / f"{i:05d}.png"
        dst_msk = mask_video_dir / f"{i:05d}.png"

        if not dst_img.exists():
            os.symlink(src.resolve(), dst_img)
        if not dst_msk.exists():
            os.symlink(msk.resolve(), dst_msk)

    return image_root, mask_root


def copy_results(work_dir: Path, output_dir: Path) -> int:
    """Copy VideoMaMa results to output directory.

    Returns:
        Number of frames copied
    """
    results_dir = work_dir / "results" / "video"

    if not results_dir.exists():
        results_dir = work_dir / "results"
        video_dirs = [d for d in results_dir.iterdir() if d.is_dir()] if results_dir.exists() else []
        if video_dirs:
            results_dir = video_dirs[0]

    if not results_dir.exists():
        print_error(f"Results directory not found: {results_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob("*.png"))
    if not result_files:
        result_files = sorted(results_dir.glob("*/*.png"))

    count = 0
    for i, src in enumerate(result_files):
        dst = output_dir / f"matte_{i+1:05d}.png"
        shutil.copy2(src, dst)
        count += 1

    return count


def run_videomama(
    conda_exe: str,
    image_root: Path,
    mask_root: Path,
    output_dir: Path,
    num_frames: int = 16,
    width: int = 1024,
    height: int = 576,
) -> bool:
    """Run VideoMaMa inference."""
    inference_script = VIDEOMAMA_TOOLS_DIR / "inference_onestep_folder.py"

    if not inference_script.exists():
        print_error(f"Inference script not found: {inference_script}")
        return False

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

    print_info(f"Running VideoMaMa inference...")
    print_info(f"  Images: {image_root}")
    print_info(f"  Masks: {mask_root}")
    print_info(f"  Output: {output_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=VIDEOMAMA_TOOLS_DIR,
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"VideoMaMa inference failed with code {e.returncode}")
        return False


def process_project(
    project_dir: Path,
    num_frames: int = 16,
    width: int = 1024,
    height: int = 576,
) -> bool:
    """Process a project's roto/person masks with VideoMaMa.

    Args:
        project_dir: Project directory containing source/frames and roto/person
        num_frames: Number of frames per batch for VideoMaMa
        width: Processing width
        height: Processing height

    Returns:
        True if successful
    """
    print(f"\n=== VideoMaMa Processing ===")
    print(f"  Project: {project_dir}")

    source_frames = project_dir / "source" / "frames"
    mask_dir = project_dir / "roto" / "person"
    output_dir = project_dir / "roto" / "person_mama"

    if not source_frames.exists():
        print_error(f"Source frames not found: {source_frames}")
        return False

    if not mask_dir.exists():
        print_error(f"Person masks not found: {mask_dir}")
        print_info("Run the roto stage first: python run_pipeline.py <video> -s roto")
        return False

    mask_files = list(mask_dir.glob("*.png"))
    if not mask_files:
        print_error(f"No mask PNG files in {mask_dir}")
        return False

    print_info(f"Found {len(mask_files)} mask frames")

    if not check_installation():
        return False

    conda_exe = find_conda()
    if not conda_exe:
        print_error("Conda not found")
        return False

    with tempfile.TemporaryDirectory(prefix="videomama_") as work_dir:
        work_path = Path(work_dir)

        print_info("Preparing input structure...")
        try:
            image_root, mask_root = prepare_input_structure(
                source_frames, mask_dir, work_path
            )
        except ValueError as e:
            print_error(str(e))
            return False

        if not run_videomama(
            conda_exe,
            image_root,
            mask_root,
            work_path,
            num_frames=num_frames,
            width=width,
            height=height,
        ):
            return False

        print_info("Copying results...")
        count = copy_results(work_path, output_dir)

        if count == 0:
            print_error("No output frames generated")
            return False

        print_success(f"Generated {count} refined matte frames")
        print_info(f"Output: {output_dir}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine SAM3 masks with VideoMaMa diffusion matting"
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing source/frames and roto/person"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames per batch (default: 16)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Processing width (default: 1024)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="Processing height (default: 576)"
    )

    args = parser.parse_args()

    if not args.project_dir.exists():
        print_error(f"Project directory not found: {args.project_dir}")
        return 1

    success = process_project(
        args.project_dir.resolve(),
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
