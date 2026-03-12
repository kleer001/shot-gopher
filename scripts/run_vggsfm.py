#!/usr/bin/env python3
"""VGGSfM v2 reconstruction wrapper for the VFX pipeline.

Runs VGGSfM Structure-from-Motion on a frame sequence to produce
camera poses and a sparse 3D point cloud in COLMAP format.

VGGSfM is designed for video sequences with narrow baselines where
COLMAP's SIFT features struggle (e.g., handheld tracking shots with
large dynamic foreground).

Usage:
    python run_vggsfm.py <project_dir> [--max-size N]

Output:
    mmcam/sparse/0/cameras.bin, images.bin, points3D.bin
    camera/extrinsics.json, intrinsics.json
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from env_config import require_conda_env, INSTALL_DIR
from log_manager import LogCapture


VGGSFM_INSTALL_DIR = INSTALL_DIR / "tools" / "vggsfm"
VGGSFM_CONDA_ENV = "vggsfm"


def _vram_gb() -> float:
    """Get GPU VRAM in GB, or 0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


def _default_max_query_pts() -> int:
    """Choose default max_query_pts based on available GPU VRAM.

    Triangulation VRAM scales with point count. Default 2048
    causes OOM on 24GB GPUs during eigenvalue decomposition.

    Returns:
        Maximum query points for triangulation.
    """
    vram = _vram_gb()
    if vram >= 40:
        return 2048
    if vram >= 20:
        return 1024
    return 512


def _max_batch_frames() -> int:
    """Maximum frames for demo.py batch mode based on VRAM.

    Batch mode loads all frames simultaneously. Memory scales
    with frame_count * max_query_pts. Conservative limits to
    avoid OOM during triangulation.

    Returns:
        Maximum frame count for batch processing.
    """
    vram = _vram_gb()
    if vram >= 80:
        return 200
    if vram >= 40:
        return 120
    if vram >= 20:
        return 60
    return 32


def _find_conda() -> Optional[str]:
    """Find conda or mamba executable.

    Returns:
        Path to conda executable, or None if not found.
    """
    for cmd in ["conda", "mamba"]:
        if shutil.which(cmd):
            return cmd
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return str(conda_exe)
    return None


def _find_conda_base() -> Optional[Path]:
    """Find the conda base directory."""
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return Path(conda_exe).parent.parent
    for base_name in ["anaconda3", "miniconda3", "miniforge3"]:
        conda_base = Path.home() / base_name
        if conda_base.exists():
            return conda_base
    return None


def check_vggsfm_available() -> bool:
    """Check if VGGSfM is installed and its conda env exists.

    Returns:
        True if VGGSfM repo and conda environment are present.
    """
    if not VGGSFM_INSTALL_DIR.exists():
        return False
    if not (VGGSFM_INSTALL_DIR / "demo.py").exists():
        return False

    conda_exe = _find_conda()
    if not conda_exe:
        return False

    result = subprocess.run(
        [conda_exe, "env", "list"],
        capture_output=True, text=True, timeout=10,
    )
    return result.returncode == 0 and VGGSFM_CONDA_ENV in result.stdout


def diagnose_vggsfm_environment(verbose: bool = False) -> dict:
    """Diagnose VGGSfM installation and GPU support.

    Args:
        verbose: Print diagnostic information.

    Returns:
        Dict with diagnostic info.
    """
    info = {
        "vggsfm_available": False,
        "conda_env_exists": False,
        "cuda_available": False,
        "repo_exists": VGGSFM_INSTALL_DIR.exists(),
        "demo_py_exists": (VGGSFM_INSTALL_DIR / "demo.py").exists(),
        "video_demo_py_exists": (VGGSFM_INSTALL_DIR / "video_demo.py").exists(),
    }

    conda_exe = _find_conda()
    if conda_exe:
        result = subprocess.run(
            [conda_exe, "env", "list"],
            capture_output=True, text=True, timeout=10,
        )
        info["conda_env_exists"] = result.returncode == 0 and VGGSFM_CONDA_ENV in result.stdout

    if info["conda_env_exists"] and conda_exe:
        result = subprocess.run(
            [conda_exe, "run", "-n", VGGSFM_CONDA_ENV,
             "python", "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            info["cuda_available"] = result.stdout.strip() == "True"

    info["vggsfm_available"] = (
        info["repo_exists"]
        and info["demo_py_exists"]
        and info["conda_env_exists"]
    )

    if verbose:
        print(f"    DIAG: VGGSfM repo: {VGGSFM_INSTALL_DIR}")
        print(f"    DIAG: Repo exists: {info['repo_exists']}")
        print(f"    DIAG: demo.py exists: {info['demo_py_exists']}")
        print(f"    DIAG: video_demo.py exists: {info['video_demo_py_exists']}")
        print(f"    DIAG: Conda env '{VGGSFM_CONDA_ENV}': {info['conda_env_exists']}")
        print(f"    DIAG: CUDA available: {info['cuda_available']}")

    return info


def _find_mask_sources(project_dir: Path) -> list[Path]:
    """Find mask source directories for VGGSfM.

    Priority order (same as COLMAP):
    1. matte/ numbered dirs (refined alpha mattes)
    2. roto/mask/ (pre-combined from instance separation)
    3. roto/person/ (combined single-person mask)

    Args:
        project_dir: Project root directory.

    Returns:
        List of directories containing mask PNGs, empty if none found.
    """
    numbered_re = re.compile(r".+_\d+$")

    matte_dir = project_dir / "matte"
    if matte_dir.exists():
        matte_subdirs = sorted(
            d for d in matte_dir.iterdir()
            if d.is_dir()
            and numbered_re.match(d.name)
            and list(d.glob("*.png"))
        )
        if matte_subdirs:
            return matte_subdirs

    roto_dir = project_dir / "roto"
    if not roto_dir.exists():
        return []

    roto_mask_dir = roto_dir / "mask"
    if roto_mask_dir.exists() and list(roto_mask_dir.glob("*.png")):
        return [roto_mask_dir]

    roto_person_dir = roto_dir / "person"
    if roto_person_dir.exists() and list(roto_person_dir.glob("*.png")):
        return [roto_person_dir]

    return []


def prepare_vggsfm_scene(
    project_dir: Path,
    scene_dir: Path,
    use_masks: bool = True,
    subsample_step: int = 1,
) -> tuple[int, int]:
    """Prepare a VGGSfM-compatible scene directory.

    VGGSfM expects:
        scene_dir/images/  — input frames
        scene_dir/masks/   — optional binary masks (1=exclude, 0=keep)

    This function symlinks frames into images/ and optionally prepares
    masks by unioning multi-instance masks.

    Args:
        project_dir: Project root (contains source/frames/, roto/, matte/).
        scene_dir: Output scene directory for VGGSfM.
        use_masks: Prepare masks if available.
        subsample_step: Take every Nth frame (1 = all frames).

    Returns:
        Tuple of (total_frame_count, linked_frame_count).
    """
    images_dir = scene_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    frames_dir = project_dir / "source" / "frames"
    all_frame_files = sorted(
        list(frames_dir.glob("*.png"))
        + list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
    )
    total_count = len(all_frame_files)

    frame_files = all_frame_files[::subsample_step]

    for frame_file in frame_files:
        (images_dir / frame_file.name).symlink_to(frame_file)

    if use_masks:
        mask_sources = _find_mask_sources(project_dir)
        if mask_sources:
            _prepare_masks(mask_sources, frame_files, scene_dir / "masks")

    return total_count, len(frame_files)


def _extract_frame_number(filename: str) -> int:
    """Extract numeric index from a frame/mask filename.

    Handles patterns like 'frame_0001.png', 'matte_00001.png',
    'mask_00001_.png', 'person_00_00024_.png'.
    Returns the last contiguous digit group as an integer, or -1.
    """
    match = re.search(r"(\d+)\D*$", Path(filename).stem)
    if match:
        return int(match.group(1))
    return -1


def _prepare_masks(
    mask_sources: list[Path],
    frame_files: list[Path],
    masks_dir: Path,
) -> None:
    """Prepare VGGSfM masks from roto/matte sources.

    VGGSfM mask convention: 1 (white) = exclude, 0 (black) = keep.
    Roto/matte convention: white = subject (to exclude).
    These match, so masks can be used directly. For multi-instance,
    we union all masks per frame.

    Masks are matched to frames by extracted numeric index, so different
    naming conventions (frame_0001 vs matte_00001) are handled correctly.

    Args:
        mask_sources: Directories containing mask PNGs.
        frame_files: Source frame files (for index matching).
        masks_dir: Output directory for VGGSfM masks.
    """
    from PIL import Image
    import numpy as np

    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True)

    masks_by_number_per_source = []
    for d in mask_sources:
        by_number = {
            _extract_frame_number(f.name): f
            for f in sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
        }
        by_number.pop(-1, None)
        masks_by_number_per_source.append(by_number)

    primary_count = len(masks_by_number_per_source[0])
    source_desc = ", ".join(d.name for d in mask_sources)
    print(f"    Mask source: {source_desc} ({primary_count} frames)")

    single_source = len(masks_by_number_per_source) == 1
    matched = 0

    for frame_file in frame_files:
        frame_num = _extract_frame_number(frame_file.name)
        primary_mask = masks_by_number_per_source[0].get(frame_num)
        if primary_mask is None:
            continue

        if single_source:
            shutil.copy2(primary_mask, masks_dir / frame_file.name)
        else:
            gray = np.array(Image.open(primary_mask).convert("L"))
            for source_masks in masks_by_number_per_source[1:]:
                instance_mask = source_masks.get(frame_num)
                if instance_mask is not None:
                    instance = np.array(Image.open(instance_mask).convert("L"))
                    np.maximum(gray, instance, out=gray)
            Image.fromarray(gray).save(masks_dir / frame_file.name)

        matched += 1

    print(f"    Matched {matched} masks to {len(frame_files)} frames")


def run_vggsfm_pipeline(
    project_dir: Path,
    max_image_size: int = -1,
    max_gap: int = 12,
    use_masks: bool = True,
) -> bool:
    """Run VGGSfM sparse reconstruction pipeline.

    Uses demo.py (batch mode) exclusively. For long sequences that
    exceed VRAM limits, frames are subsampled and missing camera
    poses are interpolated during export.

    Args:
        project_dir: Project directory containing source/frames/.
        max_image_size: Maximum image dimension for processing (-1 for default 1024).
        max_gap: Maximum frame gap to interpolate for missing frames.
        use_masks: Use roto/matte masks if available.

    Returns:
        True if reconstruction succeeded.
    """
    if not check_vggsfm_available():
        print("Error: VGGSfM not available.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Install with: python scripts/install_wizard.py", file=sys.stderr)
        print("Select VGGSfM during installation.", file=sys.stderr)
        return False

    pipeline_start = time.time()

    diag = diagnose_vggsfm_environment(verbose=True)

    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}", file=sys.stderr)
        return False

    frame_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("*.jpg")))
    if frame_count == 0:
        print(f"Error: No images found in {frames_dir}", file=sys.stderr)
        return False

    max_frames = _max_batch_frames()
    subsample_step = max(1, (frame_count + max_frames - 1) // max_frames)

    print(f"\n{'='*60}")
    print(f"VGGSfM Reconstruction")
    print(f"{'='*60}")
    print(f"Project: {project_dir}")
    print(f"Frames: {frame_count}")
    if subsample_step > 1:
        effective = (frame_count + subsample_step - 1) // subsample_step
        print(f"Subsampling: every {subsample_step} frames ({effective} batch frames)")
    print(f"Mode: demo.py (batch)")
    if max_image_size > 0:
        print(f"Max image size: {max_image_size}px")
    if not diag["cuda_available"]:
        print("Warning: CUDA not available in vggsfm env — this will be slow")
    print()

    scene_dir = project_dir / "vggsfm_work"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)

    print("[1/3] Preparing scene directory")
    total_count, linked_count = prepare_vggsfm_scene(
        project_dir, scene_dir, use_masks=use_masks, subsample_step=subsample_step,
    )
    print(f"    Linked {linked_count} of {total_count} frames to {scene_dir / 'images'}")
    if (scene_dir / "masks").exists():
        mask_count = len(list((scene_dir / "masks").glob("*.png")))
        print(f"    Prepared {mask_count} masks")
        print(f"    Dynamic scene masking: Enabled")
    else:
        print(f"    Dynamic scene masking: Disabled (no masks found)")

    print(f"\n[2/3] Running VGGSfM inference")
    img_size = max_image_size if max_image_size > 0 else 1024
    max_pts = _default_max_query_pts()

    hydra_overrides = [
        f"SCENE_DIR={scene_dir}",
        f"img_size={img_size}",
        f"max_query_pts={max_pts}",
        "shared_camera=True",
        "save_to_disk=True",
        "auto_download_ckpt=True",
        "filter_invalid_frame=False",
        "query_frame_num=5",
        "BA_iters=2",
        "camera_type=SIMPLE_RADIAL",
    ]

    conda_exe = _find_conda()
    cmd = [
        conda_exe, "run", "-n", VGGSFM_CONDA_ENV, "--no-capture-output",
        "python", str(VGGSFM_INSTALL_DIR / "demo.py"),
    ] + hydra_overrides

    print(f"    Script: demo.py")
    print(f"    Image size: {img_size}")
    print(f"    Max query points: {max_pts}")
    print(f"    Running...")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    result = subprocess.run(
        cmd,
        cwd=str(VGGSFM_INSTALL_DIR),
        env=env,
        timeout=7200,
    )

    if result.returncode != 0:
        print(f"VGGSfM reconstruction failed (exit code {result.returncode})", file=sys.stderr)
        return False

    vggsfm_sparse = scene_dir / "sparse"
    if not vggsfm_sparse.exists() or not (vggsfm_sparse / "cameras.bin").exists():
        print("Error: VGGSfM did not produce expected output", file=sys.stderr)
        print(f"  Expected: {vggsfm_sparse / 'cameras.bin'}", file=sys.stderr)
        if vggsfm_sparse.exists():
            contents = list(vggsfm_sparse.iterdir())
            print(f"  Found: {[f.name for f in contents]}", file=sys.stderr)
        return False

    print(f"\n[3/3] Exporting camera data")
    mmcam_sparse_0 = project_dir / "mmcam" / "sparse" / "0"
    mmcam_sparse_0.mkdir(parents=True, exist_ok=True)

    for bin_file in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = vggsfm_sparse / bin_file
        dst = mmcam_sparse_0 / bin_file
        if src.exists():
            shutil.copy2(src, dst)

    print(f"    Copied COLMAP binary to {mmcam_sparse_0}")

    from run_matchmove_camera import export_colmap_to_pipeline_format

    interpolation_gap = max(max_gap, subsample_step + 2)
    camera_dir = project_dir / "camera"
    if not export_colmap_to_pipeline_format(
        mmcam_sparse_0,
        camera_dir,
        total_frames=frame_count,
        max_gap=interpolation_gap,
    ):
        print("Camera export failed", file=sys.stderr)
        return False

    shutil.rmtree(scene_dir)

    pipeline_end = time.time()
    total_seconds = pipeline_end - pipeline_start
    total_minutes = total_seconds / 60
    per_frame = total_seconds / frame_count if frame_count > 0 else 0

    print(f"\n{'='*60}")
    print(f"VGGSfM Reconstruction Complete")
    print(f"{'='*60}")
    print(f"Sparse model: {mmcam_sparse_0}")
    print(f"Camera data: {camera_dir}")
    print()
    print(f"TOTAL TIME: {total_minutes:.1f} minutes ({per_frame:.2f}s per frame)")
    print()

    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run VGGSfM reconstruction on a frame sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing source/frames/",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=-1,
        help="Maximum image dimension for VGGSfM (-1 for default 1024)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=12,
        help="Maximum frame gap to interpolate (default: 12)",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Disable use of roto/matte masks",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if VGGSfM is available and exit",
    )

    args = parser.parse_args()

    require_conda_env()

    if args.check:
        if check_vggsfm_available():
            print("VGGSfM is available")
            diagnose_vggsfm_environment(verbose=True)
            sys.exit(0)
        else:
            print("VGGSfM is not available")
            diagnose_vggsfm_environment(verbose=True)
            sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_vggsfm_pipeline(
        project_dir=project_dir,
        max_image_size=args.max_size,
        max_gap=args.max_gap,
        use_masks=not args.no_masks,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
