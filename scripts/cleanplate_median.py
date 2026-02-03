#!/usr/bin/env python3
"""Temporal median cleanplate generation.

Creates a cleanplate by computing the temporal median of masked frames.
This works well for static camera shots where the foreground object moves
enough to reveal all background pixels at some point.

Usage:
    python cleanplate_median.py <project_dir> [options]

Example:
    python cleanplate_median.py /path/to/projects/My_Shot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _gather_frame_paths(
    source_dir: Path,
    mask_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Gather and validate source and mask frame paths.

    Args:
        source_dir: Directory containing source frames
        mask_dir: Directory containing mask frames

    Returns:
        Tuple of (source_frames, mask_frames) sorted by filename

    Raises:
        ValueError: If no frames found or frame counts don't match
    """
    source_frames = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg"))
    mask_frames = sorted(mask_dir.glob("*.png"))

    if not source_frames:
        raise ValueError(f"No source frames found in {source_dir}")

    if not mask_frames:
        raise ValueError(f"No mask frames found in {mask_dir}")

    if len(source_frames) != len(mask_frames):
        print(f"Warning: Frame count mismatch - {len(source_frames)} sources, {len(mask_frames)} masks")
        frame_count = min(len(source_frames), len(mask_frames))
        source_frames = source_frames[:frame_count]
        mask_frames = mask_frames[:frame_count]

    return source_frames, mask_frames


def _load_frame_pair(
    src_path: Path,
    mask_path: Path,
    target_size: tuple[int, int],
    threshold: float,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a source/mask frame pair.

    Args:
        src_path: Path to source frame
        mask_path: Path to mask frame
        target_size: Target (height, width) for resizing if needed
        threshold: Mask threshold (0-1)
        channels: Number of color channels

    Returns:
        Tuple of (source_array, background_mask_boolean)
    """
    height, width = target_size

    src_img = np.array(Image.open(src_path))
    mask_img = np.array(Image.open(mask_path).convert('L'))

    if src_img.shape[:2] != (height, width):
        src_img = np.array(Image.open(src_path).resize((width, height), Image.LANCZOS))
    if mask_img.shape[:2] != (height, width):
        mask_img = np.array(Image.open(mask_path).convert('L').resize((width, height), Image.NEAREST))

    if len(src_img.shape) == 2:
        src_img = np.stack([src_img] * 3, axis=-1)

    return src_img[:, :, :channels], mask_img < (threshold * 255)


def _compute_masked_median(
    pixel_values: np.ndarray,
    pixel_valid: np.ndarray,
    channels: int,
) -> np.ndarray:
    """Compute per-pixel median using only valid (unmasked) samples.

    Args:
        pixel_values: Array of shape (H, W, C, T) with pixel values
        pixel_valid: Boolean array of shape (H, W, T) indicating valid samples
        channels: Number of color channels

    Returns:
        Cleanplate array of shape (H, W, C)
    """
    height, width = pixel_values.shape[:2]
    cleanplate = np.zeros((height, width, channels), dtype=np.uint8)

    for c in range(channels):
        for y in range(height):
            valid_row = pixel_valid[y, :, :]
            values_row = pixel_values[y, :, c, :]

            for x in range(width):
                valid_mask = valid_row[x, :]
                if np.any(valid_mask):
                    cleanplate[y, x, c] = np.median(values_row[x, valid_mask])
                else:
                    cleanplate[y, x, c] = np.median(values_row[x, :])

    return cleanplate


def generate_temporal_median_cleanplate(
    source_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    sample_count: int = 0,
    threshold: float = 0.5,
) -> bool:
    """Generate a cleanplate using temporal median of unmasked pixels.

    For each output pixel, collects values from frames where that pixel
    is not masked (background), then takes the median. This effectively
    removes moving foreground objects while preserving the static background.

    Args:
        source_dir: Directory containing source frames (PNG/JPG)
        mask_dir: Directory containing mask frames (white = foreground to remove)
        output_dir: Directory to write cleanplate frames
        sample_count: Max frames to sample (0 = all frames)
        threshold: Mask threshold (0-1), pixels above this are considered masked

    Returns:
        True if successful
    """
    try:
        source_frames, mask_frames = _gather_frame_paths(source_dir, mask_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    first_frame = np.array(Image.open(source_frames[0]))
    height, width = first_frame.shape[:2]
    channels = first_frame.shape[2] if len(first_frame.shape) > 2 else 3

    print(f"Generating temporal median cleanplate...")
    print(f"  Source frames: {len(source_frames)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Mask threshold: {threshold}")

    all_source_frames = source_frames

    if sample_count > 0 and sample_count < len(source_frames):
        indices = np.linspace(0, len(source_frames) - 1, sample_count, dtype=int)
        source_frames = [source_frames[i] for i in indices]
        mask_frames = [mask_frames[i] for i in indices]
        print(f"  Sampling: {sample_count} frames")

    pixel_values = np.zeros((height, width, channels, len(source_frames)), dtype=np.uint8)
    pixel_valid = np.zeros((height, width, len(source_frames)), dtype=bool)

    for i, (src_path, mask_path) in enumerate(zip(source_frames, mask_frames)):
        if i % 10 == 0:
            print(f"  Loading frame {i + 1}/{len(source_frames)}...")

        src_img, bg_mask = _load_frame_pair(
            src_path, mask_path, (height, width), threshold, channels
        )
        pixel_values[:, :, :, i] = src_img
        pixel_valid[:, :, i] = bg_mask

    print("  Computing temporal median...")

    cleanplate = _compute_masked_median(pixel_values, pixel_valid, channels)

    valid_count = np.sum(pixel_valid, axis=2)
    never_visible_count = np.sum(valid_count == 0)
    if never_visible_count > 0:
        pct = 100 * never_visible_count / (height * width)
        print(f"  Warning: {never_visible_count} pixels ({pct:.1f}%) never unmasked - using full median")

    cleanplate_img = Image.fromarray(cleanplate)

    for src_path in all_source_frames:
        output_path = output_dir / f"{src_path.stem}.png"
        cleanplate_img.save(output_path)

    print(f"  Wrote {len(all_source_frames)} cleanplate frames to {output_dir}")
    return True


def run_cleanplate_median(
    project_dir: Path,
    sample_count: int = 0,
    threshold: float = 0.5,
) -> bool:
    """Run temporal median cleanplate generation for a project.

    Args:
        project_dir: Project directory
        sample_count: Max frames to sample (0 = all)
        threshold: Mask threshold

    Returns:
        True if successful
    """
    source_dir = project_dir / "source" / "frames"
    roto_dir = project_dir / "roto"
    output_dir = project_dir / "cleanplate"

    if not list(roto_dir.glob("*.png")):
        print(f"Error: No mask files found in {roto_dir}", file=sys.stderr)
        print("  (masks should be prepared by prepare_roto_for_cleanplate first)")
        return False

    return generate_temporal_median_cleanplate(
        source_dir, roto_dir, output_dir, sample_count=sample_count, threshold=threshold
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate cleanplate using temporal median"
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory"
    )
    parser.add_argument(
        "--sample-count", "-n",
        type=int,
        default=0,
        help="Max frames to sample (0 = all frames)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Mask threshold 0-1 (default: 0.5)"
    )
    args = parser.parse_args()

    if not args.project_dir.exists():
        print(f"Error: Project directory not found: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_cleanplate_median(
        args.project_dir,
        sample_count=args.sample_count,
        threshold=args.threshold,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
