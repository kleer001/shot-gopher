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
from typing import Optional

import numpy as np
from PIL import Image


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
    source_frames = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg"))
    mask_frames = sorted(mask_dir.glob("*.png"))

    if not source_frames:
        print(f"Error: No source frames found in {source_dir}", file=sys.stderr)
        return False

    if not mask_frames:
        print(f"Error: No mask frames found in {mask_dir}", file=sys.stderr)
        return False

    if len(source_frames) != len(mask_frames):
        print(f"Warning: Frame count mismatch - {len(source_frames)} sources, {len(mask_frames)} masks")
        frame_count = min(len(source_frames), len(mask_frames))
        source_frames = source_frames[:frame_count]
        mask_frames = mask_frames[:frame_count]

    output_dir.mkdir(parents=True, exist_ok=True)

    first_frame = np.array(Image.open(source_frames[0]))
    height, width = first_frame.shape[:2]
    channels = first_frame.shape[2] if len(first_frame.shape) > 2 else 1

    print(f"Generating temporal median cleanplate...")
    print(f"  Source frames: {len(source_frames)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Mask threshold: {threshold}")

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

        src_img = np.array(Image.open(src_path))
        mask_img = np.array(Image.open(mask_path).convert('L'))

        if src_img.shape[:2] != (height, width):
            src_img = np.array(Image.open(src_path).resize((width, height), Image.LANCZOS))
        if mask_img.shape[:2] != (height, width):
            mask_img = np.array(Image.open(mask_path).convert('L').resize((width, height), Image.NEAREST))

        if len(src_img.shape) == 2:
            src_img = np.stack([src_img] * 3, axis=-1)

        background_mask = mask_img < (threshold * 255)

        pixel_values[:, :, :, i] = src_img[:, :, :channels]
        pixel_valid[:, :, i] = background_mask

    print("  Computing temporal median...")

    cleanplate = np.zeros((height, width, channels), dtype=np.uint8)
    valid_count = np.sum(pixel_valid, axis=2)

    for c in range(channels):
        channel_values = pixel_values[:, :, c, :]

        for y in range(height):
            for x in range(width):
                valid_mask = pixel_valid[y, x, :]
                if np.any(valid_mask):
                    valid_pixels = channel_values[y, x, valid_mask]
                    cleanplate[y, x, c] = np.median(valid_pixels)
                else:
                    cleanplate[y, x, c] = np.median(channel_values[y, x, :])

    never_visible = valid_count == 0
    never_visible_count = np.sum(never_visible)
    if never_visible_count > 0:
        pct = 100 * never_visible_count / (height * width)
        print(f"  Warning: {never_visible_count} pixels ({pct:.1f}%) never unmasked - using full median")

    cleanplate_img = Image.fromarray(cleanplate)

    for i, src_path in enumerate(sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg"))):
        output_path = output_dir / f"{src_path.stem}.png"
        cleanplate_img.save(output_path)

    print(f"  Wrote {len(list(output_dir.glob('*.png')))} cleanplate frames to {output_dir}")
    return True


def generate_temporal_median_cleanplate_chunked(
    source_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    chunk_size: int = 50,
    threshold: float = 0.5,
) -> bool:
    """Memory-efficient temporal median using chunked processing.

    Processes the image in spatial chunks to reduce memory usage for
    high-resolution or long sequences.

    Args:
        source_dir: Directory containing source frames
        mask_dir: Directory containing mask frames
        output_dir: Directory to write cleanplate frames
        chunk_size: Size of spatial chunks to process
        threshold: Mask threshold (0-1)

    Returns:
        True if successful
    """
    source_frames = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg"))
    mask_frames = sorted(mask_dir.glob("*.png"))

    if not source_frames or not mask_frames:
        print("Error: No frames found", file=sys.stderr)
        return False

    frame_count = min(len(source_frames), len(mask_frames))
    source_frames = source_frames[:frame_count]
    mask_frames = mask_frames[:frame_count]

    output_dir.mkdir(parents=True, exist_ok=True)

    first_frame = np.array(Image.open(source_frames[0]))
    height, width = first_frame.shape[:2]
    channels = first_frame.shape[2] if len(first_frame.shape) > 2 else 1

    print(f"Generating temporal median cleanplate (chunked)...")
    print(f"  Source frames: {len(source_frames)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Chunk size: {chunk_size}")

    cleanplate = np.zeros((height, width, channels), dtype=np.uint8)

    y_chunks = range(0, height, chunk_size)
    x_chunks = range(0, width, chunk_size)
    total_chunks = len(list(y_chunks)) * len(list(x_chunks))

    chunk_idx = 0
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)

        for x_start in range(0, width, chunk_size):
            x_end = min(x_start + chunk_size, width)
            chunk_idx += 1

            if chunk_idx % 10 == 0:
                print(f"  Processing chunk {chunk_idx}/{total_chunks}...")

            chunk_h = y_end - y_start
            chunk_w = x_end - x_start

            pixel_values = []
            pixel_valid = []

            for src_path, mask_path in zip(source_frames, mask_frames):
                src_img = np.array(Image.open(src_path))
                mask_img = np.array(Image.open(mask_path).convert('L'))

                src_chunk = src_img[y_start:y_end, x_start:x_end]
                mask_chunk = mask_img[y_start:y_end, x_start:x_end]

                if len(src_chunk.shape) == 2:
                    src_chunk = np.stack([src_chunk] * 3, axis=-1)

                pixel_values.append(src_chunk[:, :, :channels])
                pixel_valid.append(mask_chunk < (threshold * 255))

            pixel_values = np.stack(pixel_values, axis=-1)
            pixel_valid = np.stack(pixel_valid, axis=-1)

            for c in range(channels):
                for y in range(chunk_h):
                    for x in range(chunk_w):
                        valid_mask = pixel_valid[y, x, :]
                        if np.any(valid_mask):
                            cleanplate[y_start + y, x_start + x, c] = np.median(
                                pixel_values[y, x, c, valid_mask]
                            )
                        else:
                            cleanplate[y_start + y, x_start + x, c] = np.median(
                                pixel_values[y, x, c, :]
                            )

    cleanplate_img = Image.fromarray(cleanplate)

    all_source_frames = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg"))
    for src_path in all_source_frames:
        output_path = output_dir / f"{src_path.stem}.png"
        cleanplate_img.save(output_path)

    print(f"  Wrote {len(all_source_frames)} cleanplate frames to {output_dir}")
    return True


def run_cleanplate_median(
    project_dir: Path,
    sample_count: int = 0,
    threshold: float = 0.5,
    use_chunked: bool = False,
) -> bool:
    """Run temporal median cleanplate generation for a project.

    Args:
        project_dir: Project directory
        sample_count: Max frames to sample (0 = all)
        threshold: Mask threshold
        use_chunked: Use memory-efficient chunked processing

    Returns:
        True if successful
    """
    source_dir = project_dir / "source" / "frames"
    roto_dir = project_dir / "roto"
    output_dir = project_dir / "cleanplate"

    combined_dir = roto_dir / "combined"
    if combined_dir.exists() and list(combined_dir.glob("*.png")):
        mask_dir = combined_dir
    elif list(roto_dir.glob("*.png")):
        mask_dir = roto_dir
    else:
        print(f"Error: No masks found in {roto_dir}", file=sys.stderr)
        return False

    if use_chunked:
        return generate_temporal_median_cleanplate_chunked(
            source_dir, mask_dir, output_dir, threshold=threshold
        )
    else:
        return generate_temporal_median_cleanplate(
            source_dir, mask_dir, output_dir, sample_count=sample_count, threshold=threshold
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
    parser.add_argument(
        "--chunked", "-c",
        action="store_true",
        help="Use memory-efficient chunked processing"
    )
    args = parser.parse_args()

    if not args.project_dir.exists():
        print(f"Error: Project directory not found: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_cleanplate_median(
        args.project_dir,
        sample_count=args.sample_count,
        threshold=args.threshold,
        use_chunked=args.chunked,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
