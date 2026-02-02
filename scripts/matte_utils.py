"""Matte and mask combination utilities.

Functions for combining multiple matte/mask sequences into unified outputs.
"""

from pathlib import Path

__all__ = [
    "combine_mattes",
    "combine_mask_sequences",
    "prepare_roto_for_cleanplate",
]


def combine_mattes(
    input_dirs: list,
    output_dir: Path,
    output_prefix: str = "combined"
) -> bool:
    """Combine multiple matte directories into a single combined output.

    Takes the maximum (union) of all mattes at each frame.

    Args:
        input_dirs: List of directories containing matte images
        output_dir: Directory to write combined mattes
        output_prefix: Prefix for output filenames

    Returns:
        True if successful, False otherwise
    """
    from PIL import Image
    import numpy as np

    if not input_dirs:
        print("  → No input directories to combine")
        return False

    first_dir = input_dirs[0]
    frame_files = sorted(first_dir.glob("*.png"))
    if not frame_files:
        print(f"  → No PNG files found in {first_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(frame_files)
    print(f"  → Combining {len(input_dirs)} matte directories ({num_frames} frames)")

    for frame_idx in range(num_frames):
        combined = None

        for input_dir in input_dirs:
            dir_files = sorted(input_dir.glob("*.png"))
            if frame_idx >= len(dir_files):
                continue

            frame_file = dir_files[frame_idx]
            img = Image.open(frame_file).convert('L')
            matte = np.array(img, dtype=np.float32)

            if combined is None:
                combined = matte
            else:
                combined = np.maximum(combined, matte)

        if combined is not None:
            out_file = output_dir / f"{output_prefix}_{frame_idx:05d}_.png"
            result = Image.fromarray(combined.astype(np.uint8))
            result.save(out_file)

        if (frame_idx + 1) % 50 == 0:
            print(f"    Combined {frame_idx + 1}/{num_frames} frames...")

    print(f"  → Combined mattes written to: {output_dir}")
    return True


def combine_mask_sequences(
    source_dirs: list[Path],
    output_dir: Path,
    prefix: str = "combined"
) -> int:
    """Combine multiple mask sequences by OR-ing them together.

    Args:
        source_dirs: List of directories containing mask sequences
        output_dir: Output directory for combined masks
        prefix: Filename prefix for combined masks

    Returns:
        Number of frames processed
    """
    from PIL import Image
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dirs:
        return 0

    frame_files = sorted(source_dirs[0].glob("*.png"))
    if not frame_files:
        return 0

    count = 0
    for i, frame_file in enumerate(frame_files):
        combined = None
        for src_dir in source_dirs:
            src_files = sorted(src_dir.glob("*.png"))
            if i < len(src_files):
                img = Image.open(src_files[i]).convert('L')
                arr = np.array(img)
                if combined is None:
                    combined = arr
                else:
                    combined = np.maximum(combined, arr)

        if combined is not None:
            out_name = f"{prefix}_{i+1:05d}.png"
            result = Image.fromarray(combined)
            result.save(output_dir / out_name)
            count += 1

    return count


def prepare_roto_for_cleanplate(roto_dir: Path) -> tuple[bool, str]:
    """Prepare roto masks for cleanplate processing.

    Finds roto sequences and copies the appropriate one to roto root as mask_*.png.

    Logic:
    - If one sequence in roto/ → use it (regardless of name)
    - If multiple sequences → prefer combined/ if exists, else OR them together

    Args:
        roto_dir: Path to roto directory

    Returns:
        Tuple of (success, message describing what was done)
    """
    import shutil

    all_roto_dirs = []
    if roto_dir.exists():
        for subdir in sorted(roto_dir.iterdir()):
            if subdir.is_dir() and list(subdir.glob("*.png")):
                all_roto_dirs.append(subdir)

    if not all_roto_dirs:
        return False, "No roto sequences found"

    for old_file in roto_dir.glob("*.png"):
        old_file.unlink()

    if len(all_roto_dirs) == 1:
        source_dir = all_roto_dirs[0]
        for i, mask_file in enumerate(sorted(source_dir.glob("*.png"))):
            out_name = f"mask_{i+1:05d}.png"
            shutil.copy2(mask_file, roto_dir / out_name)
        return True, f"Using masks from {source_dir.name}/"

    combined_dir = roto_dir / "combined"
    if combined_dir in all_roto_dirs:
        for i, mask_file in enumerate(sorted(combined_dir.glob("*.png"))):
            out_name = f"mask_{i+1:05d}.png"
            shutil.copy2(mask_file, roto_dir / out_name)
        return True, "Using combined mattes from roto/combined/"

    count = combine_mask_sequences(all_roto_dirs, roto_dir, prefix="mask")
    return True, f"Consolidated {count} frames from {len(all_roto_dirs)} mask sources"
