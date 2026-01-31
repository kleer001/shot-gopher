"""Media utilities for video and frame detection."""

from pathlib import Path
from typing import Optional, Tuple, List

VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]


def find_video_or_frames(source_dir: Path) -> Tuple[Optional[Path], bool, int]:
    """Find video file or existing frames in a source directory.

    Args:
        source_dir: Path to source directory (typically project/source/)

    Returns:
        Tuple of (video_path, has_frames, frame_count):
        - video_path: Path to video file if found, None otherwise
        - has_frames: True if frames exist in source/frames/
        - frame_count: Number of frames found (0 if none)
    """
    frames_dir = source_dir / "frames"
    video_path = None
    has_frames = False
    frame_count = 0

    if frames_dir.exists():
        frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        if frame_files:
            has_frames = True
            frame_count = len(frame_files)

    if not has_frames:
        video_path = find_video_file(source_dir)

    return video_path, has_frames, frame_count


def find_video_file(source_dir: Path) -> Optional[Path]:
    """Find a video file in the source directory.

    Prefers files named 'input.*', then falls back to any video file.

    Args:
        source_dir: Directory to search

    Returns:
        Path to video file or None if not found
    """
    if not source_dir.exists():
        return None

    for ext in VIDEO_EXTENSIONS:
        candidate = source_dir / f"input{ext}"
        if candidate.exists():
            return candidate

    for ext in VIDEO_EXTENSIONS:
        for video_file in source_dir.glob(f"*{ext}"):
            if video_file.is_file():
                return video_file

    return None


def get_frame_files(frames_dir: Path) -> List[Path]:
    """Get list of frame files in a directory.

    Args:
        frames_dir: Directory containing frames

    Returns:
        Sorted list of frame file paths
    """
    if not frames_dir.exists():
        return []

    frames = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
    return sorted(frames)


def get_dir_size_bytes(directory: Path) -> int:
    """Calculate total size of all files in a directory recursively.

    Args:
        directory: Directory to measure

    Returns:
        Total size in bytes
    """
    if not directory.exists():
        return 0

    total = 0
    try:
        for item in directory.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except (OSError, IOError):
                    pass
    except (OSError, IOError):
        pass

    return total


def get_dir_size_gb(directory: Path) -> float:
    """Calculate total size of directory in gigabytes.

    Args:
        directory: Directory to measure

    Returns:
        Size in GB (rounded to 2 decimal places)
    """
    size_bytes = get_dir_size_bytes(directory)
    return round(size_bytes / (1024 ** 3), 2)
