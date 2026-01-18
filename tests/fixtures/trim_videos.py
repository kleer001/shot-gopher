#!/usr/bin/env python3
"""Trim all downloaded test videos to 2 seconds."""

import subprocess
from pathlib import Path


def trim_video(video_path: Path, duration: float = 2.0) -> bool:
    """Trim video to specified duration, replacing original."""
    if not video_path.exists():
        return False

    temp_path = video_path.with_suffix(".tmp.mp4")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-t", str(duration),
                "-c", "copy",
                str(temp_path)
            ],
            check=True,
            capture_output=True
        )
        temp_path.replace(video_path)
        print(f"Trimmed: {video_path.name}")
        return True
    except subprocess.CalledProcessError:
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    fixtures_dir = Path(__file__).parent
    videos = list(fixtures_dir.glob("*.mp4"))

    if not videos:
        print("No videos found")
        return

    for video in videos:
        trim_video(video)


if __name__ == "__main__":
    main()
