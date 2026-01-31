"""Video analysis service for extracting metadata from video files."""

import json
import subprocess
from pathlib import Path
from typing import Optional

from install_wizard.platform import PlatformManager


class VideoService:
    """Service for video file analysis and metadata extraction."""

    def get_video_info(self, video_path: Path) -> dict:
        """Extract video metadata using ffprobe."""
        try:
            ffprobe_path = PlatformManager.find_tool("ffprobe")
            if not ffprobe_path:
                return {}

            cmd = [
                str(ffprobe_path), "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {}

            data = json.loads(result.stdout)

            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                return {}

            duration = float(data.get("format", {}).get("duration", 0))
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))

            fps_str = video_stream.get("r_frame_rate", "24/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 24.0
            else:
                fps = float(fps_str)

            frame_count = int(duration * fps) if duration > 0 else 0

            return {
                "duration": round(duration, 2),
                "fps": round(fps, 2),
                "resolution": [width, height],
                "frame_count": frame_count,
            }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}

    def get_resolution_from_frames(self, frames_dir: Path) -> Optional[tuple]:
        """Get resolution from first frame in directory."""
        try:
            from PIL import Image
            frames = sorted(frames_dir.glob("*.png"))
            if not frames:
                frames = sorted(frames_dir.glob("*.jpg"))
            if frames:
                with Image.open(frames[0]) as img:
                    return img.size
        except Exception:
            pass
        return None


_video_service = None


def get_video_service() -> VideoService:
    """Get singleton video service instance."""
    global _video_service
    if _video_service is None:
        _video_service = VideoService()
    return _video_service
