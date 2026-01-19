"""Pipeline utility functions.

General-purpose utilities for FFmpeg operations, subprocess handling,
and GPU memory management. No pipeline-specific logic.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "run_command",
    "get_frame_count",
    "extract_frames",
    "get_video_info",
    "generate_preview_movie",
    "get_image_dimensions",
    "clear_gpu_memory",
]


def clear_gpu_memory() -> None:
    """Clear GPU VRAM to free memory after a stage completes.

    This helps prevent out-of-memory errors when running multiple
    GPU-intensive stages sequentially. Safe to call even if CUDA
    is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  → Cleared GPU memory")
    except ImportError:
        pass
    except Exception as e:
        print(f"  → Warning: Could not clear GPU memory: {e}")


def run_command(
    cmd: list[str],
    description: str,
    check: bool = True,
    stream: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command with logging and optional streaming output.

    Args:
        cmd: Command and arguments
        description: Human-readable description
        check: Raise exception on non-zero exit
        stream: If True, stream stdout/stderr in real-time (default: True)

    Returns:
        CompletedProcess-like object with returncode, stdout, stderr

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
    """
    print(f"  → {description}")
    print(f"    $ {' '.join(cmd)}")
    sys.stdout.flush()

    if stream:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stdout_lines = []
        for line in iter(process.stdout.readline, ''):
            stdout_lines.append(line)
            print(line.rstrip())
            sys.stdout.flush()

        process.wait()

        stdout = ''.join(stdout_lines)
        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, "")

        class Result:
            def __init__(self):
                self.returncode = process.returncode
                self.stdout = stdout
                self.stderr = ""
        return Result()
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"    Error: {result.stderr}", file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result


def get_frame_count(input_path: Path) -> int:
    """Get total frame count from video using ffprobe.

    Args:
        input_path: Path to video file

    Returns:
        Number of frames, or 0 if count could not be determined
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0


def extract_frames(
    input_path: Path,
    output_dir: Path,
    start_frame: int = 1,
    fps: Optional[float] = None
) -> int:
    """Extract frames from video file using ffmpeg.

    Args:
        input_path: Path to input video
        output_dir: Directory to write frames
        start_frame: Starting frame number for output naming
        fps: Optional frame rate override

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%04d.png"

    print(f"  → Analyzing video for frame count...")
    total_frames = get_frame_count(input_path)
    if total_frames > 0:
        print(f"    Video contains {total_frames} frames")
    else:
        print(f"    Frame count unknown, progress will be estimated")

    cmd = ["ffmpeg", "-i", str(input_path)]

    if fps:
        cmd.extend(["-vf", f"fps={fps}"])

    cmd.extend([
        "-start_number", str(start_frame),
        "-q:v", "2",
        "-progress", "pipe:1",
        "-nostats",
        str(output_pattern),
        "-y"
    ])

    print(f"  → Extracting frames to {output_dir}")
    print(f"    $ {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    last_reported = 0
    report_interval = max(1, total_frames // 100) if total_frames > 0 else 10

    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line.startswith("frame="):
            try:
                current_frame = int(line.split("=")[1])
                if current_frame - last_reported >= report_interval or current_frame == total_frames:
                    if total_frames > 0:
                        print(f"[FFmpeg] Extracting frame {current_frame}/{total_frames}")
                    else:
                        print(f"[FFmpeg] Extracting frame {current_frame}")
                    last_reported = current_frame
                    sys.stdout.flush()
            except (ValueError, IndexError):
                pass

    process.wait()

    if process.returncode != 0:
        stderr_output = process.stderr.read() if process.stderr else ""
        print(f"    Error during extraction: {stderr_output}", file=sys.stderr)
        raise subprocess.CalledProcessError(process.returncode, cmd)

    frames = list(output_dir.glob("frame_*.png"))
    return len(frames)


def get_video_info(input_path: Path) -> dict:
    """Get video metadata using ffprobe.

    Args:
        input_path: Path to video file

    Returns:
        Dict with 'streams' and 'format' keys, or empty dict on error
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def generate_preview_movie(
    image_dir: Path,
    output_path: Path,
    fps: float = 24.0,
    pattern: str = "*.png",
    crf: int = 23,
) -> bool:
    """Generate a preview MP4 from an image sequence.

    Args:
        image_dir: Directory containing images
        output_path: Output MP4 path
        fps: Frame rate
        pattern: Glob pattern for images
        crf: Quality (lower = better, 18-28 typical)

    Returns:
        True if successful
    """
    images = sorted(image_dir.glob(pattern))
    if not images:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(image_dir / pattern),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    → Created: {output_path.name}")
            return True
        else:
            first_image = images[0]
            match = re.search(r'(\d+)', first_image.stem)
            if match:
                num_digits = len(match.group(1))
                prefix = first_image.stem[:match.start()]
                suffix = first_image.suffix
                input_pattern = str(image_dir / f"{prefix}%0{num_digits}d{suffix}")
                start_num = int(match.group(1))

                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-start_number", str(start_num),
                    "-i", input_pattern,
                    "-c:v", "libx264",
                    "-crf", str(crf),
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"    → Created: {output_path.name}")
                    return True
            return False
    except Exception:
        return False


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get width and height from an image file using ffprobe.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height), defaults to (1920, 1080) on error
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(image_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        width, height = result.stdout.strip().split("x")
        return int(width), int(height)
    except (ValueError, AttributeError):
        return 1920, 1080
