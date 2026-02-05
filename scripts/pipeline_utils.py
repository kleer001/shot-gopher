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

from install_wizard.platform import PlatformManager

# Cache for tool paths
_ffmpeg_path: Optional[str] = None
_ffprobe_path: Optional[str] = None


def _get_ffmpeg() -> str:
    """Get FFmpeg executable path, using PlatformManager for Windows."""
    global _ffmpeg_path
    if _ffmpeg_path is None:
        found = PlatformManager.find_tool("ffmpeg")
        _ffmpeg_path = str(found) if found else "ffmpeg"
    return _ffmpeg_path


def _get_ffprobe() -> str:
    """Get FFprobe executable path, using PlatformManager for Windows."""
    global _ffprobe_path
    if _ffprobe_path is None:
        found = PlatformManager.find_tool("ffprobe")
        _ffprobe_path = str(found) if found else "ffprobe"
    return _ffprobe_path

__all__ = [
    "run_command",
    "get_frame_count",
    "extract_frames",
    "get_video_info",
    "generate_preview_movie",
    "get_image_dimensions",
    "clear_gpu_memory",
    "clear_output_directory",
    "get_gpu_vram_gb",
]


def get_gpu_vram_gb() -> float | None:
    """Detect GPU VRAM in gigabytes.

    Tries PyTorch first, falls back to nvidia-smi.

    Returns:
        VRAM in GB, or None if no GPU detected.
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            return vram_bytes / (1024 ** 3)
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            vram_mb = int(result.stdout.strip().split("\n")[0])
            return vram_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return None


def clear_output_directory(output_dir: Path) -> int:
    """Clear existing files from output directory to prevent accumulation.

    Completely removes the directory and recreates it empty to ensure
    no stale files remain and ComfyUI counters reset properly.

    Args:
        output_dir: Path to the output directory

    Returns:
        Number of files that were in the directory
    """
    import shutil

    print(f"  → Output directory: {output_dir}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0

    file_count = sum(1 for _ in output_dir.rglob("*") if _.is_file())

    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_count > 0:
        print(f"  → Cleared directory ({file_count} files removed)")

    return file_count


def clear_gpu_memory(comfyui_url: str = None) -> None:
    """Clear GPU VRAM to free memory after a stage completes.

    This helps prevent out-of-memory errors when running multiple
    GPU-intensive stages sequentially. Unloads ComfyUI's cached models
    and clears PyTorch's CUDA cache.

    Args:
        comfyui_url: ComfyUI API URL for model unloading (optional)
    """
    from comfyui_utils import free_comfyui_memory, DEFAULT_COMFYUI_URL

    url = comfyui_url or DEFAULT_COMFYUI_URL
    models_freed = free_comfyui_memory(url, unload_models=True)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if models_freed:
                print("  → Cleared GPU memory (models unloaded)")
            else:
                print("  → Cleared GPU memory")
    except ImportError:
        if models_freed:
            print("  → Cleared GPU memory (models unloaded)")
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
        _get_ffprobe(), "-v", "error",
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

    cmd = [_get_ffmpeg(), "-i", str(input_path)]

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

    frames = list(output_dir.glob("*.png"))
    return len(frames)


def get_video_info(input_path: Path) -> dict:
    """Get video metadata using ffprobe.

    Args:
        input_path: Path to video file

    Returns:
        Dict with 'streams' and 'format' keys, or empty dict on error
    """
    cmd = [
        _get_ffprobe(), "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


GSIR_FRAME_SKIPS = [1, 4, 8, 16]


def create_frame_subsets(
    source_frames_dir: Path,
    skip_factors: list[int] = None,
) -> dict[int, Path]:
    """Create subsampled frame directories using symlinks for GS-IR fallback.

    For slow-moving camera shots, GS-IR may fail due to insufficient camera
    baseline. This creates sparser frame subsets that can be tried as fallbacks.

    Args:
        source_frames_dir: Directory containing all extracted frames (frames/)
        skip_factors: List of skip factors (e.g., [4, 8, 16] for every 4th, 8th, 16th frame)

    Returns:
        Dict mapping skip factor to subset directory path
    """
    if skip_factors is None:
        skip_factors = [s for s in GSIR_FRAME_SKIPS if s > 1]

    source_frames = sorted(source_frames_dir.glob("frame_*.png"))
    if not source_frames:
        return {}

    subsets = {}
    parent_dir = source_frames_dir.parent

    for skip in skip_factors:
        subset_dir = parent_dir / f"frames_{skip}s"
        subset_dir.mkdir(parents=True, exist_ok=True)

        for old_file in subset_dir.glob("*.png"):
            old_file.unlink()

        new_frame_num = 1
        for i, frame_path in enumerate(source_frames):
            if i % skip == 0:
                new_name = f"frame_{new_frame_num:04d}.png"
                link_path = subset_dir / new_name
                if not link_path.exists():
                    link_path.symlink_to(frame_path.resolve())
                new_frame_num += 1

        subsets[skip] = subset_dir
        frame_count = len(list(subset_dir.glob("*.png")))
        print(f"    Created {subset_dir.name}/ with {frame_count} frames (every {skip}th)")

    return subsets


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
        _get_ffmpeg(), "-y",
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
                    _get_ffmpeg(), "-y",
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
        Tuple of (width, height), or (0, 0) if dimensions cannot be determined
    """
    try:
        cmd = [
            _get_ffprobe(), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(image_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        width, height = result.stdout.strip().split("x")
        return int(width), int(height)
    except (ValueError, AttributeError, FileNotFoundError, OSError):
        print(f"  → Warning: Could not determine dimensions for {image_path.name}")
        return 0, 0
