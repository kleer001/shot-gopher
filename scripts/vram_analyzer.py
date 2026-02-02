"""VRAM analyzer for per-project GPU memory recommendations.

Analyzes GPU VRAM availability against stage requirements and generates
per-project recommendations stored in the project directory.

Run at video ingest time to pre-calculate VRAM warnings for the web UI.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pipeline_utils import get_gpu_vram_gb


class VramStatus(str, Enum):
    OK = "ok"
    CHUNKED = "chunked"
    WARNING = "warning"
    INSUFFICIENT = "insufficient"


@dataclass
class StageVramConfig:
    """VRAM configuration for a pipeline stage.

    VRAM estimation formula:
        estimated = base_vram_gb + (megapixels * frames * gb_per_mpx_frame)

    Where megapixels = (width * height) / 1,000,000
    """
    base_vram_gb: float
    gb_per_mpx_frame: float = 0.0
    chunked: bool = False
    chunk_formula: Optional[str] = None


STAGE_VRAM_REQUIREMENTS: dict[str, StageVramConfig] = {
    "ingest": StageVramConfig(base_vram_gb=0),
    "depth": StageVramConfig(base_vram_gb=3),
    "roto": StageVramConfig(base_vram_gb=2, gb_per_mpx_frame=0.04),
    "interactive": StageVramConfig(base_vram_gb=2, gb_per_mpx_frame=0.04),
    "mama": StageVramConfig(
        base_vram_gb=4,
        gb_per_mpx_frame=0.08,
        chunked=True,
        chunk_formula="vram_to_chunk_size"
    ),
    "cleanplate": StageVramConfig(base_vram_gb=2, gb_per_mpx_frame=0.09),
    "colmap": StageVramConfig(base_vram_gb=2),
    "gsir": StageVramConfig(base_vram_gb=4, gb_per_mpx_frame=0.04),
    "mocap": StageVramConfig(base_vram_gb=6),
    "camera": StageVramConfig(base_vram_gb=0),
}


def get_mama_chunk_size(vram_gb: float) -> int:
    """Calculate optimal chunk size for VideoMaMa based on available VRAM.

    Mirrors the logic in video_mama.py for consistency.
    """
    available_vram = vram_gb * 0.9

    if available_vram >= 43:
        return 20
    elif available_vram >= 21:
        return 14
    elif available_vram >= 14:
        return 10
    elif available_vram >= 10:
        return 8
    elif available_vram >= 7:
        return 6
    else:
        return 4


@dataclass
class StageAnalysis:
    """Analysis result for a single stage."""
    stage: str
    status: VramStatus
    base_vram_gb: float
    estimated_vram_gb: float
    user_vram_gb: float
    message: str
    chunk_size: Optional[int] = None
    scales_with_frames: bool = False


@dataclass
class VramAnalysisResult:
    """Complete VRAM analysis for a project."""
    gpu_vram_gb: Optional[float]
    gpu_name: Optional[str]
    frame_count: int
    resolution: list[int]
    fps: float
    stages: dict[str, dict]
    analyzed_at: str
    warnings: list[str]


def get_gpu_name() -> Optional[str]:
    """Get the GPU name if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def calculate_estimated_vram(
    config: StageVramConfig,
    frame_count: int,
    resolution: tuple[int, int]
) -> float:
    """Calculate estimated VRAM needed based on resolution and frame count.

    Formula: base + (megapixels * frames * coefficient)
    Where megapixels = (width * height) / 1,000,000
    """
    megapixels = (resolution[0] * resolution[1]) / 1_000_000
    scaling = megapixels * frame_count * config.gb_per_mpx_frame
    return config.base_vram_gb + scaling


def analyze_stage(
    stage: str,
    vram_gb: Optional[float],
    frame_count: int,
    resolution: tuple[int, int],
) -> StageAnalysis:
    """Analyze VRAM requirements for a single stage."""
    config = STAGE_VRAM_REQUIREMENTS.get(stage)

    if config is None:
        return StageAnalysis(
            stage=stage,
            status=VramStatus.OK,
            base_vram_gb=0,
            estimated_vram_gb=0,
            user_vram_gb=vram_gb or 0,
            message="Unknown stage",
        )

    estimated_vram = calculate_estimated_vram(config, frame_count, resolution)
    scales_with_frames = config.gb_per_mpx_frame > 0

    if vram_gb is None:
        return StageAnalysis(
            stage=stage,
            status=VramStatus.WARNING,
            base_vram_gb=config.base_vram_gb,
            estimated_vram_gb=estimated_vram,
            user_vram_gb=0,
            message="GPU not detected - cannot estimate",
            scales_with_frames=scales_with_frames,
        )

    if config.base_vram_gb == 0:
        return StageAnalysis(
            stage=stage,
            status=VramStatus.OK,
            base_vram_gb=0,
            estimated_vram_gb=0,
            user_vram_gb=vram_gb,
            message="Minimal VRAM required",
        )

    headroom = vram_gb - estimated_vram

    if config.chunked:
        chunk_size = get_mama_chunk_size(vram_gb)
        if headroom >= 4:
            status = VramStatus.OK
            message = f"Chunk size: {chunk_size} frames"
        elif headroom >= 0:
            status = VramStatus.CHUNKED
            message = f"Will process in small chunks ({chunk_size} frames) - slower but works"
        else:
            status = VramStatus.WARNING
            message = f"Needs ~{estimated_vram:.1f}GB for {frame_count} frames, will use small chunks"

        return StageAnalysis(
            stage=stage,
            status=status,
            base_vram_gb=config.base_vram_gb,
            estimated_vram_gb=estimated_vram,
            user_vram_gb=vram_gb,
            message=message,
            chunk_size=chunk_size,
            scales_with_frames=scales_with_frames,
        )

    if scales_with_frames:
        if headroom >= 4:
            status = VramStatus.OK
            message = f"~{estimated_vram:.1f}GB needed for {frame_count} frames"
        elif headroom >= 0:
            status = VramStatus.WARNING
            message = f"Tight: ~{estimated_vram:.1f}GB needed for {frame_count} frames"
        else:
            status = VramStatus.INSUFFICIENT
            message = f"Needs ~{estimated_vram:.1f}GB for {frame_count} frames, you have {vram_gb:.1f}GB"

        return StageAnalysis(
            stage=stage,
            status=status,
            base_vram_gb=config.base_vram_gb,
            estimated_vram_gb=estimated_vram,
            user_vram_gb=vram_gb,
            message=message,
            scales_with_frames=True,
        )

    if headroom >= 2:
        status = VramStatus.OK
        message = "Should work well"
    elif headroom >= 0:
        status = VramStatus.WARNING
        message = "Tight on VRAM - close other GPU applications"
    else:
        status = VramStatus.INSUFFICIENT
        message = f"Needs {estimated_vram:.1f}GB, you have {vram_gb:.1f}GB"

    return StageAnalysis(
        stage=stage,
        status=status,
        base_vram_gb=config.base_vram_gb,
        estimated_vram_gb=estimated_vram,
        user_vram_gb=vram_gb,
        message=message,
    )


def analyze_project_vram(
    frame_count: int,
    resolution: tuple[int, int],
    fps: float,
    vram_gb: Optional[float] = None,
) -> VramAnalysisResult:
    """Analyze VRAM requirements for all stages given project parameters.

    Args:
        frame_count: Number of frames in the video
        resolution: Video resolution as (width, height)
        fps: Frames per second
        vram_gb: GPU VRAM in GB (auto-detected if None)

    Returns:
        Complete VRAM analysis result
    """
    if vram_gb is None:
        vram_gb = get_gpu_vram_gb()

    gpu_name = get_gpu_name()

    stages_analysis = {}
    warnings = []

    for stage in STAGE_VRAM_REQUIREMENTS:
        analysis = analyze_stage(stage, vram_gb, frame_count, resolution)
        stages_analysis[stage] = {
            "status": analysis.status.value,
            "base_vram_gb": analysis.base_vram_gb,
            "estimated_vram_gb": analysis.estimated_vram_gb,
            "message": analysis.message,
        }

        if analysis.chunk_size is not None:
            stages_analysis[stage]["chunk_size"] = analysis.chunk_size

        if analysis.scales_with_frames:
            stages_analysis[stage]["scales_with_frames"] = True

        if analysis.status in (VramStatus.WARNING, VramStatus.INSUFFICIENT):
            warnings.append(f"{stage}: {analysis.message}")

    return VramAnalysisResult(
        gpu_vram_gb=vram_gb,
        gpu_name=gpu_name,
        frame_count=frame_count,
        resolution=list(resolution),
        fps=fps,
        stages=stages_analysis,
        analyzed_at=datetime.now().isoformat(),
        warnings=warnings,
    )


def save_vram_analysis(project_dir: Path, analysis: VramAnalysisResult) -> Path:
    """Save VRAM analysis to project directory.

    Args:
        project_dir: Path to project directory
        analysis: VRAM analysis result

    Returns:
        Path to the saved analysis file
    """
    output_path = project_dir / "vram_analysis.json"

    analysis_dict = {
        "gpu_vram_gb": analysis.gpu_vram_gb,
        "gpu_name": analysis.gpu_name,
        "frame_count": analysis.frame_count,
        "resolution": analysis.resolution,
        "fps": analysis.fps,
        "stages": analysis.stages,
        "analyzed_at": analysis.analyzed_at,
        "warnings": analysis.warnings,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, indent=2)

    return output_path


def load_vram_analysis(project_dir: Path) -> Optional[dict]:
    """Load VRAM analysis from project directory.

    Args:
        project_dir: Path to project directory

    Returns:
        Analysis dict or None if not found
    """
    analysis_path = project_dir / "vram_analysis.json"

    if not analysis_path.exists():
        return None

    with open(analysis_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_and_save(
    project_dir: Path,
    frame_count: int,
    resolution: tuple[int, int],
    fps: float,
) -> dict:
    """Analyze VRAM requirements and save to project directory.

    Convenience function that combines analysis and saving.

    Args:
        project_dir: Path to project directory
        frame_count: Number of frames in the video
        resolution: Video resolution as (width, height)
        fps: Frames per second

    Returns:
        Analysis result as dictionary
    """
    analysis = analyze_project_vram(frame_count, resolution, fps)
    save_vram_analysis(project_dir, analysis)

    return {
        "gpu_vram_gb": analysis.gpu_vram_gb,
        "gpu_name": analysis.gpu_name,
        "frame_count": analysis.frame_count,
        "resolution": analysis.resolution,
        "fps": analysis.fps,
        "stages": analysis.stages,
        "analyzed_at": analysis.analyzed_at,
        "warnings": analysis.warnings,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze GPU VRAM requirements for a project"
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Path to project directory"
    )
    parser.add_argument(
        "--frames",
        type=int,
        required=True,
        help="Number of frames in the video"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video width (auto-detected from source frames if available)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video height (auto-detected from source frames if available)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second (default: 24.0)"
    )
    parser.add_argument(
        "--vram",
        type=float,
        help="Override GPU VRAM detection (GB)"
    )

    args = parser.parse_args()

    width = args.width
    height = args.height

    if width is None or height is None:
        source_frames_dir = args.project_dir / "source" / "frames"
        if source_frames_dir.exists():
            frames = sorted(source_frames_dir.glob("*.png"))
            if not frames:
                frames = sorted(source_frames_dir.glob("*.jpg"))
            if frames:
                from pipeline_utils import get_image_dimensions
                w, h = get_image_dimensions(frames[0])
                if w > 0 and h > 0:
                    width = width or w
                    height = height or h
                    print(f"Detected resolution: {width}x{height}")

        if width is None or height is None:
            width = width or 1920
            height = height or 1080
            print(f"Using default resolution: {width}x{height}")

    if args.vram:
        vram = args.vram
    else:
        vram = get_gpu_vram_gb()
        if vram:
            print(f"Detected GPU VRAM: {vram:.1f} GB")
        else:
            print("Could not detect GPU VRAM")

    analysis = analyze_project_vram(
        frame_count=args.frames,
        resolution=(width, height),
        fps=args.fps,
        vram_gb=vram,
    )

    if args.project_dir.exists():
        output_path = save_vram_analysis(args.project_dir, analysis)
        print(f"Analysis saved to: {output_path}")
    else:
        print("Project directory does not exist, printing analysis:")

    print(json.dumps({
        "gpu_vram_gb": analysis.gpu_vram_gb,
        "gpu_name": analysis.gpu_name,
        "frame_count": analysis.frame_count,
        "warnings": analysis.warnings,
        "stages": analysis.stages,
    }, indent=2))
