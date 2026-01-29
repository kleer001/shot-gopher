"""Pipeline configuration dataclass.

Centralizes all pipeline configuration into a single typed object,
reducing parameter count and improving maintainability.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from env_config import DEFAULT_PROJECTS_DIR
from comfyui_utils import DEFAULT_COMFYUI_URL
from pipeline_constants import STAGE_ORDER


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run.

    Consolidates all pipeline parameters into a single object.
    """

    input_path: Optional[Path] = None
    project_name: Optional[str] = None
    projects_dir: Path = field(default_factory=lambda: DEFAULT_PROJECTS_DIR)
    stages: list[str] = field(default_factory=lambda: STAGE_ORDER.copy())
    comfyui_url: str = DEFAULT_COMFYUI_URL
    fps: Optional[float] = None
    skip_existing: bool = False
    overwrite: bool = True
    auto_movie: bool = False
    auto_start_comfyui: bool = True

    colmap_quality: str = "medium"
    colmap_dense: bool = False
    colmap_mesh: bool = False
    colmap_use_masks: bool = True
    colmap_max_size: int = -1

    gsir_iterations: int = 35000
    gsir_path: Optional[str] = None

    mocap_method: str = "auto"

    roto_prompt: Optional[str] = None
    roto_start_frame: Optional[int] = None
    separate_instances: bool = True

    @classmethod
    def from_args(cls, args, input_path: Optional[Path] = None, project_dir: Optional[Path] = None) -> "PipelineConfig":
        """Create config from argparse namespace.

        Args:
            args: Parsed argparse namespace
            input_path: Resolved input path (may differ from args.input)
            project_dir: Existing project directory (if resuming)

        Returns:
            PipelineConfig instance
        """
        if project_dir:
            return cls(
                input_path=None,
                project_name=project_dir.name,
                projects_dir=project_dir.parent,
                stages=args.stages if hasattr(args, 'stages') and isinstance(args.stages, list) else STAGE_ORDER.copy(),
                comfyui_url=args.comfyui_url,
                fps=args.fps,
                skip_existing=args.skip_existing,
                overwrite=not args.no_overwrite,
                auto_movie=args.auto_movie,
                auto_start_comfyui=not args.no_auto_comfyui,
                colmap_quality=args.colmap_quality,
                colmap_dense=args.colmap_dense,
                colmap_mesh=args.colmap_mesh,
                colmap_use_masks=not args.colmap_no_masks,
                colmap_max_size=args.colmap_max_size,
                gsir_iterations=args.gsir_iterations,
                gsir_path=args.gsir_path,
                mocap_method=args.mocap_method,
                roto_prompt=args.prompt,
                roto_start_frame=args.start_frame,
                separate_instances=args.separate_instances,
            )

        return cls(
            input_path=input_path,
            project_name=args.name,
            projects_dir=args.projects_dir,
            stages=args.stages if hasattr(args, 'stages') and isinstance(args.stages, list) else STAGE_ORDER.copy(),
            comfyui_url=args.comfyui_url,
            fps=args.fps,
            skip_existing=args.skip_existing,
            overwrite=not args.no_overwrite,
            auto_movie=args.auto_movie,
            auto_start_comfyui=not args.no_auto_comfyui,
            colmap_quality=args.colmap_quality,
            colmap_dense=args.colmap_dense,
            colmap_mesh=args.colmap_mesh,
            colmap_use_masks=not args.colmap_no_masks,
            colmap_max_size=args.colmap_max_size,
            gsir_iterations=args.gsir_iterations,
            gsir_path=args.gsir_path,
            mocap_method=args.mocap_method,
            roto_prompt=args.prompt,
            roto_start_frame=args.start_frame,
            separate_instances=args.separate_instances,
        )


@dataclass
class StageContext:
    """Runtime context passed to stage handlers.

    Contains resolved paths and state needed during stage execution.
    """

    project_dir: Path
    source_frames: Path
    workflows_dir: Path
    comfyui_url: str
    total_frames: int
    fps: float
    skip_existing: bool
    overwrite: bool
    auto_movie: bool

    @classmethod
    def from_config(cls, config: PipelineConfig, project_dir: Path, total_frames: int, fps: float) -> "StageContext":
        """Create context from config and runtime values.

        Args:
            config: Pipeline configuration
            project_dir: Resolved project directory
            total_frames: Number of source frames
            fps: Frames per second

        Returns:
            StageContext instance
        """
        return cls(
            project_dir=project_dir,
            source_frames=project_dir / "source" / "frames",
            workflows_dir=Path(__file__).parent.parent / "workflow_templates",
            comfyui_url=config.comfyui_url,
            total_frames=total_frames,
            fps=fps,
            skip_existing=config.skip_existing,
            overwrite=config.overwrite,
            auto_movie=config.auto_movie,
        )
