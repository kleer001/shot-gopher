"""Domain entities - internal representation (framework-agnostic)."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List


class ProjectStatus(str, Enum):
    """Project status enumeration."""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    """Pipeline job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Project:
    """Project domain entity."""
    name: str
    path: Path
    status: ProjectStatus
    video_path: Optional[Path]
    stages: List[str]
    created_at: datetime
    updated_at: datetime

    @property
    def source_dir(self) -> Path:
        """Source directory path."""
        return self.path / "source"

    @property
    def frames_dir(self) -> Path:
        """Frames directory path."""
        return self.path / "source" / "frames"

    @property
    def state_file(self) -> Path:
        """Project state file path."""
        return self.path / "project_state.json"


@dataclass
class PipelineJob:
    """Pipeline job domain entity."""
    project_name: str
    stages: List[str]
    status: JobStatus
    current_stage: Optional[str]
    progress: float
    message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
