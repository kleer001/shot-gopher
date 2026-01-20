"""Data Transfer Objects - API contracts."""

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, List, Dict, Any
from web.models.domain import ProjectStatus, JobStatus


class ProjectDTO(BaseModel):
    """Project data for API responses."""
    name: str
    status: ProjectStatus
    video_path: Optional[str] = None
    stages: List[str] = []
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProjectCreateRequest(BaseModel):
    """Request to create a new project."""
    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    stages: List[str] = Field(default_factory=list)


class ProjectListResponse(BaseModel):
    """Response with list of projects."""
    projects: List[ProjectDTO]
    total: int


class VideoUploadResponse(BaseModel):
    """Response after video upload."""
    project_id: str
    name: str
    project_dir: str
    video_info: Optional[Dict[str, Any]] = None


class JobDTO(BaseModel):
    """Pipeline job data for API responses."""
    project_name: str
    stages: List[str]
    status: JobStatus
    current_stage: Optional[str] = None
    progress: float = Field(ge=0.0, le=1.0)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class JobStartRequest(BaseModel):
    """Request to start a pipeline job."""
    stages: List[str] = Field(..., min_length=1)
    roto_prompt: str = "person"
    skip_existing: bool = False


class JobStartResponse(BaseModel):
    """Response after starting a job."""
    status: str
    project_id: str
    job: Optional[JobDTO] = None


class ProgressUpdate(BaseModel):
    """Real-time progress update (WebSocket)."""
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    status: JobStatus
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """System status response."""
    comfyui: bool
    disk_space_gb: float
    projects_dir: str
    install_dir: str


class ProjectOutputsResponse(BaseModel):
    """Response with project outputs."""
    project_id: str
    project_dir: str
    outputs: Dict[str, Dict[str, Any]]
