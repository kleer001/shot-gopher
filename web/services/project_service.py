"""Project service - business logic for project management."""

from pathlib import Path
from datetime import datetime
from typing import Optional, List

from web.models.domain import Project, ProjectStatus
from web.models.dto import (
    ProjectDTO,
    ProjectCreateRequest,
    ProjectListResponse,
    VideoUploadResponse
)
from web.repositories.project_repository import ProjectRepository
from web.services.config_service import get_config_service


class ProjectService:
    """
    Service for project management business logic.

    Responsibilities:
    - Enforce business rules (unique names, valid states, etc.)
    - Orchestrate project operations
    - Convert between domain entities and DTOs

    Does NOT:
    - Handle HTTP requests (that's API layer)
    - Access files directly (that's repository layer)
    """

    def __init__(self, project_repo: ProjectRepository):
        self.project_repo = project_repo
        self.config_service = get_config_service()

    def list_projects(self) -> ProjectListResponse:
        """List all projects."""
        projects = self.project_repo.list()

        return ProjectListResponse(
            projects=[self._to_dto(p) for p in projects],
            total=len(projects),
        )

    def get_project(self, name: str) -> Optional[ProjectDTO]:
        """Get project by name."""
        project = self.project_repo.get(name)

        if not project:
            return None

        return self._to_dto(project)

    def create_project(
        self,
        request: ProjectCreateRequest,
        projects_dir: Path
    ) -> ProjectDTO:
        """
        Create a new project.

        Business rules:
        - Project name must be unique
        - Name must be valid (alphanumeric, dash, underscore only)
        - Stages must be valid stage names
        """
        existing = self.project_repo.get(request.name)
        if existing:
            raise ValueError(f"Project '{request.name}' already exists")

        valid_stages = set(self.config_service.get_stages().keys())
        invalid_stages = set(request.stages) - valid_stages
        if invalid_stages:
            raise ValueError(f"Invalid stages: {invalid_stages}. Valid: {valid_stages}")

        now = datetime.now()
        project = Project(
            name=request.name,
            path=projects_dir / request.name,
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=request.stages,
            created_at=now,
            updated_at=now,
        )

        project = self.project_repo.save(project)

        return self._to_dto(project)

    def save_uploaded_video(
        self,
        project_name: str,
        video_filename: str,
        video_content: bytes
    ) -> ProjectDTO:
        """Save uploaded video to project."""
        project = self.project_repo.get(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found")

        video_path = project.source_dir / video_filename
        video_path.parent.mkdir(parents=True, exist_ok=True)

        with open(video_path, "wb") as f:
            f.write(video_content)

        project.video_path = video_path
        project.updated_at = datetime.now()

        project = self.project_repo.save(project)

        return self._to_dto(project)

    def delete_project(self, name: str) -> bool:
        """Delete a project."""
        return self.project_repo.delete(name)

    def update_project_status(
        self,
        name: str,
        status: ProjectStatus
    ) -> Optional[ProjectDTO]:
        """Update project status."""
        project = self.project_repo.get(name)
        if not project:
            return None

        project.status = status
        project.updated_at = datetime.now()

        project = self.project_repo.save(project)

        return self._to_dto(project)

    @staticmethod
    def _to_dto(project: Project) -> ProjectDTO:
        """Convert domain entity to DTO."""
        return ProjectDTO(
            name=project.name,
            status=project.status,
            video_path=str(project.video_path) if project.video_path else None,
            stages=project.stages,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )
