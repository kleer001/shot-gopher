"""Pipeline service - business logic for pipeline orchestration."""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from web.models.domain import PipelineJob, JobStatus, Project, ProjectStatus
from web.models.dto import JobDTO, JobStartRequest, JobStartResponse
from web.repositories.job_repository import JobRepository
from web.repositories.project_repository import ProjectRepository
from web.services.config_service import get_config_service


class PipelineService:
    """
    Service for pipeline execution orchestration.

    Responsibilities:
    - Enforce business rules for job execution
    - Manage job lifecycle (start, stop, status)
    - Coordinate with pipeline_runner for actual execution
    - Update job state in repository

    Does NOT:
    - Handle HTTP requests (that's API layer)
    - Directly execute subprocesses (that's pipeline_runner)
    """

    def __init__(
        self,
        job_repo: JobRepository,
        project_repo: ProjectRepository
    ):
        self.job_repo = job_repo
        self.project_repo = project_repo
        self.config_service = get_config_service()

    def start_job(
        self,
        project_name: str,
        request: JobStartRequest
    ) -> JobStartResponse:
        """
        Start a pipeline job for a project.

        Business rules:
        - Project must exist
        - No job already running for project
        - Stages must be valid
        - Stage dependencies must be satisfied
        """
        project = self.project_repo.get(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found")

        existing_job = self.job_repo.get(project_name)
        if existing_job and existing_job.status == JobStatus.RUNNING:
            raise ValueError(f"Job already running for project '{project_name}'")

        valid_stages = set(self.config_service.get_stages().keys())
        invalid_stages = set(request.stages) - valid_stages
        if invalid_stages:
            raise ValueError(f"Invalid stages: {invalid_stages}. Valid: {valid_stages}")

        now = datetime.now()
        job = PipelineJob(
            project_name=project_name,
            stages=request.stages,
            status=JobStatus.RUNNING,
            current_stage=request.stages[0] if request.stages else None,
            progress=0.0,
            message="Starting pipeline...",
            started_at=now,
            completed_at=None,
            error=None,
        )

        self.job_repo.save(job)

        project.status = ProjectStatus.PROCESSING
        project.updated_at = now
        self.project_repo.save(project)

        from web.pipeline_runner import start_pipeline
        start_pipeline(
            project_id=project_name,
            project_dir=str(project.path),
            stages=request.stages,
            roto_prompt=request.roto_prompt,
            skip_existing=request.skip_existing,
        )

        return JobStartResponse(
            status="started",
            project_id=project_name,
            job=self._to_dto(job),
        )

    def get_job_status(self, project_name: str) -> Optional[JobDTO]:
        """Get current job status for a project."""
        job = self.job_repo.get(project_name)

        if not job:
            return None

        return self._to_dto(job)

    def stop_job(self, project_name: str) -> bool:
        """
        Stop a running pipeline job.

        Returns True if job was stopped, False if not running.
        """
        job = self.job_repo.get(project_name)

        if not job or job.status != JobStatus.RUNNING:
            return False

        from web.pipeline_runner import stop_pipeline
        stopped = stop_pipeline(project_name)

        if stopped:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.message = "Job cancelled by user"
            self.job_repo.save(job)

            project = self.project_repo.get(project_name)
            if project:
                project.status = ProjectStatus.FAILED
                project.updated_at = datetime.now()
                self.project_repo.save(project)

        return stopped

    def update_job_progress(
        self,
        project_name: str,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        status: Optional[JobStatus] = None,
        error: Optional[str] = None,
    ):
        """Update job progress (called by pipeline_runner)."""
        job = self.job_repo.get(project_name)

        if not job:
            return

        if stage is not None:
            job.current_stage = stage
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message
        if status is not None:
            job.status = status
        if error is not None:
            job.error = error

        if status == JobStatus.COMPLETE:
            job.completed_at = datetime.now()
            project = self.project_repo.get(project_name)
            if project:
                project.status = ProjectStatus.COMPLETE
                project.updated_at = datetime.now()
                self.project_repo.save(project)

        if status == JobStatus.FAILED:
            job.completed_at = datetime.now()
            project = self.project_repo.get(project_name)
            if project:
                project.status = ProjectStatus.FAILED
                project.updated_at = datetime.now()
                self.project_repo.save(project)

        self.job_repo.save(job)

    def list_active_jobs(self) -> List[JobDTO]:
        """List all active jobs."""
        jobs = self.job_repo.get_active_jobs()
        return [self._to_dto(job) for job in jobs]

    @staticmethod
    def _to_dto(job: PipelineJob) -> JobDTO:
        """Convert domain entity to DTO."""
        return JobDTO(
            project_name=job.project_name,
            stages=job.stages,
            status=job.status,
            current_stage=job.current_stage,
            progress=job.progress,
            message=job.message,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
        )
