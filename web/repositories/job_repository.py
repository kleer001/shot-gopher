"""Job repository - in-memory implementation."""

from typing import Optional, Dict, List
from web.models.domain import PipelineJob, JobStatus
from web.repositories.base import Repository


class JobRepository(Repository[PipelineJob]):
    """
    Repository for pipeline job data.

    Current implementation: In-memory (dict)
    Rationale: Jobs are transient, don't need persistence across restarts
    Future: Could persist to DB if needed for job history
    """

    def __init__(self):
        self._jobs: Dict[str, PipelineJob] = {}

    def get(self, project_name: str) -> Optional[PipelineJob]:
        """Get job by project name."""
        return self._jobs.get(project_name)

    def list(self) -> List[PipelineJob]:
        """List all jobs."""
        return list(self._jobs.values())

    def save(self, job: PipelineJob) -> PipelineJob:
        """Save job to memory."""
        self._jobs[job.project_name] = job
        return job

    def delete(self, project_name: str) -> bool:
        """Delete job from memory."""
        if project_name in self._jobs:
            del self._jobs[project_name]
            return True
        return False

    def get_active_jobs(self) -> List[PipelineJob]:
        """Get all running or pending jobs."""
        return [
            job for job in self._jobs.values()
            if job.status in (JobStatus.RUNNING, JobStatus.PENDING)
        ]
