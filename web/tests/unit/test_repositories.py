"""Unit tests for repositories."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from web.repositories.project_repository import ProjectRepository
from web.repositories.job_repository import JobRepository
from web.models.domain import Project, ProjectStatus, PipelineJob, JobStatus


class TestProjectRepository:
    """Test ProjectRepository filesystem operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_save_and_get_project(self, temp_dir):
        """Test saving and retrieving a project."""
        repo = ProjectRepository(temp_dir)

        now = datetime.now()
        project = Project(
            name="test_project",
            path=temp_dir / "test_project",
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=["ingest", "depth"],
            created_at=now,
            updated_at=now,
        )

        saved = repo.save(project)
        assert saved.name == "test_project"

        retrieved = repo.get("test_project")
        assert retrieved is not None
        assert retrieved.name == "test_project"
        assert retrieved.status == ProjectStatus.CREATED
        assert retrieved.stages == ["ingest", "depth"]

    def test_list_projects(self, temp_dir):
        """Test listing all projects."""
        repo = ProjectRepository(temp_dir)

        now = datetime.now()
        for i in range(3):
            project = Project(
                name=f"project{i}",
                path=temp_dir / f"project{i}",
                status=ProjectStatus.CREATED,
                video_path=None,
                stages=[],
                created_at=now,
                updated_at=now,
            )
            repo.save(project)

        projects = repo.list()
        assert len(projects) == 3

    def test_delete_project(self, temp_dir):
        """Test deleting a project."""
        repo = ProjectRepository(temp_dir)

        now = datetime.now()
        project = Project(
            name="to_delete",
            path=temp_dir / "to_delete",
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=[],
            created_at=now,
            updated_at=now,
        )
        repo.save(project)

        assert repo.delete("to_delete") is True
        assert repo.get("to_delete") is None

    def test_get_nonexistent_project(self, temp_dir):
        """Test getting a project that doesn't exist."""
        repo = ProjectRepository(temp_dir)
        assert repo.get("nonexistent") is None


class TestJobRepository:
    """Test JobRepository in-memory operations."""

    def test_save_and_get_job(self):
        """Test saving and retrieving a job."""
        repo = JobRepository()

        now = datetime.now()
        job = PipelineJob(
            project_name="test_project",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.5,
            message="Processing...",
            started_at=now,
            completed_at=None,
            error=None,
        )

        saved = repo.save(job)
        assert saved.project_name == "test_project"

        retrieved = repo.get("test_project")
        assert retrieved is not None
        assert retrieved.project_name == "test_project"
        assert retrieved.status == JobStatus.RUNNING
        assert retrieved.progress == 0.5

    def test_list_jobs(self):
        """Test listing all jobs."""
        repo = JobRepository()

        now = datetime.now()
        for i in range(3):
            job = PipelineJob(
                project_name=f"project{i}",
                stages=["ingest"],
                status=JobStatus.RUNNING,
                current_stage="ingest",
                progress=0.0,
                message=None,
                started_at=now,
                completed_at=None,
                error=None,
            )
            repo.save(job)

        jobs = repo.list()
        assert len(jobs) == 3

    def test_delete_job(self):
        """Test deleting a job."""
        repo = JobRepository()

        now = datetime.now()
        job = PipelineJob(
            project_name="to_delete",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.0,
            message=None,
            started_at=now,
            completed_at=None,
            error=None,
        )
        repo.save(job)

        assert repo.delete("to_delete") is True
        assert repo.get("to_delete") is None

    def test_get_active_jobs(self):
        """Test getting only active jobs."""
        repo = JobRepository()

        now = datetime.now()
        repo.save(PipelineJob(
            project_name="running",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.0,
            message=None,
            started_at=now,
            completed_at=None,
            error=None,
        ))

        repo.save(PipelineJob(
            project_name="completed",
            stages=["ingest"],
            status=JobStatus.COMPLETE,
            current_stage="ingest",
            progress=1.0,
            message=None,
            started_at=now,
            completed_at=now,
            error=None,
        ))

        active_jobs = repo.get_active_jobs()
        assert len(active_jobs) == 1
        assert active_jobs[0].project_name == "running"
