"""Unit tests for PipelineService."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime

from web.services.pipeline_service import PipelineService
from web.models.dto import JobStartRequest
from web.models.domain import PipelineJob, JobStatus, Project, ProjectStatus


class TestPipelineService:
    """Test PipelineService business logic."""

    def test_start_job_success(self):
        """Test successfully starting a pipeline job."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        project = Project(
            name="test_project",
            path=Path("/workspace/test_project"),
            status=ProjectStatus.CREATED,
            video_path=Path("/workspace/test_project/source/input.mp4"),
            stages=["ingest"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_project_repo.get.return_value = project
        mock_job_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["ingest"])

        with patch("web.pipeline_runner.start_pipeline"):
            response = service.start_job("test_project", request)

        assert response.status == "started"
        assert response.project_id == "test_project"
        assert response.job.status == JobStatus.RUNNING
        mock_job_repo.save.assert_called_once()
        mock_project_repo.save.assert_called_once()

    def test_start_job_project_not_found(self):
        """Test starting job for non-existent project raises error."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()
        mock_project_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["ingest"])

        with pytest.raises(ValueError, match="not found"):
            service.start_job("nonexistent", request)

    def test_start_job_already_running(self):
        """Test starting job when one is already running raises error."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        project = Project(
            name="test_project",
            path=Path("/workspace/test_project"),
            status=ProjectStatus.PROCESSING,
            video_path=None,
            stages=["ingest"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_project_repo.get.return_value = project

        existing_job = PipelineJob(
            project_name="test_project",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.5,
            message="Running...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        mock_job_repo.get.return_value = existing_job

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["ingest"])

        with pytest.raises(ValueError, match="already running"):
            service.start_job("test_project", request)

    def test_start_job_invalid_stages(self):
        """Test starting job with invalid stages raises error."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        project = Project(
            name="test_project",
            path=Path("/workspace/test_project"),
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_project_repo.get.return_value = project
        mock_job_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["invalid_stage"])

        with pytest.raises(ValueError, match="Invalid stages"):
            service.start_job("test_project", request)

    def test_get_job_status(self):
        """Test getting job status."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        job = PipelineJob(
            project_name="test_project",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.5,
            message="Processing...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        mock_job_repo.get.return_value = job

        service = PipelineService(mock_job_repo, mock_project_repo)
        result = service.get_job_status("test_project")

        assert result is not None
        assert result.status == JobStatus.RUNNING
        assert result.progress == 0.5

    def test_get_job_status_not_found(self):
        """Test getting status for non-existent job returns None."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()
        mock_job_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        result = service.get_job_status("nonexistent")

        assert result is None

    def test_stop_job_success(self):
        """Test successfully stopping a running job."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        job = PipelineJob(
            project_name="test_project",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.5,
            message="Running...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        mock_job_repo.get.return_value = job

        project = Project(
            name="test_project",
            path=Path("/workspace/test_project"),
            status=ProjectStatus.PROCESSING,
            video_path=None,
            stages=["ingest"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_project_repo.get.return_value = project

        service = PipelineService(mock_job_repo, mock_project_repo)

        with patch("web.pipeline_runner.stop_pipeline", return_value=True):
            result = service.stop_job("test_project")

        assert result is True
        mock_job_repo.save.assert_called_once()
        mock_project_repo.save.assert_called_once()

    def test_stop_job_not_running(self):
        """Test stopping job that's not running returns False."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()
        mock_job_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        result = service.stop_job("test_project")

        assert result is False

    def test_update_job_progress(self):
        """Test updating job progress."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        job = PipelineJob(
            project_name="test_project",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.5,
            message="Processing...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        mock_job_repo.get.return_value = job

        service = PipelineService(mock_job_repo, mock_project_repo)
        service.update_job_progress(
            "test_project",
            stage="ingest",
            progress=0.75,
            message="Almost done..."
        )

        mock_job_repo.save.assert_called_once()
        saved_job = mock_job_repo.save.call_args[0][0]
        assert saved_job.progress == 0.75
        assert saved_job.message == "Almost done..."

    def test_update_job_progress_on_complete(self):
        """Test updating job progress on completion updates project status."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        job = PipelineJob(
            project_name="test_project",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.9,
            message="Finishing...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        mock_job_repo.get.return_value = job

        project = Project(
            name="test_project",
            path=Path("/workspace/test_project"),
            status=ProjectStatus.PROCESSING,
            video_path=None,
            stages=["ingest"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_project_repo.get.return_value = project

        service = PipelineService(mock_job_repo, mock_project_repo)
        service.update_job_progress(
            "test_project",
            status=JobStatus.COMPLETE,
            progress=1.0,
            message="Complete"
        )

        mock_project_repo.save.assert_called_once()
        saved_project = mock_project_repo.save.call_args[0][0]
        assert saved_project.status == ProjectStatus.COMPLETE

    def test_update_job_progress_on_failure(self):
        """Test updating job progress on failure updates project status."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        job = PipelineJob(
            project_name="test_project",
            stages=["ingest"],
            status=JobStatus.RUNNING,
            current_stage="ingest",
            progress=0.5,
            message="Processing...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        mock_job_repo.get.return_value = job

        project = Project(
            name="test_project",
            path=Path("/workspace/test_project"),
            status=ProjectStatus.PROCESSING,
            video_path=None,
            stages=["ingest"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_project_repo.get.return_value = project

        service = PipelineService(mock_job_repo, mock_project_repo)
        service.update_job_progress(
            "test_project",
            status=JobStatus.FAILED,
            error="Something went wrong"
        )

        mock_project_repo.save.assert_called_once()
        saved_project = mock_project_repo.save.call_args[0][0]
        assert saved_project.status == ProjectStatus.FAILED

    def test_list_active_jobs(self):
        """Test listing all active jobs."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        active_jobs = [
            PipelineJob(
                project_name="project1",
                stages=["ingest"],
                status=JobStatus.RUNNING,
                current_stage="ingest",
                progress=0.5,
                message=None,
                started_at=datetime.now(),
                completed_at=None,
                error=None,
            ),
            PipelineJob(
                project_name="project2",
                stages=["depth"],
                status=JobStatus.PENDING,
                current_stage=None,
                progress=0.0,
                message=None,
                started_at=datetime.now(),
                completed_at=None,
                error=None,
            ),
        ]
        mock_job_repo.get_active_jobs.return_value = active_jobs

        service = PipelineService(mock_job_repo, mock_project_repo)
        result = service.list_active_jobs()

        assert len(result) == 2
        assert result[0].project_name == "project1"
        assert result[1].project_name == "project2"
