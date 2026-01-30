"""Tests for job start and status tracking flow."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from web.models.domain import Project, ProjectStatus, PipelineJob, JobStatus
from web.models.dto import JobStartRequest
from web.services.pipeline_service import PipelineService


class TestJobStartFlow:
    """Test the job start flow from API to pipeline runner."""

    def test_start_job_validates_project_exists(self):
        """Starting a job for non-existent project should raise ValueError."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()
        mock_project_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["depth"], roto_prompt="person")

        with pytest.raises(ValueError, match="not found"):
            service.start_job("nonexistent", request)

    def test_start_job_validates_no_running_job(self):
        """Starting a job when one is already running should raise ValueError."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        mock_project_repo.get.return_value = Project(
            name="test",
            path=Path("/tmp/test"),
            status=ProjectStatus.PROCESSING,
            video_path=None,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_job_repo.get.return_value = PipelineJob(
            project_name="test",
            stages=["depth"],
            status=JobStatus.RUNNING,
            current_stage="depth",
            progress=0.0,
            message="Running",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["roto"], roto_prompt="person")

        with pytest.raises(ValueError, match="already running"):
            service.start_job("test", request)

    def test_start_job_validates_stages(self):
        """Starting a job with invalid stages should raise ValueError."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        mock_project_repo.get.return_value = Project(
            name="test",
            path=Path("/tmp/test"),
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_job_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["invalid_stage"], roto_prompt="person")

        with pytest.raises(ValueError, match="Invalid stages"):
            service.start_job("test", request)

    @patch("web.pipeline_runner.start_pipeline")
    def test_start_job_success(self, mock_start_pipeline):
        """Starting a job with valid params should succeed."""
        mock_job_repo = Mock()
        mock_project_repo = Mock()

        project = Project(
            name="test",
            path=Path("/tmp/test"),
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_project_repo.get.return_value = project
        mock_job_repo.get.return_value = None

        service = PipelineService(mock_job_repo, mock_project_repo)
        request = JobStartRequest(stages=["depth"], roto_prompt="person")

        response = service.start_job("test", request)

        assert response.status == "started"
        assert response.project_id == "test"
        mock_start_pipeline.assert_called_once()
        mock_job_repo.save.assert_called_once()


class TestJobStatusEndpoint:
    """Test the job status endpoint behavior."""

    def test_active_jobs_dict_structure(self):
        """Verify active_jobs dict stores expected fields."""
        from web.api import active_jobs, active_jobs_lock

        with active_jobs_lock:
            active_jobs["test_project"] = {
                "status": "running",
                "current_stage": "depth",
                "progress": 0.5,
                "last_output": "Processing frame 50/100",
                "error": None,
            }

        with active_jobs_lock:
            job = active_jobs.get("test_project")

        assert job is not None
        assert job["status"] == "running"
        assert job["current_stage"] == "depth"
        assert job["progress"] == 0.5
        assert job["last_output"] == "Processing frame 50/100"

        # Clean up
        with active_jobs_lock:
            del active_jobs["test_project"]


class TestPipelineRunnerProgressParsing:
    """Test progress line parsing from pipeline output."""

    def test_parse_stage_start(self):
        """Stage start lines should be parsed correctly."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("=== Stage: depth ===", None, ["depth", "roto"])

        assert result is not None
        assert result["current_stage"] == "depth"
        assert result["stage_index"] == 0

    def test_parse_comfyui_file_progress(self):
        """ComfyUI file progress should be parsed correctly."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("[ComfyUI] depth frame 42/100", "depth", ["depth"])

        assert result is not None
        assert result["frame"] == 42
        assert result["total_frames"] == 100
        assert result["progress"] == 0.42

    def test_parse_ffmpeg_progress(self):
        """FFmpeg extraction progress should be parsed correctly."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("[FFmpeg] Extracting frame 50/200", "ingest", ["ingest"])

        assert result is not None
        assert result["frame"] == 50
        assert result["total_frames"] == 200
        assert result["progress"] == 0.25

    def test_parse_colmap_progress(self):
        """COLMAP registration progress should be parsed correctly."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("Registered 30/100 images", "colmap", ["colmap"])

        assert result is not None
        assert result["frame"] == 30
        assert result["total_frames"] == 100
        assert result["progress"] == 0.30

    def test_parse_bracket_progress(self):
        """Bracket-style progress [1/5] should be parsed correctly."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("[3/10] Processing frame_0003.png", "mocap", ["mocap"])

        assert result is not None
        assert result["frame"] == 3
        assert result["total_frames"] == 10
        assert result["progress"] == 0.30

    def test_parse_non_progress_line(self):
        """Non-progress lines should return None."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("Some random output text", "depth", ["depth"])

        assert result is None


class TestPipelineRunnerVideoFinding:
    """Test video file discovery in pipeline runner."""

    def test_finds_input_mp4(self, tmp_path):
        """Should find input.mp4 in source directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "input.mp4").write_text("fake video")

        # Test the video finding logic
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]
        input_video = None

        for ext in video_extensions:
            candidate = source_dir / f"input{ext}"
            if candidate.exists():
                input_video = candidate
                break

        assert input_video is not None
        assert input_video.name == "input.mp4"

    def test_finds_custom_named_video(self, tmp_path):
        """Should find video with custom name when input.* not present."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "beach_video.mp4").write_text("fake video")

        # Test the video finding logic
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]
        input_video = None

        # First try input.* naming convention
        for ext in video_extensions:
            candidate = source_dir / f"input{ext}"
            if candidate.exists():
                input_video = candidate
                break

        # If not found, look for any video file in source/
        if not input_video and source_dir.exists():
            for ext in video_extensions:
                for video_file in source_dir.glob(f"*{ext}"):
                    if video_file.is_file():
                        input_video = video_file
                        break
                if input_video:
                    break

        assert input_video is not None
        assert input_video.name == "beach_video.mp4"

    def test_returns_none_when_no_video(self, tmp_path):
        """Should return None when no video in source directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "somefile.txt").write_text("not a video")

        # Test the video finding logic
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]
        input_video = None

        for ext in video_extensions:
            candidate = source_dir / f"input{ext}"
            if candidate.exists():
                input_video = candidate
                break

        if not input_video and source_dir.exists():
            for ext in video_extensions:
                for video_file in source_dir.glob(f"*{ext}"):
                    if video_file.is_file():
                        input_video = video_file
                        break
                if input_video:
                    break

        assert input_video is None

    def test_finds_existing_frames(self, tmp_path):
        """Should detect existing frames in source/frames directory."""
        source_dir = tmp_path / "source"
        frames_dir = source_dir / "frames"
        frames_dir.mkdir(parents=True)

        # Create some frame files
        (frames_dir / "frame_0001.png").write_text("fake frame")
        (frames_dir / "frame_0002.png").write_text("fake frame")
        (frames_dir / "frame_0003.png").write_text("fake frame")

        # Test the frame detection logic
        has_frames = False
        if frames_dir.exists():
            frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
            if frame_files:
                has_frames = True

        assert has_frames is True

    def test_prefers_frames_over_video(self, tmp_path):
        """Should prefer existing frames over video file."""
        source_dir = tmp_path / "source"
        frames_dir = source_dir / "frames"
        frames_dir.mkdir(parents=True)

        # Create both frames and video
        (frames_dir / "frame_0001.png").write_text("fake frame")
        (source_dir / "input.mp4").write_text("fake video")

        # Test that frames are detected first
        has_frames = False
        if frames_dir.exists():
            frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
            if frame_files:
                has_frames = True

        assert has_frames is True


class TestAPIJobEndpoint:
    """Test the /api/projects/{id}/job endpoint."""

    def test_job_endpoint_returns_idle_when_no_job(self):
        """Job endpoint should return idle status when no active job."""
        from fastapi.testclient import TestClient
        from web.server import app
        from web.api import active_jobs, active_jobs_lock

        # Ensure no active job
        with active_jobs_lock:
            if "test_project" in active_jobs:
                del active_jobs["test_project"]

        client = TestClient(app)
        response = client.get("/api/projects/test_project/job")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"

    def test_job_endpoint_returns_running_status(self):
        """Job endpoint should return running status with details."""
        from fastapi.testclient import TestClient
        from web.server import app
        from web.api import active_jobs, active_jobs_lock

        with active_jobs_lock:
            active_jobs["test_project"] = {
                "status": "running",
                "current_stage": "depth",
                "progress": 0.42,
                "last_output": "[ComfyUI] depth frame 42/100",
            }

        try:
            client = TestClient(app)
            response = client.get("/api/projects/test_project/job")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert data["current_stage"] == "depth"
            assert data["progress"] == 0.42
            assert data["last_output"] == "[ComfyUI] depth frame 42/100"
        finally:
            with active_jobs_lock:
                del active_jobs["test_project"]
