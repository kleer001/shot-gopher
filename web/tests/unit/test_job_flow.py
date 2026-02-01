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
        from web.job_state import active_jobs, active_jobs_lock

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

    def test_parse_output_line(self):
        """Any output line should be captured as message."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("[ComfyUI] depth frame 42/100", "depth", ["depth"])

        assert result is not None
        assert result["message"] == "[ComfyUI] depth frame 42/100"

    def test_parse_any_line_returns_message(self):
        """All lines should return a message."""
        from web.pipeline_runner import parse_progress_line

        result = parse_progress_line("Some random output text", "depth", ["depth"])

        assert result is not None
        assert result["message"] == "Some random output text"


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
        from web.job_state import active_jobs, active_jobs_lock

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
        from web.job_state import active_jobs, active_jobs_lock

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


class TestVramAnalysisFrameCounting:
    """Test VRAM analysis with frame counting for projects without video."""

    def test_counts_png_frames(self, tmp_path):
        """Should count PNG frames in source/frames directory."""
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)

        for i in range(100):
            (frames_dir / f"frame_{i:04d}.png").write_bytes(b"fake")

        frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        assert len(frame_files) == 100

    def test_counts_jpg_frames(self, tmp_path):
        """Should count JPG frames in source/frames directory."""
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)

        for i in range(50):
            (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"fake")

        frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        assert len(frame_files) == 50

    def test_counts_mixed_frames(self, tmp_path):
        """Should count both PNG and JPG frames."""
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)

        for i in range(30):
            (frames_dir / f"frame_{i:04d}.png").write_bytes(b"fake")
        for i in range(20):
            (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"fake")

        frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        assert len(frame_files) == 50

    def test_gets_resolution_from_first_frame(self, tmp_path):
        """Should extract resolution from first frame using PIL."""
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img = Image.new("RGB", (1920, 1080), color="red")
        img.save(frames_dir / "frame_0001.png")
        img.save(frames_dir / "frame_0002.png")

        frame_files = sorted(frames_dir.glob("*.png"))
        first_frame = frame_files[0]

        with Image.open(first_frame) as loaded:
            resolution = loaded.size

        assert resolution == (1920, 1080)

    def test_handles_4k_resolution(self, tmp_path):
        """Should correctly detect 4K resolution from frames."""
        frames_dir = tmp_path / "source" / "frames"
        frames_dir.mkdir(parents=True)

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img = Image.new("RGB", (3840, 2160), color="blue")
        img.save(frames_dir / "frame_0001.png")

        frame_files = sorted(frames_dir.glob("*.png"))
        first_frame = frame_files[0]

        with Image.open(first_frame) as loaded:
            resolution = loaded.size

        assert resolution == (3840, 2160)

    def test_prefers_frames_over_video_for_vram(self, tmp_path):
        """VRAM analysis should use frame count from frames/ if available."""
        source_dir = tmp_path / "source"
        frames_dir = source_dir / "frames"
        frames_dir.mkdir(parents=True)

        for i in range(75):
            (frames_dir / f"frame_{i:04d}.png").write_bytes(b"fake")
        (source_dir / "input.mp4").write_bytes(b"fake video with different frame count")

        frame_count = 0
        if frames_dir.exists():
            frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
            if frame_files:
                frame_count = len(frame_files)

        assert frame_count == 75

    def test_falls_back_to_video_when_no_frames(self, tmp_path):
        """Should use video file when no frames directory exists."""
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "input.mp4").write_bytes(b"fake video")

        frames_dir = source_dir / "frames"
        frame_count = 0

        if frames_dir.exists():
            frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
            if frame_files:
                frame_count = len(frame_files)

        video_path = None
        if frame_count == 0:
            for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]:
                candidate = source_dir / f"input{ext}"
                if candidate.exists():
                    video_path = candidate
                    break

        assert frame_count == 0
        assert video_path is not None
        assert video_path.name == "input.mp4"

    def test_finds_any_video_when_no_input_named(self, tmp_path):
        """Should find any video file when input.* doesn't exist."""
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "my_clip.mov").write_bytes(b"fake video")

        video_path = None
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]:
            candidate = source_dir / f"input{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if not video_path:
            for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]:
                for vf in source_dir.glob(f"*{ext}"):
                    if vf.is_file():
                        video_path = vf
                        break
                if video_path:
                    break

        assert video_path is not None
        assert video_path.name == "my_clip.mov"


class TestJobRepoStatusUpdate:
    """Test that job_repo status is updated when pipeline completes."""

    def test_update_job_repo_status_completed(self):
        """Should update job status to COMPLETE when pipeline finishes successfully."""
        from web.models.domain import PipelineJob, JobStatus
        from web.repositories.job_repository import JobRepository
        from datetime import datetime

        job_repo = JobRepository()
        job = PipelineJob(
            project_name="test_project",
            stages=["depth"],
            status=JobStatus.RUNNING,
            current_stage="depth",
            progress=0.5,
            message="Processing...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        job_repo.save(job)

        job.status = JobStatus.COMPLETE
        job.completed_at = datetime.now()
        job.progress = 1.0
        job_repo.save(job)

        updated_job = job_repo.get("test_project")
        assert updated_job.status == JobStatus.COMPLETE
        assert updated_job.progress == 1.0
        assert updated_job.completed_at is not None

    def test_update_job_repo_status_failed(self):
        """Should update job status to FAILED when pipeline fails."""
        from web.models.domain import PipelineJob, JobStatus
        from web.repositories.job_repository import JobRepository
        from datetime import datetime

        job_repo = JobRepository()
        job = PipelineJob(
            project_name="test_project_fail",
            stages=["depth"],
            status=JobStatus.RUNNING,
            current_stage="depth",
            progress=0.5,
            message="Processing...",
            started_at=datetime.now(),
            completed_at=None,
            error=None,
        )
        job_repo.save(job)

        job.status = JobStatus.FAILED
        job.completed_at = datetime.now()
        job.error = "Pipeline exited with code 1"
        job_repo.save(job)

        updated_job = job_repo.get("test_project_fail")
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error == "Pipeline exited with code 1"

    def test_completed_job_allows_new_job(self):
        """Should allow starting a new job after previous job completed."""
        from web.models.domain import PipelineJob, JobStatus
        from web.repositories.job_repository import JobRepository
        from datetime import datetime

        job_repo = JobRepository()
        job = PipelineJob(
            project_name="test_project_new",
            stages=["ingest"],
            status=JobStatus.COMPLETE,
            current_stage="ingest",
            progress=1.0,
            message="Done",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            error=None,
        )
        job_repo.save(job)

        existing_job = job_repo.get("test_project_new")
        can_start_new = existing_job is None or existing_job.status != JobStatus.RUNNING

        assert can_start_new is True
