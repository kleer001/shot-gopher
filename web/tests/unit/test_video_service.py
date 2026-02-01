"""Tests for video analysis service."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from web.services.video_service import VideoService, get_video_service


class TestVideoService:
    """Tests for VideoService class."""

    def test_get_video_service_returns_singleton(self):
        """get_video_service should return same instance."""
        service1 = get_video_service()
        service2 = get_video_service()
        assert service1 is service2

    def test_get_video_info_returns_empty_when_no_ffprobe(self):
        """Returns empty dict when ffprobe not found."""
        service = VideoService()
        with patch("web.services.video_service.PlatformManager.find_tool", return_value=None):
            result = service.get_video_info(Path("/fake/video.mp4"))
        assert result == {}

    def test_get_video_info_parses_ffprobe_output(self):
        """Parses ffprobe JSON output correctly."""
        service = VideoService()
        mock_output = '{"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "24/1"}], "format": {"duration": "10.0"}}'

        with patch("web.services.video_service.PlatformManager.find_tool", return_value="/usr/bin/ffprobe"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
                result = service.get_video_info(Path("/fake/video.mp4"))

        assert result["duration"] == 10.0
        assert result["fps"] == 24.0
        assert result["resolution"] == [1920, 1080]
        assert result["frame_count"] == 240

    def test_get_video_info_handles_fractional_fps(self):
        """Parses fractional frame rate like 30000/1001."""
        service = VideoService()
        mock_output = '{"streams": [{"codec_type": "video", "width": 1280, "height": 720, "r_frame_rate": "30000/1001"}], "format": {"duration": "5.0"}}'

        with patch("web.services.video_service.PlatformManager.find_tool", return_value="/usr/bin/ffprobe"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
                result = service.get_video_info(Path("/fake/video.mp4"))

        assert result["fps"] == pytest.approx(29.97, rel=0.01)

    def test_get_resolution_from_frames_returns_none_for_empty_dir(self, tmp_path):
        """Returns None when frames directory is empty."""
        service = VideoService()
        result = service.get_resolution_from_frames(tmp_path)
        assert result is None

    def test_get_resolution_from_frames_returns_none_for_missing_dir(self, tmp_path):
        """Returns None when directory doesn't exist."""
        service = VideoService()
        result = service.get_resolution_from_frames(tmp_path / "nonexistent")
        assert result is None

    def test_get_resolution_from_frames_reads_png(self, tmp_path):
        """Reads resolution from PNG frames."""
        PIL = pytest.importorskip("PIL.Image")
        service = VideoService()

        frame = tmp_path / "frame_0001.png"
        img = PIL.new("RGB", (1920, 1080))
        img.save(frame)

        result = service.get_resolution_from_frames(tmp_path)
        assert result == (1920, 1080)

    def test_get_resolution_from_frames_reads_jpg(self, tmp_path):
        """Reads resolution from JPG frames when no PNG."""
        PIL = pytest.importorskip("PIL.Image")
        service = VideoService()

        frame = tmp_path / "frame_0001.jpg"
        img = PIL.new("RGB", (1280, 720))
        img.save(frame)

        result = service.get_resolution_from_frames(tmp_path)
        assert result == (1280, 720)

    def test_get_video_info_handles_missing_streams(self):
        """Returns empty dict when no video stream in file."""
        service = VideoService()
        mock_output = '{"streams": [{"codec_type": "audio"}], "format": {"duration": "10.0"}}'

        with patch("web.services.video_service.PlatformManager.find_tool", return_value="/usr/bin/ffprobe"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
                result = service.get_video_info(Path("/fake/video.mp4"))

        assert result == {}

    def test_get_video_info_handles_subprocess_timeout(self):
        """Returns empty dict on subprocess timeout."""
        import subprocess
        service = VideoService()

        with patch("web.services.video_service.PlatformManager.find_tool", return_value="/usr/bin/ffprobe"):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
                result = service.get_video_info(Path("/fake/video.mp4"))

        assert result == {}

    def test_get_video_info_handles_zero_duration(self):
        """Handles video with zero duration."""
        service = VideoService()
        mock_output = '{"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "24/1"}], "format": {"duration": "0"}}'

        with patch("web.services.video_service.PlatformManager.find_tool", return_value="/usr/bin/ffprobe"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
                result = service.get_video_info(Path("/fake/video.mp4"))

        assert result["frame_count"] == 0
        assert result["duration"] == 0
