"""Tests for system status service."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from web.services.system_service import SystemService, get_system_service


class TestSystemService:
    """Tests for SystemService class."""

    def test_get_system_service_returns_singleton(self):
        """get_system_service should return same instance."""
        service1 = get_system_service()
        service2 = get_system_service()
        assert service1 is service2

    def test_check_comfyui_status_returns_false_on_connection_error(self):
        """Returns False when ComfyUI is not reachable."""
        service = SystemService()
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = service.check_comfyui_status()
        assert result is False

    def test_check_comfyui_status_returns_true_on_success(self):
        """Returns True when ComfyUI responds with 200."""
        service = SystemService()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = service.check_comfyui_status()
        assert result is True

    def test_get_gpu_info_returns_unknown_on_failure(self):
        """Returns unknown when nvidia-smi fails."""
        service = SystemService()
        with patch("subprocess.run", side_effect=Exception("Command not found")):
            result = service.get_gpu_info()
        assert result == {"name": "Unknown", "vram_gb": 0}

    def test_get_gpu_info_parses_nvidia_smi_output(self):
        """Parses nvidia-smi output correctly."""
        service = SystemService()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="NVIDIA GeForce RTX 4090, 24576"
            )
            result = service.get_gpu_info()

        assert result["name"] == "NVIDIA GeForce RTX 4090"
        assert result["vram_gb"] == 24.0

    def test_get_disk_usage_returns_stats(self, tmp_path):
        """Returns disk usage statistics for existing path."""
        service = SystemService()
        result = service.get_disk_usage(tmp_path)

        assert result is not None
        assert "free_gb" in result
        assert "total_gb" in result
        assert "used_percent" in result
        assert result["total_gb"] > 0
        assert 0 <= result["used_percent"] <= 100

    def test_get_disk_usage_returns_none_for_invalid_path(self, tmp_path):
        """Returns None for non-existent path with non-existent parent."""
        service = SystemService()
        fake_path = Path("/nonexistent/path/that/does/not/exist")
        result = service.get_disk_usage(fake_path)
        assert result is None

    def test_get_gpu_info_handles_malformed_output(self):
        """Returns unknown for malformed nvidia-smi output."""
        service = SystemService()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="malformed output without comma"
            )
            result = service.get_gpu_info()

        assert result == {"name": "Unknown", "vram_gb": 0}

    def test_get_gpu_info_handles_non_numeric_vram(self):
        """Returns unknown when VRAM is not a number."""
        service = SystemService()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="NVIDIA GPU, not_a_number"
            )
            result = service.get_gpu_info()

        assert result == {"name": "Unknown", "vram_gb": 0}

    def test_get_disk_usage_uses_parent_if_path_missing(self, tmp_path):
        """Falls back to parent directory if path doesn't exist."""
        service = SystemService()
        missing_subdir = tmp_path / "missing_subdir"
        result = service.get_disk_usage(missing_subdir)

        assert result is not None
        assert result["total_gb"] > 0
