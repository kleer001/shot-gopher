"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil
import json

from web.server import app


@pytest.fixture
def temp_projects_dir():
    """Create temporary projects directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client(temp_projects_dir, monkeypatch):
    """Create test client with temporary projects directory."""
    monkeypatch.setenv("VFX_PROJECTS_DIR", str(temp_projects_dir))
    return TestClient(app)


class TestConfigAPI:
    """Test configuration API endpoints."""

    def test_get_config(self, client):
        """Test getting pipeline configuration."""
        response = client.get("/api/config")

        assert response.status_code == 200
        data = response.json()
        assert "stages" in data
        assert "presets" in data
        assert "supportedVideoFormats" in data


class TestSystemAPI:
    """Test system status API endpoints."""

    def test_system_status(self, client):
        """Test getting system status."""
        response = client.get("/api/system/status")

        assert response.status_code == 200
        data = response.json()
        assert "comfyui" in data
        assert "disk_space_gb" in data
        assert "projects_dir" in data
