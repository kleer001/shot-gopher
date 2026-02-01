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

    import importlib
    import sys
    if "env_config" in sys.modules:
        importlib.reload(sys.modules["env_config"])
    if "web.api" in sys.modules:
        importlib.reload(sys.modules["web.api"])

    import web.api as api_module
    api_module._project_repo = None
    api_module._job_repo = None
    api_module._project_service = None
    api_module._pipeline_service = None

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


class TestProjectAPI:
    """Test project CRUD API endpoints."""

    def test_create_project(self, client):
        """Test creating a new project via POST."""
        response = client.post(
            "/api/projects",
            json={"name": "test_project", "stages": ["ingest"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_project"
        assert data["status"] == "created"
        assert data["stages"] == ["ingest"]

    def test_create_project_duplicate_name(self, client):
        """Test creating project with duplicate name returns 400."""
        client.post("/api/projects", json={"name": "duplicate", "stages": []})

        response = client.post("/api/projects", json={"name": "duplicate", "stages": []})

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_project_invalid_name(self, client):
        """Test creating project with invalid name returns 422."""
        response = client.post(
            "/api/projects",
            json={"name": "invalid name with spaces", "stages": []}
        )

        assert response.status_code == 422

    def test_create_project_invalid_stages(self, client):
        """Test creating project with invalid stages returns 400."""
        response = client.post(
            "/api/projects",
            json={"name": "test_invalid", "stages": ["nonexistent_stage"]}
        )

        assert response.status_code == 400
        assert "Invalid stages" in response.json()["detail"]

    def test_get_projects(self, client):
        """Test listing projects."""
        client.post("/api/projects", json={"name": "project1", "stages": []})
        client.post("/api/projects", json={"name": "project2", "stages": []})

        response = client.get("/api/projects")

        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert len(data["projects"]) >= 2
        project_names = [p["name"] for p in data["projects"]]
        assert "project1" in project_names
        assert "project2" in project_names

    def test_get_project(self, client):
        """Test getting single project details."""
        client.post("/api/projects", json={"name": "test_get", "stages": []})

        response = client.get("/api/projects/test_get")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_get"
        assert "status" in data
        assert "created_at" in data

    def test_get_project_not_found(self, client):
        """Test getting non-existent project returns 404."""
        response = client.get("/api/projects/nonexistent")

        assert response.status_code == 404

    def test_delete_project(self, client):
        """Test deleting a project."""
        client.post("/api/projects", json={"name": "to_delete", "stages": []})

        response = client.delete("/api/projects/to_delete")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["project_id"] == "to_delete"

        get_response = client.get("/api/projects/to_delete")
        assert get_response.status_code == 404

    def test_delete_project_not_found(self, client):
        """Test deleting non-existent project returns 404."""
        response = client.delete("/api/projects/nonexistent")

        assert response.status_code == 404

    def test_interactive_complete_creates_signal_file(self, client):
        """Test that interactive-complete creates the signal file."""
        client.post("/api/projects", json={"name": "interactive_test", "stages": []})

        response = client.post("/api/projects/interactive_test/interactive-complete")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "signaled"
        assert data["project_id"] == "interactive_test"

    def test_interactive_complete_not_found(self, client):
        """Test interactive-complete returns 404 for non-existent project."""
        response = client.post("/api/projects/nonexistent/interactive-complete")

        assert response.status_code == 404

    def test_interactive_complete_signal_file_exists(self, client, temp_projects_dir):
        """Test that signal file is actually created on disk."""
        client.post("/api/projects", json={"name": "signal_test", "stages": []})

        client.post("/api/projects/signal_test/interactive-complete")

        signal_file = temp_projects_dir / "signal_test" / ".interactive_done"
        assert signal_file.exists()


class TestSystemAPI:
    """Test system status API endpoints."""

    def test_system_status(self, client):
        """Test getting system status."""
        response = client.get("/api/system/status")

        assert response.status_code == 200
        data = response.json()
        assert "comfyui" in data
        assert "disk_free_gb" in data
        assert "projects_dir" in data
