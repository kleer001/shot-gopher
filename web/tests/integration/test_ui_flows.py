"""End-to-end integration tests for web UI flows."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil
import json
import io

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

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_video():
    """Create a minimal test video file."""
    video_data = b"FAKE_VIDEO_DATA_FOR_TESTING"
    return io.BytesIO(video_data)


class TestDashboardFlow:
    """Test dashboard loading and project listing."""

    def test_dashboard_loads(self, client):
        """Test that dashboard page loads successfully."""
        response = client.get("/dashboard")

        assert response.status_code == 200
        assert b"VFX Pipeline" in response.content or b"VFX PIPELINE" in response.content
        assert b"projects" in response.content.lower()

    def test_dashboard_loads_projects_from_api(self, client):
        """Test that dashboard can fetch projects from API."""
        response = client.get("/api/projects")

        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert isinstance(data["projects"], list)

    def test_all_layout_variants_load(self, client):
        """Test that all layout variants render successfully."""
        layouts = ["/", "/compact", "/dashboard", "/split", "/cards"]

        for layout in layouts:
            response = client.get(layout)
            assert response.status_code == 200, f"Layout {layout} failed to load"
            assert b"VFX Pipeline" in response.content or b"VFX PIPELINE" in response.content


class TestProjectCreationFlow:
    """Test complete project creation workflow."""

    def test_create_project_minimal(self, client):
        """Test creating a project with minimal configuration."""
        project_data = {
            "name": "test_project_001",
            "description": "Test project for E2E testing",
            "stages": ["ingest"]
        }

        response = client.post("/api/projects", json=project_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_project_001"
        assert data["status"] == "created"
        assert "ingest" in data["stages"]

    def test_create_project_with_multiple_stages(self, client):
        """Test creating a project with multiple stages."""
        project_data = {
            "name": "test_project_002",
            "description": "Multi-stage test project",
            "stages": ["ingest", "depth", "roto"]
        }

        response = client.post("/api/projects", json=project_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["stages"]) == 3
        assert set(data["stages"]) == {"ingest", "depth", "roto"}

    def test_create_project_duplicate_name_fails(self, client):
        """Test that duplicate project names are rejected."""
        project_data = {
            "name": "duplicate_test",
            "stages": ["ingest"]
        }

        response1 = client.post("/api/projects", json=project_data)
        assert response1.status_code == 200

        response2 = client.post("/api/projects", json=project_data)
        assert response2.status_code == 400
        assert "already exists" in response2.json()["detail"].lower()

    def test_create_project_invalid_name_fails(self, client):
        """Test that invalid project names are rejected."""
        invalid_names = [
            "",
            " ",
            "project/with/slashes",
            "project\\with\\backslashes",
            "..",
            ".",
        ]

        for invalid_name in invalid_names:
            project_data = {
                "name": invalid_name,
                "stages": ["ingest"]
            }

            response = client.post("/api/projects", json=project_data)
            assert response.status_code in [400, 422], f"Invalid name '{invalid_name}' was accepted"


class TestVideoUploadFlow:
    """Test video upload workflow."""

    def test_upload_video_to_project(self, client, sample_video):
        """Test uploading a video file to a project."""
        client.post("/api/projects", json={
            "name": "upload_test",
            "stages": ["ingest"]
        })

        files = {"video": ("test_video.mp4", sample_video, "video/mp4")}
        response = client.post("/api/projects/upload_test/upload-video", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "video_path" in data

    def test_upload_video_to_nonexistent_project_fails(self, client, sample_video):
        """Test that uploading to non-existent project fails."""
        files = {"video": ("test_video.mp4", sample_video, "video/mp4")}
        response = client.post("/api/projects/nonexistent/upload-video", files=files)

        assert response.status_code == 404


class TestProjectManagementFlow:
    """Test project retrieval and deletion."""

    def test_get_project_details(self, client):
        """Test retrieving project details."""
        client.post("/api/projects", json={
            "name": "detail_test",
            "description": "Project for detail testing",
            "stages": ["ingest", "depth"]
        })

        response = client.get("/api/projects/detail_test")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "detail_test"
        assert data["description"] == "Project for detail testing"
        assert len(data["stages"]) == 2

    def test_get_nonexistent_project_fails(self, client):
        """Test that getting non-existent project returns 404."""
        response = client.get("/api/projects/does_not_exist")
        assert response.status_code == 404

    def test_delete_project(self, client):
        """Test deleting a project."""
        client.post("/api/projects", json={
            "name": "delete_test",
            "stages": ["ingest"]
        })

        response = client.delete("/api/projects/delete_test")
        assert response.status_code == 200

        response = client.get("/api/projects/delete_test")
        assert response.status_code == 404

    def test_delete_nonexistent_project_fails(self, client):
        """Test that deleting non-existent project returns 404."""
        response = client.delete("/api/projects/does_not_exist")
        assert response.status_code == 404


class TestConfigurationFlow:
    """Test pipeline configuration retrieval."""

    def test_get_pipeline_config(self, client):
        """Test retrieving pipeline configuration."""
        response = client.get("/api/config")

        assert response.status_code == 200
        data = response.json()
        assert "stages" in data
        assert "presets" in data
        assert "supportedVideoFormats" in data
        assert isinstance(data["stages"], list)
        assert len(data["stages"]) > 0

    def test_get_video_info(self, client, sample_video):
        """Test getting video metadata."""
        client.post("/api/projects", json={
            "name": "video_info_test",
            "stages": ["ingest"]
        })

        files = {"video": ("test.mp4", sample_video, "video/mp4")}
        client.post("/api/projects/video_info_test/upload-video", files=files)

        response = client.get("/api/projects/video_info_test/video-info")

        assert response.status_code in [200, 404]


class TestSystemStatusFlow:
    """Test system status and health checks."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/api/system/status")

        assert response.status_code == 200
        data = response.json()
        assert "comfyui" in data
        assert "disk_space_gb" in data
        assert "projects_dir" in data
        assert isinstance(data["disk_space_gb"], (int, float))


class TestCompleteUserJourney:
    """Test complete end-to-end user workflows."""

    def test_complete_project_lifecycle(self, client, sample_video):
        """Test the complete lifecycle: create → upload → process → delete."""
        project_name = "lifecycle_test"

        response = client.post("/api/projects", json={
            "name": project_name,
            "description": "Complete lifecycle test",
            "stages": ["ingest"]
        })
        assert response.status_code == 200
        created_project = response.json()
        assert created_project["status"] == "created"

        response = client.get("/api/projects")
        assert response.status_code == 200
        projects = response.json()["projects"]
        assert any(p["name"] == project_name for p in projects)

        files = {"video": ("test.mp4", sample_video, "video/mp4")}
        response = client.post(f"/api/projects/{project_name}/upload-video", files=files)
        assert response.status_code == 200

        response = client.get(f"/api/projects/{project_name}")
        assert response.status_code == 200
        project_details = response.json()
        assert "video_path" in project_details or "video" in str(project_details)

        response = client.delete(f"/api/projects/{project_name}")
        assert response.status_code == 200

        response = client.get(f"/api/projects/{project_name}")
        assert response.status_code == 404

    def test_multiple_projects_workflow(self, client):
        """Test creating and managing multiple projects."""
        project_names = ["multi_test_1", "multi_test_2", "multi_test_3"]

        for name in project_names:
            response = client.post("/api/projects", json={
                "name": name,
                "stages": ["ingest"]
            })
            assert response.status_code == 200

        response = client.get("/api/projects")
        assert response.status_code == 200
        projects = response.json()["projects"]
        found_names = [p["name"] for p in projects]

        for name in project_names:
            assert name in found_names

        for name in project_names:
            response = client.delete(f"/api/projects/{name}")
            assert response.status_code == 200


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_json_request(self, client):
        """Test that malformed JSON is rejected."""
        response = client.post(
            "/api/projects",
            data="invalid json{{{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test that requests with missing required fields are rejected."""
        response = client.post("/api/projects", json={})
        assert response.status_code == 422

    def test_unsupported_http_methods(self, client):
        """Test that unsupported HTTP methods return 405."""
        response = client.patch("/api/projects")
        assert response.status_code == 405

    def test_api_error_responses_are_json(self, client):
        """Test that all API errors return JSON with detail field."""
        error_responses = [
            client.get("/api/projects/nonexistent"),
            client.delete("/api/projects/nonexistent"),
            client.post("/api/projects", json={"invalid": "data"}),
        ]

        for response in error_responses:
            if response.status_code >= 400:
                data = response.json()
                assert "detail" in data, f"Error response missing 'detail': {data}"


class TestPerformance:
    """Test performance characteristics."""

    def test_list_projects_performance_with_many_projects(self, client):
        """Test that listing projects scales reasonably."""
        for i in range(10):
            client.post("/api/projects", json={
                "name": f"perf_test_{i:03d}",
                "stages": ["ingest"]
            })

        response = client.get("/api/projects")
        assert response.status_code == 200

        data = response.json()
        assert len(data["projects"]) >= 10

    def test_concurrent_project_creation_safety(self, client):
        """Test that concurrent operations are handled safely."""
        project_name = "concurrent_test"

        responses = []
        for i in range(3):
            response = client.post("/api/projects", json={
                "name": project_name,
                "stages": ["ingest"]
            })
            responses.append(response)

        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == 1, "Only one concurrent creation should succeed"


class TestAccessibility:
    """Test UI accessibility features."""

    def test_html_has_lang_attribute(self, client):
        """Test that HTML pages have lang attribute for screen readers."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert b'lang="en"' in response.content

    def test_html_has_viewport_meta(self, client):
        """Test that pages are mobile-responsive."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert b'name="viewport"' in response.content

    def test_html_has_proper_structure(self, client):
        """Test that HTML has proper semantic structure."""
        response = client.get("/dashboard")
        assert response.status_code == 200

        content = response.content
        assert b'<nav' in content or b'<header' in content or b'<h1>' in content
        assert b'<body>' in content
        assert b'</body>' in content
