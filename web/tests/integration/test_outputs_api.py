"""Integration tests for project outputs API endpoint."""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def test_project_with_outputs(tmp_path, monkeypatch):
    """Create a test project with sample output files."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    # Create a project
    project_dir = projects_dir / "test_project"
    project_dir.mkdir()

    # Create output directories with files
    # source/frames (for ingest)
    (project_dir / "source" / "frames").mkdir(parents=True)
    for i in range(5):
        (project_dir / "source" / "frames" / f"frame_{i:04d}.png").touch()

    # depth
    (project_dir / "depth").mkdir()
    for i in range(5):
        (project_dir / "depth" / f"depth_{i:04d}.exr").touch()

    # roto
    (project_dir / "roto").mkdir()
    for i in range(5):
        (project_dir / "roto" / f"mask_{i:04d}.png").touch()

    # colmap with nested structure
    (project_dir / "colmap" / "sparse").mkdir(parents=True)
    (project_dir / "colmap" / "sparse" / "cameras.bin").touch()
    (project_dir / "colmap" / "sparse" / "images.bin").touch()
    (project_dir / "colmap" / "sparse" / "points3D.bin").touch()

    # Create project state file
    import json
    from datetime import datetime
    state = {
        "name": "test_project",
        "status": "created",
        "video_path": None,
        "stages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    (project_dir / "project_state.json").write_text(json.dumps(state))

    monkeypatch.setenv("VFX_PROJECTS_DIR", str(projects_dir))

    return {
        "projects_dir": projects_dir,
        "project_dir": project_dir,
        "project_name": "test_project",
    }


class TestOutputsAPIEndpoint:
    """Test the /api/projects/{id}/outputs endpoint."""

    def test_outputs_returns_normalized_keys(self, test_project_with_outputs, monkeypatch):
        """API should return normalized keys like 'source' not 'source/frames'."""
        # Import after monkeypatching env vars
        from web.server import app

        client = TestClient(app)
        response = client.get(f"/api/projects/{test_project_with_outputs['project_name']}/outputs")

        assert response.status_code == 200
        data = response.json()

        # Check that outputs dict exists
        assert "outputs" in data

        outputs = data["outputs"]

        # Check normalized keys are present
        assert "source" in outputs, "Should have 'source' key (from source/frames)"
        assert "depth" in outputs, "Should have 'depth' key"
        assert "roto" in outputs, "Should have 'roto' key"
        assert "colmap" in outputs, "Should have 'colmap' key"

        # Check that 'source/frames' is NOT a key
        assert "source/frames" not in outputs, "Should NOT have 'source/frames' key"

    def test_outputs_counts_files_recursively(self, test_project_with_outputs, monkeypatch):
        """API should count files in subdirectories."""
        from web.server import app

        client = TestClient(app)
        response = client.get(f"/api/projects/{test_project_with_outputs['project_name']}/outputs")

        assert response.status_code == 200
        data = response.json()
        outputs = data["outputs"]

        # source should have 5 files (from source/frames)
        assert outputs["source"]["count"] == 5

        # colmap should have 3 files (from colmap/sparse/)
        assert outputs["colmap"]["count"] == 3

    def test_outputs_empty_project(self, tmp_path, monkeypatch):
        """API should handle project with no outputs gracefully."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        project_dir = projects_dir / "empty_project"
        project_dir.mkdir()

        # Create minimal project state
        import json
        from datetime import datetime
        state = {
            "name": "empty_project",
            "status": "created",
            "video_path": None,
            "stages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        (project_dir / "project_state.json").write_text(json.dumps(state))

        monkeypatch.setenv("VFX_PROJECTS_DIR", str(projects_dir))

        from web.server import app
        client = TestClient(app)
        response = client.get("/api/projects/empty_project/outputs")

        assert response.status_code == 200
        data = response.json()
        assert data["outputs"] == {}


class TestProjectsListAPI:
    """Test the /api/projects endpoint."""

    def test_projects_list_returns_name_field(self, test_project_with_outputs, monkeypatch):
        """API should return projects with 'name' field."""
        from web.server import app

        client = TestClient(app)
        response = client.get("/api/projects")

        assert response.status_code == 200
        data = response.json()

        assert "projects" in data
        assert len(data["projects"]) > 0

        project = data["projects"][0]
        assert "name" in project, "Project should have 'name' field"
        assert project["name"] == "test_project"

        # project_id should NOT be in the response
        assert "project_id" not in project, "Project should NOT have 'project_id' field"
