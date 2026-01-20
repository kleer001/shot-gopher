"""Pytest configuration and fixtures."""

import sys
import os
from pathlib import Path

# Add paths for imports - do this before other imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

import pytest
import tempfile
import shutil
from unittest.mock import MagicMock


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment before any tests run."""
    # Create mock ComfyUI directory
    temp_dir = Path(tempfile.mkdtemp())
    comfyui_dir = temp_dir / ".vfx_pipeline" / "ComfyUI"
    comfyui_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal ComfyUI structure
    (comfyui_dir / "models").mkdir(exist_ok=True)
    (comfyui_dir / "custom_nodes").mkdir(exist_ok=True)
    (comfyui_dir / "output").mkdir(exist_ok=True)
    (comfyui_dir / "main.py").write_text("# Mock ComfyUI main.py\n")

    # Set environment variables
    os.environ["INSTALL_DIR"] = str(temp_dir)
    original_projects_dir = os.environ.get("VFX_PROJECTS_DIR")
    original_models_dir = os.environ.get("VFX_MODELS_DIR")

    yield {
        "comfyui_dir": comfyui_dir,
        "temp_dir": temp_dir,
    }

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Restore original environment variables
    if original_projects_dir:
        os.environ["VFX_PROJECTS_DIR"] = original_projects_dir
    if original_models_dir:
        os.environ["VFX_MODELS_DIR"] = original_models_dir


@pytest.fixture(autouse=True)
def mock_comfyui_http(monkeypatch):
    """Mock ComfyUI HTTP endpoints for testing."""
    def mock_urlopen(req, timeout=None):
        """Mock urllib.request.urlopen for ComfyUI endpoints."""
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'{"status": "ok"}'

        # Create a context manager for 'with' statement
        class ResponseContext:
            def __enter__(self):
                return response
            def __exit__(self, *args):
                pass

        return ResponseContext()

    # Only mock if ComfyUI is not actually running
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8188/system_stats", method="GET")
        urllib.request.urlopen(req, timeout=1)
        # ComfyUI is running, don't mock
    except Exception:
        # ComfyUI not running, apply mock
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)


@pytest.fixture
def mock_env_vars(monkeypatch, tmp_path):
    """Set up mock environment variables for testing."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(exist_ok=True)

    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)

    monkeypatch.setenv("VFX_PROJECTS_DIR", str(projects_dir))
    monkeypatch.setenv("VFX_MODELS_DIR", str(models_dir))

    return {
        "projects_dir": projects_dir,
        "models_dir": models_dir,
    }
