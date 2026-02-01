"""Tests for interactive signal file mechanism."""

import pytest
import tempfile
import shutil
import threading
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from stage_runners import (
    wait_for_interactive_signal,
    INTERACTIVE_SIGNAL_FILE,
)


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestInteractiveSignal:
    """Test interactive signal file mechanism."""

    def test_signal_file_detection(self, temp_project_dir):
        """Test that wait_for_interactive_signal returns when signal file is created."""
        result = {"completed": False, "error": None}

        def wait_thread():
            try:
                wait_for_interactive_signal(temp_project_dir, poll_interval=0.1)
                result["completed"] = True
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=wait_thread)
        thread.start()

        time.sleep(0.2)
        signal_file = temp_project_dir / INTERACTIVE_SIGNAL_FILE
        signal_file.touch()

        thread.join(timeout=2.0)

        assert result["completed"] is True
        assert result["error"] is None
        assert not signal_file.exists()

    def test_cleans_up_stale_signal_file(self, temp_project_dir):
        """Test that stale signal file is removed on start."""
        signal_file = temp_project_dir / INTERACTIVE_SIGNAL_FILE
        signal_file.touch()
        assert signal_file.exists()

        result = {"completed": False}

        def wait_thread():
            wait_for_interactive_signal(temp_project_dir, poll_interval=0.1)
            result["completed"] = True

        thread = threading.Thread(target=wait_thread)
        thread.start()

        time.sleep(0.2)
        assert not signal_file.exists()

        signal_file.touch()

        thread.join(timeout=2.0)
        assert result["completed"] is True

    def test_raises_when_project_dir_deleted(self, temp_project_dir):
        """Test that FileNotFoundError is raised when project dir is deleted."""
        result = {"error": None}

        def wait_thread():
            try:
                wait_for_interactive_signal(temp_project_dir, poll_interval=0.1)
            except FileNotFoundError as e:
                result["error"] = e

        thread = threading.Thread(target=wait_thread)
        thread.start()

        time.sleep(0.2)
        shutil.rmtree(temp_project_dir)

        thread.join(timeout=2.0)

        assert result["error"] is not None
        assert "deleted" in str(result["error"]).lower() or "cannot access" in str(result["error"]).lower()

    def test_signal_file_constant(self):
        """Test that signal file constant is correct."""
        assert INTERACTIVE_SIGNAL_FILE == ".interactive_done"
