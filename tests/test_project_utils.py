"""Tests for project_utils.py - Project metadata and last-project tracking."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from project_utils import (
    ProjectMetadata,
    get_last_project_file,
    save_last_project,
    get_last_project,
)


class TestProjectMetadata:
    def test_load_nonexistent_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)

            result = metadata.load()

            assert result == {}

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)

            metadata.save({"name": "test", "fps": 24})

            assert metadata.path.exists()
            with open(metadata.path) as f:
                data = json.load(f)
            assert data["name"] == "test"
            assert data["fps"] == 24

    def test_save_creates_project_dir_if_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "nonexistent" / "project"
            metadata = ProjectMetadata(project_dir)

            metadata.save({"name": "test"})

            assert project_dir.exists()
            assert metadata.path.exists()

    def test_load_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)

            with open(metadata.path, "w") as f:
                json.dump({"name": "my_project", "fps": 30}, f)

            result = metadata.load()

            assert result["name"] == "my_project"
            assert result["fps"] == 30

    def test_load_invalid_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)

            with open(metadata.path, "w") as f:
                f.write("not valid json {{{")

            result = metadata.load()

            assert result == {}

    def test_update_merges_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            metadata.save({"name": "original", "fps": 24})

            result = metadata.update(width=1920, height=1080)

            assert result["name"] == "original"
            assert result["fps"] == 24
            assert result["width"] == 1920
            assert result["height"] == 1080

    def test_update_overwrites_existing_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            metadata.save({"name": "original", "fps": 24})

            result = metadata.update(fps=30)

            assert result["fps"] == 30
            assert result["name"] == "original"

    def test_get_returns_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            metadata.save({"name": "test", "fps": 24})

            assert metadata.get("fps") == 24
            assert metadata.get("name") == "test"

    def test_get_returns_default_for_missing_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            metadata.save({"name": "test"})

            assert metadata.get("fps") is None
            assert metadata.get("fps", 30) == 30

    def test_get_fps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            metadata.save({"fps": 29.97})

            assert metadata.get_fps() == 29.97

    def test_get_fps_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            metadata.save({})

            assert metadata.get_fps() is None

    def test_set_source_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            source_path = Path("/videos/clip.mov")

            metadata.set_source_info(source_path, 23.976)

            data = metadata.load()
            assert data["source"] == "/videos/clip.mov"
            assert data["fps"] == 23.976

    def test_set_frame_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)

            metadata.set_frame_info(count=100, width=1920, height=1080)

            data = metadata.load()
            assert data["frame_count"] == 100
            assert data["width"] == 1920
            assert data["height"] == 1080

    def test_initialize_new_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)

            result = metadata.initialize(
                name="my_shot",
                fps=24,
                source_path=Path("/videos/source.mov")
            )

            assert result["name"] == "my_shot"
            assert result["fps"] == 24
            assert result["source"] == "/videos/source.mov"
            assert result["start_frame"] == 1

    def test_initialize_preserves_start_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)
            metadata.save({"start_frame": 1001})

            result = metadata.initialize(name="updated", fps=30)

            assert result["start_frame"] == 1001

    def test_initialize_without_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            metadata = ProjectMetadata(project_dir)

            result = metadata.initialize(name="my_shot", fps=24)

            assert "source" not in result
            assert result["name"] == "my_shot"


class TestLastProjectTracking:
    def test_get_last_project_file_returns_path(self):
        result = get_last_project_file()
        assert isinstance(result, Path)
        assert result.name == ".last_project"

    def test_save_and_get_last_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()

            with patch("project_utils.LAST_PROJECT_FILE", Path(tmpdir) / ".last_project"):
                with patch("project_utils.INSTALL_DIR", Path(tmpdir)):
                    save_last_project(project_dir)
                    result = get_last_project()

            assert result == project_dir

    def test_get_last_project_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("project_utils.LAST_PROJECT_FILE", Path(tmpdir) / ".last_project"):
                result = get_last_project()

            assert result is None

    def test_get_last_project_invalid_path_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            last_project_file = Path(tmpdir) / ".last_project"
            last_project_file.write_text("/nonexistent/project/path")

            with patch("project_utils.LAST_PROJECT_FILE", last_project_file):
                result = get_last_project()

            assert result is None

    def test_get_last_project_file_not_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            not_a_dir = Path(tmpdir) / "regular_file"
            not_a_dir.touch()

            last_project_file = Path(tmpdir) / ".last_project"
            last_project_file.write_text(str(not_a_dir))

            with patch("project_utils.LAST_PROJECT_FILE", last_project_file):
                result = get_last_project()

            assert result is None
