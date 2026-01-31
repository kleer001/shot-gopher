"""Tests for setup_project.py"""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from setup_project import (
    create_project_structure,
    populate_workflow,
    copy_and_populate_workflows,
    PROJECT_DIRS,
)


class TestCreateProjectStructure:
    def test_creates_all_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()

            create_project_structure(project_dir)

            for subdir in PROJECT_DIRS:
                assert (project_dir / subdir).exists()
                assert (project_dir / subdir).is_dir()

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()

            # Run twice - should not fail
            create_project_structure(project_dir)
            create_project_structure(project_dir)

            for subdir in PROJECT_DIRS:
                assert (project_dir / subdir).exists()


class TestPopulateWorkflow:
    def test_replaces_relative_paths(self):
        project_dir = Path("/projects/My_Shot")
        workflow = {
            "nodes": [
                {
                    "widgets_values": ["source/frames/", "depth/"]
                }
            ]
        }

        result = populate_workflow(workflow, project_dir)

        assert result["nodes"][0]["widgets_values"][0] == "/projects/My_Shot/source/frames/"
        assert result["nodes"][0]["widgets_values"][1] == "/projects/My_Shot/depth/"

    def test_replaces_camera_paths(self):
        project_dir = Path("/projects/My_Shot")
        workflow = {
            "nodes": [
                {"widgets_values": ["camera/extrinsics.json"]},
                {"widgets_values": ["camera/intrinsics.json"]},
            ]
        }

        result = populate_workflow(workflow, project_dir)

        assert result["nodes"][0]["widgets_values"][0] == "/projects/My_Shot/camera/extrinsics.json"
        assert result["nodes"][1]["widgets_values"][0] == "/projects/My_Shot/camera/intrinsics.json"

    def test_handles_nested_structures(self):
        project_dir = Path("/projects/My_Shot")
        workflow = {
            "nodes": [
                {
                    "inputs": [{"path": "source/frames"}],
                    "outputs": [{"path": "depth/"}]  # Must have slash to be recognized as path
                }
            ]
        }

        result = populate_workflow(workflow, project_dir)

        assert result["nodes"][0]["inputs"][0]["path"] == "/projects/My_Shot/source/frames"
        assert result["nodes"][0]["outputs"][0]["path"] == "/projects/My_Shot/depth/"

    def test_preserves_filename_patterns(self):
        """Bare names like 'depth' should NOT be replaced (could be filename patterns)."""
        project_dir = Path("/projects/My_Shot")
        workflow = {
            "nodes": [
                {
                    "widgets_values": ["depth_%04d", "roto_mask"]
                }
            ]
        }

        result = populate_workflow(workflow, project_dir)

        # These should NOT be modified - they're filename patterns, not paths
        assert result["nodes"][0]["widgets_values"][0] == "depth_%04d"
        assert result["nodes"][0]["widgets_values"][1] == "roto_mask"

    def test_preserves_non_path_strings(self):
        project_dir = Path("/projects/My_Shot")
        workflow = {
            "nodes": [
                {"type": "LoadImage", "title": "My Node"}
            ]
        }

        result = populate_workflow(workflow, project_dir)

        assert result["nodes"][0]["type"] == "LoadImage"
        assert result["nodes"][0]["title"] == "My Node"


class TestCopyAndPopulateWorkflows:
    def test_copies_and_populates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create source workflow
            workflows_dir = tmpdir / "workflows"
            workflows_dir.mkdir()
            source_workflow = {
                "nodes": [{"widgets_values": ["source/frames/"]}]
            }
            with open(workflows_dir / "test.json", "w", encoding='utf-8') as f:
                json.dump(source_workflow, f)

            # Create project
            project_dir = tmpdir / "my_project"
            project_dir.mkdir()

            # Copy and populate
            created = copy_and_populate_workflows(project_dir, workflows_dir)

            assert len(created) == 1
            assert created[0].exists()

            # Verify populated content
            with open(created[0], encoding='utf-8') as f:
                result = json.load(f)

            expected_path = str(project_dir / "source/frames") + "/"
            assert result["nodes"][0]["widgets_values"][0] == expected_path

    def test_missing_workflows_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()

            with pytest.raises(FileNotFoundError):
                copy_and_populate_workflows(
                    project_dir,
                    Path("/nonexistent/workflows")
                )
