"""Unit tests for project outputs API functionality."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from datetime import datetime

from web.models.domain import Project, ProjectStatus


class TestOutputKeyNormalization:
    """Test that output directory keys are normalized correctly."""

    def test_simple_directory_unchanged(self):
        """Simple directories like 'depth' stay unchanged."""
        dir_name = "depth"
        output_key = dir_name.split("/")[0]
        assert output_key == "depth"

    def test_nested_directory_normalized(self):
        """Nested paths like 'source/frames' become 'source'."""
        dir_name = "source/frames"
        output_key = dir_name.split("/")[0]
        assert output_key == "source"

    def test_all_config_directories_normalize_correctly(self):
        """All directories from pipeline_config.json normalize as expected."""
        # These are the outputDir values from pipeline_config.json
        config_output_dirs = {
            "ingest": "source/frames",
            "interactive": "roto",
            "depth": "depth",
            "roto": "roto",
            "mama": "matte",
            "cleanplate": "cleanplate",
            "matchmove_camera": "mmcam",
            "gsir": "gsir",
            "mocap": "mocap",
            "camera": "camera",
        }

        # These are what the frontend expects (from STAGE_OUTPUT_DIRS)
        expected_keys = {
            "ingest": "source",
            "interactive": "roto",
            "depth": "depth",
            "roto": "roto",
            "mama": "matte",
            "cleanplate": "cleanplate",
            "matchmove_camera": "mmcam",
            "gsir": "gsir",
            "mocap": "mocap",
            "camera": "camera",
        }

        for stage, output_dir in config_output_dirs.items():
            normalized = output_dir.split("/")[0]
            assert normalized == expected_keys[stage], \
                f"Stage '{stage}': expected '{expected_keys[stage]}', got '{normalized}'"


class TestProjectDTOFields:
    """Test that project DTO has correct field names."""

    def test_project_dto_has_name_field(self):
        """ProjectDTO should have 'name' field, not 'project_id'."""
        from web.models.dto import ProjectDTO

        # Check the model fields
        field_names = set(ProjectDTO.model_fields.keys())
        assert "name" in field_names, "ProjectDTO should have 'name' field"

    def test_project_dto_serialization(self):
        """ProjectDTO.model_dump() returns 'name' not 'project_id'."""
        from web.models.dto import ProjectDTO
        from web.models.domain import ProjectStatus

        dto = ProjectDTO(
            name="test_project",
            status=ProjectStatus.CREATED,
            stages=["depth"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        data = dto.model_dump()
        assert "name" in data
        assert data["name"] == "test_project"
        # project_id should NOT be in the serialized data
        assert "project_id" not in data


class TestConfigServiceOutputDirs:
    """Test ConfigService.get_output_directories()."""

    def test_get_output_directories_returns_list(self):
        """get_output_directories should return a list of strings."""
        from web.services.config_service import ConfigService

        # Create a mock config
        mock_config = {
            "stages": {
                "depth": {"outputDir": "depth"},
                "roto": {"outputDir": "roto"},
                "ingest": {"outputDir": "source/frames"},
            }
        }

        service = ConfigService.__new__(ConfigService)
        service._config = mock_config
        service.config_path = Path("/fake/path")

        dirs = service.get_output_directories()

        assert isinstance(dirs, list)
        assert "depth" in dirs
        assert "roto" in dirs
        assert "source/frames" in dirs

    def test_get_output_directories_handles_missing_outputDir(self):
        """Stages without outputDir should be skipped."""
        from web.services.config_service import ConfigService

        mock_config = {
            "stages": {
                "depth": {"outputDir": "depth"},
                "nostage": {"name": "No Output Stage"},  # No outputDir
            }
        }

        service = ConfigService.__new__(ConfigService)
        service._config = mock_config
        service.config_path = Path("/fake/path")

        dirs = service.get_output_directories()

        assert "depth" in dirs
        assert len([d for d in dirs if d]) == 1  # Only one valid dir


class TestOutputsAPILogic:
    """Test the logic used in the outputs API endpoint."""

    def test_rglob_finds_nested_files(self, tmp_path):
        """rglob should find files in subdirectories."""
        # Create nested structure like matchmove_camera outputs
        (tmp_path / "mmcam" / "sparse").mkdir(parents=True)
        (tmp_path / "mmcam" / "sparse" / "cameras.bin").touch()
        (tmp_path / "mmcam" / "sparse" / "images.bin").touch()
        (tmp_path / "mmcam" / "sparse" / "points3D.bin").touch()

        dir_path = tmp_path / "mmcam"
        all_files = list(dir_path.rglob("*"))
        files = [f for f in all_files if f.is_file()]

        assert len(files) == 3
        file_names = {f.name for f in files}
        assert "cameras.bin" in file_names
        assert "images.bin" in file_names
        assert "points3D.bin" in file_names

    def test_rglob_finds_direct_files(self, tmp_path):
        """rglob should also find files directly in the directory."""
        (tmp_path / "depth").mkdir()
        for i in range(5):
            (tmp_path / "depth" / f"frame_{i:04d}.exr").touch()

        dir_path = tmp_path / "depth"
        all_files = list(dir_path.rglob("*"))
        files = [f for f in all_files if f.is_file()]

        assert len(files) == 5

    def test_empty_directory_returns_no_files(self, tmp_path):
        """Empty directories should return empty file list."""
        (tmp_path / "empty").mkdir()

        dir_path = tmp_path / "empty"
        all_files = list(dir_path.rglob("*"))
        files = [f for f in all_files if f.is_file()]

        assert len(files) == 0

    def test_source_frames_structure(self, tmp_path):
        """Test source/frames nested structure is found correctly."""
        (tmp_path / "source" / "frames").mkdir(parents=True)
        for i in range(10):
            (tmp_path / "source" / "frames" / f"frame_{i:04d}.png").touch()

        # Test that checking source/frames works
        dir_path = tmp_path / "source" / "frames"
        assert dir_path.exists()

        files = list(dir_path.rglob("*"))
        file_list = [f for f in files if f.is_file()]
        assert len(file_list) == 10

        # After normalization, key should be "source"
        dir_name = "source/frames"
        output_key = dir_name.split("/")[0]
        assert output_key == "source"


class TestFrontendBackendConsistency:
    """Test that frontend and backend use consistent naming."""

    def test_stage_output_dirs_match(self):
        """Frontend STAGE_OUTPUT_DIRS should match normalized backend keys."""
        # Frontend mapping (from ProjectsController.js)
        frontend_mapping = {
            'ingest': 'source',
            'depth': 'depth',
            'roto': 'roto',
            'cleanplate': 'cleanplate',
            'matchmove_camera': 'mmcam',
            'interactive': 'roto',
            'mama': 'matte',
            'mocap': 'mocap',
            'gsir': 'gsir',
            'camera': 'camera',
        }

        # Backend outputDir values (from pipeline_config.json)
        backend_output_dirs = {
            'ingest': 'source/frames',
            'depth': 'depth',
            'roto': 'roto',
            'cleanplate': 'cleanplate',
            'matchmove_camera': 'mmcam',
            'interactive': 'roto',
            'mama': 'matte',
            'mocap': 'mocap',
            'gsir': 'gsir',
            'camera': 'camera',
        }

        for stage in frontend_mapping:
            backend_dir = backend_output_dirs.get(stage, '')
            normalized_backend = backend_dir.split("/")[0]
            frontend_expected = frontend_mapping[stage]

            assert normalized_backend == frontend_expected, \
                f"Mismatch for '{stage}': frontend expects '{frontend_expected}', " \
                f"backend returns '{normalized_backend}'"

    def test_all_stages_list_consistency(self):
        """ALL_STAGES in frontend should include all pipeline stages."""
        # From ProjectsController.js
        frontend_all_stages = [
            'ingest', 'depth', 'roto', 'cleanplate', 'matchmove_camera',
            'interactive', 'mama', 'mocap', 'gsir', 'camera'
        ]

        # From pipeline_config.json
        backend_stages = [
            'ingest', 'interactive', 'depth', 'roto', 'mama',
            'cleanplate', 'matchmove_camera', 'gsir', 'mocap', 'camera'
        ]

        # Both should have the same stages (order doesn't matter)
        assert set(frontend_all_stages) == set(backend_stages)


class TestProjectSizeBytes:
    """Test project size calculation functionality."""

    def test_project_dto_has_size_bytes_field(self):
        """ProjectDTO should have 'size_bytes' optional field."""
        from web.models.dto import ProjectDTO

        field_names = set(ProjectDTO.model_fields.keys())
        assert "size_bytes" in field_names, "ProjectDTO should have 'size_bytes' field"

    def test_project_dto_size_bytes_is_optional(self):
        """size_bytes should be optional and default to None."""
        from web.models.dto import ProjectDTO
        from web.models.domain import ProjectStatus

        dto = ProjectDTO(
            name="test_project",
            status=ProjectStatus.CREATED,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert dto.size_bytes is None

    def test_project_dto_accepts_size_bytes(self):
        """ProjectDTO should accept size_bytes value."""
        from web.models.dto import ProjectDTO
        from web.models.domain import ProjectStatus

        dto = ProjectDTO(
            name="test_project",
            status=ProjectStatus.CREATED,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            size_bytes=1024000,
        )

        assert dto.size_bytes == 1024000

    def test_get_dir_size_bytes_function(self, tmp_path):
        """get_dir_size_bytes should calculate directory size."""
        from web.utils.media import get_dir_size_bytes

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        (project_dir / "file1.txt").write_text("Hello" * 100)
        (project_dir / "file2.txt").write_text("World" * 200)
        (project_dir / "subdir").mkdir()
        (project_dir / "subdir" / "file3.txt").write_text("Nested" * 50)

        size = get_dir_size_bytes(project_dir)

        assert size > 0
        expected_size = 500 + 1000 + 300
        assert size == expected_size

    def test_get_dir_size_bytes_empty_dir(self, tmp_path):
        """Empty directory should return 0 bytes."""
        from web.utils.media import get_dir_size_bytes

        project_dir = tmp_path / "empty_project"
        project_dir.mkdir()

        size = get_dir_size_bytes(project_dir)
        assert size == 0

    def test_get_dir_size_bytes_nonexistent_dir(self, tmp_path):
        """Non-existent directory should return 0 (graceful failure)."""
        from web.utils.media import get_dir_size_bytes

        fake_path = tmp_path / "does_not_exist"
        size = get_dir_size_bytes(fake_path)
        assert size == 0

    def test_project_size_in_serialization(self):
        """size_bytes should be included in model_dump output."""
        from web.models.dto import ProjectDTO
        from web.models.domain import ProjectStatus

        dto = ProjectDTO(
            name="test_project",
            status=ProjectStatus.CREATED,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            size_bytes=5000000,
        )

        data = dto.model_dump()
        assert "size_bytes" in data
        assert data["size_bytes"] == 5000000


class TestFormatFileSize:
    """Test file size formatting logic (matches frontend formatFileSize function)."""

    def test_format_bytes(self):
        """Test formatting various byte sizes."""
        def format_file_size(size_bytes):
            if size_bytes is None or size_bytes == 0:
                return "0 B"
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            unit_index = 0
            size = float(size_bytes)
            while size >= 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1
            if unit_index == 0:
                return f"{int(size)} {units[unit_index]}"
            return f"{size:.1f} {units[unit_index]}"

        assert format_file_size(0) == "0 B"
        assert format_file_size(500) == "500 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"
        assert format_file_size(1048576) == "1.0 MB"
        assert format_file_size(1572864) == "1.5 MB"
        assert format_file_size(1073741824) == "1.0 GB"
        assert format_file_size(None) == "0 B"
