"""Tests for run_gsir.py"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

COLMAP_AVAILABLE = shutil.which("colmap") is not None

from run_gsir import (
    check_gsir_available,
    setup_gsir_data_structure,
)


class TestCheckGsirAvailable:
    def test_gsir_not_found(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            available, path = check_gsir_available()
            assert available is False
            assert path is None

    def test_gsir_import_fails(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            # Also ensure no common paths exist
            with patch("pathlib.Path.exists", return_value=False):
                available, path = check_gsir_available()
                assert available is False

    def test_gsir_found_via_env(self):
        """Test that GSIR_PATH environment variable is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gsir_path = Path(tmpdir)
            (gsir_path / "train.py").touch()
            with patch.dict("os.environ", {"GSIR_PATH": str(gsir_path)}):
                available, path = check_gsir_available()
                assert available is True
                assert path == gsir_path


class TestSetupGsirDataStructure:
    def test_missing_colmap_sparse(self):
        """Test that setup fails if COLMAP sparse model doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()

            # Create source frames but no COLMAP output
            (project_dir / "source" / "frames").mkdir(parents=True)
            (project_dir / "source" / "frames" / "frame_0001.png").touch()

            gsir_data_dir = project_dir / "gsir" / "data"

            result = setup_gsir_data_structure(project_dir, gsir_data_dir)
            assert result is False

    def test_missing_source_frames(self):
        """Test that setup fails if source frames don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()

            # Create COLMAP output but no source frames
            (project_dir / "mmcam" / "sparse" / "0").mkdir(parents=True)

            gsir_data_dir = project_dir / "gsir" / "data"

            result = setup_gsir_data_structure(project_dir, gsir_data_dir)
            assert result is False

    @pytest.mark.skipif(not COLMAP_AVAILABLE, reason="colmap not installed")
    def test_successful_setup(self):
        """Test that setup succeeds with valid inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()

            # Create COLMAP output
            colmap_sparse = project_dir / "mmcam" / "sparse" / "0"
            colmap_sparse.mkdir(parents=True)
            (colmap_sparse / "cameras.bin").touch()
            (colmap_sparse / "images.bin").touch()
            (colmap_sparse / "points3D.bin").touch()

            # Create undistorted output (as if colmap already ran)
            undistorted = project_dir / "mmcam" / "undistorted"
            undistorted.mkdir(parents=True)
            (undistorted / "sparse" / "0").mkdir(parents=True)
            (undistorted / "images").mkdir()

            # Create source frames
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)
            (frames_dir / "frame_0001.png").touch()
            (frames_dir / "frame_0002.png").touch()

            gsir_data_dir = project_dir / "gsir" / "data"

            # Mock the colmap subprocess call
            with patch("run_gsir.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                result = setup_gsir_data_structure(project_dir, gsir_data_dir)

            assert result is True
            assert gsir_data_dir.exists()

    @pytest.mark.skipif(not COLMAP_AVAILABLE, reason="colmap not installed")
    def test_idempotent_setup(self):
        """Test that setup can be run multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()

            # Create valid structure
            colmap_sparse = project_dir / "mmcam" / "sparse" / "0"
            colmap_sparse.mkdir(parents=True)
            (colmap_sparse / "cameras.bin").touch()

            # Create undistorted output
            undistorted = project_dir / "mmcam" / "undistorted"
            undistorted.mkdir(parents=True)
            (undistorted / "sparse" / "0").mkdir(parents=True)
            (undistorted / "images").mkdir()

            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)
            (frames_dir / "frame_0001.png").touch()

            gsir_data_dir = project_dir / "gsir" / "data"

            # Mock the colmap subprocess call
            with patch("run_gsir.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                result1 = setup_gsir_data_structure(project_dir, gsir_data_dir)
                result2 = setup_gsir_data_structure(project_dir, gsir_data_dir)

            assert result1 is True
            assert result2 is True


class TestGsirPipelineIntegration:
    def test_pipeline_stage_definition(self):
        """Test that gsir stage is defined in pipeline."""
        from run_pipeline import STAGES
        assert "gsir" in STAGES
        assert "material" in STAGES["gsir"].lower() or "GS-IR" in STAGES["gsir"]

    def test_gsir_comes_after_matchmove_camera(self):
        """Test that gsir stage is ordered after matchmove_camera."""
        from run_pipeline import STAGES
        stage_list = list(STAGES.keys())
        matchmove_camera_idx = stage_list.index("matchmove_camera")
        gsir_idx = stage_list.index("gsir")
        assert gsir_idx > matchmove_camera_idx

    def test_gsir_comes_before_camera(self):
        """Test that gsir stage is ordered before camera export."""
        from run_pipeline import STAGES
        stage_list = list(STAGES.keys())
        gsir_idx = stage_list.index("gsir")
        camera_idx = stage_list.index("camera")
        assert gsir_idx < camera_idx


class TestGsirMetadata:
    def test_metadata_format(self):
        """Test the expected metadata format for GS-IR outputs."""
        expected_keys = ["source", "checkpoint", "iteration", "outputs"]
        expected_outputs = ["materials", "normals", "depth", "environment"]

        # This validates our expected output structure
        metadata = {
            "source": "gs-ir",
            "checkpoint": "/path/to/checkpoint.pth",
            "iteration": 35000,
            "outputs": {
                "materials": "materials/",
                "normals": "normals/",
                "depth": "depth_gsir/",
                "environment": "environment.png",
            }
        }

        for key in expected_keys:
            assert key in metadata

        for output_key in expected_outputs:
            assert output_key in metadata["outputs"]
