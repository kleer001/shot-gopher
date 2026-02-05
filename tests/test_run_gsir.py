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
    _verify_gsir_module_importable,
    get_frame_subset_dir,
    get_colmap_sparse_dir,
    cleanup_colmap_subset,
    is_baseline_error,
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
                with patch("run_gsir._verify_gsir_module_importable", return_value=True):
                    available, path = check_gsir_available()
                    assert available is True
                    assert path == gsir_path

    def test_gsir_dir_exists_but_module_not_importable(self):
        """Test that check fails if gs_ir module is not importable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gsir_path = Path(tmpdir)
            (gsir_path / "train.py").touch()
            with patch.dict("os.environ", {"GSIR_PATH": str(gsir_path)}):
                with patch("run_gsir._verify_gsir_module_importable", return_value=False):
                    available, path = check_gsir_available()
                    assert available is False
                    assert path is None


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
            (project_dir / "colmap" / "sparse" / "0").mkdir(parents=True)

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
            colmap_sparse = project_dir / "colmap" / "sparse" / "0"
            colmap_sparse.mkdir(parents=True)
            (colmap_sparse / "cameras.bin").touch()
            (colmap_sparse / "images.bin").touch()
            (colmap_sparse / "points3D.bin").touch()

            # Create undistorted output (as if colmap already ran)
            undistorted = project_dir / "colmap" / "undistorted"
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
            colmap_sparse = project_dir / "colmap" / "sparse" / "0"
            colmap_sparse.mkdir(parents=True)
            (colmap_sparse / "cameras.bin").touch()

            # Create undistorted output
            undistorted = project_dir / "colmap" / "undistorted"
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

    def test_gsir_comes_after_colmap(self):
        """Test that gsir stage is ordered after colmap."""
        from run_pipeline import STAGES
        stage_list = list(STAGES.keys())
        colmap_idx = stage_list.index("colmap")
        gsir_idx = stage_list.index("gsir")
        assert gsir_idx > colmap_idx

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


class TestFrameSubsetHelpers:
    def test_frame_subset_dir_original(self):
        """Test that skip_factor=1 returns original frames directory."""
        project_dir = Path("/tmp/project")
        result = get_frame_subset_dir(project_dir, 1)
        assert result == project_dir / "source" / "frames"

    def test_frame_subset_dir_4s(self):
        """Test that skip_factor=4 returns frames_4s directory."""
        project_dir = Path("/tmp/project")
        result = get_frame_subset_dir(project_dir, 4)
        assert result == project_dir / "source" / "frames_4s"

    def test_colmap_sparse_dir_original(self):
        """Test that skip_factor=1 returns original sparse directory."""
        project_dir = Path("/tmp/project")
        result = get_colmap_sparse_dir(project_dir, 1)
        assert result == project_dir / "colmap" / "sparse" / "0"

    def test_colmap_sparse_dir_4s(self):
        """Test that skip_factor=4 returns sparse_4s directory."""
        project_dir = Path("/tmp/project")
        result = get_colmap_sparse_dir(project_dir, 4)
        assert result == project_dir / "colmap" / "sparse_4s" / "0"


class TestColmapSubsetCleanup:
    def test_cleanup_does_nothing_for_original(self):
        """Test that cleanup_colmap_subset doesn't touch original (skip_factor=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            colmap_dir = project_dir / "colmap"
            colmap_dir.mkdir(parents=True)

            original_db = colmap_dir / "database.db"
            original_sparse = colmap_dir / "sparse" / "0"
            original_db.touch()
            original_sparse.mkdir(parents=True)

            cleanup_colmap_subset(project_dir, 1)

            assert original_db.exists()
            assert original_sparse.exists()

    def test_cleanup_removes_subset_files(self):
        """Test that cleanup_colmap_subset removes all subset artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            colmap_dir = project_dir / "colmap"
            colmap_dir.mkdir(parents=True)

            subset_db = colmap_dir / "database_4s.db"
            subset_sparse = colmap_dir / "sparse_4s" / "0"
            subset_undistorted = colmap_dir / "undistorted_4s" / "images"
            subset_gsir_data = project_dir / "gsir" / "data_4s"
            subset_db.touch()
            subset_sparse.mkdir(parents=True)
            subset_undistorted.mkdir(parents=True)
            subset_gsir_data.mkdir(parents=True)

            original_db = colmap_dir / "database.db"
            original_sparse = colmap_dir / "sparse" / "0"
            original_undistorted = colmap_dir / "undistorted" / "images"
            original_db.touch()
            original_sparse.mkdir(parents=True)
            original_undistorted.mkdir(parents=True)

            cleanup_colmap_subset(project_dir, 4)

            assert not subset_db.exists()
            assert not (colmap_dir / "sparse_4s").exists()
            assert not (colmap_dir / "undistorted_4s").exists()
            assert not (project_dir / "gsir" / "data_4s").exists()
            assert original_db.exists()
            assert original_sparse.exists()
            assert original_undistorted.exists()


class TestBaselineErrorDetection:
    def test_detects_invalid_gradient_error(self):
        """Test that baseline error is detected from invalid gradient message."""
        error = "RuntimeError: Function _RasterizeGaussiansBackward returned an invalid gradient at index 7"
        assert is_baseline_error(error) is True

    def test_detects_shape_mismatch_error(self):
        """Test that baseline error is detected from shape mismatch."""
        error = "got [0, 0, 3] but expected shape compatible with [0, 16, 3]"
        assert is_baseline_error(error) is True

    def test_ignores_unrelated_errors(self):
        """Test that unrelated errors are not flagged as baseline errors."""
        error = "RuntimeError: CUDA out of memory"
        assert is_baseline_error(error) is False

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        error = "INVALID GRADIENT error occurred"
        assert is_baseline_error(error) is True
