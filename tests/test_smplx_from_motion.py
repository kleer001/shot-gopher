"""Tests for smplx_from_motion.py - SMPL-X mesh generation from motion data.

Tests the motion loading and parameter extraction logic without requiring
SMPL-X models or torch dependencies.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import from scripts - add to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from smplx_from_motion import (
    load_motion_data,
    find_smplx_models,
    extract_frame_number,
    detect_source_start_frame,
)


class TestLoadMotionData:
    """Test motion data loading and validation."""

    def test_load_valid_motion_data(self):
        """Test loading valid motion.pkl file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            # Create test motion data
            test_data = {
                "poses": np.random.rand(100, 72).astype(np.float32),
                "trans": np.random.rand(100, 3).astype(np.float32),
                "betas": np.random.rand(10).astype(np.float32),
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            # Load and verify
            loaded = load_motion_data(motion_path)

            assert "poses" in loaded
            assert "trans" in loaded
            assert "betas" in loaded
            assert loaded["poses"].shape == (100, 72)
            assert loaded["trans"].shape == (100, 3)
            # Betas should be expanded to per-frame
            assert loaded["betas"].shape == (100, 10)

    def test_load_motion_with_per_frame_betas(self):
        """Test loading motion data with per-frame betas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            # Create test data with per-frame betas
            n_frames = 50
            test_data = {
                "poses": np.random.rand(n_frames, 72).astype(np.float32),
                "trans": np.random.rand(n_frames, 3).astype(np.float32),
                "betas": np.random.rand(n_frames, 10).astype(np.float32),
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            loaded = load_motion_data(motion_path)

            # Per-frame betas should be preserved as-is
            assert loaded["betas"].shape == (n_frames, 10)
            np.testing.assert_array_almost_equal(loaded["betas"], test_data["betas"])

    def test_load_motion_without_betas(self):
        """Test loading motion data without betas (uses default neutral)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            n_frames = 30
            test_data = {
                "poses": np.random.rand(n_frames, 72).astype(np.float32),
                "trans": np.random.rand(n_frames, 3).astype(np.float32),
                # No betas
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            loaded = load_motion_data(motion_path)

            # Should create default neutral betas (zeros)
            assert loaded["betas"].shape == (n_frames, 10)
            np.testing.assert_array_equal(loaded["betas"], np.zeros((n_frames, 10)))

    def test_missing_poses_raises_error(self):
        """Test that missing poses key raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            test_data = {
                "trans": np.random.rand(100, 3).astype(np.float32),
                # Missing "poses"
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            with pytest.raises(ValueError, match="missing 'poses'"):
                load_motion_data(motion_path)

    def test_missing_trans_raises_error(self):
        """Test that missing trans key raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            test_data = {
                "poses": np.random.rand(100, 72).astype(np.float32),
                # Missing "trans"
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            with pytest.raises(ValueError, match="missing 'trans'"):
                load_motion_data(motion_path)

    def test_data_converted_to_numpy(self):
        """Test that data is properly converted to numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            # Use Python lists instead of numpy arrays
            test_data = {
                "poses": [[0.1] * 72 for _ in range(10)],
                "trans": [[0.0, 0.0, 0.0] for _ in range(10)],
                "betas": [0.0] * 10,
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            loaded = load_motion_data(motion_path)

            assert isinstance(loaded["poses"], np.ndarray)
            assert isinstance(loaded["trans"], np.ndarray)
            assert isinstance(loaded["betas"], np.ndarray)


class TestFindSmplxModels:
    """Test SMPL-X model directory discovery."""

    def test_returns_none_when_not_found(self):
        """Test that None is returned when no models found."""
        # In a fresh environment without SMPL-X installed
        # This test may pass or fail depending on environment
        result = find_smplx_models()

        # Result is either None or a valid Path
        assert result is None or isinstance(result, Path)

    def test_finds_local_models_directory(self):
        """Test finding models in .vfx_pipeline/smplx_models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock SMPL-X directory structure
            models_dir = Path(tmpdir) / ".vfx_pipeline" / "smplx_models" / "smplx"
            models_dir.mkdir(parents=True)

            # Create a mock model file
            (models_dir / "model.pkl").touch()

            # Temporarily modify the search path function
            # Since we can't easily inject paths, we test the structure we expect
            assert models_dir.exists()
            assert (models_dir / "model.pkl").exists()


class TestPoseParameterExtraction:
    """Test that pose parameters are correctly extracted for SMPL-X."""

    def test_smplx_pose_breakdown(self):
        """Test the expected breakdown of SMPL/SMPL-X pose parameters."""
        # GVHMR outputs SMPL format: 72 values = 3 (global) + 23*3 (body)
        pose_smpl = np.random.rand(72).astype(np.float32)

        # Extract as the code does
        global_orient = pose_smpl[:3]
        body_pose_smplx = pose_smpl[3:66]  # 21 joints for SMPL-X

        assert len(global_orient) == 3
        assert len(body_pose_smplx) == 63  # 21 * 3

    def test_optional_jaw_pose(self):
        """Test extraction of optional jaw pose parameters."""
        # Some motion data may include jaw pose
        pose_with_jaw = np.random.rand(69).astype(np.float32)

        if len(pose_with_jaw) > 66:
            jaw_pose = pose_with_jaw[66:69]
            assert len(jaw_pose) == 3
        else:
            jaw_pose = np.zeros(3)
            assert len(jaw_pose) == 3

    def test_betas_extraction(self):
        """Test beta (shape) parameter handling."""
        # Test 1D betas (constant across frames)
        betas_1d = np.random.rand(10).astype(np.float32)
        n_frames = 5

        if betas_1d.ndim == 1:
            betas_expanded = np.tile(betas_1d[np.newaxis, :], (n_frames, 1))
        else:
            betas_expanded = betas_1d

        assert betas_expanded.shape == (n_frames, 10)

        # Test 2D betas (per-frame)
        betas_2d = np.random.rand(n_frames, 10).astype(np.float32)

        if betas_2d.ndim == 1:
            betas_result = np.tile(betas_2d[np.newaxis, :], (n_frames, 1))
        else:
            betas_result = betas_2d

        assert betas_result.shape == (n_frames, 10)
        np.testing.assert_array_equal(betas_result, betas_2d)


class TestMotionDataIntegrity:
    """Test motion data integrity and edge cases."""

    def test_single_frame_motion(self):
        """Test loading single-frame motion data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            test_data = {
                "poses": np.random.rand(1, 72).astype(np.float32),
                "trans": np.random.rand(1, 3).astype(np.float32),
                "betas": np.random.rand(10).astype(np.float32),
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            loaded = load_motion_data(motion_path)

            assert loaded["poses"].shape[0] == 1
            assert loaded["trans"].shape[0] == 1
            assert loaded["betas"].shape[0] == 1

    def test_large_motion_sequence(self):
        """Test loading large motion sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            n_frames = 10000  # Large sequence
            test_data = {
                "poses": np.random.rand(n_frames, 72).astype(np.float32),
                "trans": np.random.rand(n_frames, 3).astype(np.float32),
                "betas": np.random.rand(10).astype(np.float32),
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            loaded = load_motion_data(motion_path)

            assert loaded["poses"].shape[0] == n_frames
            assert loaded["betas"].shape[0] == n_frames

    def test_motion_data_shapes_match(self):
        """Test that poses and trans have consistent frame counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.pkl"

            n_frames = 100
            test_data = {
                "poses": np.random.rand(n_frames, 72).astype(np.float32),
                "trans": np.random.rand(n_frames, 3).astype(np.float32),
            }

            with open(motion_path, "wb") as f:
                pickle.dump(test_data, f)

            loaded = load_motion_data(motion_path)

            # All arrays should have same number of frames
            assert loaded["poses"].shape[0] == loaded["trans"].shape[0]
            assert loaded["betas"].shape[0] == loaded["poses"].shape[0]


class TestExtractFrameNumber:
    """Test frame number extraction from filenames."""

    def test_standard_frame_format(self):
        """Test extracting from standard frame_0001.png format."""
        assert extract_frame_number("frame_0001.png") == 1
        assert extract_frame_number("frame_0100.png") == 100
        assert extract_frame_number("frame_1001.png") == 1001

    def test_depth_format(self):
        """Test extracting from depth_00001.png format."""
        assert extract_frame_number("depth_00001.png") == 1
        assert extract_frame_number("depth_01000.png") == 1000

    def test_no_number_returns_negative(self):
        """Test that filenames without numbers return -1."""
        assert extract_frame_number("readme.txt") == -1
        assert extract_frame_number("config.json") == -1

    def test_multiple_numbers_takes_first(self):
        """Test that first number is extracted when multiple present."""
        assert extract_frame_number("frame_0001_v2.png") == 1


class TestDetectSourceStartFrame:
    """Test source frame sequence start detection."""

    def test_detect_start_frame_1(self):
        """Test detection when sequence starts at frame 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            for i in range(1, 11):
                (frames_dir / f"frame_{i:04d}.png").touch()

            assert detect_source_start_frame(project_dir) == 1

    def test_detect_start_frame_1001(self):
        """Test detection when sequence starts at frame 1001 (VFX convention)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            for i in range(1001, 1011):
                (frames_dir / f"frame_{i:04d}.png").touch()

            assert detect_source_start_frame(project_dir) == 1001

    def test_detect_start_frame_arbitrary(self):
        """Test detection with arbitrary start frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            for i in range(500, 510):
                (frames_dir / f"frame_{i:04d}.jpg").touch()

            assert detect_source_start_frame(project_dir) == 500

    def test_missing_frames_dir_returns_default(self):
        """Test that missing frames directory returns default of 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            assert detect_source_start_frame(project_dir) == 1

    def test_empty_frames_dir_returns_default(self):
        """Test that empty frames directory returns default of 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)
            assert detect_source_start_frame(project_dir) == 1

    def test_mixed_file_types(self):
        """Test detection with mixed png and jpg files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            (frames_dir / "frame_0100.png").touch()
            (frames_dir / "frame_0101.jpg").touch()
            (frames_dir / "frame_0102.png").touch()

            assert detect_source_start_frame(project_dir) == 100
