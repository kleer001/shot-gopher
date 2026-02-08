"""Tests for run_matchmove_camera.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_matchmove_camera import (
    check_colmap_available,
    quaternion_to_rotation_matrix,
    colmap_to_camera_matrices,
    export_colmap_to_pipeline_format,
    QUALITY_PRESETS,
)


class TestCheckColmapAvailable:
    def test_colmap_not_found(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert check_colmap_available() is False

    def test_colmap_found(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_colmap_available() is True

    def test_colmap_timeout(self):
        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="colmap", timeout=10)
            assert check_colmap_available() is False


class TestQuaternionToRotationMatrix:
    def test_identity_quaternion(self):
        # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
        quat = [1, 0, 0, 0]
        R = quaternion_to_rotation_matrix(quat)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_90_degree_z_rotation(self):
        # 90 degrees around Z axis: w=cos(45deg), z=sin(45deg)
        import math
        angle = math.radians(90)
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        quat = [w, 0, 0, z]
        R = quaternion_to_rotation_matrix(quat)

        # Expected rotation matrix for 90 degrees around Z
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(R, expected, decimal=5)

    def test_180_degree_x_rotation(self):
        # 180 degrees around X axis: w=0, x=1
        quat = [0, 1, 0, 0]
        R = quaternion_to_rotation_matrix(quat)

        # Expected: flip Y and Z
        expected = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        np.testing.assert_array_almost_equal(R, expected, decimal=5)


class TestColmapToCameraMatrices:
    def test_single_camera_identity(self):
        """Test conversion with identity pose."""
        images = {
            "frame_0001.png": {
                "image_id": 1,
                "quat": [1, 0, 0, 0],  # Identity rotation
                "trans": [0, 0, 0],     # No translation
                "camera_id": 1,
            }
        }
        cameras = {
            1: {
                "model": "PINHOLE",
                "width": 1920,
                "height": 1080,
                "params": [1000, 1000, 960, 540],  # fx, fy, cx, cy
            }
        }

        extrinsics, intrinsics = colmap_to_camera_matrices(images, cameras)

        assert len(extrinsics) == 1
        np.testing.assert_array_almost_equal(extrinsics[0], np.eye(4))
        assert intrinsics["fx"] == 1000
        assert intrinsics["fy"] == 1000
        assert intrinsics["cx"] == 960
        assert intrinsics["cy"] == 540

    def test_multiple_frames_sorted(self):
        """Test that frames are sorted by name."""
        images = {
            "frame_0003.png": {"image_id": 3, "quat": [1, 0, 0, 0], "trans": [3, 0, 0], "camera_id": 1},
            "frame_0001.png": {"image_id": 1, "quat": [1, 0, 0, 0], "trans": [1, 0, 0], "camera_id": 1},
            "frame_0002.png": {"image_id": 2, "quat": [1, 0, 0, 0], "trans": [2, 0, 0], "camera_id": 1},
        }
        cameras = {
            1: {"model": "PINHOLE", "width": 1920, "height": 1080, "params": [1000, 1000, 960, 540]}
        }

        extrinsics, intrinsics = colmap_to_camera_matrices(images, cameras)

        assert len(extrinsics) == 3
        # Should be sorted: frame_0001, frame_0002, frame_0003
        # COLMAP trans is world-to-camera, so camera position = -R.T @ t
        # With identity rotation, camera position = -t
        np.testing.assert_array_almost_equal(extrinsics[0][:3, 3], [-1, 0, 0])
        np.testing.assert_array_almost_equal(extrinsics[1][:3, 3], [-2, 0, 0])
        np.testing.assert_array_almost_equal(extrinsics[2][:3, 3], [-3, 0, 0])

    def test_opencv_camera_model(self):
        """Test OPENCV camera model parsing."""
        images = {
            "frame_0001.png": {"image_id": 1, "quat": [1, 0, 0, 0], "trans": [0, 0, 0], "camera_id": 1}
        }
        cameras = {
            1: {
                "model": "OPENCV",
                "width": 1920,
                "height": 1080,
                "params": [1200, 1200, 960, 540, 0.1, -0.2, 0.01, 0.02],  # fx, fy, cx, cy, k1, k2, p1, p2
            }
        }

        extrinsics, intrinsics = colmap_to_camera_matrices(images, cameras)

        assert intrinsics["fx"] == 1200
        assert intrinsics["fy"] == 1200
        assert intrinsics["model"] == "OPENCV"

    def test_simple_radial_camera_model(self):
        """Test SIMPLE_RADIAL camera model parsing."""
        images = {
            "frame_0001.png": {"image_id": 1, "quat": [1, 0, 0, 0], "trans": [0, 0, 0], "camera_id": 1}
        }
        cameras = {
            1: {
                "model": "SIMPLE_RADIAL",
                "width": 1920,
                "height": 1080,
                "params": [1000, 960, 540, 0.1],  # f, cx, cy, k1
            }
        }

        extrinsics, intrinsics = colmap_to_camera_matrices(images, cameras)

        # SIMPLE_RADIAL uses single focal length for both fx and fy
        assert intrinsics["fx"] == 1000
        assert intrinsics["fy"] == 1000


class TestExportColmapToPipelineFormat:
    def test_export_creates_files(self):
        """Test that export creates the expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sparse_path = Path(tmpdir) / "sparse"
            sparse_path.mkdir()
            output_dir = Path(tmpdir) / "camera"

            # Create mock COLMAP text output files
            with open(sparse_path / "cameras.txt", "w") as f:
                f.write("# Camera list\n")
                f.write("1 PINHOLE 1920 1080 1000 1000 960 540\n")

            with open(sparse_path / "images.txt", "w") as f:
                f.write("# Image list\n")
                f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
                f.write("1 1 0 0 0 0 0 0 1 frame_0001.png\n")
                f.write("100.0 200.0 -1\n")  # Dummy 2D point (X, Y, POINT3D_ID=-1 means no 3D point)
                f.write("2 1 0 0 0 0 0 1 1 frame_0002.png\n")
                f.write("150.0 250.0 -1\n")  # Dummy 2D point

            result = export_colmap_to_pipeline_format(sparse_path, output_dir)

            assert result is True
            assert (output_dir / "extrinsics.json").exists()
            assert (output_dir / "intrinsics.json").exists()
            assert (output_dir / "colmap_raw.json").exists()

            # Verify extrinsics content
            with open(output_dir / "extrinsics.json") as f:
                extrinsics = json.load(f)
            assert len(extrinsics) == 2

            # Verify intrinsics content
            with open(output_dir / "intrinsics.json") as f:
                intrinsics = json.load(f)
            assert intrinsics["fx"] == 1000


class TestQualityPresets:
    def test_presets_exist(self):
        """Test that all quality presets are defined."""
        assert "low" in QUALITY_PRESETS
        assert "medium" in QUALITY_PRESETS
        assert "high" in QUALITY_PRESETS

    def test_presets_have_required_keys(self):
        """Test that presets have all required settings."""
        required_keys = ["sift_max_features", "matcher", "ba_refine_focal", "dense_max_size"]
        for preset_name, preset in QUALITY_PRESETS.items():
            for key in required_keys:
                assert key in preset, f"Missing {key} in {preset_name} preset"

    def test_low_preset_is_fastest(self):
        """Test that low preset has fewer features than high."""
        assert QUALITY_PRESETS["low"]["sift_max_features"] < QUALITY_PRESETS["high"]["sift_max_features"]

    def test_high_preset_uses_exhaustive_matcher(self):
        """Test that high preset uses exhaustive matching for accuracy."""
        assert QUALITY_PRESETS["high"]["matcher"] == "exhaustive"


class TestCameraMatrixInversion:
    def test_world_to_camera_to_camera_to_world(self):
        """Test that camera matrix inversion is correct.

        COLMAP stores world-to-camera (R, t) where X_cam = R @ X_world + t
        We want camera-to-world where the camera position in world coords is -R.T @ t
        """
        import math

        # Create a test case with known rotation and translation
        angle = math.radians(30)
        R_wc = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        t_wc = np.array([1, 2, 3])

        # Convert to quaternion (w, x, y, z)
        # For rotation around Z axis: w = cos(theta/2), z = sin(theta/2)
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        quat = [w, 0, 0, z]

        images = {
            "frame_0001.png": {
                "image_id": 1,
                "quat": quat,
                "trans": t_wc.tolist(),
                "camera_id": 1,
            }
        }
        cameras = {
            1: {"model": "PINHOLE", "width": 1920, "height": 1080, "params": [1000, 1000, 960, 540]}
        }

        extrinsics, _ = colmap_to_camera_matrices(images, cameras)
        matrix = extrinsics[0]

        # The camera position in world coordinates should be -R_wc.T @ t_wc
        expected_pos = -R_wc.T @ t_wc
        np.testing.assert_array_almost_equal(matrix[:3, 3], expected_pos, decimal=5)

        # The rotation should be the inverse (transpose) of R_wc
        expected_rot = R_wc.T
        np.testing.assert_array_almost_equal(matrix[:3, :3], expected_rot, decimal=5)
