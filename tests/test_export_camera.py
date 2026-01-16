"""Tests for export_camera.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import from scripts - add to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from export_camera import (
    decompose_matrix,
    rotation_matrix_to_euler,
    compute_fov_from_intrinsics,
    export_json_camera,
    export_after_effects_jsx,
    load_camera_data,
)


class TestDecomposeMatrix:
    def test_identity_matrix(self):
        matrix = np.eye(4)
        translation, rotation, scale = decompose_matrix(matrix)

        np.testing.assert_array_almost_equal(translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(rotation, np.eye(3))
        np.testing.assert_array_almost_equal(scale, [1, 1, 1])

    def test_translation_only(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [10, 20, 30]
        translation, rotation, scale = decompose_matrix(matrix)

        np.testing.assert_array_almost_equal(translation, [10, 20, 30])
        np.testing.assert_array_almost_equal(scale, [1, 1, 1])

    def test_uniform_scale(self):
        matrix = np.eye(4) * 2
        matrix[3, 3] = 1  # Keep homogeneous coordinate
        translation, rotation, scale = decompose_matrix(matrix)

        np.testing.assert_array_almost_equal(scale, [2, 2, 2])


class TestRotationToEuler:
    def test_identity_rotation(self):
        rotation = np.eye(3)
        euler = rotation_matrix_to_euler(rotation)
        np.testing.assert_array_almost_equal(euler, [0, 0, 0])

    def test_90_degree_z_rotation(self):
        # 90 degrees around Z axis
        angle = np.radians(90)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        euler = rotation_matrix_to_euler(rotation)
        # Z rotation should be ~90 degrees
        assert abs(euler[2] - 90) < 0.1


class TestComputeFOV:
    def test_standard_intrinsics(self):
        intrinsics = {"fx": 1000, "fy": 1000}
        h_fov, v_fov = compute_fov_from_intrinsics(intrinsics, 1920, 1080)

        # With fx=1000 and width=1920: FOV = 2*atan(960/1000) ~ 87.4 degrees
        assert 80 < h_fov < 95
        assert 50 < v_fov < 60


class TestLoadCameraData:
    def test_load_valid_data_da3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            camera_dir = Path(tmpdir)

            # Create test extrinsics (2 frames, identity matrices)
            extrinsics = [np.eye(4).tolist(), np.eye(4).tolist()]
            with open(camera_dir / "extrinsics.json", "w") as f:
                json.dump(extrinsics, f)

            # Create test intrinsics
            intrinsics = {"fx": 1000, "fy": 1000, "cx": 960, "cy": 540}
            with open(camera_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f)

            loaded_ext, loaded_int, source = load_camera_data(camera_dir)

            assert len(loaded_ext) == 2
            assert loaded_int["fx"] == 1000
            assert source == "da3"

    def test_load_valid_data_colmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            camera_dir = Path(tmpdir)

            # Create test extrinsics (2 frames, identity matrices)
            extrinsics = [np.eye(4).tolist(), np.eye(4).tolist()]
            with open(camera_dir / "extrinsics.json", "w") as f:
                json.dump(extrinsics, f)

            # Create test intrinsics (COLMAP format with extra fields)
            intrinsics = {
                "fx": 1000, "fy": 1000, "cx": 960, "cy": 540,
                "width": 1920, "height": 1080, "model": "OPENCV"
            }
            with open(camera_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f)

            # Create colmap_raw.json to indicate COLMAP source
            with open(camera_dir / "colmap_raw.json", "w") as f:
                json.dump({"source": "colmap"}, f)

            loaded_ext, loaded_int, source = load_camera_data(camera_dir)

            assert len(loaded_ext) == 2
            assert loaded_int["fx"] == 1000
            assert loaded_int["model"] == "OPENCV"
            assert source == "colmap"

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            camera_dir = Path(tmpdir)
            with pytest.raises(FileNotFoundError):
                load_camera_data(camera_dir)


class TestExportJsonCamera:
    def test_export_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "camera.json"

            extrinsics = [np.eye(4), np.eye(4)]
            intrinsics = {"fx": 1000, "fy": 1000}

            export_json_camera(
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                output_path=output_path,
                start_frame=1001,
                fps=24.0
            )

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert data["start_frame"] == 1001
            assert data["end_frame"] == 1002
            assert data["fps"] == 24.0
            assert len(data["frames"]) == 2


class TestExportAfterEffectsJsx:
    def test_export_creates_jsx_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "camera.jsx"

            extrinsics = [np.eye(4), np.eye(4)]
            intrinsics = {"fx": 1000, "fy": 1000, "width": 1920, "height": 1080}

            export_after_effects_jsx(
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                output_path=output_path,
                start_frame=1,
                fps=24.0,
                camera_name="test_camera"
            )

            assert output_path.exists()

            # Verify file content
            with open(output_path) as f:
                content = f.read()

            # Check for key JSX elements
            assert "// After Effects Camera Import Script" in content
            assert "var cameraName = \"test_camera\"" in content
            assert "var fps = 24" in content
            assert "var numFrames = 2" in content
            assert "addCamera" in content
            assert "setValueAtTime" in content
