"""Tests for gravity alignment of camera extrinsics."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_matchmove_camera import _gravity_align_extrinsics


def _make_camera_c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix from rotation and translation."""
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def _rotation_around_x(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix around X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


class TestGravityAlignExtrinsics:
    def test_empty_list_returns_empty(self):
        assert _gravity_align_extrinsics([]) == []

    def test_already_aligned_is_unchanged(self):
        R_aligned = np.diag([1.0, -1.0, -1.0])
        cameras = [_make_camera_c2w(R_aligned, np.array([0.0, 0.0, float(i)])) for i in range(5)]
        aligned = _gravity_align_extrinsics(cameras)

        for orig, result in zip(cameras, aligned):
            np.testing.assert_array_almost_equal(orig, result, decimal=5)

    def test_tilted_cameras_become_y_up(self):
        tilt = np.radians(30)
        R_tilt = _rotation_around_x(tilt)
        cameras = [
            _make_camera_c2w(R_tilt, np.array([float(i), 0.0, 0.0]))
            for i in range(10)
        ]

        aligned = _gravity_align_extrinsics(cameras)

        for mat in aligned:
            cam_y = mat[:3, 1]
            np.testing.assert_array_almost_equal(cam_y, [0, -1, 0], decimal=5)

    def test_preserves_rotation_determinant(self):
        tilt = np.radians(45)
        R_tilt = _rotation_around_x(tilt)
        cameras = [_make_camera_c2w(R_tilt, np.zeros(3)) for _ in range(3)]

        aligned = _gravity_align_extrinsics(cameras)

        for mat in aligned:
            R = mat[:3, :3]
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_180_degree_flip(self):
        R_flip = np.diag([1.0, -1.0, -1.0])
        cameras = [_make_camera_c2w(R_flip, np.zeros(3)) for _ in range(3)]

        aligned = _gravity_align_extrinsics(cameras)

        for mat in aligned:
            cam_y = mat[:3, 1]
            np.testing.assert_array_almost_equal(cam_y, [0, -1, 0], decimal=5)
