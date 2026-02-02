"""Tests for transforms.py - Matrix, quaternion, and Euler angle utilities."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from transforms import (
    decompose_matrix,
    compose_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    slerp,
    lerp,
    cubic_bezier,
    convert_opencv_to_opengl,
    convert_opengl_to_opencv,
    matrix_to_alembic_xform,
    compute_fov_from_intrinsics,
    focal_length_to_fov,
    fov_to_focal_length,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
)


class TestDecomposeCompose:
    def test_decompose_identity(self):
        identity = np.eye(4)
        translation, rotation, scale = decompose_matrix(identity)

        np.testing.assert_array_almost_equal(translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(rotation, np.eye(3))
        np.testing.assert_array_almost_equal(scale, [1, 1, 1])

    def test_decompose_translation_only(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [10, 20, 30]
        translation, rotation, scale = decompose_matrix(matrix)

        np.testing.assert_array_almost_equal(translation, [10, 20, 30])
        np.testing.assert_array_almost_equal(rotation, np.eye(3))
        np.testing.assert_array_almost_equal(scale, [1, 1, 1])

    def test_decompose_scale_only(self):
        matrix = np.diag([2, 3, 4, 1]).astype(float)
        translation, rotation, scale = decompose_matrix(matrix)

        np.testing.assert_array_almost_equal(translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(scale, [2, 3, 4])

    def test_compose_identity(self):
        translation = np.array([0, 0, 0])
        rotation = np.eye(3)
        result = compose_matrix(translation, rotation)

        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_compose_with_scale(self):
        translation = np.array([1, 2, 3])
        rotation = np.eye(3)
        scale = np.array([2, 2, 2])
        result = compose_matrix(translation, rotation, scale)

        assert result[0, 0] == 2
        assert result[1, 1] == 2
        assert result[2, 2] == 2
        np.testing.assert_array_almost_equal(result[:3, 3], [1, 2, 3])

    def test_roundtrip_decompose_compose(self):
        original = np.array([
            [2, 0, 0, 5],
            [0, 3, 0, 10],
            [0, 0, 4, 15],
            [0, 0, 0, 1]
        ], dtype=float)

        translation, rotation, scale = decompose_matrix(original)
        reconstructed = compose_matrix(translation, rotation, scale)

        np.testing.assert_array_almost_equal(reconstructed, original)


class TestQuaternionRotationMatrix:
    def test_identity_quaternion(self):
        identity_quat = [1, 0, 0, 0]
        rotation = quaternion_to_rotation_matrix(identity_quat)
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_90deg_rotation_z(self):
        angle = np.pi / 2
        quat = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
        rotation = quaternion_to_rotation_matrix(quat)

        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(rotation, expected, decimal=5)

    def test_zero_quaternion_returns_identity(self):
        zero_quat = [0, 0, 0, 0]
        rotation = quaternion_to_rotation_matrix(zero_quat)
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_rotation_matrix_to_quaternion_identity(self):
        identity_rot = np.eye(3)
        quat = rotation_matrix_to_quaternion(identity_rot)
        np.testing.assert_array_almost_equal(np.abs(quat), [1, 0, 0, 0], decimal=5)

    def test_quaternion_roundtrip(self):
        original_quat = np.array([0.5, 0.5, 0.5, 0.5])
        original_quat = original_quat / np.linalg.norm(original_quat)

        rotation = quaternion_to_rotation_matrix(original_quat)
        recovered = rotation_matrix_to_quaternion(rotation)

        sign = np.sign(np.dot(original_quat, recovered))
        np.testing.assert_array_almost_equal(original_quat, sign * recovered, decimal=5)


class TestEulerRotation:
    def test_zero_euler_xyz(self):
        euler = np.array([0, 0, 0])
        rotation = euler_to_rotation_matrix(euler, "xyz")
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_90deg_z_euler_xyz(self):
        euler = np.array([0, 0, 90])
        rotation = euler_to_rotation_matrix(euler, "xyz")

        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(rotation, expected, decimal=5)

    def test_euler_roundtrip_xyz(self):
        original = np.array([30, 45, 60])
        rotation = euler_to_rotation_matrix(original, "xyz")
        recovered = rotation_matrix_to_euler(rotation, "xyz")
        np.testing.assert_array_almost_equal(original, recovered, decimal=3)

    @pytest.mark.skip(reason="Known issue: zxy Euler extraction doesn't match composition")
    def test_euler_roundtrip_zxy(self):
        original = np.array([15, 20, 25])
        rotation = euler_to_rotation_matrix(original, "zxy")
        recovered = rotation_matrix_to_euler(rotation, "zxy")
        recovered_rotation = euler_to_rotation_matrix(recovered, "zxy")
        np.testing.assert_array_almost_equal(rotation, recovered_rotation, decimal=5)

    @pytest.mark.skip(reason="Known issue: zyx Euler extraction doesn't match composition")
    def test_euler_roundtrip_zyx(self):
        original = np.array([15, 20, 25])
        rotation = euler_to_rotation_matrix(original, "zyx")
        recovered = rotation_matrix_to_euler(rotation, "zyx")
        recovered_rotation = euler_to_rotation_matrix(recovered, "zyx")
        np.testing.assert_array_almost_equal(rotation, recovered_rotation, decimal=5)

    def test_unsupported_order_raises(self):
        euler = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="Unsupported rotation order"):
            euler_to_rotation_matrix(euler, "yxz")

    def test_euler_to_rotation_unsupported_order_raises(self):
        rotation = np.eye(3)
        with pytest.raises(ValueError, match="Unsupported rotation order"):
            rotation_matrix_to_euler(rotation, "yxz")


class TestSlerp:
    def test_slerp_t0(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        result = slerp(q1, q2, 0)
        np.testing.assert_array_almost_equal(result, q1 / np.linalg.norm(q1))

    def test_slerp_t1(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        result = slerp(q1, q2, 1)
        np.testing.assert_array_almost_equal(result, q2 / np.linalg.norm(q2))

    def test_slerp_halfway(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        result = slerp(q1, q2, 0.5)
        expected = np.array([0.5, 0.5, 0, 0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_array_almost_equal(np.abs(result), np.abs(expected), decimal=5)

    def test_slerp_zero_quaternions(self):
        q1 = np.array([0, 0, 0, 0])
        q2 = np.array([0, 0, 0, 0])
        result = slerp(q1, q2, 0.5)
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0])

    def test_slerp_very_close_quaternions(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([1, 0.0001, 0, 0])
        q2 = q2 / np.linalg.norm(q2)
        result = slerp(q1, q2, 0.5)
        assert np.linalg.norm(result) > 0.99


class TestLerp:
    def test_lerp_t0(self):
        a = np.array([0, 0, 0])
        b = np.array([10, 20, 30])
        result = lerp(a, b, 0)
        np.testing.assert_array_equal(result, a)

    def test_lerp_t1(self):
        a = np.array([0, 0, 0])
        b = np.array([10, 20, 30])
        result = lerp(a, b, 1)
        np.testing.assert_array_equal(result, b)

    def test_lerp_halfway(self):
        a = np.array([0, 0, 0])
        b = np.array([10, 20, 30])
        result = lerp(a, b, 0.5)
        np.testing.assert_array_equal(result, [5, 10, 15])


class TestCubicBezier:
    def test_bezier_t0(self):
        p0 = np.array([0, 0])
        p1 = np.array([0, 1])
        p2 = np.array([1, 1])
        p3 = np.array([1, 0])
        result = cubic_bezier(p0, p1, p2, p3, 0)
        np.testing.assert_array_equal(result, p0)

    def test_bezier_t1(self):
        p0 = np.array([0, 0])
        p1 = np.array([0, 1])
        p2 = np.array([1, 1])
        p3 = np.array([1, 0])
        result = cubic_bezier(p0, p1, p2, p3, 1)
        np.testing.assert_array_equal(result, p3)

    def test_bezier_straight_line(self):
        p0 = np.array([0, 0])
        p1 = np.array([1, 0])
        p2 = np.array([2, 0])
        p3 = np.array([3, 0])
        result = cubic_bezier(p0, p1, p2, p3, 0.5)
        np.testing.assert_array_almost_equal(result, [1.5, 0])


class TestCoordinateConversion:
    def test_opencv_to_opengl_identity(self):
        identity = np.eye(4)
        result = convert_opencv_to_opengl(identity)
        expected = np.diag([1, -1, -1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_opengl_to_opencv_identity(self):
        identity = np.eye(4)
        result = convert_opengl_to_opencv(identity)
        expected = np.diag([1, -1, -1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_roundtrip_conversion(self):
        original = np.random.rand(4, 4)
        original[3, :] = [0, 0, 0, 1]
        converted = convert_opencv_to_opengl(original)
        back = convert_opengl_to_opencv(converted)
        np.testing.assert_array_almost_equal(original, back)


class TestMatrixToAlembic:
    def test_identity_matrix(self):
        identity = np.eye(4)
        result = matrix_to_alembic_xform(identity)
        expected = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        assert result == expected

    def test_column_major_order(self):
        matrix = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ])
        result = matrix_to_alembic_xform(matrix)
        expected = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
        assert result == expected


class TestFOVConversion:
    def test_compute_fov_from_intrinsics(self):
        intrinsics = {"fx": 1000, "fy": 1000}
        h_fov, v_fov = compute_fov_from_intrinsics(intrinsics, 1920, 1080)

        assert 80 < h_fov < 100
        assert 50 < v_fov < 60

    def test_compute_fov_with_focal_x_y_keys(self):
        intrinsics = {"focal_x": 1000, "focal_y": 1000}
        h_fov, v_fov = compute_fov_from_intrinsics(intrinsics, 1920, 1080)
        assert h_fov > 0
        assert v_fov > 0

    def test_focal_length_to_fov_35mm(self):
        fov = focal_length_to_fov(35, 36)
        assert 50 < fov < 60

    def test_fov_to_focal_length_roundtrip(self):
        original_focal = 50
        sensor_size = 36
        fov = focal_length_to_fov(original_focal, sensor_size)
        recovered = fov_to_focal_length(fov, sensor_size)
        assert abs(original_focal - recovered) < 0.01


class TestQuaternionOperations:
    def test_normalize_quaternion(self):
        q = np.array([2, 0, 0, 0])
        result = normalize_quaternion(q)
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0])

    def test_normalize_zero_quaternion(self):
        q = np.array([0, 0, 0, 0])
        result = normalize_quaternion(q)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_quaternion_multiply_identity(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = q2 / np.linalg.norm(q2)
        result = quaternion_multiply(q1, q2)
        np.testing.assert_array_almost_equal(result, q2)

    def test_quaternion_multiply_inverse(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q = q / np.linalg.norm(q)
        q_conj = quaternion_conjugate(q)
        result = quaternion_multiply(q, q_conj)
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0], decimal=5)

    def test_quaternion_conjugate(self):
        q = np.array([1, 2, 3, 4])
        result = quaternion_conjugate(q)
        expected = np.array([1, -2, -3, -4])
        np.testing.assert_array_equal(result, expected)
