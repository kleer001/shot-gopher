"""Tests for camera_alignment.py — body-aligned camera from SMPL orients."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from camera_alignment import (
    camera_rotation_from_orients,
    camera_translation_from_transl,
    compute_aligned_camera,
)
from transforms import axis_angle_to_rotation_matrix_batch


class TestCameraRotationFromOrients:
    def test_identical_orients_give_identity(self):
        aa = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        R_w2c = camera_rotation_from_orients(aa, aa)
        for i in range(len(aa)):
            np.testing.assert_array_almost_equal(R_w2c[i], np.eye(3), decimal=10)

    def test_known_rotation(self):
        orient_w = np.array([[0.0, 0.0, np.pi / 2]])
        orient_c = np.array([[0.0, 0.0, 0.0]])
        R_w2c = camera_rotation_from_orients(orient_w, orient_c)
        R_w = axis_angle_to_rotation_matrix_batch(orient_w)
        expected = R_w[0].T
        np.testing.assert_array_almost_equal(R_w2c[0], expected, decimal=10)

    def test_result_is_valid_rotation(self):
        rng = np.random.default_rng(42)
        orient_w = rng.standard_normal((10, 3))
        orient_c = rng.standard_normal((10, 3))
        R_w2c = camera_rotation_from_orients(orient_w, orient_c)
        for i in range(10):
            np.testing.assert_array_almost_equal(
                R_w2c[i] @ R_w2c[i].T, np.eye(3), decimal=10
            )
            assert abs(np.linalg.det(R_w2c[i]) - 1.0) < 1e-10

    def test_formula_R_c_times_R_w_transpose(self):
        rng = np.random.default_rng(99)
        orient_w = rng.standard_normal((5, 3))
        orient_c = rng.standard_normal((5, 3))
        R_w = axis_angle_to_rotation_matrix_batch(orient_w)
        R_c = axis_angle_to_rotation_matrix_batch(orient_c)
        expected = np.einsum("nij,nkj->nik", R_c, R_w)
        result = camera_rotation_from_orients(orient_w, orient_c)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)


class TestCameraTranslationFromTransl:
    def test_identity_rotation_gives_difference(self):
        R_w2c = np.tile(np.eye(3), (3, 1, 1))
        transl_global = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        transl_incam = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]])
        pelvis = np.array([0.0, -0.37, 0.0])
        t_w2c = camera_translation_from_transl(R_w2c, transl_global, transl_incam, pelvis)
        expected = transl_incam - transl_global
        np.testing.assert_array_almost_equal(t_w2c, expected, decimal=10)

    def test_zero_translations_give_zero_with_identity_rotation(self):
        R_w2c = np.tile(np.eye(3), (5, 1, 1))
        zeros = np.zeros((5, 3))
        pelvis = np.array([0.003, -0.37, 0.008])
        t_w2c = camera_translation_from_transl(R_w2c, zeros, zeros, pelvis)
        np.testing.assert_array_almost_equal(t_w2c, zeros, decimal=10)

    def test_pelvis_correction_with_180_degree_rotation(self):
        R_z180 = np.array([[[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]])
        pelvis = np.array([0.0, -0.37, 0.0])
        zeros = np.zeros((1, 3))
        t_w2c = camera_translation_from_transl(R_z180, zeros, zeros, pelvis)
        expected = np.array([[0.0, -0.74, 0.0]])
        np.testing.assert_array_almost_equal(t_w2c, expected, decimal=10)

    def test_zero_pelvis_matches_naive_formula(self):
        rng = np.random.default_rng(77)
        n = 5
        orient_w = rng.standard_normal((n, 3)) * 0.5
        orient_c = rng.standard_normal((n, 3)) * 0.5
        R_w2c = camera_rotation_from_orients(orient_w, orient_c)
        transl_w = rng.standard_normal((n, 3))
        transl_c = rng.standard_normal((n, 3))
        pelvis = np.zeros(3)
        t_w2c = camera_translation_from_transl(R_w2c, transl_w, transl_c, pelvis)
        naive = transl_c - np.einsum("nij,nj->ni", R_w2c, transl_w)
        np.testing.assert_array_almost_equal(t_w2c, naive, decimal=10)


class TestComputeAlignedCamera:
    @pytest.fixture()
    def static_gvhmr_data(self):
        n = 10
        orient = np.zeros((n, 3))
        transl = np.zeros((n, 3))
        K = np.array([[500.0, 0, 960.0], [0, 500.0, 540.0], [0, 0, 1]])
        return {
            "smpl_params_global": {"global_orient": orient, "transl": transl},
            "smpl_params_incam": {"global_orient": orient, "transl": transl},
            "K_fullimg": K,
        }

    def test_static_camera_gives_identity_extrinsics(self, static_gvhmr_data):
        pelvis = np.array([0.003, -0.37, 0.008])
        result = compute_aligned_camera(static_gvhmr_data, 1920, 1080, pelvis)
        for mat in result["extrinsics"]:
            np.testing.assert_array_almost_equal(mat, np.eye(4), decimal=10)

    def test_intrinsics_extracted(self, static_gvhmr_data):
        pelvis = np.zeros(3)
        result = compute_aligned_camera(static_gvhmr_data, 1920, 1080, pelvis)
        intr = result["intrinsics"]
        assert intr["fx"] == 500.0
        assert intr["fy"] == 500.0
        assert intr["cx"] == 960.0
        assert intr["cy"] == 540.0
        assert intr["width"] == 1920
        assert intr["height"] == 1080

    def test_metadata_source(self, static_gvhmr_data):
        pelvis = np.zeros(3)
        result = compute_aligned_camera(static_gvhmr_data, 1920, 1080, pelvis)
        assert result["metadata"]["source"] == "gvhmr_aligned"

    def test_roundtrip_w2c_to_c2w(self):
        rng = np.random.default_rng(123)
        n = 8
        orient_w = rng.standard_normal((n, 3)) * 0.5
        orient_c = rng.standard_normal((n, 3)) * 0.5
        transl_w = rng.standard_normal((n, 3))
        transl_c = rng.standard_normal((n, 3))
        pelvis = rng.standard_normal(3) * 0.3
        K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])

        gvhmr_data = {
            "smpl_params_global": {"global_orient": orient_w, "transl": transl_w},
            "smpl_params_incam": {"global_orient": orient_c, "transl": transl_c},
            "K_fullimg": K,
        }
        result = compute_aligned_camera(gvhmr_data, 1920, 1080, pelvis)

        R_w2c = camera_rotation_from_orients(orient_w, orient_c)
        t_w2c = camera_translation_from_transl(R_w2c, transl_w, transl_c, pelvis)

        for i, c2w in enumerate(result["extrinsics"]):
            T_c2w = np.array(c2w)
            T_w2c = np.eye(4)
            T_w2c[:3, :3] = R_w2c[i]
            T_w2c[:3, 3] = t_w2c[i]
            product = T_w2c @ T_c2w
            np.testing.assert_array_almost_equal(product, np.eye(4), decimal=10)

    def test_3d_K_fullimg_handled(self):
        n = 3
        orient = np.zeros((n, 3))
        transl = np.zeros((n, 3))
        pelvis = np.zeros(3)
        K_3d = np.array([[[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]]] * n)
        gvhmr_data = {
            "smpl_params_global": {"global_orient": orient, "transl": transl},
            "smpl_params_incam": {"global_orient": orient, "transl": transl},
            "K_fullimg": K_3d,
        }
        result = compute_aligned_camera(gvhmr_data, 640, 480, pelvis)
        assert result["intrinsics"]["fx"] == 800.0
