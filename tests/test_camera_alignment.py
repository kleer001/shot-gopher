"""Tests for camera_alignment.py — body-aligned camera from SMPL orients."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from camera_alignment import (
    align_colmap_trajectory,
    camera_rotation_from_orients,
    camera_translation_from_transl,
    compute_aligned_camera,
    compute_chordal_mean_rotation,
    compute_world_transform,
    smooth_camera_trajectory,
)
from export_mocap import transform_meshes_to_world
from transforms import (
    axis_angle_to_rotation_matrix,
    axis_angle_to_rotation_matrix_batch,
)


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


class TestChordalMeanRotation:
    def test_identical_rotations_return_same(self):
        R = axis_angle_to_rotation_matrix(np.array([0.5, -0.3, 0.8]))
        Rs = np.tile(R, (20, 1, 1))
        result = compute_chordal_mean_rotation(Rs)
        np.testing.assert_array_almost_equal(result, R, decimal=10)

    def test_identity_inputs_return_identity(self):
        Rs = np.tile(np.eye(3), (10, 1, 1))
        result = compute_chordal_mean_rotation(Rs)
        np.testing.assert_array_almost_equal(result, np.eye(3), decimal=10)

    def test_result_is_valid_rotation(self):
        rng = np.random.default_rng(42)
        Rs = np.zeros((20, 3, 3))
        for i in range(20):
            Rs[i] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.5)
        result = compute_chordal_mean_rotation(Rs)
        np.testing.assert_array_almost_equal(result @ result.T, np.eye(3), decimal=10)
        assert abs(np.linalg.det(result) - 1.0) < 1e-10

    def test_noise_averages_out(self):
        rng = np.random.default_rng(77)
        R_true = axis_angle_to_rotation_matrix(np.array([0.3, -0.5, 0.2]))
        Rs = np.zeros((200, 3, 3))
        for i in range(200):
            noise = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.05)
            Rs[i] = noise @ R_true
        result = compute_chordal_mean_rotation(Rs)
        angle_err = np.arccos(np.clip((np.trace(result @ R_true.T) - 1) / 2, -1, 1))
        assert angle_err < np.radians(2.0)


class TestWorldTransform:
    def test_identity_when_same_frame(self):
        rng = np.random.default_rng(42)
        n = 20
        c2w = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            c2w[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.2)
            c2w[i, :3, 3] = rng.standard_normal(3)
        wt = compute_world_transform(c2w, c2w)
        np.testing.assert_array_almost_equal(wt["R"], np.eye(3), decimal=8)
        assert abs(wt["s"] - 1.0) < 1e-6
        np.testing.assert_array_almost_equal(wt["t"], np.zeros(3), decimal=8)

    def test_pure_scale(self):
        n = 15
        src = np.tile(np.eye(4), (n, 1, 1))
        src[:, :3, 3] = np.linspace([0, 0, 0], [1, 0, 0], n)
        tgt = np.tile(np.eye(4), (n, 1, 1))
        tgt[:, :3, 3] = np.linspace([0, 0, 0], [3, 0, 0], n)
        wt = compute_world_transform(src, tgt)
        assert abs(wt["s"] - 3.0) < 0.01
        np.testing.assert_array_almost_equal(wt["R"], np.eye(3), decimal=5)

    def test_pure_translation_offset(self):
        n = 10
        src = np.tile(np.eye(4), (n, 1, 1))
        src[:, :3, 3] = np.array([0.0, 0.0, 0.0])
        tgt = np.tile(np.eye(4), (n, 1, 1))
        tgt[:, :3, 3] = np.array([5.0, 10.0, 15.0])
        wt = compute_world_transform(src, tgt)
        np.testing.assert_array_almost_equal(wt["t"], [5.0, 10.0, 15.0], decimal=5)

    def test_transform_maps_positions_correctly(self):
        rng = np.random.default_rng(99)
        n = 30
        R_true = axis_angle_to_rotation_matrix(np.array([0.3, -0.2, 0.5]))
        s_true = 2.5
        t_true = np.array([1.0, -3.0, 7.0])

        src = np.tile(np.eye(4), (n, 1, 1))
        tgt = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            R_i = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.3)
            pos_i = rng.standard_normal(3) * 2
            src[i, :3, :3] = R_i
            src[i, :3, 3] = pos_i
            tgt[i, :3, :3] = R_true @ R_i
            tgt[i, :3, 3] = s_true * R_true @ pos_i + t_true

        wt = compute_world_transform(src, tgt)
        np.testing.assert_array_almost_equal(wt["R"], R_true, decimal=5)
        assert abs(wt["s"] - s_true) < 0.01
        np.testing.assert_array_almost_equal(wt["t"], t_true, decimal=2)

    def test_result_has_valid_rotation(self):
        rng = np.random.default_rng(55)
        n = 20
        src = np.tile(np.eye(4), (n, 1, 1))
        tgt = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            src[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.3)
            src[i, :3, 3] = rng.standard_normal(3)
            tgt[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.3)
            tgt[i, :3, 3] = rng.standard_normal(3)
        wt = compute_world_transform(src, tgt)
        R = wt["R"]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestAlignColmapTrajectory:
    def test_first_frame_orientation_matches_gvhmr(self):
        """After alignment, first frame rotation should match GVHMR's."""
        rng = np.random.default_rng(42)
        n = 20
        R_offset = axis_angle_to_rotation_matrix(np.array([0.5, -0.3, 0.8]))

        colmap = np.tile(np.eye(4), (n, 1, 1))
        gvhmr = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            R_colmap = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.1)
            colmap[i, :3, :3] = R_colmap
            colmap[i, :3, 3] = rng.standard_normal(3)
            gvhmr[i, :3, :3] = R_offset @ R_colmap
            gvhmr[i, :3, 3] = rng.standard_normal(3)

        result = align_colmap_trajectory(colmap, gvhmr)
        np.testing.assert_array_almost_equal(
            result[0, :3, :3], gvhmr[0, :3, :3], decimal=10
        )

    def test_preserves_relative_rotation_angle(self):
        """Frame-to-frame rotation angles from COLMAP must be preserved."""
        rng = np.random.default_rng(7)
        n = 20
        colmap = np.tile(np.eye(4), (n, 1, 1))
        gvhmr = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            colmap[i, :3, :3] = axis_angle_to_rotation_matrix(
                np.array([0.01 * i, 0.005 * i, -0.003 * i])
            )
            colmap[i, :3, 3] = rng.standard_normal(3) * 0.5
            gvhmr[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.3)
            gvhmr[i, :3, 3] = rng.standard_normal(3)

        result = align_colmap_trajectory(colmap, gvhmr)

        for i in range(1, n):
            delta_colmap = colmap[i, :3, :3] @ colmap[i - 1, :3, :3].T
            delta_result = result[i, :3, :3] @ result[i - 1, :3, :3].T
            angle_colmap = np.arccos(np.clip((np.trace(delta_colmap) - 1) / 2, -1, 1))
            angle_result = np.arccos(np.clip((np.trace(delta_result) - 1) / 2, -1, 1))
            np.testing.assert_almost_equal(angle_colmap, angle_result, decimal=10)

    def test_identity_when_already_aligned(self):
        rng = np.random.default_rng(11)
        n = 20
        c2w = np.tile(np.eye(4), (n, 1, 1))
        c2w[:, :3, 3] = rng.standard_normal((n, 3))
        for i in range(n):
            c2w[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.2)

        result = align_colmap_trajectory(c2w, c2w)
        np.testing.assert_array_almost_equal(result, c2w, decimal=8)

    def test_pure_scale(self):
        n = 15
        src = np.tile(np.eye(4), (n, 1, 1))
        src[:, :3, 3] = np.linspace([0, 0, 0], [1, 0, 0], n)

        tgt = np.tile(np.eye(4), (n, 1, 1))
        tgt[:, :3, 3] = np.linspace([0, 0, 0], [3, 0, 0], n)

        result = align_colmap_trajectory(src, tgt)
        np.testing.assert_array_almost_equal(
            result[:, :3, 3], tgt[:, :3, 3], decimal=10
        )

    def test_static_camera_offset(self):
        n = 10
        src = np.tile(np.eye(4), (n, 1, 1))
        src[:, :3, 3] = np.array([5.0, 5.0, 5.0])

        tgt = np.tile(np.eye(4), (n, 1, 1))
        tgt[:, :3, 3] = np.array([1.0, 2.0, 3.0])

        result = align_colmap_trajectory(src, tgt)
        np.testing.assert_array_almost_equal(
            result[:, :3, 3], tgt[:, :3, 3], decimal=10
        )

    def test_output_rotations_are_valid(self):
        rng = np.random.default_rng(99)
        n = 20
        src = np.tile(np.eye(4), (n, 1, 1))
        tgt = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            src[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.3)
            src[i, :3, 3] = rng.standard_normal(3)
            tgt[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.3)
            tgt[i, :3, 3] = rng.standard_normal(3)

        result = align_colmap_trajectory(src, tgt)
        for i in range(n):
            R = result[i, :3, :3]
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_complementary_filter_reduces_jitter(self):
        """Noisy GVHMR + smooth COLMAP -> result smoother than GVHMR."""
        rng = np.random.default_rng(42)
        n = 100

        colmap = np.tile(np.eye(4), (n, 1, 1))
        gvhmr = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            t = i / n
            colmap[i, :3, 3] = [t * 5, 0, 0]
            gvhmr[i, :3, 3] = [t * 5, 0, 0]
        gvhmr[:, :3, 3] += rng.standard_normal((n, 3)) * 0.1

        result = align_colmap_trajectory(colmap, gvhmr)

        gvhmr_jitter = np.max(np.abs(np.diff(gvhmr[:, :3, 3], axis=0)))
        result_jitter = np.max(np.abs(np.diff(result[:, :3, 3], axis=0)))
        assert result_jitter < gvhmr_jitter * 0.5


class TestSmoothCameraTrajectory:
    def test_constant_trajectory_unchanged(self):
        n = 20
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, :3, 3] = np.array([1.0, 2.0, 3.0])
        result = smooth_camera_trajectory(T)
        np.testing.assert_array_almost_equal(result, T, decimal=10)

    def test_reduces_noise(self):
        rng = np.random.default_rng(55)
        n = 50
        T = np.tile(np.eye(4), (n, 1, 1))
        clean_trans = np.linspace([0, 0, 0], [1, 2, 3], n)
        noisy_trans = clean_trans + rng.standard_normal((n, 3)) * 0.05
        T[:, :3, 3] = noisy_trans

        result = smooth_camera_trajectory(T, translation_only=True)
        smoothed_trans = result[:, :3, 3]

        noise_before = np.max(np.abs(np.diff(noisy_trans, axis=0)), axis=0)
        noise_after = np.max(np.abs(np.diff(smoothed_trans, axis=0)), axis=0)
        assert np.all(noise_after < noise_before)

    def test_translation_only_preserves_rotation(self):
        rng = np.random.default_rng(88)
        n = 20
        T = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            aa = rng.standard_normal(3) * 0.3
            T[i, :3, :3] = axis_angle_to_rotation_matrix(aa)
        T[:, :3, 3] = rng.standard_normal((n, 3))

        result = smooth_camera_trajectory(T, translation_only=True)
        for i in range(n):
            np.testing.assert_array_almost_equal(
                result[i, :3, :3], T[i, :3, :3], decimal=10
            )

    def test_short_sequence_no_crash(self):
        T = np.tile(np.eye(4), (3, 1, 1))
        T[:, :3, 3] = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
        result = smooth_camera_trajectory(T)
        assert result.shape == (3, 4, 4)

    def test_very_short_sequence_returns_copy(self):
        T = np.tile(np.eye(4), (2, 1, 1))
        result = smooth_camera_trajectory(T)
        assert result.shape == (2, 4, 4)
        np.testing.assert_array_almost_equal(result, T)


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

    def test_metadata_source_without_colmap(self, static_gvhmr_data):
        pelvis = np.zeros(3)
        result = compute_aligned_camera(static_gvhmr_data, 1920, 1080, pelvis)
        assert result["metadata"]["source"] == "gvhmr_aligned"
        assert result["metadata"]["rotation_source"] == "smpl_orient"
        assert result["metadata"]["translation_source"] == "smpl_transl"
        assert "world_transform" not in result

    def test_metadata_source_with_colmap(self, static_gvhmr_data):
        n = 10
        pelvis = np.zeros(3)
        colmap_c2w = np.tile(np.eye(4), (n, 1, 1))
        result = compute_aligned_camera(
            static_gvhmr_data, 1920, 1080, pelvis, colmap_extrinsics=colmap_c2w,
        )
        assert result["metadata"]["rotation_source"] == "mmcam_direct"
        assert result["metadata"]["translation_source"] == "mmcam_direct"
        assert "world_transform" in result
        wt = result["world_transform"]
        assert "R" in wt and "s" in wt and "t" in wt

    def test_colmap_path_produces_valid_extrinsics(self):
        rng = np.random.default_rng(123)
        n = 30
        orient_w = rng.standard_normal((n, 3)) * 0.3
        orient_c = rng.standard_normal((n, 3)) * 0.3
        transl_w = rng.standard_normal((n, 3))
        transl_c = rng.standard_normal((n, 3))
        K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])

        gvhmr_data = {
            "smpl_params_global": {"global_orient": orient_w, "transl": transl_w},
            "smpl_params_incam": {"global_orient": orient_c, "transl": transl_c},
            "K_fullimg": K,
        }

        colmap_c2w = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            aa = rng.standard_normal(3) * 0.2
            colmap_c2w[i, :3, :3] = axis_angle_to_rotation_matrix(aa)
            colmap_c2w[i, :3, 3] = rng.standard_normal(3)

        pelvis = rng.standard_normal(3) * 0.3

        result = compute_aligned_camera(
            gvhmr_data, 1920, 1080, pelvis, colmap_extrinsics=colmap_c2w,
        )

        assert len(result["extrinsics"]) == n
        for mat in result["extrinsics"]:
            mat = np.array(mat)
            R = mat[:3, :3]
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)
            assert abs(np.linalg.det(R) - 1.0) < 1e-5

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


class TestTransformMeshesToWorld:
    def test_identity_c2w_preserves_vertices(self):
        verts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        faces = np.array([[0, 1, 0]])
        meshes = [(verts, faces)] * 3
        c2w = np.tile(np.eye(4), (3, 1, 1))
        result = transform_meshes_to_world(meshes, c2w)
        for v_world, f in result:
            np.testing.assert_array_almost_equal(v_world, verts)
            np.testing.assert_array_equal(f, faces)

    def test_pure_translation(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        faces = np.array([[0, 1, 0]])
        c2w = np.eye(4).reshape(1, 4, 4).copy()
        c2w[0, :3, 3] = [10.0, 20.0, 30.0]
        result = transform_meshes_to_world([(verts, faces)], c2w)
        expected = verts + np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result[0][0], expected)

    def test_rotation_applied(self):
        verts = np.array([[1.0, 0.0, 0.0]])
        faces = np.array([[0, 0, 0]])
        c2w = np.eye(4).reshape(1, 4, 4).copy()
        c2w[0, :3, :3] = axis_angle_to_rotation_matrix(np.array([0.0, 0.0, np.pi / 2]))
        result = transform_meshes_to_world([(verts, faces)], c2w)
        np.testing.assert_array_almost_equal(result[0][0], [[0.0, 1.0, 0.0]], decimal=10)

    def test_frame_count_mismatch_raises(self):
        meshes = [(np.zeros((3, 3)), np.zeros((1, 3)))] * 5
        c2w = np.tile(np.eye(4), (3, 1, 1))
        with pytest.raises(ValueError, match="Frame count mismatch"):
            transform_meshes_to_world(meshes, c2w)

    def test_roundtrip_c2w_then_w2c(self):
        """Transforming cam->world->cam should recover original vertices."""
        rng = np.random.default_rng(42)
        n = 5
        verts_cam = rng.standard_normal((10, 3)).astype(np.float64)
        faces = np.array([[0, 1, 2]])
        meshes = [(verts_cam.copy(), faces)] * n

        c2w = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
        for i in range(n):
            c2w[i, :3, :3] = axis_angle_to_rotation_matrix(rng.standard_normal(3) * 0.5)
            c2w[i, :3, 3] = rng.standard_normal(3) * 2

        world_meshes = transform_meshes_to_world(meshes, c2w)

        w2c = np.zeros_like(c2w)
        for i in range(n):
            R = c2w[i, :3, :3]
            t = c2w[i, :3, 3]
            w2c[i, :3, :3] = R.T
            w2c[i, :3, 3] = -R.T @ t
            w2c[i, 3, 3] = 1.0

        recovered = transform_meshes_to_world(world_meshes, w2c)
        for i in range(n):
            np.testing.assert_array_almost_equal(recovered[i][0], verts_cam, decimal=10)
