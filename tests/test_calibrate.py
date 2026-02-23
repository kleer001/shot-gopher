"""Tests for calibrate.py - Camera trajectory alignment."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from calibrate import (
    apply_transform_to_trajectory,
    compare_trajectories,
    extract_positions,
    extract_rotations,
    load_alignment_transform,
    load_trajectory,
    rotation_geodesic_distance,
    save_alignment_transform,
    save_trajectory,
    umeyama_alignment,
)
from transforms import euler_to_rotation_matrix


def _make_camera_matrix(
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix from translation and Euler angles."""
    R = euler_to_rotation_matrix(np.array(euler), "xyz")
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = [tx, ty, tz]
    return m


class TestExtractPositions:
    def test_extracts_translation(self) -> None:
        matrices = [_make_camera_matrix(1, 2, 3), _make_camera_matrix(4, 5, 6)]
        positions = extract_positions(matrices)
        np.testing.assert_array_almost_equal(positions, [[1, 2, 3], [4, 5, 6]])

    def test_single_frame(self) -> None:
        matrices = [_make_camera_matrix(10, 20, 30)]
        positions = extract_positions(matrices)
        assert positions.shape == (1, 3)
        np.testing.assert_array_almost_equal(positions[0], [10, 20, 30])


class TestExtractRotations:
    def test_extracts_rotation(self) -> None:
        m = _make_camera_matrix(euler=(45, 0, 0))
        rotations = extract_rotations([m])
        assert len(rotations) == 1
        assert rotations[0].shape == (3, 3)
        np.testing.assert_array_almost_equal(rotations[0], m[:3, :3])


class TestRotationGeodesicDistance:
    def test_identical_rotations(self) -> None:
        R = euler_to_rotation_matrix(np.array([30, 45, 60]), "xyz")
        assert rotation_geodesic_distance(R, R) == pytest.approx(0.0, abs=1e-6)

    def test_known_angle(self) -> None:
        R1 = np.eye(3)
        R2 = euler_to_rotation_matrix(np.array([0, 0, 90]), "xyz")
        assert rotation_geodesic_distance(R1, R2) == pytest.approx(90.0, abs=0.1)

    def test_symmetric(self) -> None:
        R1 = euler_to_rotation_matrix(np.array([10, 20, 30]), "xyz")
        R2 = euler_to_rotation_matrix(np.array([40, 50, 60]), "xyz")
        assert rotation_geodesic_distance(R1, R2) == pytest.approx(
            rotation_geodesic_distance(R2, R1), abs=1e-6
        )


class TestUmeyamaAlignment:
    def test_identity_alignment(self) -> None:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        R, t, s = umeyama_alignment(points, points)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=5)
        np.testing.assert_array_almost_equal(t, [0, 0, 0], decimal=5)
        assert s == pytest.approx(1.0, abs=1e-5)

    def test_pure_translation(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        offset = np.array([5, 10, 15])
        target = source + offset
        R, t, s = umeyama_alignment(source, target)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=4)
        np.testing.assert_array_almost_equal(t, offset, decimal=4)
        assert s == pytest.approx(1.0, abs=1e-4)

    def test_pure_scale(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        target = source * 3.0
        R, t, s = umeyama_alignment(source, target)
        assert s == pytest.approx(3.0, abs=1e-4)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=4)

    def test_rotation_90deg_z(self) -> None:
        source = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=np.float64)
        R_true = euler_to_rotation_matrix(np.array([0, 0, 90]), "xyz")
        target = (R_true @ source.T).T
        R, t, s = umeyama_alignment(source, target)
        np.testing.assert_array_almost_equal(R, R_true, decimal=4)
        assert s == pytest.approx(1.0, abs=1e-4)

    def test_combined_transform(self) -> None:
        source = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=np.float64,
        )
        R_true = euler_to_rotation_matrix(np.array([0, 0, 45]), "xyz")
        s_true = 2.5
        t_true = np.array([10, 20, 30])
        target = s_true * (R_true @ source.T).T + t_true
        R, t, s = umeyama_alignment(source, target)
        assert s == pytest.approx(s_true, abs=1e-3)
        np.testing.assert_array_almost_equal(t, t_true, decimal=2)
        np.testing.assert_array_almost_equal(R, R_true, decimal=3)

    def test_too_few_points_raises(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 3"):
            umeyama_alignment(source, source)

    def test_shape_mismatch_raises(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        target = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            umeyama_alignment(source, target)


class TestApplyTransform:
    def test_identity_transform(self) -> None:
        matrices = [_make_camera_matrix(1, 2, 3, (10, 20, 30))]
        aligned = apply_transform_to_trajectory(matrices, np.eye(3), np.zeros(3), 1.0)
        np.testing.assert_array_almost_equal(aligned[0], matrices[0])

    def test_translation_only(self) -> None:
        matrices = [_make_camera_matrix(0, 0, 0)]
        t = np.array([5, 10, 15])
        aligned = apply_transform_to_trajectory(matrices, np.eye(3), t, 1.0)
        np.testing.assert_array_almost_equal(aligned[0][:3, 3], t)

    def test_scale_only(self) -> None:
        matrices = [_make_camera_matrix(1, 2, 3)]
        aligned = apply_transform_to_trajectory(matrices, np.eye(3), np.zeros(3), 2.0)
        np.testing.assert_array_almost_equal(aligned[0][:3, 3], [2, 4, 6])

    def test_roundtrip_with_umeyama(self) -> None:
        source_matrices = [
            _make_camera_matrix(0, 0, 0),
            _make_camera_matrix(1, 0, 0),
            _make_camera_matrix(0, 1, 0),
            _make_camera_matrix(0, 0, 1),
        ]
        R_true = euler_to_rotation_matrix(np.array([0, 0, 30]), "xyz")
        s_true = 1.5
        t_true = np.array([5, 10, 15])
        target_matrices = apply_transform_to_trajectory(
            source_matrices, R_true, t_true, s_true
        )
        src_pos = extract_positions(source_matrices)
        tgt_pos = extract_positions(target_matrices)
        R, t, s = umeyama_alignment(src_pos, tgt_pos)
        assert s == pytest.approx(s_true, abs=1e-4)
        np.testing.assert_array_almost_equal(R, R_true, decimal=4)
        np.testing.assert_array_almost_equal(t, t_true, decimal=3)


class TestCompareTrajectories:
    def test_identical_trajectories(self) -> None:
        matrices = [
            _make_camera_matrix(0, 0, 0),
            _make_camera_matrix(1, 0, 0),
            _make_camera_matrix(2, 0, 0),
        ]
        metrics = compare_trajectories(matrices, matrices)
        assert metrics["mean_position_error"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mean_rotation_error"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["n_frames"] == 3
        assert metrics["scale_ratio"] == pytest.approx(1.0, abs=1e-6)

    def test_offset_trajectories(self) -> None:
        source = [_make_camera_matrix(0, 0, 0), _make_camera_matrix(1, 0, 0)]
        target = [_make_camera_matrix(0, 0, 10), _make_camera_matrix(1, 0, 10)]
        metrics = compare_trajectories(source, target)
        assert metrics["mean_position_error"] == pytest.approx(10.0, abs=1e-6)
        assert metrics["mean_rotation_error"] == pytest.approx(0.0, abs=1e-6)

    def test_different_lengths_uses_min(self) -> None:
        source = [_make_camera_matrix(i, 0, 0) for i in range(5)]
        target = [_make_camera_matrix(i, 0, 0) for i in range(3)]
        metrics = compare_trajectories(source, target)
        assert metrics["n_frames"] == 3

    def test_scale_ratio(self) -> None:
        source = [_make_camera_matrix(i, 0, 0) for i in range(4)]
        target = [_make_camera_matrix(i * 2, 0, 0) for i in range(4)]
        metrics = compare_trajectories(source, target)
        assert metrics["scale_ratio"] == pytest.approx(2.0, abs=1e-6)


class TestLoadSaveTrajectory:
    def test_roundtrip(self, tmp_path: Path) -> None:
        matrices = [_make_camera_matrix(1, 2, 3, (10, 20, 30))]
        intrinsics = {"fx": 1000, "fy": 1000, "cx": 960, "cy": 540}
        save_trajectory(matrices, intrinsics, tmp_path)
        loaded_matrices, loaded_intrinsics = load_trajectory(tmp_path)
        assert len(loaded_matrices) == 1
        np.testing.assert_array_almost_equal(loaded_matrices[0], matrices[0], decimal=6)
        assert loaded_intrinsics["fx"] == 1000

    def test_missing_extrinsics_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="extrinsics"):
            load_trajectory(tmp_path)

    def test_missing_intrinsics_returns_empty(self, tmp_path: Path) -> None:
        extrinsics = [np.eye(4).tolist()]
        with open(tmp_path / "extrinsics.json", "w") as f:
            json.dump(extrinsics, f)
        _, intrinsics = load_trajectory(tmp_path)
        assert intrinsics == {}


class TestSaveLoadAlignmentTransform:
    def test_roundtrip(self, tmp_path: Path) -> None:
        R = euler_to_rotation_matrix(np.array([10, 20, 30]), "xyz")
        t = np.array([1.0, 2.0, 3.0])
        s = 2.5
        path = tmp_path / "transform.json"
        save_alignment_transform(R, t, s, path)
        R_loaded, t_loaded, s_loaded = load_alignment_transform(path)
        np.testing.assert_array_almost_equal(R_loaded, R)
        np.testing.assert_array_almost_equal(t_loaded, t)
        assert s_loaded == pytest.approx(s)

    def test_includes_metrics(self, tmp_path: Path) -> None:
        path = tmp_path / "transform.json"
        metrics = {"mean_position_error": 0.5, "n_frames": 10}
        save_alignment_transform(np.eye(3), np.zeros(3), 1.0, path, pre_metrics=metrics)
        with open(path) as f:
            data = json.load(f)
        assert "pre_alignment" in data
        assert data["pre_alignment"]["mean_position_error"] == 0.5
