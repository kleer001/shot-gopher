"""Tests for GVHMR integration in run_mocap.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_mocap import (
    colmap_intrinsics_to_focal_mm,
    detect_static_camera,
    find_or_create_video,
)


class TestColmapIntrinsicsToFocalMm:
    def test_valid_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intrinsics_path = Path(tmpdir) / "intrinsics.json"
            intrinsics = {"fx": 1000, "fy": 1000, "width": 1920, "height": 1080}
            with open(intrinsics_path, "w") as f:
                json.dump(intrinsics, f)

            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path, sensor_width_mm=36.0)

            assert focal_mm is not None
            expected = 1000 * 36.0 / 1920
            assert abs(focal_mm - expected) < 0.01

    def test_missing_file(self):
        focal_mm = colmap_intrinsics_to_focal_mm(Path("/nonexistent/intrinsics.json"))
        assert focal_mm is None

    def test_missing_fx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intrinsics_path = Path(tmpdir) / "intrinsics.json"
            intrinsics = {"fy": 1000, "width": 1920}
            with open(intrinsics_path, "w") as f:
                json.dump(intrinsics, f)

            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path)
            assert focal_mm is None

    def test_default_sensor_width(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intrinsics_path = Path(tmpdir) / "intrinsics.json"
            intrinsics = {"fx": 1000, "width": 1920}
            with open(intrinsics_path, "w") as f:
                json.dump(intrinsics, f)

            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path)

            assert focal_mm is not None
            expected = 1000 * 36.0 / 1920
            assert abs(focal_mm - expected) < 0.01


class TestDetectStaticCamera:
    def test_static_camera(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extrinsics_path = Path(tmpdir) / "extrinsics.json"
            identity = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
            extrinsics = [identity, identity, identity]
            with open(extrinsics_path, "w") as f:
                json.dump(extrinsics, f)

            is_static = detect_static_camera(extrinsics_path)
            assert is_static is True

    def test_moving_camera(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extrinsics_path = Path(tmpdir) / "extrinsics.json"
            frame1 = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
            frame2 = [
                [1, 0, 0, 1.0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
            extrinsics = [frame1, frame2]
            with open(extrinsics_path, "w") as f:
                json.dump(extrinsics, f)

            is_static = detect_static_camera(extrinsics_path, threshold_meters=0.01)
            assert is_static is False

    def test_missing_file(self):
        is_static = detect_static_camera(Path("/nonexistent/extrinsics.json"))
        assert is_static is False

    def test_single_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extrinsics_path = Path(tmpdir) / "extrinsics.json"
            identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            extrinsics = [identity]
            with open(extrinsics_path, "w") as f:
                json.dump(extrinsics, f)

            is_static = detect_static_camera(extrinsics_path)
            assert is_static is False


class TestFindOrCreateVideo:
    def test_find_existing_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            source_dir = project_dir / "source"
            source_dir.mkdir()

            video_path = source_dir / "test_video.mp4"
            video_path.touch()

            result = find_or_create_video(project_dir)
            assert result == video_path

    def test_skip_underscore_prefixed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            source_dir = project_dir / "source"
            source_dir.mkdir()

            gvhmr_video = source_dir / "_gvhmr_input.mp4"
            gvhmr_video.touch()
            real_video = source_dir / "footage.mp4"
            real_video.touch()

            result = find_or_create_video(project_dir)
            assert result == real_video

    def test_no_video_no_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            source_dir = project_dir / "source"
            source_dir.mkdir()

            result = find_or_create_video(project_dir)
            assert result is None


class TestGvhmrOutputConversion:
    def test_conversion_smpl_params_global(self):
        pytest.importorskip("numpy")
        import numpy as np
        import pickle

        from run_mocap import convert_gvhmr_to_wham_format

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            n_frames = 10
            gvhmr_data = {
                'smpl_params_global': {
                    'body_pose': np.zeros((n_frames, 63)),
                    'global_orient': np.zeros((n_frames, 3)),
                    'transl': np.zeros((n_frames, 3)),
                    'betas': np.zeros(10),
                }
            }

            gvhmr_output = gvhmr_dir / "output.pkl"
            with open(gvhmr_output, 'wb') as f:
                pickle.dump(gvhmr_data, f)

            wham_output = Path(tmpdir) / "wham" / "motion.pkl"

            success = convert_gvhmr_to_wham_format(gvhmr_dir, wham_output)
            assert success is True
            assert wham_output.exists()

            with open(wham_output, 'rb') as f:
                wham_data = pickle.load(f)

            assert 'poses' in wham_data
            assert 'trans' in wham_data
            assert 'betas' in wham_data
            assert wham_data['poses'].shape == (n_frames, 72)
            assert wham_data['trans'].shape == (n_frames, 3)
            assert wham_data['betas'].shape == (10,)

    def test_conversion_no_files(self):
        from run_mocap import convert_gvhmr_to_wham_format

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            wham_output = Path(tmpdir) / "wham" / "motion.pkl"

            success = convert_gvhmr_to_wham_format(gvhmr_dir, wham_output)
            assert success is False

    def test_conversion_nonexistent_dir(self):
        """Test conversion when GVHMR output directory doesn't exist."""
        from run_mocap import convert_gvhmr_to_wham_format

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "nonexistent"
            wham_output = Path(tmpdir) / "wham" / "motion.pkl"

            success = convert_gvhmr_to_wham_format(gvhmr_dir, wham_output)
            assert success is False

    def test_conversion_direct_global_orient(self):
        """Test conversion when global_orient is at top level (not under smpl_params_global)."""
        pytest.importorskip("numpy")
        import numpy as np
        import pickle

        from run_mocap import convert_gvhmr_to_wham_format

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            n_frames = 5
            gvhmr_data = {
                'body_pose': np.zeros((n_frames, 63)),
                'global_orient': np.zeros((n_frames, 3)),
                'transl': np.zeros((n_frames, 3)),
                'betas': np.zeros(10),
            }

            gvhmr_output = gvhmr_dir / "global_output.pkl"
            with open(gvhmr_output, 'wb') as f:
                pickle.dump(gvhmr_data, f)

            wham_output = Path(tmpdir) / "wham" / "motion.pkl"

            success = convert_gvhmr_to_wham_format(gvhmr_dir, wham_output)
            assert success is True
            assert wham_output.exists()

            with open(wham_output, 'rb') as f:
                wham_data = pickle.load(f)

            assert wham_data['poses'].shape == (n_frames, 72)

    def test_conversion_short_body_pose(self):
        """Test conversion when body_pose has fewer than 63 elements."""
        pytest.importorskip("numpy")
        import numpy as np
        import pickle

        from run_mocap import convert_gvhmr_to_wham_format

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            n_frames = 5
            gvhmr_data = {
                'smpl_params_global': {
                    'body_pose': np.zeros((n_frames, 45)),
                    'global_orient': np.zeros((n_frames, 3)),
                    'transl': np.zeros((n_frames, 3)),
                    'betas': np.zeros(10),
                }
            }

            gvhmr_output = gvhmr_dir / "output.pkl"
            with open(gvhmr_output, 'wb') as f:
                pickle.dump(gvhmr_data, f)

            wham_output = Path(tmpdir) / "wham" / "motion.pkl"

            success = convert_gvhmr_to_wham_format(gvhmr_dir, wham_output)
            assert success is True

            with open(wham_output, 'rb') as f:
                wham_data = pickle.load(f)

            assert wham_data['poses'].shape == (n_frames, 72)

    def test_conversion_multi_person(self):
        """Test conversion with multiple person directories."""
        pytest.importorskip("numpy")
        import numpy as np
        import pickle

        from run_mocap import convert_gvhmr_to_wham_format

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            for person_idx in range(3):
                person_dir = gvhmr_dir / f"person_{person_idx}"
                person_dir.mkdir()

                n_frames = 10 + person_idx
                gvhmr_data = {
                    'smpl_params_global': {
                        'body_pose': np.ones((n_frames, 63)) * person_idx,
                        'global_orient': np.zeros((n_frames, 3)),
                        'transl': np.ones((n_frames, 3)) * person_idx,
                        'betas': np.zeros(10),
                    }
                }

                gvhmr_output = person_dir / "output.pkl"
                with open(gvhmr_output, 'wb') as f:
                    pickle.dump(gvhmr_data, f)

            wham_output = Path(tmpdir) / "wham" / "motion.pkl"

            success = convert_gvhmr_to_wham_format(gvhmr_dir, wham_output)
            assert success is True

            assert wham_output.exists()
            assert (wham_output.parent / "motion_person_0.pkl").exists()
            assert (wham_output.parent / "motion_person_1.pkl").exists()
            assert (wham_output.parent / "motion_person_2.pkl").exists()

            with open(wham_output, 'rb') as f:
                wham_data = pickle.load(f)
            assert wham_data['poses'].shape == (10, 72)

            with open(wham_output.parent / "motion_person_2.pkl", 'rb') as f:
                person2_data = pickle.load(f)
            assert person2_data['poses'].shape == (12, 72)


class TestRunGvhmrMotionTracking:
    def test_gvhmr_not_installed(self):
        """Test graceful failure when GVHMR is not installed."""
        from run_mocap import run_gvhmr_motion_tracking

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "source").mkdir()

            result = run_gvhmr_motion_tracking(project_dir)
            assert result is False


class TestRunMocapPipeline:
    def test_missing_dependencies(self):
        """Test that pipeline handles missing dependencies gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "source").mkdir()

            from run_mocap import run_mocap_pipeline
            result = run_mocap_pipeline(project_dir, method="auto")
            assert result is False

    def test_method_selection_auto(self):
        """Test that auto method selection works."""
        from run_mocap import run_mocap_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "source").mkdir()

            result = run_mocap_pipeline(project_dir, method="auto")
            assert result is False
