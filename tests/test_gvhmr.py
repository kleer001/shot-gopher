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
    composite_frames_with_matte,
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

    def test_verbose_warning_empty_extrinsics(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            extrinsics_path = Path(tmpdir) / "extrinsics.json"
            with open(extrinsics_path, "w") as f:
                json.dump([], f)

            is_static = detect_static_camera(extrinsics_path, verbose=True)
            assert is_static is False

            captured = capsys.readouterr()
            assert "empty" in captured.out.lower()

    def test_verbose_warning_invalid_json(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            extrinsics_path = Path(tmpdir) / "extrinsics.json"
            with open(extrinsics_path, "w") as f:
                f.write("{not valid json")

            is_static = detect_static_camera(extrinsics_path, verbose=True)
            assert is_static is False

            captured = capsys.readouterr()
            assert "Invalid JSON" in captured.out

    def test_verbose_suppressed(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            extrinsics_path = Path(tmpdir) / "extrinsics.json"
            with open(extrinsics_path, "w") as f:
                json.dump([], f)

            is_static = detect_static_camera(extrinsics_path, verbose=False)
            assert is_static is False

            captured = capsys.readouterr()
            assert captured.out == ""


class TestColmapIntrinsicsWarnings:
    def test_verbose_warning_missing_fx(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            intrinsics_path = Path(tmpdir) / "intrinsics.json"
            with open(intrinsics_path, "w") as f:
                json.dump({"fy": 1000}, f)

            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path, verbose=True)
            assert focal_mm is None

            captured = capsys.readouterr()
            assert "fx" in captured.out.lower()

    def test_verbose_warning_invalid_json(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            intrinsics_path = Path(tmpdir) / "intrinsics.json"
            with open(intrinsics_path, "w") as f:
                f.write("{not valid}")

            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path, verbose=True)
            assert focal_mm is None

            captured = capsys.readouterr()
            assert "Invalid JSON" in captured.out

    def test_verbose_warning_dir_exists_file_missing(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            camera_dir = Path(tmpdir) / "camera"
            camera_dir.mkdir()
            intrinsics_path = camera_dir / "intrinsics.json"

            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path, verbose=True)
            assert focal_mm is None

            captured = capsys.readouterr()
            assert "directory exists" in captured.out.lower()
            assert "intrinsics.json missing" in captured.out

    def test_verbose_suppressed(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            intrinsics_path = Path(tmpdir) / "intrinsics.json"
            with open(intrinsics_path, "w") as f:
                json.dump({"fy": 1000}, f)

            focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path, verbose=False)
            assert focal_mm is None

            captured = capsys.readouterr()
            assert captured.out == ""


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

    def test_find_uppercase_extension(self):
        """Test that uppercase video extensions are found (e.g., .MP4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            source_dir = project_dir / "source"
            source_dir.mkdir()

            video_path = source_dir / "footage.MP4"
            video_path.touch()

            result = find_or_create_video(project_dir)
            assert result == video_path


class TestGvhmrOutputConversion:
    def test_conversion_smpl_params_global(self):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        import numpy as np
        import torch

        from run_mocap import save_motion_output

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

            gvhmr_output = gvhmr_dir / "hmr4d_results.pt"
            torch.save(gvhmr_data, gvhmr_output)

            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is True
            assert motion_output.exists()

            with open(motion_output, 'rb') as f:
                motion_data = pickle.load(f)

            assert 'poses' in motion_data
            assert 'trans' in motion_data
            assert 'betas' in motion_data
            assert motion_data['poses'].shape == (n_frames, 72)
            assert motion_data['trans'].shape == (n_frames, 3)
            assert motion_data['betas'].shape == (10,)

    def test_conversion_no_files(self):
        from run_mocap import save_motion_output

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is False

    def test_conversion_nonexistent_dir(self):
        """Test conversion when GVHMR output directory doesn't exist."""
        from run_mocap import save_motion_output

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "nonexistent"
            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is False

    def test_conversion_empty_body_pose(self):
        """Test conversion fails gracefully with empty body_pose."""
        pytest.importorskip("numpy")
        import numpy as np
        import pickle

        from run_mocap import save_motion_output

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            gvhmr_data = {
                'smpl_params_global': {
                    'body_pose': np.zeros((0, 63)),
                    'global_orient': np.zeros((0, 3)),
                    'transl': np.zeros((0, 3)),
                    'betas': np.zeros(10),
                }
            }

            gvhmr_output = gvhmr_dir / "output.pkl"
            with open(gvhmr_output, 'wb') as f:
                pickle.dump(gvhmr_data, f)

            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is False

    def test_conversion_direct_global_orient(self):
        """Test conversion when global_orient is at top level (not under smpl_params_global)."""
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        import numpy as np
        import torch

        from run_mocap import save_motion_output

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

            gvhmr_output = gvhmr_dir / "hmr4d_results.pt"
            torch.save(gvhmr_data, gvhmr_output)

            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is True
            assert motion_output.exists()

            with open(motion_output, 'rb') as f:
                motion_data = pickle.load(f)

            assert motion_data['poses'].shape == (n_frames, 72)

    def test_conversion_short_body_pose(self):
        """Test conversion when body_pose has fewer than 63 elements."""
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        import numpy as np
        import torch

        from run_mocap import save_motion_output

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

            gvhmr_output = gvhmr_dir / "hmr4d_results.pt"
            torch.save(gvhmr_data, gvhmr_output)

            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is True

            with open(motion_output, 'rb') as f:
                motion_data = pickle.load(f)

            assert motion_data['poses'].shape == (n_frames, 72)

    def test_conversion_multi_person(self):
        """Test conversion with multiple person directories."""
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        import numpy as np
        import torch

        from run_mocap import save_motion_output

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

                gvhmr_output = person_dir / "hmr4d_results.pt"
                torch.save(gvhmr_data, gvhmr_output)

            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is True

            assert motion_output.exists()
            assert (motion_output.parent / "motion_person_0.pkl").exists()
            assert (motion_output.parent / "motion_person_1.pkl").exists()
            assert (motion_output.parent / "motion_person_2.pkl").exists()

            with open(motion_output, 'rb') as f:
                motion_data = pickle.load(f)
            assert motion_data['poses'].shape == (10, 72)

            with open(motion_output.parent / "motion_person_2.pkl", 'rb') as f:
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


class TestFindOrCreateVideoFrameRange:
    """Tests for frame range functionality in find_or_create_video."""

    def test_frame_range_returns_different_path(self):
        """Test that specifying frame range creates a different video path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            source_dir = project_dir / "source"
            frames_dir = source_dir / "frames"
            frames_dir.mkdir(parents=True)

            for i in range(1, 101):
                (frames_dir / f"frame_{i:04d}.png").touch()

            result_full = find_or_create_video(project_dir)
            assert result_full is None or "_gvhmr_input" in str(result_full)

    def test_frame_range_only_start(self):
        """Test specifying only start_frame (should go to end)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            source_dir = project_dir / "source"
            source_dir.mkdir()

            video_path = source_dir / "test.mp4"
            video_path.touch()

            result = find_or_create_video(project_dir, start_frame=50)
            assert result is None or "_trimmed" in str(result) or result == video_path

    def test_frame_range_only_end(self):
        """Test specifying only end_frame (should start from beginning)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            source_dir = project_dir / "source"
            source_dir.mkdir()

            video_path = source_dir / "test.mp4"
            video_path.touch()

            result = find_or_create_video(project_dir, end_frame=100)
            assert result is None or "_trimmed" in str(result) or result == video_path


class TestSaveMotionOutputMultiPerson:
    """Tests for multi-person handling in save_motion_output."""

    def test_multi_person_defaults_to_first(self):
        """Test that multi-person output defaults to person_0 as primary."""
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        import numpy as np
        import torch

        from run_mocap import save_motion_output

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir) / "gvhmr"
            gvhmr_dir.mkdir()

            for person_idx in range(2):
                person_dir = gvhmr_dir / f"person_{person_idx}"
                person_dir.mkdir()

                n_frames = 10 if person_idx == 0 else 20
                gvhmr_data = {
                    'smpl_params_global': {
                        'body_pose': np.zeros((n_frames, 63)),
                        'global_orient': np.zeros((n_frames, 3)),
                        'transl': np.zeros((n_frames, 3)),
                        'betas': np.zeros(10),
                    }
                }

                gvhmr_output = person_dir / "hmr4d_results.pt"
                torch.save(gvhmr_data, gvhmr_output)

            motion_output = Path(tmpdir) / "mocap" / "motion.pkl"

            success = save_motion_output(gvhmr_dir, motion_output)
            assert success is True

            with open(motion_output, 'rb') as f:
                motion_data = pickle.load(f)

            assert motion_data['poses'].shape == (10, 72)


class TestListDetectedPersons:
    """Tests for list_detected_persons function."""

    def test_list_multiple_persons(self):
        """Test listing multiple detected persons."""
        from run_mocap import list_detected_persons

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir)
            (gvhmr_dir / "person_0").mkdir()
            (gvhmr_dir / "person_1").mkdir()
            (gvhmr_dir / "person_2").mkdir()

            detected = list_detected_persons(gvhmr_dir)
            assert detected == ["person_0", "person_1", "person_2"]

    def test_list_single_person(self):
        """Test listing a single detected person."""
        from run_mocap import list_detected_persons

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir)
            (gvhmr_dir / "person_0").mkdir()

            detected = list_detected_persons(gvhmr_dir)
            assert detected == ["person_0"]

    def test_list_no_persons(self):
        """Test listing when no persons detected (other files exist)."""
        from run_mocap import list_detected_persons

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir)
            (gvhmr_dir / "output.pkl").touch()

            detected = list_detected_persons(gvhmr_dir)
            assert detected == []

    def test_list_nonexistent_dir(self):
        """Test listing when directory doesn't exist."""
        from run_mocap import list_detected_persons

        detected = list_detected_persons(Path("/nonexistent/gvhmr"))
        assert detected == []

    def test_list_ignores_non_person_dirs(self):
        """Test that non-person directories are ignored."""
        from run_mocap import list_detected_persons

        with tempfile.TemporaryDirectory() as tmpdir:
            gvhmr_dir = Path(tmpdir)
            (gvhmr_dir / "person_0").mkdir()
            (gvhmr_dir / "person_1").mkdir()
            (gvhmr_dir / "other_dir").mkdir()
            (gvhmr_dir / "output.pkl").touch()

            detected = list_detected_persons(gvhmr_dir)
            assert detected == ["person_0", "person_1"]


class TestCompositeFramesWithMatte:
    """Tests for composite_frames_with_matte function."""

    def test_composite_basic(self):
        """Test basic frame compositing with matte."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        from PIL import Image
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            matte_dir = Path(tmpdir) / "matte"
            output_dir = Path(tmpdir) / "output"
            frames_dir.mkdir()
            matte_dir.mkdir()

            for i in range(3):
                img = Image.new("RGB", (100, 100), color=(255, 128, 64))
                img.save(frames_dir / f"frame_{i+1:04d}.png")

                matte = Image.new("L", (100, 100), color=128)
                matte.save(matte_dir / f"frame_{i+1:04d}.png")

            result = composite_frames_with_matte(frames_dir, matte_dir, output_dir)

            assert len(result) == 3
            assert all(p.exists() for p in result)

            result_img = Image.open(result[0])
            result_array = np.array(result_img)
            assert result_array.shape == (100, 100, 3)
            expected_r = int(255 * 128 / 255)
            assert abs(result_array[50, 50, 0] - expected_r) < 2

    def test_composite_missing_frames(self):
        """Test compositing with missing source frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            matte_dir = Path(tmpdir) / "matte"
            output_dir = Path(tmpdir) / "output"
            matte_dir.mkdir()

            result = composite_frames_with_matte(frames_dir, matte_dir, output_dir)
            assert result == []

    def test_composite_missing_mattes(self):
        """Test compositing with missing matte frames."""
        pytest.importorskip("PIL")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            matte_dir = Path(tmpdir) / "matte"
            output_dir = Path(tmpdir) / "output"
            frames_dir.mkdir()

            img = Image.new("RGB", (100, 100), color=(255, 128, 64))
            img.save(frames_dir / "frame_0001.png")

            result = composite_frames_with_matte(frames_dir, matte_dir, output_dir)
            assert result == []

    def test_composite_frame_count_mismatch(self):
        """Test compositing with different frame counts."""
        pytest.importorskip("PIL")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            matte_dir = Path(tmpdir) / "matte"
            output_dir = Path(tmpdir) / "output"
            frames_dir.mkdir()
            matte_dir.mkdir()

            for i in range(5):
                img = Image.new("RGB", (100, 100), color=(255, 128, 64))
                img.save(frames_dir / f"frame_{i+1:04d}.png")

            for i in range(3):
                matte = Image.new("L", (100, 100), color=255)
                matte.save(matte_dir / f"frame_{i+1:04d}.png")

            result = composite_frames_with_matte(frames_dir, matte_dir, output_dir)
            assert result == []

    def test_composite_with_frame_range(self):
        """Test compositing with start/end frame range."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            matte_dir = Path(tmpdir) / "matte"
            output_dir = Path(tmpdir) / "output"
            frames_dir.mkdir()
            matte_dir.mkdir()

            for i in range(10):
                img = Image.new("RGB", (100, 100), color=(255, 128, 64))
                img.save(frames_dir / f"frame_{i+1:04d}.png")

                matte = Image.new("L", (100, 100), color=255)
                matte.save(matte_dir / f"frame_{i+1:04d}.png")

            result = composite_frames_with_matte(
                frames_dir, matte_dir, output_dir,
                start_frame=3, end_frame=7
            )

            assert len(result) == 5


class TestRunMocapPipeline:
    def test_missing_gvhmr(self):
        """Test that pipeline handles missing GVHMR gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "source").mkdir()

            from run_mocap import run_mocap_pipeline
            result = run_mocap_pipeline(project_dir)
            assert result is False
