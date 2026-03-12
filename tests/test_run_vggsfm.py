"""Tests for run_vggsfm.py"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_vggsfm import (
    check_vggsfm_available,
    diagnose_vggsfm_environment,
    prepare_vggsfm_scene,
    _find_mask_sources,
    _prepare_masks,
    _extract_frame_number,
    _max_batch_frames,
    _vram_gb,
    VGGSFM_CONDA_ENV,
    VGGSFM_INSTALL_DIR,
)


@pytest.fixture()
def project_dir(tmp_path):
    """Create a minimal project directory with frames."""
    frames_dir = tmp_path / "source" / "frames"
    frames_dir.mkdir(parents=True)
    for i in range(5):
        frame = frames_dir / f"frame_{i:04d}.png"
        frame.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return tmp_path


class TestCheckVggsfmAvailable:
    def test_missing_install_dir(self):
        with patch("run_vggsfm.VGGSFM_INSTALL_DIR", Path("/nonexistent")):
            assert check_vggsfm_available() is False

    def test_missing_demo_py(self, tmp_path):
        with patch("run_vggsfm.VGGSFM_INSTALL_DIR", tmp_path):
            assert check_vggsfm_available() is False

    def test_missing_conda_env(self, tmp_path):
        (tmp_path / "demo.py").touch()
        with patch("run_vggsfm.VGGSFM_INSTALL_DIR", tmp_path):
            with patch("run_vggsfm._find_conda", return_value="conda"):
                mock_result = MagicMock(returncode=0, stdout="base\nsome_other_env\n")
                with patch("subprocess.run", return_value=mock_result):
                    assert check_vggsfm_available() is False

    def test_all_present(self, tmp_path):
        (tmp_path / "demo.py").touch()
        with patch("run_vggsfm.VGGSFM_INSTALL_DIR", tmp_path):
            with patch("run_vggsfm._find_conda", return_value="conda"):
                mock_result = MagicMock(
                    returncode=0,
                    stdout=f"base\n{VGGSFM_CONDA_ENV}\n",
                )
                with patch("subprocess.run", return_value=mock_result):
                    assert check_vggsfm_available() is True


class TestDiagnoseEnvironment:
    def test_diagnose_empty(self, tmp_path):
        with patch("run_vggsfm.VGGSFM_INSTALL_DIR", tmp_path):
            with patch("run_vggsfm._find_conda", return_value=None):
                info = diagnose_vggsfm_environment()
                assert info["vggsfm_available"] is False
                assert info["conda_env_exists"] is False

    def test_diagnose_verbose(self, tmp_path, capsys):
        (tmp_path / "demo.py").touch()
        (tmp_path / "video_demo.py").touch()
        with patch("run_vggsfm.VGGSFM_INSTALL_DIR", tmp_path):
            with patch("run_vggsfm._find_conda", return_value=None):
                diagnose_vggsfm_environment(verbose=True)
                captured = capsys.readouterr()
                assert "DIAG:" in captured.out
                assert "demo.py exists: True" in captured.out


class TestFindMaskSources:
    def test_no_masks(self, project_dir):
        assert _find_mask_sources(project_dir) == []

    def test_matte_numbered_dirs(self, project_dir):
        matte_dir = project_dir / "matte"
        person_00 = matte_dir / "person_00"
        person_00.mkdir(parents=True)
        (person_00 / "frame_0000.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        result = _find_mask_sources(project_dir)
        assert len(result) == 1
        assert result[0].name == "person_00"

    def test_multiple_matte_dirs(self, project_dir):
        matte_dir = project_dir / "matte"
        for name in ["person_00", "person_01"]:
            d = matte_dir / name
            d.mkdir(parents=True)
            (d / "frame_0000.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        result = _find_mask_sources(project_dir)
        assert len(result) == 2

    def test_roto_mask_dir(self, project_dir):
        mask_dir = project_dir / "roto" / "mask"
        mask_dir.mkdir(parents=True)
        (mask_dir / "frame_0000.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        result = _find_mask_sources(project_dir)
        assert len(result) == 1
        assert result[0].name == "mask"

    def test_roto_person_dir(self, project_dir):
        person_dir = project_dir / "roto" / "person"
        person_dir.mkdir(parents=True)
        (person_dir / "frame_0000.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        result = _find_mask_sources(project_dir)
        assert len(result) == 1
        assert result[0].name == "person"

    def test_matte_takes_priority_over_roto(self, project_dir):
        matte_dir = project_dir / "matte" / "person_00"
        matte_dir.mkdir(parents=True)
        (matte_dir / "frame_0000.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        roto_dir = project_dir / "roto" / "person"
        roto_dir.mkdir(parents=True)
        (roto_dir / "frame_0000.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        result = _find_mask_sources(project_dir)
        assert len(result) == 1
        assert "matte" in str(result[0])


class TestPrepareVggsfmScene:
    def test_symlinks_frames(self, project_dir):
        scene_dir = project_dir / "vggsfm_work"
        total, linked = prepare_vggsfm_scene(project_dir, scene_dir, use_masks=False)

        assert total == 5
        assert linked == 5
        images_dir = scene_dir / "images"
        assert images_dir.exists()
        linked_files = sorted(images_dir.iterdir())
        assert len(linked_files) == 5
        assert all(f.is_symlink() for f in linked_files)

    def test_subsample_step(self, project_dir):
        scene_dir = project_dir / "vggsfm_work"
        total, linked = prepare_vggsfm_scene(
            project_dir, scene_dir, use_masks=False, subsample_step=2,
        )

        assert total == 5
        assert linked == 3
        images_dir = scene_dir / "images"
        assert len(list(images_dir.iterdir())) == 3

    def test_no_masks_dir_when_disabled(self, project_dir):
        scene_dir = project_dir / "vggsfm_work"
        prepare_vggsfm_scene(project_dir, scene_dir, use_masks=False)
        assert not (scene_dir / "masks").exists()

    def test_no_masks_dir_when_none_available(self, project_dir):
        scene_dir = project_dir / "vggsfm_work"
        prepare_vggsfm_scene(project_dir, scene_dir, use_masks=True)
        assert not (scene_dir / "masks").exists()

    def test_masks_prepared_when_available(self, project_dir):
        from PIL import Image

        roto_dir = project_dir / "roto" / "person"
        roto_dir.mkdir(parents=True)
        for i in range(5):
            mask = Image.new("L", (10, 10), 255)
            mask.save(roto_dir / f"frame_{i:04d}.png")

        scene_dir = project_dir / "vggsfm_work"
        prepare_vggsfm_scene(project_dir, scene_dir, use_masks=True)

        masks_dir = scene_dir / "masks"
        assert masks_dir.exists()
        assert len(list(masks_dir.glob("*.png"))) == 5

    def test_cleans_existing_scene_dir(self, project_dir):
        scene_dir = project_dir / "vggsfm_work"
        images_dir = scene_dir / "images"
        images_dir.mkdir(parents=True)
        (images_dir / "stale_file.txt").touch()

        prepare_vggsfm_scene(project_dir, scene_dir, use_masks=False)
        assert not (images_dir / "stale_file.txt").exists()
        assert len(list(images_dir.glob("*.png"))) == 5


class TestExtractFrameNumber:
    def test_frame_4digit(self):
        assert _extract_frame_number("frame_0001.png") == 1

    def test_matte_5digit(self):
        assert _extract_frame_number("matte_00001.png") == 1

    def test_mask_trailing_underscore(self):
        assert _extract_frame_number("mask_00001_.png") == 1

    def test_person_instance(self):
        assert _extract_frame_number("person_00_00024_.png") == 24

    def test_no_digits(self):
        assert _extract_frame_number("nodigits.png") == -1


class TestPrepareMasks:
    def test_single_source_different_naming(self, tmp_path):
        from PIL import Image

        src_dir = tmp_path / "masks_src"
        src_dir.mkdir()
        frames = []
        for i in range(1, 4):
            mask = Image.new("L", (10, 10), 128)
            mask.save(src_dir / f"matte_{i:05d}.png")
            frame_path = tmp_path / f"frame_{i:04d}.png"
            frame_path.touch()
            frames.append(frame_path)

        out_dir = tmp_path / "masks_out"
        _prepare_masks([src_dir], frames, out_dir)

        assert out_dir.exists()
        output_files = sorted(out_dir.glob("*.png"))
        assert len(output_files) == 3
        assert output_files[0].name == "frame_0001.png"

    def test_multi_source_unions_masks(self, tmp_path):
        from PIL import Image

        src1 = tmp_path / "person_00"
        src2 = tmp_path / "person_01"
        src1.mkdir()
        src2.mkdir()

        frames = []
        for i in range(1, 3):
            mask1 = Image.new("L", (10, 10), 100)
            mask1.save(src1 / f"matte_{i:05d}.png")

            mask2 = Image.new("L", (10, 10), 200)
            mask2.save(src2 / f"matte_{i:05d}.png")

            frame_path = tmp_path / f"frame_{i:04d}.png"
            frame_path.touch()
            frames.append(frame_path)

        out_dir = tmp_path / "masks_out"
        _prepare_masks([src1, src2], frames, out_dir)

        result_mask = Image.open(out_dir / "frame_0001.png")
        arr = np.array(result_mask)
        assert arr.max() == 200

    def test_subsampled_frames_match_by_number(self, tmp_path):
        from PIL import Image

        src_dir = tmp_path / "masks_src"
        src_dir.mkdir()

        all_frames = []
        for i in range(1, 11):
            mask = Image.new("L", (10, 10), i * 25)
            mask.save(src_dir / f"mask_{i:05d}_.png")
            frame_path = tmp_path / f"frame_{i:04d}.png"
            frame_path.touch()
            all_frames.append(frame_path)

        subsampled = all_frames[::3]
        out_dir = tmp_path / "masks_out"
        _prepare_masks([src_dir], subsampled, out_dir)

        assert len(list(out_dir.glob("*.png"))) == len(subsampled)
        result = Image.open(out_dir / "frame_0004.png")
        arr = np.array(result)
        assert arr.flat[0] == 100


class TestBatchFrameLimits:
    def test_max_batch_frames_returns_positive(self):
        assert _max_batch_frames() > 0

    def test_vram_gb_returns_float(self):
        result = _vram_gb()
        assert isinstance(result, float)


class TestPipelineConfig:
    def test_mmcam_engine_default(self):
        from pipeline_config import PipelineConfig

        config = PipelineConfig()
        assert config.mmcam_engine == "vggsfm"

    def test_mmcam_engine_vggsfm(self):
        from pipeline_config import PipelineConfig

        config = PipelineConfig(mmcam_engine="vggsfm")
        assert config.mmcam_engine == "vggsfm"
