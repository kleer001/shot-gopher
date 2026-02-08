"""Tests for the dense pipeline stage.

Tests cover:
- COLMAP binary array reader (depth/normal maps)
- Sparse PLY export
- Depth map EXR conversion
- Normal map EXR conversion
- Stage dependency injection (sanitize_stages)
- Stage registration
"""

import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_matchmove_camera import (
    read_colmap_array,
    export_sparse_ply,
)
from pipeline_constants import STAGES, STAGE_ORDER, STAGES_REQUIRING_FRAMES
from stage_runners import STAGE_HANDLERS


class TestDenseStageRegistration:
    def test_dense_in_stages_dict(self):
        assert "dense" in STAGES

    def test_dense_in_stage_order(self):
        assert "dense" in STAGE_ORDER

    def test_dense_after_matchmove_camera_in_order(self):
        mmcam_idx = STAGE_ORDER.index("matchmove_camera")
        dense_idx = STAGE_ORDER.index("dense")
        assert dense_idx == mmcam_idx + 1

    def test_dense_requires_frames(self):
        assert "dense" in STAGES_REQUIRING_FRAMES

    def test_dense_in_stage_handlers(self):
        assert "dense" in STAGE_HANDLERS

    def test_dense_handler_is_callable(self):
        assert callable(STAGE_HANDLERS["dense"])


class TestSanitizeStagesDenseDependency:
    def test_dense_injects_matchmove_camera(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_pipeline import sanitize_stages

        result = sanitize_stages(["dense"])
        assert "matchmove_camera" in result
        assert "ingest" in result
        assert "dense" in result

    def test_dense_preserves_order(self):
        from run_pipeline import sanitize_stages

        result = sanitize_stages(["dense"])
        mmcam_idx = result.index("matchmove_camera")
        dense_idx = result.index("dense")
        assert mmcam_idx < dense_idx

    def test_dense_with_matchmove_no_duplicate(self):
        from run_pipeline import sanitize_stages

        result = sanitize_stages(["matchmove_camera", "dense"])
        assert result.count("matchmove_camera") == 1
        assert result.count("dense") == 1

    def test_matchmove_alone_no_dense(self):
        from run_pipeline import sanitize_stages

        result = sanitize_stages(["matchmove_camera"])
        assert "dense" not in result


class TestReadColmapArray:
    def _write_colmap_bin(self, path: Path, data: np.ndarray) -> None:
        """Write a COLMAP-format binary array file."""
        if data.ndim == 2:
            height, width = data.shape
            channels = 1
        else:
            height, width, channels = data.shape

        header = f"{width}&{height}&{channels}&"
        transposed = data.reshape(height, width, channels).transpose(1, 0, 2)
        flat = transposed.flatten(order="F")

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(struct.pack(f"<{len(flat)}f", *flat))

    def test_read_single_channel_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            path = Path(tmpdir) / "test.geometric.bin"
            self._write_colmap_bin(path, data)

            result = read_colmap_array(path)
            np.testing.assert_array_almost_equal(result, data)

    def test_read_three_channel_normals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(4, 6, 3).astype(np.float32)
            path = Path(tmpdir) / "test.geometric.bin"
            self._write_colmap_bin(path, data)

            result = read_colmap_array(path)
            assert result.shape == (4, 6, 3)
            np.testing.assert_array_almost_equal(result, data)

    def test_read_larger_depth_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(100, 200).astype(np.float32)
            path = Path(tmpdir) / "depth.geometric.bin"
            self._write_colmap_bin(path, data)

            result = read_colmap_array(path)
            assert result.shape == (100, 200)
            np.testing.assert_array_almost_equal(result, data)


class TestExportSparsePly:
    def test_no_points3d_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sparse_dir = Path(tmpdir) / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            output = Path(tmpdir) / "output.ply"

            result = export_sparse_ply(sparse_dir, output)
            assert result is False

    def test_calls_model_converter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sparse_dir = Path(tmpdir) / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            (sparse_dir / "points3D.bin").touch()
            output = Path(tmpdir) / "output.ply"

            with patch("run_matchmove_camera.run_colmap_command") as mock_cmd:
                mock_cmd.return_value = None
                output.touch()

                result = export_sparse_ply(sparse_dir, output)

                mock_cmd.assert_called_once()
                call_args = mock_cmd.call_args
                assert call_args[0][0] == "model_converter"
                assert call_args[0][1]["output_type"] == "PLY"


class TestConvertDepthMapsToExr:
    def test_no_depth_maps_dir_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dense_path = Path(tmpdir) / "dense"
            output_dir = Path(tmpdir) / "depth"

            from run_matchmove_camera import convert_depth_maps_to_exr
            result = convert_depth_maps_to_exr(dense_path, output_dir)
            assert result == 0

    def test_no_geometric_files_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            depth_maps_dir = Path(tmpdir) / "dense" / "stereo" / "depth_maps"
            depth_maps_dir.mkdir(parents=True)
            output_dir = Path(tmpdir) / "depth"

            from run_matchmove_camera import convert_depth_maps_to_exr
            result = convert_depth_maps_to_exr(Path(tmpdir) / "dense", output_dir)
            assert result == 0


class TestConvertNormalMapsToExr:
    def test_no_normal_maps_dir_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dense_path = Path(tmpdir) / "dense"
            output_dir = Path(tmpdir) / "normals"

            from run_matchmove_camera import convert_normal_maps_to_exr
            result = convert_normal_maps_to_exr(dense_path, output_dir)
            assert result == 0


class TestPipelineConfigNoDenseFields:
    def test_no_mmcam_dense_field(self):
        from pipeline_config import PipelineConfig
        config = PipelineConfig()
        assert not hasattr(config, "mmcam_dense")

    def test_no_mmcam_mesh_field(self):
        from pipeline_config import PipelineConfig
        config = PipelineConfig()
        assert not hasattr(config, "mmcam_mesh")
