"""Tests for validate_gsir.py"""

import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_gsir import (
    ValidationResult,
    GsirValidationReport,
    count_mmcam_images,
    check_output_directories,
    check_file_counts,
    check_image_validity,
    check_content_quality,
    check_environment_map,
    check_metadata,
    validate_gsir_output,
)


class TestValidationResult:
    def test_valid_result_is_truthy(self):
        result = ValidationResult(valid=True, message="OK")
        assert result
        assert bool(result) is True

    def test_invalid_result_is_falsy(self):
        result = ValidationResult(valid=False, message="Failed")
        assert not result
        assert bool(result) is False

    def test_details_default_to_empty_list(self):
        result = ValidationResult(valid=True, message="OK")
        assert result.details == []


class TestGsirValidationReport:
    def test_empty_report_is_valid(self):
        report = GsirValidationReport(project_dir=Path("/test"))
        assert report.valid is True
        assert report.warnings == []

    def test_report_with_passing_checks_is_valid(self):
        report = GsirValidationReport(project_dir=Path("/test"))
        report.add_check("test1", ValidationResult(valid=True, message="OK"))
        report.add_check("test2", ValidationResult(valid=True, message="OK"))
        assert report.valid is True

    def test_report_with_failing_check_is_invalid(self):
        report = GsirValidationReport(project_dir=Path("/test"))
        report.add_check("test1", ValidationResult(valid=True, message="OK"))
        report.add_check("test2", ValidationResult(valid=False, message="Failed"))
        assert report.valid is False

    def test_warnings_lists_failing_checks(self):
        report = GsirValidationReport(project_dir=Path("/test"))
        report.add_check("test1", ValidationResult(valid=True, message="OK"))
        report.add_check("test2", ValidationResult(valid=False, message="Missing files"))
        warnings = report.warnings
        assert len(warnings) == 1
        assert "test2" in warnings[0]
        assert "Missing files" in warnings[0]

    def test_summary_includes_all_checks(self):
        report = GsirValidationReport(project_dir=Path("/test/project"))
        report.add_check("check1", ValidationResult(valid=True, message="Passed"))
        report.add_check("check2", ValidationResult(valid=False, message="Failed"))
        summary = report.summary()
        assert "project" in summary
        assert "PASS" in summary
        assert "FAIL" in summary
        assert "1/2" in summary


class TestCountMmcamImages:
    def test_returns_zero_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert count_mmcam_images(Path(tmpdir)) == 0

    def test_reads_image_count_from_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_bin = Path(tmpdir) / "images.bin"
            with open(images_bin, "wb") as f:
                f.write(struct.pack("<Q", 42))
                f.write(b"\x00" * 100)
            assert count_mmcam_images(Path(tmpdir)) == 42


class TestCheckOutputDirectories:
    def test_all_directories_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "materials").mkdir()
            (output_dir / "normals").mkdir()
            (output_dir / "depth_gsir").mkdir()

            result = check_output_directories(output_dir)
            assert result.valid is True
            assert "present" in result.message.lower()

    def test_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "materials").mkdir()

            result = check_output_directories(output_dir)
            assert result.valid is False
            assert "normals" in result.message or "depth_gsir" in result.message

    def test_no_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_output_directories(Path(tmpdir))
            assert result.valid is False


class TestCheckFileCounts:
    def test_correct_file_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            for dirname in ["materials", "normals", "depth_gsir"]:
                d = output_dir / dirname
                d.mkdir()
                for i in range(5):
                    (d / f"frame_{i:04d}.png").touch()

            result = check_file_counts(output_dir, expected_count=5)
            assert result.valid is True

    def test_mismatched_file_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            materials = output_dir / "materials"
            materials.mkdir()
            for i in range(3):
                (materials / f"frame_{i:04d}.png").touch()

            normals = output_dir / "normals"
            normals.mkdir()

            depth = output_dir / "depth_gsir"
            depth.mkdir()

            result = check_file_counts(output_dir, expected_count=5)
            assert result.valid is False

    def test_empty_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            for dirname in ["materials", "normals", "depth_gsir"]:
                (output_dir / dirname).mkdir()

            result = check_file_counts(output_dir, expected_count=5)
            assert result.valid is False
            assert "no PNG files" in str(result.details)


class TestCheckImageValidity:
    def test_skips_when_pil_unavailable(self):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = check_image_validity(Path(tmpdir))
                assert result.valid is True
                assert "skipped" in result.message.lower() or "PIL" in str(result.details)

    def test_valid_images(self):
        pytest.importorskip("PIL")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            materials = output_dir / "materials"
            materials.mkdir()

            for i in range(3):
                img = Image.new("RGB", (100, 100), color=(128, 128, 128))
                img.save(materials / f"frame_{i:04d}.png")

            result = check_image_validity(output_dir, sample_size=3)
            assert result.valid is True

    def test_corrupt_image(self):
        pytest.importorskip("PIL")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            materials = output_dir / "materials"
            materials.mkdir()

            corrupt_file = materials / "frame_0001.png"
            corrupt_file.write_bytes(b"not a valid png file")

            result = check_image_validity(output_dir, sample_size=3)
            assert result.valid is False


class TestCheckContentQuality:
    def test_skips_when_dependencies_unavailable(self):
        with patch.dict("sys.modules", {"PIL": None, "numpy": None}):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = check_content_quality(Path(tmpdir))
                assert result.valid is True

    def test_detects_all_black_image(self):
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            materials = output_dir / "materials"
            materials.mkdir()

            black_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
            black_img.save(materials / "frame_0001.png")

            result = check_content_quality(output_dir, sample_size=1)
            assert result.valid is False
            assert "black" in str(result.details).lower()

    def test_detects_all_white_image(self):
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            materials = output_dir / "materials"
            materials.mkdir()

            white_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
            white_img.save(materials / "frame_0001.png")

            result = check_content_quality(output_dir, sample_size=1)
            assert result.valid is False
            assert "white" in str(result.details).lower()

    def test_accepts_normal_content(self):
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        from PIL import Image
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            materials = output_dir / "materials"
            materials.mkdir()

            arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(materials / "frame_0001.png")

            result = check_content_quality(output_dir, sample_size=1)
            assert result.valid is True


class TestCheckEnvironmentMap:
    def test_missing_environment_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_environment_map(Path(tmpdir))
            assert result.valid is False
            assert "not found" in result.message.lower()

    def test_valid_environment_map(self):
        pytest.importorskip("PIL")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            env_map = output_dir / "environment.png"
            img = Image.new("RGB", (512, 256), color=(128, 128, 200))
            img.save(env_map)

            result = check_environment_map(output_dir)
            assert result.valid is True

    def test_empty_environment_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            env_map = output_dir / "environment.png"
            env_map.touch()

            result = check_environment_map(output_dir)
            assert result.valid is False


class TestCheckMetadata:
    def test_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_metadata(Path(tmpdir))
            assert result.valid is False
            assert "not found" in result.message.lower()

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metadata_file = output_dir / "gsir_metadata.json"
            metadata_file.write_text("{ invalid json }")

            result = check_metadata(output_dir)
            assert result.valid is False
            assert "json" in result.message.lower()

    def test_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metadata_file = output_dir / "gsir_metadata.json"
            metadata_file.write_text('{"source": "gs-ir"}')

            result = check_metadata(output_dir)
            assert result.valid is False
            assert "missing" in result.message.lower()

    def test_valid_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            (output_dir / "materials").mkdir()
            (output_dir / "normals").mkdir()
            (output_dir / "depth_gsir").mkdir()
            (output_dir / "environment.png").touch()

            metadata = {
                "source": "gs-ir",
                "checkpoint": "/path/to/checkpoint.pth",
                "iteration": 35000,
                "outputs": {
                    "materials": "materials/",
                    "normals": "normals/",
                    "depth": "depth_gsir/",
                    "environment": "environment.png",
                },
            }
            metadata_file = output_dir / "gsir_metadata.json"
            metadata_file.write_text(json.dumps(metadata))

            result = check_metadata(output_dir)
            assert result.valid is True

    def test_inconsistent_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            metadata = {
                "source": "gs-ir",
                "checkpoint": "/path/to/checkpoint.pth",
                "iteration": 35000,
                "outputs": {
                    "materials": "materials/",
                    "normals": "normals/",
                    "depth": "depth_gsir/",
                    "environment": "environment.png",
                },
            }
            metadata_file = output_dir / "gsir_metadata.json"
            metadata_file.write_text(json.dumps(metadata))

            result = check_metadata(output_dir)
            assert result.valid is False
            assert "inconsistent" in result.message.lower()


class TestValidateGsirOutput:
    def test_full_validation_on_empty_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            (project_dir / "camera").mkdir()

            report = validate_gsir_output(project_dir)
            assert report.valid is False

    def test_full_validation_on_valid_project(self):
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        from PIL import Image
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            output_dir = project_dir / "camera"

            for dirname in ["materials", "normals", "depth_gsir"]:
                d = output_dir / dirname
                d.mkdir(parents=True)
                for i in range(5):
                    arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
                    img = Image.fromarray(arr)
                    img.save(d / f"frame_{i:04d}.png")

            env_img = Image.new("RGB", (512, 256), color=(128, 128, 200))
            env_img.save(output_dir / "environment.png")

            colmap_undistorted = project_dir / "mmcam" / "undistorted" / "sparse" / "0"
            colmap_undistorted.mkdir(parents=True)
            with open(colmap_undistorted / "images.bin", "wb") as f:
                f.write(struct.pack("<Q", 5))
                f.write(b"\x00" * 100)

            metadata = {
                "source": "gs-ir",
                "checkpoint": "/path/to/checkpoint.pth",
                "iteration": 35000,
                "outputs": {
                    "materials": "materials/",
                    "normals": "normals/",
                    "depth": "depth_gsir/",
                    "environment": "environment.png",
                },
            }
            (output_dir / "gsir_metadata.json").write_text(json.dumps(metadata))

            report = validate_gsir_output(project_dir)
            assert report.valid is True
