"""Tests for matte_utils.py - Matte and mask combination utilities."""

import tempfile
from pathlib import Path
import shutil

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from matte_utils import (
    combine_mattes,
    combine_mask_sequences,
    prepare_roto_for_cleanplate,
)


def create_test_mask(path: Path, value: int, width: int = 100, height: int = 100):
    """Helper to create a test mask image."""
    from PIL import Image
    img = Image.new('L', (width, height), value)
    img.save(path)


class TestCombineMattes:
    def test_combine_empty_list_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            result = combine_mattes([], output_dir)

            assert result is False

    def test_combine_single_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input1"
            input_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            create_test_mask(input_dir / "frame_00001.png", 128)
            create_test_mask(input_dir / "frame_00002.png", 255)

            result = combine_mattes([input_dir], output_dir)

            assert result is True
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 2

    def test_combine_takes_maximum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input1 = tmpdir / "input1"
            input2 = tmpdir / "input2"
            input1.mkdir()
            input2.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            create_test_mask(input1 / "frame_00001.png", 100)
            create_test_mask(input2 / "frame_00001.png", 200)

            result = combine_mattes([input1, input2], output_dir)

            assert result is True

            from PIL import Image
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 1
            img = Image.open(output_files[0])
            pixel = img.getpixel((50, 50))
            assert pixel == 200

    def test_combine_no_png_files_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            result = combine_mattes([input_dir], output_dir)

            assert result is False

    def test_combine_mismatched_frame_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input1 = tmpdir / "input1"
            input2 = tmpdir / "input2"
            input1.mkdir()
            input2.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            create_test_mask(input1 / "frame_00001.png", 100)
            create_test_mask(input1 / "frame_00002.png", 100)
            create_test_mask(input2 / "frame_00001.png", 200)

            result = combine_mattes([input1, input2], output_dir)

            assert result is True
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 2


class TestCombineMaskSequences:
    def test_combine_empty_list_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            count = combine_mask_sequences([], output_dir)

            assert count == 0

    def test_combine_single_sequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_dir = tmpdir / "source"
            source_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            create_test_mask(source_dir / "mask_00001.png", 128)
            create_test_mask(source_dir / "mask_00002.png", 255)

            count = combine_mask_sequences([source_dir], output_dir)

            assert count == 2
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 2

    def test_combine_multiple_sequences_takes_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source1 = tmpdir / "source1"
            source2 = tmpdir / "source2"
            source1.mkdir()
            source2.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            create_test_mask(source1 / "mask_00001.png", 50)
            create_test_mask(source2 / "mask_00001.png", 150)

            count = combine_mask_sequences([source1, source2], output_dir)

            assert count == 1

            from PIL import Image
            output_file = list(output_dir.glob("*.png"))[0]
            img = Image.open(output_file)
            pixel = img.getpixel((50, 50))
            assert pixel == 150

    def test_combine_uses_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_dir = tmpdir / "source"
            source_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            create_test_mask(source_dir / "mask_00001.png", 128)

            count = combine_mask_sequences([source_dir], output_dir, prefix="custom")

            assert count == 1
            output_files = list(output_dir.glob("custom_*.png"))
            assert len(output_files) == 1

    def test_combine_no_source_files_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_dir = tmpdir / "source"
            source_dir.mkdir()
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            count = combine_mask_sequences([source_dir], output_dir)

            assert count == 0


class TestPrepareRotoForCleanplate:
    def test_no_roto_dir_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roto_dir = Path(tmpdir) / "nonexistent_roto"

            success, message = prepare_roto_for_cleanplate(roto_dir)

            assert success is False
            assert "No roto sequences found" in message

    def test_empty_roto_dir_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roto_dir = Path(tmpdir) / "roto"
            roto_dir.mkdir()

            success, message = prepare_roto_for_cleanplate(roto_dir)

            assert success is False
            assert "No roto sequences found" in message

    def test_single_sequence_copies_to_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roto_dir = Path(tmpdir) / "roto"
            roto_dir.mkdir()
            sequence_dir = roto_dir / "person"
            sequence_dir.mkdir()

            create_test_mask(sequence_dir / "mask_00001.png", 255)
            create_test_mask(sequence_dir / "mask_00002.png", 255)

            success, message = prepare_roto_for_cleanplate(roto_dir)

            assert success is True
            assert "person" in message

            root_masks = list(roto_dir.glob("mask_*.png"))
            assert len(root_masks) == 2

    def test_multiple_sequences_uses_combined_if_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roto_dir = Path(tmpdir) / "roto"
            roto_dir.mkdir()
            person_dir = roto_dir / "person"
            person_dir.mkdir()
            combined_dir = roto_dir / "combined"
            combined_dir.mkdir()

            create_test_mask(person_dir / "mask_00001.png", 100)
            create_test_mask(combined_dir / "combined_00001.png", 200)

            success, message = prepare_roto_for_cleanplate(roto_dir)

            assert success is True
            assert "combined" in message

            root_masks = list(roto_dir.glob("mask_*.png"))
            assert len(root_masks) == 1

            from PIL import Image
            img = Image.open(root_masks[0])
            pixel = img.getpixel((50, 50))
            assert pixel == 200

    def test_multiple_sequences_consolidates_when_no_combined(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roto_dir = Path(tmpdir) / "roto"
            roto_dir.mkdir()
            person_dir = roto_dir / "person"
            bg_dir = roto_dir / "background"
            person_dir.mkdir()
            bg_dir.mkdir()

            create_test_mask(person_dir / "mask_00001.png", 100)
            create_test_mask(bg_dir / "mask_00001.png", 50)

            success, message = prepare_roto_for_cleanplate(roto_dir)

            assert success is True
            assert "Consolidated" in message
            assert "2 mask sources" in message

    def test_clears_existing_root_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roto_dir = Path(tmpdir) / "roto"
            roto_dir.mkdir()
            sequence_dir = roto_dir / "person"
            sequence_dir.mkdir()

            create_test_mask(roto_dir / "old_mask.png", 128)
            create_test_mask(sequence_dir / "new_00001.png", 255)

            success, message = prepare_roto_for_cleanplate(roto_dir)

            assert success is True
            assert not (roto_dir / "old_mask.png").exists()

    def test_empty_subdir_not_counted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roto_dir = Path(tmpdir) / "roto"
            roto_dir.mkdir()
            empty_dir = roto_dir / "empty"
            empty_dir.mkdir()
            sequence_dir = roto_dir / "person"
            sequence_dir.mkdir()

            create_test_mask(sequence_dir / "mask_00001.png", 255)

            success, message = prepare_roto_for_cleanplate(roto_dir)

            assert success is True
            assert "person" in message
