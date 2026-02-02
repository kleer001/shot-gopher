"""Tests for run_cleanplate_batched.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_cleanplate_batched import (
    ChunkInfo,
    calculate_chunks,
    count_source_frames,
    generate_chunk_workflow,
    blend_images,
    find_chunk_outputs,
    blend_overlaps,
    load_batch_state,
    save_batch_state,
    load_template_workflow,
    save_chunk_workflow,
)


class TestCalculateChunks:
    """Tests for calculate_chunks() - the core chunking algorithm."""

    def test_single_chunk_when_frames_less_than_batch(self):
        chunks = calculate_chunks(total_frames=5, batch_size=10, overlap=2)
        assert len(chunks) == 1
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 4
        assert chunks[0].frame_count == 5

    def test_single_chunk_when_frames_equal_batch(self):
        chunks = calculate_chunks(total_frames=10, batch_size=10, overlap=2)
        assert len(chunks) == 1
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 9
        assert chunks[0].frame_count == 10

    def test_two_chunks_with_overlap(self):
        chunks = calculate_chunks(total_frames=18, batch_size=10, overlap=2)
        assert len(chunks) == 2
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 9
        assert chunks[1].start_frame == 8
        assert chunks[1].end_frame == 17

    def test_overlap_frames_correct(self):
        chunks = calculate_chunks(total_frames=20, batch_size=10, overlap=2)
        overlap_start = chunks[1].start_frame
        overlap_end = chunks[0].end_frame
        actual_overlap = overlap_end - overlap_start + 1
        assert actual_overlap == 2

    def test_three_chunks_with_overlap(self):
        chunks = calculate_chunks(total_frames=26, batch_size=10, overlap=2)
        assert len(chunks) == 3
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 9
        assert chunks[1].start_frame == 8
        assert chunks[1].end_frame == 17
        assert chunks[2].start_frame == 16
        assert chunks[2].end_frame == 25

    def test_zero_overlap(self):
        chunks = calculate_chunks(total_frames=30, batch_size=10, overlap=0)
        assert len(chunks) == 3
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 9
        assert chunks[1].start_frame == 10
        assert chunks[1].end_frame == 19
        assert chunks[2].start_frame == 20
        assert chunks[2].end_frame == 29

    def test_large_overlap(self):
        chunks = calculate_chunks(total_frames=20, batch_size=10, overlap=5)
        assert len(chunks) == 3
        assert chunks[0].end_frame == 9
        assert chunks[1].start_frame == 5
        assert chunks[1].end_frame == 14
        assert chunks[2].start_frame == 10
        assert chunks[2].end_frame == 19

    def test_remainder_frames(self):
        chunks = calculate_chunks(total_frames=23, batch_size=10, overlap=2)
        last_chunk = chunks[-1]
        assert last_chunk.end_frame == 22
        assert last_chunk.frame_count <= 10

    def test_all_frames_covered(self):
        chunks = calculate_chunks(total_frames=120, batch_size=10, overlap=2)
        covered = set()
        for chunk in chunks:
            for frame in range(chunk.start_frame, chunk.end_frame + 1):
                covered.add(frame)
        assert covered == set(range(120))

    def test_no_chunk_exceeds_batch_size(self):
        chunks = calculate_chunks(total_frames=100, batch_size=10, overlap=2)
        for chunk in chunks:
            assert chunk.frame_count <= 10

    def test_single_frame(self):
        chunks = calculate_chunks(total_frames=1, batch_size=10, overlap=2)
        assert len(chunks) == 1
        assert chunks[0].frame_count == 1
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 0


class TestGenerateChunkWorkflow:
    """Tests for generate_chunk_workflow() - template modification."""

    @pytest.fixture
    def sample_template(self):
        return {
            "nodes": [
                {
                    "id": 1,
                    "type": "VHS_LoadImagesPath",
                    "widgets_values": ["source/frames", 10, 0, 1, "Disabled", 1920, 1080, None, None]
                },
                {
                    "id": 2,
                    "type": "VHS_LoadImagesPath",
                    "widgets_values": ["roto", 10, 0, 1, "Disabled", 1920, 1080, None, None]
                },
                {
                    "id": 5,
                    "type": "SaveImage",
                    "widgets_values": ["cleanplate/chunks/chunk/clean"]
                },
                {
                    "id": 4,
                    "type": "ProPainterInpaint",
                    "widgets_values": [1280, 720, 5, 8, 10, 2, 10, 5, "enable"]
                }
            ]
        }

    def test_updates_load_images_frame_count(self, sample_template):
        chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
        result = generate_chunk_workflow(sample_template, chunk, Path("output"), "clean")

        for node in result["nodes"]:
            if node["type"] == "VHS_LoadImagesPath":
                assert node["widgets_values"][1] == 10

    def test_updates_load_images_skip_frames(self, sample_template):
        chunk = ChunkInfo(index=1, start_frame=8, end_frame=17, frame_count=10)
        result = generate_chunk_workflow(sample_template, chunk, Path("output"), "clean")

        for node in result["nodes"]:
            if node["type"] == "VHS_LoadImagesPath":
                assert node["widgets_values"][2] == 8

    def test_updates_both_load_nodes(self, sample_template):
        chunk = ChunkInfo(index=2, start_frame=16, end_frame=25, frame_count=10)
        result = generate_chunk_workflow(sample_template, chunk, Path("output"), "clean")

        load_nodes = [n for n in result["nodes"] if n["type"] == "VHS_LoadImagesPath"]
        assert len(load_nodes) == 2
        for node in load_nodes:
            assert node["widgets_values"][1] == 10
            assert node["widgets_values"][2] == 16

    def test_updates_save_image_prefix(self, sample_template):
        chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
        result = generate_chunk_workflow(sample_template, chunk, Path("cleanplate/chunks/000_009"), "clean")

        save_node = next(n for n in result["nodes"] if n["type"] == "SaveImage")
        assert save_node["widgets_values"][0] == "clean"

    def test_does_not_mutate_template(self, sample_template):
        original_value = sample_template["nodes"][0]["widgets_values"][1]
        chunk = ChunkInfo(index=0, start_frame=50, end_frame=59, frame_count=10)

        generate_chunk_workflow(sample_template, chunk, Path("output"), "clean")

        assert sample_template["nodes"][0]["widgets_values"][1] == original_value

    def test_preserves_other_nodes(self, sample_template):
        chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
        result = generate_chunk_workflow(sample_template, chunk, Path("output"), "clean")

        propainter = next(n for n in result["nodes"] if n["type"] == "ProPainterInpaint")
        assert propainter["widgets_values"] == [1280, 720, 5, 8, 10, 2, 10, 5, "enable"]

    def test_handles_empty_widgets(self):
        template = {
            "nodes": [
                {"type": "VHS_LoadImagesPath", "widgets_values": []},
                {"type": "SaveImage", "widgets_values": ["prefix"]}
            ]
        }
        chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
        result = generate_chunk_workflow(template, chunk, Path("output"), "clean")
        assert result is not None

    def test_handles_short_widgets(self):
        template = {
            "nodes": [
                {"type": "VHS_LoadImagesPath", "widgets_values": ["path", 5]},
                {"type": "SaveImage", "widgets_values": ["prefix"]}
            ]
        }
        chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
        result = generate_chunk_workflow(template, chunk, Path("output"), "clean")
        load_node = next(n for n in result["nodes"] if n["type"] == "VHS_LoadImagesPath")
        assert load_node["widgets_values"] == ["path", 5]


class TestBlendImages:
    """Tests for blend_images() - PIL blending."""

    @pytest.fixture
    def red_image(self):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        return img

    @pytest.fixture
    def blue_image(self):
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        return img

    def test_weight_zero_returns_first_image(self, red_image, blue_image):
        result = blend_images(red_image, blue_image, 0.0)
        pixel = result.getpixel((50, 50))
        assert pixel == (255, 0, 0)

    def test_weight_one_returns_second_image(self, red_image, blue_image):
        result = blend_images(red_image, blue_image, 1.0)
        pixel = result.getpixel((50, 50))
        assert pixel == (0, 0, 255)

    def test_weight_half_blends_equally(self, red_image, blue_image):
        result = blend_images(red_image, blue_image, 0.5)
        pixel = result.getpixel((50, 50))
        assert pixel[0] == 127 or pixel[0] == 128
        assert pixel[2] == 127 or pixel[2] == 128

class TestCountSourceFrames:
    """Tests for count_source_frames() - filesystem operations."""

    def test_counts_png_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            for i in range(10):
                (frames_dir / f"frame_{i:04d}.png").touch()

            assert count_source_frames(project_dir) == 10

    def test_counts_mixed_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            for i in range(3):
                (frames_dir / f"frame_{i:04d}.png").touch()
            for i in range(3, 7):
                (frames_dir / f"frame_{i:04d}.jpg").touch()

            assert count_source_frames(project_dir) == 7

    def test_returns_zero_for_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            assert count_source_frames(project_dir) == 0

    def test_returns_zero_for_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            assert count_source_frames(project_dir) == 0

    def test_ignores_non_image_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            frames_dir = project_dir / "source" / "frames"
            frames_dir.mkdir(parents=True)

            (frames_dir / "frame_0001.png").touch()
            (frames_dir / "frame_0002.png").touch()
            (frames_dir / "metadata.json").touch()
            (frames_dir / "notes.txt").touch()

            assert count_source_frames(project_dir) == 2


class TestFindChunkOutputs:
    """Tests for find_chunk_outputs() - locating output files."""

    def test_finds_outputs_in_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)

            output_dir = project_dir / "output" / "cleanplate" / "chunks" / chunk.name
            output_dir.mkdir(parents=True)
            for i in range(10):
                (output_dir / f"clean_{i:05d}.png").touch()

            files = find_chunk_outputs(project_dir, chunk)
            assert len(files) == 10

    def test_finds_outputs_in_cleanplate_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)

            output_dir = project_dir / "cleanplate" / "chunks" / chunk.name
            output_dir.mkdir(parents=True)
            for i in range(10):
                (output_dir / f"clean_{i:05d}.png").touch()

            files = find_chunk_outputs(project_dir, chunk)
            assert len(files) == 10

    def test_returns_sorted_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)

            output_dir = project_dir / "output" / "cleanplate" / "chunks" / chunk.name
            output_dir.mkdir(parents=True)
            for i in [5, 2, 8, 1, 9]:
                (output_dir / f"clean_{i:05d}.png").touch()

            files = find_chunk_outputs(project_dir, chunk)
            names = [f.name for f in files]
            assert names == sorted(names)

    def test_returns_empty_for_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)

            files = find_chunk_outputs(project_dir, chunk)
            assert files == []

    def test_prefers_output_dir_over_cleanplate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)

            output_dir1 = project_dir / "output" / "cleanplate" / "chunks" / chunk.name
            output_dir1.mkdir(parents=True)
            (output_dir1 / "clean_00001.png").touch()
            (output_dir1 / "clean_00002.png").touch()

            output_dir2 = project_dir / "cleanplate" / "chunks" / chunk.name
            output_dir2.mkdir(parents=True)
            (output_dir2 / "clean_00001.png").touch()

            files = find_chunk_outputs(project_dir, chunk)
            assert len(files) == 2
            assert output_dir1 in files[0].parents


class TestBlendOverlaps:
    """Tests for blend_overlaps() - the full blending pipeline."""

    def _create_solid_image(self, path: Path, color: tuple):
        img = Image.new("RGB", (100, 100), color=color)
        img.save(path)

    def test_single_chunk_copies_to_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=4, frame_count=5)

            output_dir = project_dir / "output" / "cleanplate" / "chunks" / chunk.name
            output_dir.mkdir(parents=True)
            for i in range(5):
                self._create_solid_image(output_dir / f"clean_{i:05d}.png", (255, 0, 0))

            result = blend_overlaps(project_dir, [chunk], overlap=2)

            assert result is True
            final_dir = project_dir / "cleanplate" / "final"
            assert final_dir.exists()
            assert len(list(final_dir.glob("*.png"))) == 5

    def test_two_chunks_blends_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            chunk0 = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
            chunk1 = ChunkInfo(index=1, start_frame=8, end_frame=17, frame_count=10)

            output_dir0 = project_dir / "output" / "cleanplate" / "chunks" / chunk0.name
            output_dir0.mkdir(parents=True)
            for i in range(10):
                self._create_solid_image(output_dir0 / f"clean_{i:05d}.png", (255, 0, 0))

            output_dir1 = project_dir / "output" / "cleanplate" / "chunks" / chunk1.name
            output_dir1.mkdir(parents=True)
            for i in range(10):
                self._create_solid_image(output_dir1 / f"clean_{i:05d}.png", (0, 0, 255))

            result = blend_overlaps(project_dir, [chunk0, chunk1], overlap=2)

            assert result is True
            final_dir = project_dir / "cleanplate" / "final"
            assert len(list(final_dir.glob("*.png"))) == 18

    def test_blend_weights_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            chunk0 = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
            chunk1 = ChunkInfo(index=1, start_frame=8, end_frame=17, frame_count=10)

            output_dir0 = project_dir / "output" / "cleanplate" / "chunks" / chunk0.name
            output_dir0.mkdir(parents=True)
            for i in range(10):
                self._create_solid_image(output_dir0 / f"clean_{i:05d}.png", (255, 0, 0))

            output_dir1 = project_dir / "output" / "cleanplate" / "chunks" / chunk1.name
            output_dir1.mkdir(parents=True)
            for i in range(10):
                self._create_solid_image(output_dir1 / f"clean_{i:05d}.png", (0, 0, 255))

            blend_overlaps(project_dir, [chunk0, chunk1], overlap=2)

            final_dir = project_dir / "cleanplate" / "final"

            frame_9 = Image.open(final_dir / "clean_00009.png")
            pixel_9 = frame_9.getpixel((50, 50))

            frame_10 = Image.open(final_dir / "clean_00010.png")
            pixel_10 = frame_10.getpixel((50, 50))

            assert pixel_9[0] > pixel_9[2]
            assert pixel_10[2] > pixel_10[0]

    def test_returns_false_for_missing_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)

            result = blend_overlaps(project_dir, [chunk], overlap=2)

            assert result is False


class TestStateManagement:
    """Tests for load_batch_state() and save_batch_state()."""

    def test_load_returns_default_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            state = load_batch_state(project_dir)

            assert state == {"completed_chunks": [], "settings": {}}

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            original_state = {
                "completed_chunks": ["000_009", "008_017"],
                "settings": {"batch_size": 10, "overlap": 2, "total_frames": 50}
            }

            save_batch_state(project_dir, original_state)
            loaded_state = load_batch_state(project_dir)

            assert loaded_state == original_state


class TestSaveChunkWorkflow:
    """Tests for save_chunk_workflow()."""

    def test_saves_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            chunk = ChunkInfo(index=0, start_frame=0, end_frame=9, frame_count=10)
            workflow = {"nodes": [{"id": 1, "type": "Test"}]}

            path = save_chunk_workflow(workflow, project_dir, chunk)

            with open(path) as f:
                loaded = json.load(f)
            assert loaded == workflow


class TestBlendWeightCalculation:
    """Tests to verify the blend weight calculation formula."""

    def test_first_overlap_frame_low_weight(self):
        overlap = 2
        chunk_a_end = 9
        frame_idx = 8

        position_in_overlap = frame_idx - chunk_a_end + overlap - 1
        weight_b = (position_in_overlap + 1) / (overlap + 1)

        assert position_in_overlap == 0
        assert abs(weight_b - 1/3) < 0.001

    def test_second_overlap_frame_higher_weight(self):
        overlap = 2
        chunk_a_end = 9
        frame_idx = 9

        position_in_overlap = frame_idx - chunk_a_end + overlap - 1
        weight_b = (position_in_overlap + 1) / (overlap + 1)

        assert position_in_overlap == 1
        assert abs(weight_b - 2/3) < 0.001

    def test_weights_sum_to_one(self):
        overlap = 2
        chunk_a_end = 9

        for frame_idx in [8, 9]:
            position_in_overlap = frame_idx - chunk_a_end + overlap - 1
            weight_b = (position_in_overlap + 1) / (overlap + 1)
            weight_a = 1 - weight_b
            assert abs(weight_a + weight_b - 1.0) < 0.001

    def test_larger_overlap_weights(self):
        overlap = 4
        chunk_a_end = 9

        weights = []
        for frame_idx in range(6, 10):
            position_in_overlap = frame_idx - chunk_a_end + overlap - 1
            weight_b = (position_in_overlap + 1) / (overlap + 1)
            weights.append(weight_b)

        assert weights == pytest.approx([0.2, 0.4, 0.6, 0.8], rel=0.001)

    def test_weights_increase_monotonically(self):
        overlap = 3
        chunk_a_end = 19

        weights = []
        for frame_idx in range(17, 20):
            position_in_overlap = frame_idx - chunk_a_end + overlap - 1
            weight_b = (position_in_overlap + 1) / (overlap + 1)
            weights.append(weight_b)

        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]


class TestLoadTemplateWorkflow:
    """Tests for load_template_workflow()."""

    def test_raises_for_missing_template(self):
        with patch("run_cleanplate_batched.WORKFLOW_TEMPLATES_DIR", Path("/nonexistent")):
            with pytest.raises(FileNotFoundError):
                load_template_workflow()
