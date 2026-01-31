"""Tests for media utilities."""

import pytest
from pathlib import Path

from web.utils.media import (
    find_video_or_frames,
    find_video_file,
    get_frame_files,
    get_dir_size_bytes,
    get_dir_size_gb,
    VIDEO_EXTENSIONS,
)


class TestFindVideoOrFrames:
    """Tests for find_video_or_frames function."""

    def test_finds_existing_frames(self, tmp_path):
        """Should detect frames in source/frames directory."""
        source_dir = tmp_path / "source"
        frames_dir = source_dir / "frames"
        frames_dir.mkdir(parents=True)

        for i in range(10):
            (frames_dir / f"frame_{i:04d}.png").write_bytes(b"fake")

        video_path, has_frames, frame_count = find_video_or_frames(source_dir)

        assert video_path is None
        assert has_frames is True
        assert frame_count == 10

    def test_finds_video_when_no_frames(self, tmp_path):
        """Should find video file when no frames exist."""
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "input.mp4").write_bytes(b"fake video")

        video_path, has_frames, frame_count = find_video_or_frames(source_dir)

        assert video_path is not None
        assert video_path.name == "input.mp4"
        assert has_frames is False
        assert frame_count == 0

    def test_prefers_frames_over_video(self, tmp_path):
        """Should prefer frames if both exist."""
        source_dir = tmp_path / "source"
        frames_dir = source_dir / "frames"
        frames_dir.mkdir(parents=True)

        (source_dir / "input.mp4").write_bytes(b"fake video")
        for i in range(5):
            (frames_dir / f"frame_{i:04d}.png").write_bytes(b"fake")

        video_path, has_frames, frame_count = find_video_or_frames(source_dir)

        assert video_path is None
        assert has_frames is True
        assert frame_count == 5

    def test_returns_none_when_empty(self, tmp_path):
        """Should return None/False when directory is empty."""
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)

        video_path, has_frames, frame_count = find_video_or_frames(source_dir)

        assert video_path is None
        assert has_frames is False
        assert frame_count == 0

    def test_counts_mixed_frame_types(self, tmp_path):
        """Should count both PNG and JPG frames."""
        source_dir = tmp_path / "source"
        frames_dir = source_dir / "frames"
        frames_dir.mkdir(parents=True)

        for i in range(3):
            (frames_dir / f"frame_{i:04d}.png").write_bytes(b"fake")
        for i in range(2):
            (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"fake")

        video_path, has_frames, frame_count = find_video_or_frames(source_dir)

        assert has_frames is True
        assert frame_count == 5


class TestFindVideoFile:
    """Tests for find_video_file function."""

    def test_finds_input_mp4(self, tmp_path):
        """Should find input.mp4."""
        (tmp_path / "input.mp4").write_bytes(b"fake")

        result = find_video_file(tmp_path)

        assert result is not None
        assert result.name == "input.mp4"

    def test_finds_input_mov(self, tmp_path):
        """Should find input.mov."""
        (tmp_path / "input.mov").write_bytes(b"fake")

        result = find_video_file(tmp_path)

        assert result is not None
        assert result.name == "input.mov"

    def test_prefers_input_naming(self, tmp_path):
        """Should prefer input.* over other video files."""
        (tmp_path / "input.mp4").write_bytes(b"fake")
        (tmp_path / "other_video.mp4").write_bytes(b"fake")

        result = find_video_file(tmp_path)

        assert result.name == "input.mp4"

    def test_finds_any_video_when_no_input(self, tmp_path):
        """Should find any video file when input.* doesn't exist."""
        (tmp_path / "my_clip.mov").write_bytes(b"fake")

        result = find_video_file(tmp_path)

        assert result is not None
        assert result.name == "my_clip.mov"

    def test_returns_none_when_no_video(self, tmp_path):
        """Should return None when no video files exist."""
        (tmp_path / "readme.txt").write_bytes(b"text")

        result = find_video_file(tmp_path)

        assert result is None

    def test_returns_none_for_missing_dir(self, tmp_path):
        """Should return None for non-existent directory."""
        result = find_video_file(tmp_path / "nonexistent")

        assert result is None

    def test_supports_all_extensions(self, tmp_path):
        """Should support all VIDEO_EXTENSIONS."""
        for ext in VIDEO_EXTENSIONS:
            video_file = tmp_path / f"test{ext}"
            video_file.write_bytes(b"fake")

            result = find_video_file(tmp_path)
            assert result is not None

            video_file.unlink()


class TestGetFrameFiles:
    """Tests for get_frame_files function."""

    def test_returns_sorted_frames(self, tmp_path):
        """Should return frames sorted by name."""
        (tmp_path / "frame_0003.png").write_bytes(b"fake")
        (tmp_path / "frame_0001.png").write_bytes(b"fake")
        (tmp_path / "frame_0002.png").write_bytes(b"fake")

        result = get_frame_files(tmp_path)

        assert len(result) == 3
        assert result[0].name == "frame_0001.png"
        assert result[1].name == "frame_0002.png"
        assert result[2].name == "frame_0003.png"

    def test_returns_empty_for_missing_dir(self, tmp_path):
        """Should return empty list for non-existent directory."""
        result = get_frame_files(tmp_path / "nonexistent")

        assert result == []

    def test_includes_both_png_and_jpg(self, tmp_path):
        """Should include both PNG and JPG files."""
        (tmp_path / "frame_0001.png").write_bytes(b"fake")
        (tmp_path / "frame_0002.jpg").write_bytes(b"fake")

        result = get_frame_files(tmp_path)

        assert len(result) == 2


class TestGetDirSizeBytes:
    """Tests for get_dir_size_bytes function."""

    def test_calculates_size(self, tmp_path):
        """Should calculate total size of files."""
        (tmp_path / "file1.txt").write_bytes(b"a" * 100)
        (tmp_path / "file2.txt").write_bytes(b"b" * 200)

        result = get_dir_size_bytes(tmp_path)

        assert result == 300

    def test_includes_subdirectories(self, tmp_path):
        """Should include files in subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file1.txt").write_bytes(b"a" * 100)
        (subdir / "file2.txt").write_bytes(b"b" * 200)

        result = get_dir_size_bytes(tmp_path)

        assert result == 300

    def test_returns_zero_for_empty_dir(self, tmp_path):
        """Should return 0 for empty directory."""
        result = get_dir_size_bytes(tmp_path)

        assert result == 0

    def test_returns_zero_for_missing_dir(self, tmp_path):
        """Should return 0 for non-existent directory."""
        result = get_dir_size_bytes(tmp_path / "nonexistent")

        assert result == 0


class TestGetDirSizeGb:
    """Tests for get_dir_size_gb function."""

    def test_converts_to_gigabytes(self, tmp_path):
        """Should convert bytes to GB."""
        size_bytes = 1024 * 1024 * 1024
        (tmp_path / "file.bin").write_bytes(b"x" * size_bytes)

        result = get_dir_size_gb(tmp_path)

        assert result == 1.0

    def test_rounds_to_two_decimals(self, tmp_path):
        """Should round to 2 decimal places."""
        size_bytes = int(1.567 * 1024 * 1024 * 1024)
        (tmp_path / "file.bin").write_bytes(b"x" * size_bytes)

        result = get_dir_size_gb(tmp_path)

        assert result == 1.57
