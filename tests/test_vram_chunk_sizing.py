"""Tests for resolution-aware VRAM chunk sizing.

Verifies that chunk size calculations correctly scale with processing
resolution, preventing OOM at native resolutions higher than the
original 1024x576 calibration baseline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from vram_analyzer import (
    get_mama_chunk_size,
    calculate_estimated_vram,
    analyze_stage,
    MAMA_REFERENCE_MEGAPIXELS,
    STAGE_VRAM_REQUIREMENTS,
    VramStatus,
)
from video_mama import get_optimal_chunk_size, REFERENCE_MEGAPIXELS


class TestReferenceConstants:
    def test_reference_megapixels_match(self) -> None:
        expected = (1024 * 576) / 1_000_000
        assert MAMA_REFERENCE_MEGAPIXELS == expected
        assert REFERENCE_MEGAPIXELS == expected


class TestGetMamaChunkSize:
    def test_reference_resolution_unchanged(self) -> None:
        assert get_mama_chunk_size(48.0, (1024, 576)) == 20
        assert get_mama_chunk_size(24.0, (1024, 576)) == 14
        assert get_mama_chunk_size(16.0, (1024, 576)) == 10
        assert get_mama_chunk_size(12.0, (1024, 576)) == 8
        assert get_mama_chunk_size(8.0, (1024, 576)) == 6
        assert get_mama_chunk_size(6.0, (1024, 576)) == 4

    def test_default_resolution_is_reference(self) -> None:
        assert get_mama_chunk_size(48.0) == get_mama_chunk_size(48.0, (1024, 576))
        assert get_mama_chunk_size(12.0) == get_mama_chunk_size(12.0, (1024, 576))

    def test_1080p_reduces_chunk_size(self) -> None:
        res_1080p = (1920, 1080)
        assert get_mama_chunk_size(48.0, res_1080p) < get_mama_chunk_size(48.0, (1024, 576))

    def test_4k_reduces_chunk_size_further(self) -> None:
        res_4k = (3840, 2160)
        res_1080p = (1920, 1080)
        assert get_mama_chunk_size(48.0, res_4k) < get_mama_chunk_size(48.0, res_1080p)

    def test_1080p_48gb_gpu(self) -> None:
        chunk = get_mama_chunk_size(48.0, (1920, 1080))
        assert chunk <= 10

    def test_4k_48gb_gpu(self) -> None:
        chunk = get_mama_chunk_size(48.0, (3840, 2160))
        assert chunk <= 6

    def test_minimum_chunk_size(self) -> None:
        assert get_mama_chunk_size(4.0, (3840, 2160)) == 4


class TestGetOptimalChunkSize:
    def test_none_vram_returns_default(self) -> None:
        assert get_optimal_chunk_size(None) == 8
        assert get_optimal_chunk_size(None, (1920, 1080)) == 8

    def test_reference_resolution_unchanged(self) -> None:
        assert get_optimal_chunk_size(48.0, (1024, 576)) == 20
        assert get_optimal_chunk_size(24.0, (1024, 576)) == 14

    def test_1080p_reduces_chunk_size(self) -> None:
        res_1080p = (1920, 1080)
        assert get_optimal_chunk_size(48.0, res_1080p) < get_optimal_chunk_size(48.0, (1024, 576))

    def test_matches_analyzer(self) -> None:
        for vram in [6.0, 8.0, 12.0, 16.0, 24.0, 48.0]:
            for res in [(1024, 576), (1920, 1080), (3840, 2160)]:
                assert get_optimal_chunk_size(vram, res) == get_mama_chunk_size(vram, res)


class TestAnalyzeStageMamaChunked:
    """Verify analyze_stage uses chunk_size (not frame_count) for mama VRAM estimate."""

    def test_estimated_vram_uses_chunk_size_not_frame_count(self) -> None:
        result = analyze_stage("mama", 48.0, frame_count=300, resolution=(1920, 1080))
        chunk_size = result.chunk_size
        assert chunk_size is not None

        config = STAGE_VRAM_REQUIREMENTS["mama"]
        expected_vram = calculate_estimated_vram(config, chunk_size, (1920, 1080))
        assert result.estimated_vram_gb == expected_vram

    def test_mama_not_flagged_insufficient_for_normal_video(self) -> None:
        result = analyze_stage("mama", 24.0, frame_count=300, resolution=(1920, 1080))
        assert result.status != VramStatus.INSUFFICIENT

    def test_mama_1080p_48gb_is_ok(self) -> None:
        result = analyze_stage("mama", 48.0, frame_count=500, resolution=(1920, 1080))
        assert result.status == VramStatus.OK

    def test_mama_4k_8gb_warns(self) -> None:
        result = analyze_stage("mama", 8.0, frame_count=100, resolution=(3840, 2160))
        assert result.chunk_size is not None
        assert result.chunk_size <= 4

    def test_mama_reference_resolution_large_video_ok(self) -> None:
        result = analyze_stage("mama", 48.0, frame_count=1000, resolution=(1024, 576))
        assert result.status == VramStatus.OK
        assert result.chunk_size == 20
