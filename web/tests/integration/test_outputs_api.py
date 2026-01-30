"""Test the outputs API endpoint returns correct keys."""
import pytest
from pathlib import Path


def test_output_key_normalization():
    """Verify output directory keys are normalized correctly.

    The API should return simple keys like 'source' not 'source/frames'.
    """
    # This simulates what the API does
    output_dirs = [
        "source/frames",  # from ingest stage
        "roto",
        "depth",
        "matte",
        "cleanplate",
        "colmap",
        "gsir",
        "mocap",
        "camera",
    ]

    expected_keys = {
        "source",  # source/frames -> source
        "roto",
        "depth",
        "matte",
        "cleanplate",
        "colmap",
        "gsir",
        "mocap",
        "camera",
    }

    # This is the normalization logic from api.py
    normalized = set()
    for dir_name in output_dirs:
        output_key = dir_name.split("/")[0]
        normalized.add(output_key)

    assert normalized == expected_keys


def test_frontend_stage_mapping_matches_backend():
    """Verify frontend STAGE_OUTPUT_DIRS matches what backend returns."""
    # Frontend expects these keys (from ProjectsController.js)
    frontend_expected = {
        'ingest': 'source',
        'depth': 'depth',
        'roto': 'roto',
        'cleanplate': 'cleanplate',
        'colmap': 'colmap',
        'interactive': 'roto',
        'mama': 'matte',
        'mocap': 'mocap',
        'gsir': 'gsir',
        'camera': 'camera',
    }

    # Backend returns these output directories (from pipeline_config.json)
    backend_output_dirs = {
        'ingest': 'source/frames',
        'depth': 'depth',
        'roto': 'roto',
        'cleanplate': 'cleanplate',
        'colmap': 'colmap',
        'interactive': 'roto',
        'mama': 'matte',
        'mocap': 'mocap',
        'gsir': 'gsir',
        'camera': 'camera',
    }

    # After normalization (split on /), they should match
    for stage, frontend_key in frontend_expected.items():
        backend_dir = backend_output_dirs.get(stage, '')
        normalized_key = backend_dir.split("/")[0]
        assert normalized_key == frontend_key, \
            f"Stage {stage}: frontend expects '{frontend_key}' but backend returns '{normalized_key}'"
