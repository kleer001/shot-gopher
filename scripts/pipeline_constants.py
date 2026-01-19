"""Pipeline constants and stage definitions.

Central location for all pipeline-wide constants used across modules.
"""

from pathlib import Path

__all__ = [
    "START_FRAME",
    "SUPPORTED_FORMATS",
    "STAGES",
    "STAGE_ORDER",
    "STAGES_REQUIRING_FRAMES",
    "WORKFLOW_TEMPLATES_DIR",
]

START_FRAME = 1

SUPPORTED_FORMATS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf",
    ".exr", ".dpx", ".jpg", ".png"
}

STAGES = {
    "ingest": "Extract frames from movie",
    "depth": "Run depth analysis (01_analysis.json)",
    "roto": "Run segmentation (02_segmentation.json)",
    "matanyone": "Refine person mattes (04_matanyone.json)",
    "cleanplate": "Run clean plate generation (03_cleanplate.json)",
    "colmap": "Run COLMAP SfM reconstruction",
    "mocap": "Run human motion capture (WHAM)",
    "gsir": "Run GS-IR material decomposition",
    "camera": "Export camera to Alembic",
}

STAGE_ORDER = [
    "ingest", "depth", "roto", "matanyone", "cleanplate",
    "colmap", "mocap", "gsir", "camera"
]

STAGES_REQUIRING_FRAMES = {
    "depth", "roto", "matanyone", "cleanplate",
    "colmap", "mocap", "gsir", "camera"
}

WORKFLOW_TEMPLATES_DIR = Path(__file__).parent.parent / "workflow_templates"
