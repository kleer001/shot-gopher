#!/usr/bin/env python3
"""Verify all required models are present and valid."""

import os
import sys
from pathlib import Path

# Get repo root (parent of scripts/)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Model directory - use environment variable or default to repo-relative path
MODEL_DIR = Path(os.environ.get("VFX_MODELS_DIR") or _REPO_ROOT / ".vfx_pipeline" / "models")

REQUIRED_MODELS = {
    "sam3": {
        "path": MODEL_DIR / "sam3",
        "files": ["model.safetensors", "config.json"],
        "optional": False,
        "description": "Segment Anything Model 3 - for object segmentation",
    },
    "videodepthanything": {
        "path": MODEL_DIR / "videodepthanything",
        "files": ["video_depth_anything_vits.pth"],
        "optional": False,
        "description": "Video Depth Anything - for depth map generation",
    },
    "wham": {
        "path": MODEL_DIR / "wham",
        "files": ["wham_vit_w_3dpw.pth.tar"],
        "optional": False,
        "description": "WHAM - for human motion capture",
    },
    "matanyone": {
        "path": MODEL_DIR / "matanyone",
        "files": ["matanyone.pth"],
        "optional": True,
        "description": "MatAnyone - for matte refinement (optional)",
    },
    "smplx": {
        "path": MODEL_DIR / "smplx",
        "files": ["SMPLX_NEUTRAL.pkl", "SMPLX_MALE.pkl", "SMPLX_FEMALE.pkl"],
        "optional": True,
        "description": "SMPL-X - for human body modeling (requires manual download)",
    },
}


def check_model(name: str, config: dict) -> tuple[bool, str]:
    """Check if a model is present and valid.

    Args:
        name: Model name
        config: Model configuration dict

    Returns:
        Tuple of (is_valid, message)
    """
    path = config["path"]

    if not path.exists():
        return False, f"Directory not found: {path}"

    missing_files = []
    for file in config["files"]:
        if not (path / file).exists():
            missing_files.append(file)

    if missing_files:
        # Check if any files exist (might have different names)
        existing_files = list(path.glob("*"))
        if existing_files:
            return False, f"Missing expected files: {', '.join(missing_files)} (found {len(existing_files)} other files)"
        else:
            return False, f"Directory exists but is empty"

    return True, "OK"


def main():
    """Verify all models."""
    print(f"Checking models in: {MODEL_DIR}\n")

    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory does not exist: {MODEL_DIR}")
        print(f"\nCreate it and run: ./scripts/download_models.sh")
        return 1

    all_required_ok = True
    optional_missing = []

    for name, config in REQUIRED_MODELS.items():
        ok, msg = check_model(name, config)
        is_optional = config.get("optional", False)

        # Status symbol
        if ok:
            status = "✓"
            color = "\033[0;32m"  # Green
        elif is_optional:
            status = "⚠"
            color = "\033[1;33m"  # Yellow
            optional_missing.append(name)
        else:
            status = "✗"
            color = "\033[0;31m"  # Red
            all_required_ok = False

        nc = "\033[0m"  # No color

        # Print result
        desc = config.get("description", "")
        print(f"{color}{status}{nc} {name:20s} {msg}")
        if not ok and desc:
            print(f"   └─ {desc}")

    print()

    # Summary
    if all_required_ok and not optional_missing:
        print("\033[0;32m✓ All models present and valid!\033[0m")
        return 0
    elif all_required_ok:
        print("\033[1;33m⚠ Required models OK, but optional models missing:\033[0m")
        for name in optional_missing:
            print(f"   - {name}: {REQUIRED_MODELS[name]['description']}")
        print("\nOptional models can be downloaded later if needed.")
        return 0
    else:
        print("\033[0;31m✗ Some required models are missing or invalid\033[0m")
        print("\nTo download models, run:")
        print("  ./scripts/download_models.sh")
        print("\nFor SMPL-X (manual download required):")
        print("  1. Register at: https://smpl-x.is.tue.mpg.de/")
        print("  2. Download SMPL-X models")
        print(f"  3. Extract to: {MODEL_DIR}/smplx/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
