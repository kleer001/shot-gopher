#!/usr/bin/env python3
"""Patch ComfyUI to fix BrokenPipeError during model downloads.

This script patches ComfyUI's logger.py to handle flush errors gracefully,
which prevents crashes when HuggingFace downloads models with tqdm progress bars.

Usage:
    python scripts/patch_comfyui.py

Run this once after installing ComfyUI, then restart ComfyUI for changes to take effect.
"""

import re
import sys
from pathlib import Path

# Try to import env_config for INSTALL_DIR
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from env_config import INSTALL_DIR
    DEFAULT_COMFYUI_DIR = INSTALL_DIR / "ComfyUI"
except ImportError:
    DEFAULT_COMFYUI_DIR = Path(__file__).parent.parent / ".vfx_pipeline" / "ComfyUI"


def patch_logger(comfyui_path: Path) -> bool:
    """Patch ComfyUI's logger.py to handle flush errors gracefully.

    This fixes BrokenPipeError that occurs when tqdm/progress bars
    try to write to stderr and the flush fails.

    See: https://github.com/Comfy-Org/ComfyUI/pull/11629
    """
    logger_path = comfyui_path / "app" / "logger.py"

    if not logger_path.exists():
        print(f"  Logger file not found: {logger_path}")
        return False

    print(f"  Checking: {logger_path}")
    content = logger_path.read_text()

    # Check if already patched
    if "except (OSError, ValueError)" in content:
        print("  Already patched!")
        return True

    # Try different patterns that might exist in logger.py
    patterns = [
        # Pattern 1: Simple flush with 8-space indent
        (
            r"def flush\(self\):\n        super\(\)\.flush\(\)",
            """def flush(self):
        try:
            super().flush()
        except (OSError, ValueError):
            pass  # Ignore flush errors (BrokenPipe, etc.)"""
        ),
        # Pattern 2: With 4-space indent
        (
            r"def flush\(self\):\n    super\(\)\.flush\(\)",
            """def flush(self):
    try:
        super().flush()
    except (OSError, ValueError):
        pass  # Ignore flush errors (BrokenPipe, etc.)"""
        ),
    ]

    for pattern, replacement in patterns:
        if re.search(pattern, content):
            patched = re.sub(pattern, replacement, content)
            logger_path.write_text(patched)
            print("  Patched successfully!")
            return True

    # If no pattern matched, show what we found
    print("  Could not find flush method to patch.")
    print("  Looking for 'def flush' in the file...")

    # Find any flush method
    flush_match = re.search(r"(def flush\(self\):.*?)(?=\n    def |\n        def |\Z)", content, re.DOTALL)
    if flush_match:
        print(f"  Found flush method:\n{flush_match.group(1)[:200]}")
    else:
        print("  No flush method found in logger.py")

    return False


def patch_prestartup(comfyui_path: Path) -> bool:
    """Check for and patch ComfyUI-Manager's prestartup_script.py if present."""
    prestartup_paths = [
        comfyui_path / "custom_nodes" / "ComfyUI-Manager" / "prestartup_script.py",
        comfyui_path / "custom_nodes" / "comfyui-manager" / "prestartup_script.py",
    ]

    for prestartup_path in prestartup_paths:
        if not prestartup_path.exists():
            continue

        print(f"  Checking: {prestartup_path}")
        content = prestartup_path.read_text()

        # Check if already patched
        if "except (OSError, ValueError)" in content and "flush" in content:
            print("  Already patched!")
            return True

        # Look for flush function that needs patching
        pattern = r"def flush\(self\):\n        self\.original_stderr\.flush\(\)"
        replacement = """def flush(self):
        try:
            self.original_stderr.flush()
        except (OSError, ValueError):
            pass  # Ignore flush errors"""

        if re.search(pattern, content):
            patched = re.sub(pattern, replacement, content)
            prestartup_path.write_text(patched)
            print("  Patched successfully!")
            return True

        print("  No patchable flush method found")

    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Patch ComfyUI to fix BrokenPipeError during model downloads"
    )
    parser.add_argument(
        "--comfyui-path",
        type=Path,
        default=DEFAULT_COMFYUI_DIR,
        help=f"Path to ComfyUI installation (default: {DEFAULT_COMFYUI_DIR})"
    )

    args = parser.parse_args()
    comfyui_path = args.comfyui_path.resolve()

    print(f"ComfyUI Patcher")
    print(f"===============")
    print(f"ComfyUI path: {comfyui_path}")
    print()

    if not comfyui_path.exists():
        print(f"Error: ComfyUI not found at {comfyui_path}")
        print("Use --comfyui-path to specify the correct location")
        sys.exit(1)

    if not (comfyui_path / "main.py").exists():
        print(f"Error: {comfyui_path} doesn't look like a ComfyUI installation")
        print("(main.py not found)")
        sys.exit(1)

    print("Patching logger.py...")
    logger_patched = patch_logger(comfyui_path)

    print()
    print("Checking ComfyUI-Manager prestartup_script.py...")
    prestartup_patched = patch_prestartup(comfyui_path)

    print()
    print("=" * 50)
    if logger_patched or prestartup_patched:
        print("Patches applied successfully!")
        print()
        print("IMPORTANT: Restart ComfyUI for changes to take effect.")
        print()
        print("If ComfyUI is running via the pipeline, stop and restart it:")
        print("  1. Stop the current pipeline/web server")
        print("  2. Kill any running ComfyUI process")
        print("  3. Start the pipeline again")
    else:
        print("No patches were applied.")
        print("The files may already be patched or have a different structure.")


if __name__ == "__main__":
    main()
