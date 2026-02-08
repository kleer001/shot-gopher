#!/usr/bin/env python3
"""Debug script to check COLMAP images.bin parsing.

Usage:
    python debug_colmap_images.py /path/to/project
"""

import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional

from install_wizard.platform import PlatformManager

# Cache for COLMAP path
_colmap_path: Optional[str] = None


def _get_colmap() -> str:
    """Get COLMAP executable path."""
    global _colmap_path
    if _colmap_path is None:
        found = PlatformManager.find_tool("colmap")
        _colmap_path = str(found) if found else "colmap"
    return _colmap_path


def debug_images_bin(project_dir: Path) -> None:
    sparse_dir = project_dir / "mmcam" / "sparse" / "0"
    images_bin = sparse_dir / "images.bin"

    if not images_bin.exists():
        print(f"ERROR: images.bin not found at {images_bin}")
        return

    print(f"images.bin found: {images_bin}")
    print(f"Size: {images_bin.stat().st_size} bytes")
    print()

    # Convert to text
    temp_dir = sparse_dir / "_debug_txt"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    try:
        print("Converting binary to text...")
        colmap_exe = _get_colmap()
        is_bat = colmap_exe.lower().endswith('.bat')
        result = subprocess.run([
            colmap_exe, "model_converter",
            "--input_path", str(sparse_dir),
            "--output_path", str(temp_dir),
            "--output_type", "TXT"
        ], capture_output=True, text=True, shell=is_bat)

        print(f"Return code: {result.returncode}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
        if result.stdout:
            print(f"stdout: {result.stdout}")
        print()

        txt_file = temp_dir / "images.txt"
        if not txt_file.exists():
            print(f"ERROR: Converted images.txt not found!")
            return

        print(f"Converted images.txt size: {txt_file.stat().st_size} bytes")
        print()

        with open(txt_file) as f:
            all_lines = f.readlines()

        print(f"Total lines: {len(all_lines)}")

        # Count different line types
        comment_lines = 0
        empty_lines = 0
        data_lines = []

        for line in all_lines:
            stripped = line.strip()
            if not stripped:
                empty_lines += 1
            elif stripped.startswith("#"):
                comment_lines += 1
            else:
                data_lines.append(stripped)

        print(f"Comment lines: {comment_lines}")
        print(f"Empty lines: {empty_lines}")
        print(f"Data lines: {len(data_lines)}")
        print()

        # Analyze data lines
        metadata_lines = []
        keypoint_lines = []

        for line in data_lines:
            parts = line.split()
            if len(parts) >= 10:
                # Try to parse as metadata
                try:
                    image_id = int(parts[0])
                    qw = float(parts[1])
                    # If we got here, it's likely a metadata line
                    metadata_lines.append(line)
                except ValueError:
                    keypoint_lines.append(line)
            else:
                keypoint_lines.append(line)

        print(f"Detected metadata lines: {len(metadata_lines)}")
        print(f"Detected keypoint lines: {len(keypoint_lines)}")
        print()

        # Show first few metadata lines
        print("First 5 metadata lines:")
        for i, line in enumerate(metadata_lines[:5]):
            print(f"  {i+1}: {line[:100]}...")
        print()

        # Show first few keypoint lines (truncated)
        print("First 3 keypoint lines (truncated):")
        for i, line in enumerate(keypoint_lines[:3]):
            print(f"  {i+1}: {line[:80]}...")
        print()

        # Parse all metadata and show image names
        print(f"All {len(metadata_lines)} registered images:")
        for line in metadata_lines:
            parts = line.split()
            image_id = parts[0]
            name = parts[9]
            print(f"  ID {image_id}: {name}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_colmap_images.py /path/to/project")
        sys.exit(1)

    project_dir = Path(sys.argv[1]).resolve()
    debug_images_bin(project_dir)
