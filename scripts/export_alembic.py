#!/usr/bin/env python3
"""Export OBJ mesh sequences to Alembic (.abc) format.

Standalone script to convert OBJ mesh sequences to Alembic format
for import into Blender, Houdini, Maya, Nuke, etc.

Features:
  - Animated mesh sequence from OBJ files
  - UV coordinates and normals preserved
  - Uses Blender headless for reliable cross-platform export

Requirements:
  - Blender 4.2+ (auto-installed via wizard)

Usage:
    python export_alembic.py <mesh_dir> [options]
    python export_alembic.py --mesh-dir meshes/ --output animated.abc

Example:
    python export_alembic.py mocap/smplx_animated --fps 24 --start-frame 1
    python export_alembic.py --mesh-dir mocap/smplx_animated --output body.abc
"""

import argparse
import sys
from pathlib import Path

HAS_BLENDER = False
try:
    from blender import (
        check_blender_available,
        export_mesh_sequence_to_alembic,
    )
    HAS_BLENDER = True
except ImportError:
    pass


def export_mesh_alembic(
    input_dir: Path,
    output_path: Path,
    start_frame: int = 1,
    fps: float = 24.0,
) -> bool:
    """Export OBJ mesh sequence to Alembic file using Blender.

    Args:
        input_dir: Directory containing OBJ files
        output_path: Output .abc file path
        start_frame: Starting frame number
        fps: Frames per second

    Returns:
        True if export succeeded
    """
    if not HAS_BLENDER:
        print("Error: Blender module not available.", file=sys.stderr)
        print("Run the installation wizard to install Blender.", file=sys.stderr)
        return False

    try:
        return export_mesh_sequence_to_alembic(
            input_dir=input_dir,
            output_path=output_path,
            fps=fps,
            start_frame=start_frame,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False
    except ValueError as e:
        print(f"Invalid parameter: {e}", file=sys.stderr)
        return False
    except RuntimeError as e:
        print(f"Export failed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export mesh sequences to Alembic (.abc) format using Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing OBJ mesh files"
    )
    parser.add_argument(
        "--mesh-dir", "-m",
        type=Path,
        default=None,
        help="Directory containing OBJ mesh files (alternative to positional arg)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output .abc file path (default: <input_dir>/animated.abc)"
    )
    parser.add_argument(
        "--start-frame", "-s",
        type=int,
        default=1,
        help="Starting frame number (default: 1)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=24.0,
        help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if Blender is available and exit"
    )

    args = parser.parse_args()

    if args.check:
        if HAS_BLENDER:
            available, message = check_blender_available()
            print(message)
            sys.exit(0 if available else 1)
        else:
            print("Blender module not available. Run installation wizard.")
            sys.exit(1)

    input_dir = args.mesh_dir or args.input
    if input_dir is None:
        parser.error("Either positional input or --mesh-dir is required")

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input must be a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    obj_files = list(set(input_dir.glob("*.obj")) | set(input_dir.glob("*.OBJ")))
    if not obj_files:
        print(f"Error: No OBJ files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(obj_files)} OBJ files in {input_dir}")

    if args.output:
        output_path = args.output
    else:
        output_path = input_dir / "animated.abc"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = export_mesh_alembic(
        input_dir=input_dir,
        output_path=output_path,
        start_frame=args.start_frame,
        fps=args.fps,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
