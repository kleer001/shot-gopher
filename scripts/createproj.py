#!/usr/bin/env python3
"""Create shot directory structure for a team project."""

import argparse
import json
import sys
from pathlib import Path


def create_shot_directories(team: str, shot: str, base_path: Path | None = None) -> Path:
    """Create the standard shot directory structure.

    Args:
        team: Team name (e.g., "Team5")
        shot: Shot name (e.g., "Shot001")
        base_path: Base directory for project creation. Defaults to current directory.

    Returns:
        Path to the created shot directory.
    """
    if base_path is None:
        base_path = Path.cwd()

    shot_path = base_path / team / shot

    subdirs = ["blend", "geo", "textures", "vdb", "renders"]
    for subdir in subdirs:
        (shot_path / subdir).mkdir(parents=True, exist_ok=True)

    submit_file = shot_path / "submit.json"
    if not submit_file.exists():
        submit_file.write_text(json.dumps({}, indent=2) + "\n")

    return shot_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create shot directory structure for a team project."
    )
    parser.add_argument("team", help="Team name (e.g., Team5)")
    parser.add_argument("shot", help="Shot name (e.g., Shot001)")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base directory for project creation (default: current directory)",
    )

    args = parser.parse_args()

    shot_path = create_shot_directories(args.team, args.shot, args.base_path)
    print(f"Created: {shot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
