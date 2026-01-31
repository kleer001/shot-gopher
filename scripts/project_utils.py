"""Project management utilities.

Handles project metadata, last-project tracking, and project structure.
"""

import json
import os
from pathlib import Path
from typing import Optional

from env_config import INSTALL_DIR


class ProjectMetadata:
    """Manages project metadata (project.json).

    Provides a clean interface for reading and writing project metadata,
    eliminating scattered JSON operations throughout the codebase.
    """

    def __init__(self, project_dir: Path):
        """Initialize metadata manager.

        Args:
            project_dir: Path to the project directory
        """
        self.project_dir = project_dir
        self.path = project_dir / "project.json"

    def load(self) -> dict:
        """Load metadata from disk.

        Returns:
            Metadata dict, or empty dict if file doesn't exist or is invalid
        """
        if not self.path.exists():
            return {}
        try:
            with open(self.path, encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def save(self, data: dict) -> None:
        """Save metadata to disk.

        Args:
            data: Metadata dict to save
        """
        self.project_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def update(self, **kwargs) -> dict:
        """Update specific metadata fields.

        Args:
            **kwargs: Fields to update

        Returns:
            Updated metadata dict
        """
        data = self.load()
        data.update(kwargs)
        self.save(data)
        return data

    def get(self, key: str, default=None):
        """Get a specific metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Value or default
        """
        return self.load().get(key, default)

    def get_fps(self) -> Optional[float]:
        """Get fps from metadata.

        Returns:
            FPS value or None
        """
        return self.get("fps")

    def set_source_info(self, source_path: Path, fps: float) -> None:
        """Set source file information.

        Args:
            source_path: Path to source file
            fps: Frames per second
        """
        self.update(source=str(source_path), fps=fps)

    def set_frame_info(self, count: int, width: int, height: int) -> None:
        """Set frame dimension information.

        Args:
            count: Total frame count
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.update(frame_count=count, width=width, height=height)

    def initialize(self, name: str, fps: float, source_path: Optional[Path] = None) -> dict:
        """Initialize or update project metadata.

        Merges new values with existing metadata.

        Args:
            name: Project name
            fps: Frames per second
            source_path: Optional source file path

        Returns:
            Complete metadata dict
        """
        data = self.load()
        data["name"] = name
        data["fps"] = fps
        data["start_frame"] = data.get("start_frame", 1)
        if source_path:
            data["source"] = str(source_path)
        self.save(data)
        return data


LAST_PROJECT_FILE = INSTALL_DIR / ".last_project"


def get_last_project_file() -> Path:
    """Get the last-project file location.

    Returns:
        Path to last project file
    """
    return LAST_PROJECT_FILE


def save_last_project(project_dir: Path) -> None:
    """Save the last used project directory.

    Args:
        project_dir: Project directory path
    """
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        LAST_PROJECT_FILE.write_text(str(project_dir.resolve()))
    except PermissionError:
        pass


def get_last_project() -> Optional[Path]:
    """Get the last used project directory, if it exists.

    Returns:
        Path to last project, or None if not found
    """
    if LAST_PROJECT_FILE.exists():
        try:
            path = Path(LAST_PROJECT_FILE.read_text().strip())
            if path.exists() and path.is_dir():
                return path
        except Exception:
            pass
    return None
