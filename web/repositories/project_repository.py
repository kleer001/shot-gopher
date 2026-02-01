"""Project repository - filesystem implementation."""

from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime
import shutil

from web.models.domain import Project, ProjectStatus
from web.repositories.base import Repository


class ProjectRepository(Repository[Project]):
    """
    Repository for project data access.

    Current implementation: Filesystem (JSON files)
    Future: Could swap for PostgreSQL, MongoDB, etc.
    """

    def __init__(self, projects_dir: Path):
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def get(self, name: str) -> Optional[Project]:
        """Get project by name."""
        project_path = self.projects_dir / name

        if not project_path.exists():
            return None

        return self._load_project(project_path)

    def list(self) -> List[Project]:
        """List all projects, sorted by creation date (newest first)."""
        projects = []

        if not self.projects_dir.exists():
            return projects

        try:
            entries = list(self.projects_dir.iterdir())
        except OSError:
            return projects

        for project_path in entries:
            if project_path.is_dir():
                project = self._load_project(project_path)
                if project:
                    projects.append(project)

        return sorted(projects, key=lambda p: p.created_at, reverse=True)

    def save(self, project: Project) -> Project:
        """Save project to filesystem."""
        project.path.mkdir(parents=True, exist_ok=True)

        state = {
            "name": project.name,
            "status": project.status.value,
            "video_path": str(project.video_path) if project.video_path else None,
            "stages": project.stages,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
        }

        with open(project.state_file, "w", encoding='utf-8') as f:
            json.dump(state, f, indent=2)

        return project

    def delete(self, name: str) -> bool:
        """Delete project directory."""
        project_path = self.projects_dir / name

        if project_path.exists():
            shutil.rmtree(project_path)
            return True

        return False

    def _load_project(self, project_path: Path) -> Optional[Project]:
        """Load project from filesystem."""
        state_file = project_path / "project_state.json"

        try:
            with open(state_file, encoding='utf-8') as f:
                state = json.load(f)

            return Project(
                name=state.get("name", project_path.name),
                path=project_path,
                status=ProjectStatus(state.get("status", "unknown")),
                video_path=Path(state["video_path"]) if state.get("video_path") else None,
                stages=state.get("stages", []),
                created_at=datetime.fromisoformat(state["created_at"]),
                updated_at=datetime.fromisoformat(state["updated_at"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
            return self._create_and_persist_project(project_path)

    def _create_and_persist_project(self, project_path: Path) -> Optional[Project]:
        """Create project from directory and attempt to save state file."""
        try:
            project = self._create_project_from_directory(project_path)
        except OSError:
            return None

        try:
            self.save(project)
        except OSError:
            pass

        return project

    def _create_project_from_directory(self, project_path: Path) -> Project:
        """Create a Project from directory metadata when no valid state file exists."""
        dir_mtime = datetime.fromtimestamp(project_path.stat().st_mtime)
        stages = self._detect_completed_stages(project_path)

        return Project(
            name=project_path.name,
            path=project_path,
            status=ProjectStatus.UNKNOWN,
            video_path=None,
            stages=stages,
            created_at=dir_mtime,
            updated_at=dir_mtime,
        )

    def _detect_completed_stages(self, project_path: Path) -> List[str]:
        """Detect which pipeline stages have completed based on output directories."""
        stages = []
        stage_output_dirs = {
            'ingest': project_path / 'source' / 'frames',
            'depth': project_path / 'depth',
            'roto': project_path / 'roto',
            'matte': project_path / 'matte',
            'cleanplate': project_path / 'cleanplate',
            'colmap': project_path / 'colmap',
            'mocap': project_path / 'mocap',
            'camera': project_path / 'camera',
        }
        for stage_name, stage_path in stage_output_dirs.items():
            try:
                if stage_path.exists() and any(stage_path.iterdir()):
                    stages.append(stage_name)
            except OSError:
                pass
        return stages
