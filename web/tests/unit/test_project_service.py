"""Unit tests for ProjectService."""

import pytest
from unittest.mock import Mock
from pathlib import Path
from datetime import datetime

from web.services.project_service import ProjectService
from web.models.dto import ProjectCreateRequest
from web.models.domain import Project, ProjectStatus


class TestProjectService:
    """Test ProjectService business logic."""

    def test_create_project_success(self):
        """Test successful project creation."""
        mock_repo = Mock()
        mock_repo.get.return_value = None
        mock_repo.save.return_value = Project(
            name="test_project",
            path=Path("/workspace/test_project"),
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=["ingest"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        service = ProjectService(mock_repo)
        request = ProjectCreateRequest(name="test_project", stages=["ingest"])

        result = service.create_project(request, Path("/workspace"))

        assert result.name == "test_project"
        assert result.status == ProjectStatus.CREATED
        mock_repo.save.assert_called_once()

    def test_create_project_duplicate_name(self):
        """Test creating project with duplicate name raises error."""
        existing_project = Project(
            name="existing",
            path=Path("/workspace/existing"),
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        mock_repo = Mock()
        mock_repo.get.return_value = existing_project

        service = ProjectService(mock_repo)
        request = ProjectCreateRequest(name="existing", stages=[])

        with pytest.raises(ValueError, match="already exists"):
            service.create_project(request, Path("/workspace"))

    def test_create_project_invalid_stages(self):
        """Test creating project with invalid stages raises error."""
        mock_repo = Mock()
        mock_repo.get.return_value = None

        service = ProjectService(mock_repo)
        request = ProjectCreateRequest(name="test", stages=["invalid_stage"])

        with pytest.raises(ValueError, match="Invalid stages"):
            service.create_project(request, Path("/workspace"))

    def test_list_projects(self):
        """Test listing projects."""
        projects = [
            Project(
                name="project1",
                path=Path("/workspace/project1"),
                status=ProjectStatus.CREATED,
                video_path=None,
                stages=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            Project(
                name="project2",
                path=Path("/workspace/project2"),
                status=ProjectStatus.COMPLETE,
                video_path=None,
                stages=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

        mock_repo = Mock()
        mock_repo.list.return_value = projects

        service = ProjectService(mock_repo)

        result = service.list_projects()

        assert result.total == 2
        assert len(result.projects) == 2
        assert result.projects[0].name == "project1"

    def test_get_project(self):
        """Test getting a project by name."""
        project = Project(
            name="test",
            path=Path("/workspace/test"),
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        mock_repo = Mock()
        mock_repo.get.return_value = project

        service = ProjectService(mock_repo)
        result = service.get_project("test")

        assert result is not None
        assert result.name == "test"

    def test_get_project_not_found(self):
        """Test getting non-existent project returns None."""
        mock_repo = Mock()
        mock_repo.get.return_value = None

        service = ProjectService(mock_repo)
        result = service.get_project("nonexistent")

        assert result is None

    def test_delete_project(self):
        """Test deleting a project."""
        mock_repo = Mock()
        mock_repo.delete.return_value = True

        service = ProjectService(mock_repo)
        result = service.delete_project("test")

        assert result is True
        mock_repo.delete.assert_called_once_with("test")

    def test_update_project_status(self):
        """Test updating project status."""
        project = Project(
            name="test",
            path=Path("/workspace/test"),
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        mock_repo = Mock()
        mock_repo.get.return_value = project
        mock_repo.save.return_value = project

        service = ProjectService(mock_repo)
        result = service.update_project_status("test", ProjectStatus.PROCESSING)

        assert result is not None
        assert result.status == ProjectStatus.PROCESSING
        mock_repo.save.assert_called_once()
