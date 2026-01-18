# 🔌 Roadmap 2: API Backend

**Goal:** Build REST/WebSocket API with proper architecture (backend only, no UI)

**Status:** 🟡 60% Complete (Core: 80%, Testing/Docs: 10%)

**Dependencies:** Originally planned after Roadmap 1 (Docker), but developed in parallel for local mode

---

## Implementation Status

### ✅ Completed (60%)
- **Phase 2A (Partial)**: ConfigService with DRY configuration (`web/services/config_service.py`)
- **Phase 2B (Partial)**: WebSocket service for real-time updates (`web/websocket.py`)
- **Phase 2C**: REST API endpoints (`web/api.py`)
  - Project upload and creation
  - Pipeline execution
  - System health/status
  - Video metadata extraction
- **Configuration**: `web/config/pipeline_config.json` - single source of truth

### ⚪ Remaining (40%)
- **Phase 2A**: Full repository pattern implementation
- **Phase 2A**: DTO vs Domain model separation
- **Phase 2B**: Extract service layer from API layer
- **Phase 2D**: Comprehensive unit + integration tests
- **Phase 2E**: OpenAPI/Swagger documentation

---

## Overview

This roadmap builds a FastAPI backend with clean layered architecture. The core API is operational in local development mode, but needs architectural improvements (repository pattern) and comprehensive testing before production use.

### Why API-First?

1. **Validate backend independently** - Ensure business logic works before building UI
2. **Multiple frontends possible** - Web, CLI, mobile could all use same API
3. **Easier testing** - API testing is simpler than UI testing
4. **Clear contracts** - OpenAPI/Swagger documents exactly what API does
5. **Parallel development** - Backend and frontend teams can work independently (future)

### Architecture Principles

**SOLID + DRY + Clean Architecture:**
- **Single Responsibility** - Each layer has one job
- **Separation of Concerns** - Data, Logic, API are separate
- **Dependency Inversion** - Layers depend on abstractions
- **DRY** - No duplication across layers

### Layered Architecture

```
┌────────────────────────────────────────────┐
│   API Layer (FastAPI Routers)              │
│   - HTTP request/response handling         │
│   - Input validation (Pydantic)            │
│   - Delegates to services                  │
│   - Returns DTOs                           │
└────────────────────────────────────────────┘
              ↓ Service calls
┌────────────────────────────────────────────┐
│   Service Layer (Business Logic)           │
│   - ProjectService                         │
│   - PipelineService                        │
│   - WebSocketService                       │
│   - Orchestration and workflows            │
│   - Business rules enforcement             │
└────────────────────────────────────────────┘
              ↓ Repository calls
┌────────────────────────────────────────────┐
│   Repository Layer (Data Access)           │
│   - ProjectRepository                      │
│   - JobRepository                          │
│   - Abstracts storage mechanism            │
└────────────────────────────────────────────┘
              ↓ Reads/writes
┌────────────────────────────────────────────┐
│   Storage (Filesystem/DB)                  │
│   - Project directories                    │
│   - State files (JSON)                     │
│   - (Future: PostgreSQL)                   │
└────────────────────────────────────────────┘
```

### Key Principles
- **API Layer**: Thin controllers - validate input, delegate to services
- **Services**: Business logic only - orchestration, rules, workflows
- **Repositories**: Data access only - abstract storage details
- **No UI**: All testing via API tools (Postman, pytest, curl)

---

## Phase 2A: Foundation & Data Layer 🟡

**Status:** 40% Complete

**Goal:** Set up project structure with proper layering

### Deliverables
- ✅ `web/` directory structure (basic) - `services/`, `static/`, `templates/`
- ✅ ConfigService with DRY configuration management
- ⚪ Pydantic models (DTOs and domain entities) - needs separation
- ⚪ Repository layer for data access - not yet extracted
- ⚪ Database/storage abstraction - currently inline in `api.py`

### Tasks

#### Task 2A.0: ✅ ConfigService Implementation (COMPLETED)
**File:** `web/services/config_service.py`

**Implemented Features:**
- Single source of truth for pipeline configuration (`web/config/pipeline_config.json`)
- DRY configuration management (no duplication)
- Stage metadata: dependencies, output directories, time estimates
- Preset configurations (quick, full, all)
- Supported video formats configuration
- WebSocket and UI configuration
- Singleton pattern for global access

**Code Quality:**
- ✅ Self-documenting function names
- ✅ Comprehensive docstrings
- ✅ Type hints throughout
- ✅ SOLID principles (SRP, OCP, DRY)
- ✅ No hardcoded values (all in JSON config)

**Success Criteria:**
- [x] Configuration centralized in single JSON file
- [x] Service provides clean API for config access
- [x] No duplication across codebase
- [x] Follows CLAUDE.md standards

---

#### Task 2A.1: Create Layered Directory Structure

**Current Structure (Partially Implemented):**
```
web/
├── server.py                   # ✅ FastAPI app entry point
├── api.py                      # ✅ API endpoints (monolithic - needs split)
├── websocket.py                # ✅ WebSocket handler
├── pipeline_runner.py          # ✅ Pipeline execution
│
├── services/                   # 🟡 PARTIAL
│   ├── __init__.py             # ✅
│   └── config_service.py       # ✅ Configuration service (DRY pattern)
│
├── config/                     # ✅ Configuration directory
│   └── pipeline_config.json    # ✅ Single source of truth
│
├── static/                     # ✅ Frontend assets
│   ├── js/                     # ✅ Modular ES6 (Roadmap 3)
│   └── css/                    # ✅ Responsive styling
│
├── templates/                  # ✅ HTML templates
│   ├── base.html               # ✅
│   └── components/             # ✅ Reusable components
│
├── repositories/               # ⚪ NOT YET IMPLEMENTED
│   ├── __init__.py
│   ├── base.py                 # Base repository interface
│   ├── project_repository.py   # Project data access
│   └── job_repository.py       # Pipeline job data access
│
├── models/                     # ⚪ NOT YET IMPLEMENTED
│   ├── __init__.py
│   ├── domain.py               # Domain entities (internal)
│   └── dto.py                  # DTOs (API contracts)
│
└── tests/                      # ⚪ NOT YET IMPLEMENTED
    ├── __init__.py
    ├── unit/                   # Unit tests
    │   ├── test_services.py
    │   └── test_repositories.py
    └── integration/            # Integration tests
        └── test_api.py
```

**Success Criteria:**
- [x] Basic directory structure created
- [x] ConfigService extracted to services layer
- [ ] Repository layer implemented (data access abstraction)
- [ ] DTO/Domain models separated
- [ ] API endpoints split into routers
- [ ] Test structure created

---

#### Task 2A.2: Define Domain Models
**File:** `web/models/domain.py`

```python
"""Domain entities - internal representation (framework-agnostic)."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List


class ProjectStatus(str, Enum):
    """Project status enumeration."""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    """Pipeline job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Project:
    """Project domain entity."""
    name: str
    path: Path
    status: ProjectStatus
    video_path: Optional[Path]
    stages: List[str]
    created_at: datetime
    updated_at: datetime

    @property
    def source_dir(self) -> Path:
        """Source directory path."""
        return self.path / "source"

    @property
    def frames_dir(self) -> Path:
        """Frames directory path."""
        return self.path / "source" / "frames"

    @property
    def state_file(self) -> Path:
        """Project state file path."""
        return self.path / "project_state.json"


@dataclass
class PipelineJob:
    """Pipeline job domain entity."""
    project_name: str
    stages: List[str]
    status: JobStatus
    current_stage: Optional[str]
    progress: float  # 0.0 to 1.0
    message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
```

**Success Criteria:**
- [ ] Domain models represent business concepts
- [ ] No framework dependencies (FastAPI, etc.)
- [ ] Type-safe with dataclasses
- [ ] Immutable where appropriate

---

#### Task 2A.3: Define DTOs (API Contracts)
**File:** `web/models/dto.py`

```python
"""Data Transfer Objects - API contracts."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from web.models.domain import ProjectStatus, JobStatus


class ProjectDTO(BaseModel):
    """Project data for API responses."""
    name: str
    status: ProjectStatus
    video_path: Optional[str] = None
    stages: List[str] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectCreateRequest(BaseModel):
    """Request to create a new project."""
    name: str = Field(..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$")
    stages: List[str] = Field(default_factory=list)


class ProjectListResponse(BaseModel):
    """Response with list of projects."""
    projects: List[ProjectDTO]
    total: int


class JobDTO(BaseModel):
    """Pipeline job data for API responses."""
    project_name: str
    stages: List[str]
    status: JobStatus
    current_stage: Optional[str] = None
    progress: float = Field(ge=0.0, le=1.0)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class JobStartRequest(BaseModel):
    """Request to start a pipeline job."""
    stages: List[str] = Field(..., min_items=1)


class JobStartResponse(BaseModel):
    """Response after starting a job."""
    status: str
    job: JobDTO


class ProgressUpdate(BaseModel):
    """Real-time progress update (WebSocket)."""
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    status: JobStatus
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
```

**Success Criteria:**
- [ ] DTOs use Pydantic for validation
- [ ] Clear request/response models
- [ ] Input validation (lengths, patterns, ranges)
- [ ] No business logic in DTOs

---

#### Task 2A.4: Implement Repository Layer
**File:** `web/repositories/base.py`

```python
"""Base repository interface."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List

T = TypeVar('T')


class Repository(ABC, Generic[T]):
    """
    Base repository interface.

    Abstracts data access - could be filesystem, database, etc.
    Follows Repository pattern for easy testing and swapping implementations.
    """

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def list(self) -> List[T]:
        """List all entities."""
        pass

    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity (create or update)."""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity by ID. Returns True if deleted, False if not found."""
        pass
```

**File:** `web/repositories/project_repository.py`

```python
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
        """List all projects, sorted by most recently updated."""
        projects = []

        if not self.projects_dir.exists():
            return projects

        for project_path in self.projects_dir.iterdir():
            if project_path.is_dir():
                project = self._load_project(project_path)
                if project:
                    projects.append(project)

        return sorted(projects, key=lambda p: p.updated_at, reverse=True)

    def save(self, project: Project) -> Project:
        """Save project to filesystem."""
        # Ensure directory exists
        project.path.mkdir(parents=True, exist_ok=True)

        # Serialize to JSON
        state = {
            "name": project.name,
            "status": project.status.value,
            "video_path": str(project.video_path) if project.video_path else None,
            "stages": project.stages,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
        }

        # Write state file
        with open(project.state_file, "w") as f:
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

        if state_file.exists():
            with open(state_file) as f:
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
        else:
            # Legacy project without state file
            return Project(
                name=project_path.name,
                path=project_path,
                status=ProjectStatus.UNKNOWN,
                video_path=None,
                stages=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
```

**File:** `web/repositories/job_repository.py`

```python
"""Job repository - in-memory implementation."""

from typing import Optional, Dict, List
from web.models.domain import PipelineJob, JobStatus
from web.repositories.base import Repository


class JobRepository(Repository[PipelineJob]):
    """
    Repository for pipeline job data.

    Current implementation: In-memory (dict)
    Rationale: Jobs are transient, don't need persistence across restarts
    Future: Could persist to DB if needed for job history
    """

    def __init__(self):
        self._jobs: Dict[str, PipelineJob] = {}

    def get(self, project_name: str) -> Optional[PipelineJob]:
        """Get job by project name."""
        return self._jobs.get(project_name)

    def list(self) -> List[PipelineJob]:
        """List all jobs."""
        return list(self._jobs.values())

    def save(self, job: PipelineJob) -> PipelineJob:
        """Save job to memory."""
        self._jobs[job.project_name] = job
        return job

    def delete(self, project_name: str) -> bool:
        """Delete job from memory."""
        if project_name in self._jobs:
            del self._jobs[project_name]
            return True
        return False

    def get_active_jobs(self) -> List[PipelineJob]:
        """Get all running or pending jobs."""
        return [
            job for job in self._jobs.values()
            if job.status in (JobStatus.RUNNING, JobStatus.PENDING)
        ]
```

**Success Criteria:**
- [ ] Repositories handle all data access
- [ ] No SQL or file I/O outside repositories
- [ ] Repository interface allows future DB migration
- [ ] Type-safe with domain models
- [ ] Error handling for file I/O

---

### Phase 2A Exit Criteria

- [ ] Layered directory structure in place
- [ ] Domain models defined (no framework deps)
- [ ] DTOs defined (Pydantic validation)
- [ ] Repository layer implemented
- [ ] Data access abstracted from business logic
- [ ] No cross-layer violations
- [ ] Can instantiate repositories in isolation

---

## Phase 2B: Service Layer (Business Logic) ⚪

**Goal:** Implement business logic separate from API

### Deliverables
- `ProjectService` - Project management logic
- `PipelineService` - Pipeline execution orchestration
- `WebSocketService` - Real-time update broadcasting

### Tasks

#### Task 2B.1: Implement Project Service
**File:** `web/services/project_service.py`

```python
"""Project service - business logic for project management."""

from pathlib import Path
from datetime import datetime
from typing import Optional, List

from web.models.domain import Project, ProjectStatus
from web.models.dto import ProjectDTO, ProjectCreateRequest, ProjectListResponse
from web.repositories.project_repository import ProjectRepository


class ProjectService:
    """
    Service for project management business logic.

    Responsibilities:
    - Enforce business rules (unique names, valid states, etc.)
    - Orchestrate project operations
    - Convert between domain entities and DTOs

    Does NOT:
    - Handle HTTP requests (that's API layer)
    - Access files directly (that's repository layer)
    """

    def __init__(self, project_repo: ProjectRepository):
        self.project_repo = project_repo

    def list_projects(self) -> ProjectListResponse:
        """List all projects."""
        projects = self.project_repo.list()

        return ProjectListResponse(
            projects=[self._to_dto(p) for p in projects],
            total=len(projects),
        )

    def get_project(self, name: str) -> Optional[ProjectDTO]:
        """Get project by name."""
        project = self.project_repo.get(name)

        if not project:
            return None

        return self._to_dto(project)

    def create_project(
        self,
        request: ProjectCreateRequest,
        projects_dir: Path
    ) -> ProjectDTO:
        """
        Create a new project.

        Business rules:
        - Project name must be unique
        - Name must be valid (alphanumeric, dash, underscore only)
        - Stages must be valid stage names
        """
        # Business rule: Project name must be unique
        existing = self.project_repo.get(request.name)
        if existing:
            raise ValueError(f"Project '{request.name}' already exists")

        # Business rule: Validate stages (if provided)
        valid_stages = {"ingest", "depth", "roto", "matanyone", "cleanplate", "colmap", "mocap", "gsir"}
        invalid_stages = set(request.stages) - valid_stages
        if invalid_stages:
            raise ValueError(f"Invalid stages: {invalid_stages}. Valid: {valid_stages}")

        # Create project entity
        now = datetime.now()
        project = Project(
            name=request.name,
            path=projects_dir / request.name,
            status=ProjectStatus.CREATED,
            video_path=None,
            stages=request.stages,
            created_at=now,
            updated_at=now,
        )

        # Persist
        project = self.project_repo.save(project)

        return self._to_dto(project)

    def save_uploaded_video(
        self,
        project_name: str,
        video_filename: str,
        video_content: bytes
    ) -> ProjectDTO:
        """Save uploaded video to project."""
        project = self.project_repo.get(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found")

        # Save video file
        video_path = project.source_dir / video_filename
        video_path.parent.mkdir(parents=True, exist_ok=True)

        with open(video_path, "wb") as f:
            f.write(video_content)

        # Update project
        project.video_path = video_path
        project.updated_at = datetime.now()

        project = self.project_repo.save(project)

        return self._to_dto(project)

    def delete_project(self, name: str) -> bool:
        """Delete a project."""
        # Business rule: Could check if job is running first
        # For now, just delegate to repository
        return self.project_repo.delete(name)

    def update_project_status(
        self,
        name: str,
        status: ProjectStatus
    ) -> Optional[ProjectDTO]:
        """Update project status."""
        project = self.project_repo.get(name)
        if not project:
            return None

        project.status = status
        project.updated_at = datetime.now()

        project = self.project_repo.save(project)

        return self._to_dto(project)

    @staticmethod
    def _to_dto(project: Project) -> ProjectDTO:
        """Convert domain entity to DTO."""
        return ProjectDTO(
            name=project.name,
            status=project.status,
            video_path=str(project.video_path) if project.video_path else None,
            stages=project.stages,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )
```

**Success Criteria:**
- [ ] All project business logic in service
- [ ] Service validates business rules
- [ ] Service converts domain ↔ DTO
- [ ] No direct file I/O (delegates to repository)
- [ ] Raises clear exceptions for business rule violations

---

#### Task 2B.2: Implement Pipeline Service
**File:** `web/services/pipeline_service.py`

(Implementation provided in original ROADMAP-2-WEB.md - same code, just extracted to this roadmap)

**Success Criteria:**
- [ ] Pipeline orchestration in service
- [ ] Business rules enforced (one job at a time per project)
- [ ] Delegates to repositories for data
- [ ] Async/await for background execution
- [ ] Proper error handling and job cleanup

---

#### Task 2B.3: Implement WebSocket Service
**File:** `web/services/websocket_service.py`

(Implementation provided in original ROADMAP-2-WEB.md - same code)

**Success Criteria:**
- [ ] WebSocket connections managed centrally
- [ ] Broadcasting logic isolated
- [ ] Dead connection cleanup
- [ ] Thread-safe (if needed for concurrent connections)

---

### Phase 2B Exit Criteria

- [ ] All business logic in service layer
- [ ] Services delegate to repositories
- [ ] Services return DTOs (not domain entities)
- [ ] No file I/O or HTTP logic in services
- [ ] Services are testable (mockable dependencies)
- [ ] Clear exception handling with business-meaningful errors

---

## Phase 2C: API Layer (Controllers) ✅

**Status:** 90% Complete

**Goal:** Thin API controllers that delegate to services

### Deliverables
- ✅ REST API endpoints (`web/api.py`)
  - Project upload and creation
  - Pipeline execution
  - System health/status endpoints
  - Video metadata extraction (ffprobe)
- ✅ WebSocket endpoint (`web/websocket.py`)
  - Real-time progress updates
  - Connection management
- ✅ Input validation with Pydantic (BaseModel classes)
- ⚪ Proper dependency injection (partially - needs improvement)
- ⚪ OpenAPI/Swagger documentation (basic auto-gen, needs enhancement)

### Implementation Details

**File:** `web/api.py`
- Project creation with video upload
- Pipeline job management (start, status, stop)
- Video metadata extraction (duration, fps, resolution, frame count)
- System health checks
- Pydantic models for request validation

**File:** `web/websocket.py`
- WebSocket connection handling
- Progress broadcast to connected clients
- Automatic reconnection support

**File:** `web/server.py`
- FastAPI app initialization
- CORS configuration
- Static file serving
- Template rendering

### Phase 2C Exit Criteria

- [x] All core API endpoints implemented
- [~] Routers are thin (mostly - some logic should move to services)
- [~] Dependency injection working (basic - needs proper DI container)
- [x] Input validation via Pydantic
- [x] Proper HTTP status codes
- [~] No business logic in routers (mostly clean - needs service extraction)
- [~] OpenAPI docs auto-generated at `/docs` (basic - needs documentation)

---

## Phase 2D: Testing ⚪

**Goal:** Comprehensive test coverage for API backend

### Deliverables
- Unit tests for services
- Unit tests for repositories
- Integration tests for API endpoints
- WebSocket integration tests
- Test fixtures and utilities

### Tasks

#### Task 2D.1: Unit Tests - Services
**File:** `web/tests/unit/test_project_service.py`

```python
"""Unit tests for ProjectService."""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
from datetime import datetime

from web.services.project_service import ProjectService
from web.models.dto import ProjectCreateRequest
from web.models.domain import Project, ProjectStatus


class TestProjectService:
    """Test ProjectService business logic."""

    def test_create_project_success(self):
        """Test successful project creation."""
        # Arrange
        mock_repo = Mock()
        mock_repo.get.return_value = None  # Project doesn't exist
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

        # Act
        result = service.create_project(request, Path("/workspace"))

        # Assert
        assert result.name == "test_project"
        assert result.status == ProjectStatus.CREATED
        mock_repo.save.assert_called_once()

    def test_create_project_duplicate_name(self):
        """Test creating project with duplicate name raises error."""
        # Arrange
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
        mock_repo.get.return_value = existing_project  # Project exists

        service = ProjectService(mock_repo)
        request = ProjectCreateRequest(name="existing", stages=[])

        # Act & Assert
        with pytest.raises(ValueError, match="already exists"):
            service.create_project(request, Path("/workspace"))

    def test_create_project_invalid_stages(self):
        """Test creating project with invalid stages raises error."""
        # Arrange
        mock_repo = Mock()
        mock_repo.get.return_value = None

        service = ProjectService(mock_repo)
        request = ProjectCreateRequest(name="test", stages=["invalid_stage"])

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid stages"):
            service.create_project(request, Path("/workspace"))

    def test_list_projects(self):
        """Test listing projects."""
        # Arrange
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

        # Act
        result = service.list_projects()

        # Assert
        assert result.total == 2
        assert len(result.projects) == 2
        assert result.projects[0].name == "project1"
```

**Success Criteria:**
- [ ] All service methods have unit tests
- [ ] Edge cases covered (duplicates, not found, invalid input)
- [ ] Mocks used for repository layer
- [ ] Tests are fast (<1s for all unit tests)
- [ ] 90%+ code coverage for services

---

#### Task 2D.2: Integration Tests - API Endpoints
**File:** `web/tests/integration/test_projects_api.py`

```python
"""Integration tests for Projects API."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

from web.server import app


@pytest.fixture
def temp_projects_dir():
    """Create temporary projects directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def client(temp_projects_dir, monkeypatch):
    """Create test client with temporary projects directory."""
    monkeypatch.setenv("VFX_PROJECTS_DIR", str(temp_projects_dir))
    return TestClient(app)


class TestProjectsAPI:
    """Test Projects REST API endpoints."""

    def test_list_projects_empty(self, client):
        """Test listing projects when none exist."""
        response = client.get("/api/projects")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["projects"] == []

    def test_create_project(self, client):
        """Test creating a new project."""
        response = client.post(
            "/api/projects",
            json={"name": "test_project", "stages": ["ingest"]}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test_project"
        assert data["status"] == "created"
        assert data["stages"] == ["ingest"]

    def test_create_project_duplicate(self, client):
        """Test creating duplicate project returns error."""
        # Create first project
        client.post("/api/projects", json={"name": "test", "stages": []})

        # Try to create duplicate
        response = client.post("/api/projects", json={"name": "test", "stages": []})

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_project_invalid_name(self, client):
        """Test creating project with invalid name."""
        response = client.post(
            "/api/projects",
            json={"name": "invalid name with spaces", "stages": []}
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_get_project(self, client):
        """Test retrieving a project."""
        # Create project
        client.post("/api/projects", json={"name": "test", "stages": []})

        # Retrieve it
        response = client.get("/api/projects/test")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test"

    def test_get_project_not_found(self, client):
        """Test retrieving non-existent project."""
        response = client.get("/api/projects/nonexistent")

        assert response.status_code == 404

    def test_delete_project(self, client):
        """Test deleting a project."""
        # Create project
        client.post("/api/projects", json={"name": "test", "stages": []})

        # Delete it
        response = client.delete("/api/projects/test")

        assert response.status_code == 204

        # Verify deleted
        response = client.get("/api/projects/test")
        assert response.status_code == 404
```

**Success Criteria:**
- [ ] All API endpoints have integration tests
- [ ] Tests use TestClient (not real HTTP)
- [ ] Tests are isolated (temp directories, clean state)
- [ ] Error cases covered (400, 404, 500)
- [ ] Can run entire test suite in CI/CD

---

#### Task 2D.3: WebSocket Integration Tests
**File:** `web/tests/integration/test_websocket.py`

```python
"""Integration tests for WebSocket functionality."""

import pytest
from fastapi.testclient import TestClient
import json

from web.server import app


class TestWebSocketAPI:
    """Test WebSocket real-time progress updates."""

    def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        client = TestClient(app)

        with client.websocket_connect("/api/pipeline/ws/test_project") as websocket:
            # Connection successful
            assert websocket is not None

    def test_websocket_receives_progress(self):
        """Test receiving progress updates via WebSocket."""
        client = TestClient(app)

        with client.websocket_connect("/api/pipeline/ws/test_project") as websocket:
            # Simulate progress update (would come from pipeline service)
            # In real scenario, start a job and receive updates

            # For now, just verify connection works
            # Full test would involve:
            # 1. Start job via API
            # 2. Receive progress updates via WebSocket
            # 3. Verify progress data matches expected format
            pass
```

**Success Criteria:**
- [ ] WebSocket connection tested
- [ ] Progress updates validated
- [ ] Multiple concurrent connections tested
- [ ] Disconnect handling tested

---

### Phase 2D Exit Criteria

- [ ] Unit tests for all services (90%+ coverage)
- [ ] Integration tests for all API endpoints
- [ ] WebSocket tests pass
- [ ] All tests run in <30 seconds
- [ ] CI/CD pipeline configured (GitHub Actions)
- [ ] Test coverage report generated

---

## Phase 2E: Documentation & Validation ⚪

**Goal:** Document API and validate it's ready for frontend development

### Deliverables
- OpenAPI/Swagger documentation
- API usage examples
- Postman collection
- Architecture documentation

### Tasks

#### Task 2E.1: OpenAPI Documentation
**File:** `web/server.py` (update)

```python
"""FastAPI server with enhanced documentation."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="VFX Ingest Platform API",
    description="REST API for VFX pipeline project management and execution",
    version="1.0.0",
    docs_url="/api/docs",  # Swagger UI
    redoc_url="/api/redoc",  # ReDoc
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="VFX Ingest Platform API",
        version="1.0.0",
        description="""
        # VFX Ingest Platform API

        REST API for managing VFX pipeline projects and executing stages.

        ## Features
        - Create and manage projects
        - Upload video files
        - Execute pipeline stages (ingest, depth, colmap, etc.)
        - Real-time progress updates via WebSocket

        ## Authentication
        Currently no authentication required (local deployment).

        ## Rate Limiting
        No rate limiting (single-user deployment).
        """,
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

**Success Criteria:**
- [ ] OpenAPI spec accessible at `/api/docs`
- [ ] All endpoints documented
- [ ] Request/response schemas documented
- [ ] Example requests provided

---

#### Task 2E.2: API Usage Guide
**File:** `docs/API-USAGE.md`

```markdown
# API Usage Guide

## Base URL
```
http://localhost:5000/api
```

## Endpoints

### Projects

#### List All Projects
```bash
curl http://localhost:5000/api/projects
```

Response:
```json
{
  "projects": [
    {
      "name": "Shot_01",
      "status": "created",
      "stages": ["ingest", "colmap"],
      "created_at": "2026-01-17T10:00:00",
      "updated_at": "2026-01-17T10:00:00"
    }
  ],
  "total": 1
}
```

#### Create Project
```bash
curl -X POST http://localhost:5000/api/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Shot_01",
    "stages": ["ingest", "colmap"]
  }'
```

#### Upload Video
```bash
curl -X POST http://localhost:5000/api/projects/Shot_01/upload-video \
  -F "video=@/path/to/video.mp4"
```

### Pipeline

#### Start Job
```bash
curl -X POST http://localhost:5000/api/pipeline/projects/Shot_01/start \
  -H "Content-Type: application/json" \
  -d '{
    "stages": ["ingest", "depth", "colmap"]
  }'
```

#### Get Job Status
```bash
curl http://localhost:5000/api/pipeline/projects/Shot_01/status
```

#### WebSocket Progress Updates
```javascript
const ws = new WebSocket('ws://localhost:5000/api/pipeline/ws/Shot_01');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`Stage: ${update.stage}, Progress: ${update.progress}`);
};
```
```

**Success Criteria:**
- [ ] Usage guide covers all endpoints
- [ ] curl examples provided
- [ ] WebSocket usage documented
- [ ] Error responses documented

---

#### Task 2E.3: Postman Collection
**File:** `docs/VFX_Ingest_API.postman_collection.json`

Create Postman collection with:
- All REST endpoints
- Example requests
- Environment variables
- Tests for assertions

**Success Criteria:**
- [ ] Postman collection exported
- [ ] Can import and run all requests
- [ ] Environment variables documented

---

### Phase 2E Exit Criteria

- [ ] OpenAPI docs complete and accurate
- [ ] API usage guide written
- [ ] Postman collection tested
- [ ] Can use API independently (without UI)
- [ ] Ready for frontend development

---

## Roadmap 2 Success Criteria

**Ready to move to Roadmap 3 when:**

- [ ] All phases complete
- [ ] Proper layered architecture validated
- [ ] Services are unit testable (90%+ coverage)
- [ ] All API endpoints tested (integration)
- [ ] OpenAPI documentation complete
- [ ] Can manage projects via API only (no UI needed)
- [ ] WebSocket streams real-time progress
- [ ] Performance acceptable
- [ ] No cross-layer violations
- [ ] API contracts stable (unlikely to change)

**Architecture Quality Checklist:**
- [ ] SOLID principles followed
- [ ] DRY - no duplication across layers
- [ ] Separation of concerns maintained
- [ ] Easy to test (mockable dependencies)
- [ ] Easy to change (swap repositories, add endpoints)
- [ ] Easy to understand (clear responsibilities)
- [ ] API-first design (could support multiple frontends)

---

**Previous:** [Roadmap 1: Docker Migration](ROADMAP-1-DOCKER.md)
**Next:** [Roadmap 3: Web UI Frontend](ROADMAP-3-WEB-UI.md)
**Up:** [Atlas Overview](ATLAS.md)
