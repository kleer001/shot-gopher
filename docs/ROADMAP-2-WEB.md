# 🌐 Roadmap 2: Web Interface

**Goal:** Add browser-based GUI layer for artist-friendly access

**Status:** ⚪ Not Started

**Dependencies:** Roadmap 1 (Docker Migration) must be complete

---

## Overview

This roadmap builds a FastAPI web application on top of the containerized pipeline, providing an intuitive browser interface for non-technical artists.

### Key Principles
- **Zero CLI Required** - All functionality accessible via browser
- **Real-time Feedback** - Live progress updates via WebSockets
- **Artist-Centric** - Designed for visual artists, not programmers
- **One-Click Operations** - Start, stop, monitor with simple clicks
- **Project Management** - Visual dashboard for all projects

---

## Phase 2A: Web Server Foundation ⚪

**Goal:** FastAPI server with basic routing and project management

### Deliverables
- `web/` directory structure
- `web/server.py` - FastAPI application
- `web/templates/` - HTML templates (Jinja2)
- `web/static/` - CSS, JavaScript, images
- Updated Dockerfile with web server

### Tasks

#### Task 2A.1: Create Web Application Structure
**Directory Structure:**
```
web/
├── server.py              # FastAPI app entry point
├── routers/
│   ├── __init__.py
│   ├── dashboard.py       # Dashboard routes
│   ├── projects.py        # Project CRUD routes
│   ├── pipeline.py        # Pipeline execution routes
│   └── api.py             # REST API endpoints
├── templates/
│   ├── base.html          # Base template
│   ├── dashboard.html     # Main dashboard
│   ├── project_new.html   # New project form
│   ├── project_view.html  # Project details
│   └── processing.html    # Real-time processing view
├── static/
│   ├── css/
│   │   └── styles.css     # Custom styles
│   ├── js/
│   │   ├── dashboard.js   # Dashboard interactions
│   │   └── websocket.js   # WebSocket client
│   └── img/
│       └── logo.png
└── models.py              # Pydantic models for API
```

**Success Criteria:**
- [ ] Directory structure created
- [ ] Basic file stubs present

---

#### Task 2A.2: Create FastAPI Server
**File:** `web/server.py`

```python
"""FastAPI web server for VFX Ingest Platform."""

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from web.routers import dashboard, projects, pipeline, api

app = FastAPI(title="VFX Ingest Platform")

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Templates
templates = Jinja2Templates(directory="web/templates")

# Include routers
app.include_router(dashboard.router)
app.include_router(projects.router)
app.include_router(pipeline.router)
app.include_router(api.router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

**Success Criteria:**
- [ ] Server starts without errors
- [ ] `/health` endpoint responds
- [ ] Static files served correctly

---

#### Task 2A.3: Create Dashboard Router
**File:** `web/routers/dashboard.py`

```python
"""Dashboard routes."""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

PROJECTS_DIR = Path(os.environ.get("VFX_PROJECTS_DIR", "/workspace/projects"))

@router.get("/dashboard")
async def dashboard(request: Request):
    """Main dashboard view."""

    # Scan for projects
    projects = []
    if PROJECTS_DIR.exists():
        for project_path in PROJECTS_DIR.iterdir():
            if project_path.is_dir():
                # Read project metadata
                state_file = project_path / "project_state.json"
                if state_file.exists():
                    import json
                    with open(state_file) as f:
                        state = json.load(f)
                else:
                    state = {"name": project_path.name, "status": "unknown"}

                projects.append(state)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "projects": projects,
        }
    )
```

**Success Criteria:**
- [ ] Dashboard lists existing projects
- [ ] Project metadata loaded correctly
- [ ] Handles missing projects gracefully

---

#### Task 2A.4: Create Project Management Router
**File:** `web/routers/projects.py`

```python
"""Project management routes."""

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pathlib import Path
import os
import shutil
import json

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

PROJECTS_DIR = Path(os.environ.get("VFX_PROJECTS_DIR", "/workspace/projects"))

@router.get("/projects/new")
async def new_project_form(request: Request):
    """Show new project form."""
    return templates.TemplateResponse("project_new.html", {"request": request})

@router.post("/projects/create")
async def create_project(
    request: Request,
    name: str = Form(...),
    video: UploadFile = File(...),
    stages: list[str] = Form([])
):
    """Create new project from uploaded video."""

    # Create project directory
    project_dir = PROJECTS_DIR / name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = project_dir / "source" / video.filename
    video_path.parent.mkdir(exist_ok=True)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Create project state
    state = {
        "name": name,
        "status": "created",
        "video_path": str(video_path),
        "stages": stages,
        "created": str(datetime.now()),
    }

    with open(project_dir / "project_state.json", "w") as f:
        json.dump(state, f, indent=2)

    return RedirectResponse(f"/projects/{name}", status_code=303)

@router.get("/projects/{name}")
async def view_project(request: Request, name: str):
    """View project details."""

    project_dir = PROJECTS_DIR / name

    # Load state
    state_file = project_dir / "project_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
    else:
        state = {"name": name, "status": "unknown"}

    return templates.TemplateResponse(
        "project_view.html",
        {
            "request": request,
            "project": state,
        }
    )

@router.post("/projects/{name}/delete")
async def delete_project(name: str):
    """Delete a project."""

    project_dir = PROJECTS_DIR / name
    if project_dir.exists():
        shutil.rmtree(project_dir)

    return RedirectResponse("/dashboard", status_code=303)
```

**Success Criteria:**
- [ ] New project form displays
- [ ] Video upload works
- [ ] Project directory created correctly
- [ ] Project state saved

---

### Phase 2A Exit Criteria

- [ ] Web server runs and responds
- [ ] Dashboard displays projects
- [ ] Can create new projects via web form
- [ ] Can view project details
- [ ] Can delete projects
- [ ] Basic navigation works

---

## Phase 2B: Pipeline Integration ⚪

**Goal:** Execute pipeline from web interface with real-time progress

### Deliverables
- `web/routers/pipeline.py` - Pipeline execution routes
- `web/websocket.py` - WebSocket handler for progress updates
- `web/pipeline_manager.py` - Background job orchestration
- Modified `scripts/run_pipeline.py` to emit JSON progress

### Tasks

#### Task 2B.1: Create Pipeline Manager
**File:** `web/pipeline_manager.py`

```python
"""Background pipeline job management."""

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Optional
import os

class PipelineJob:
    """Represents a running pipeline job."""

    def __init__(self, project_name: str, stages: list[str]):
        self.project_name = project_name
        self.stages = stages
        self.process: Optional[subprocess.Popen] = None
        self.status = "pending"
        self.current_stage = None
        self.progress = 0.0

    async def start(self):
        """Start the pipeline job."""

        cmd = [
            "python", "/app/scripts/run_pipeline.py",
            "--name", self.project_name,
            "--stages", ",".join(self.stages),
            "--json-output",  # New flag for JSON progress
        ]

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self.status = "running"

    async def monitor(self, callback):
        """Monitor job progress and call callback with updates."""

        if not self.process:
            return

        async for line in self.process.stdout:
            try:
                data = json.loads(line.decode())

                # Update state
                if "stage" in data:
                    self.current_stage = data["stage"]
                if "progress" in data:
                    self.progress = data["progress"]
                if "status" in data:
                    self.status = data["status"]

                # Notify via callback
                await callback(data)

            except json.JSONDecodeError:
                # Regular log line, ignore
                pass

        # Wait for completion
        await self.process.wait()

        if self.process.returncode == 0:
            self.status = "complete"
        else:
            self.status = "failed"

    def stop(self):
        """Stop the job."""
        if self.process:
            self.process.terminate()
            self.status = "cancelled"


class PipelineManager:
    """Manages pipeline jobs."""

    def __init__(self):
        self.jobs: dict[str, PipelineJob] = {}

    def create_job(self, project_name: str, stages: list[str]) -> PipelineJob:
        """Create a new job."""
        job = PipelineJob(project_name, stages)
        self.jobs[project_name] = job
        return job

    def get_job(self, project_name: str) -> Optional[PipelineJob]:
        """Get a job by project name."""
        return self.jobs.get(project_name)

    def stop_job(self, project_name: str):
        """Stop a running job."""
        job = self.jobs.get(project_name)
        if job:
            job.stop()
```

**Success Criteria:**
- [ ] Can create pipeline jobs
- [ ] Jobs run in background
- [ ] Progress monitoring works
- [ ] Can stop jobs

---

#### Task 2B.2: Add JSON Output to run_pipeline.py
**File:** `scripts/run_pipeline.py`

**Modifications:**

```python
import json
import sys

def emit_progress(stage: str, progress: float, status: str, message: str = ""):
    """Emit progress as JSON to stdout."""

    if os.environ.get("JSON_OUTPUT") == "true":
        data = {
            "stage": stage,
            "progress": progress,
            "status": status,
            "message": message,
        }
        print(json.dumps(data), flush=True)
    else:
        # Regular console output
        print(f"[{stage}] {progress:.0%} - {message}")


def run_stage_ingest(project_dir, input_video):
    """Run ingest stage."""

    emit_progress("ingest", 0.0, "running", "Starting frame extraction")

    # ... existing FFmpeg logic ...

    emit_progress("ingest", 0.5, "running", "Extracting frames")

    # ... extraction ...

    emit_progress("ingest", 1.0, "complete", f"Extracted {frame_count} frames")


# Apply to all stages...
```

**Success Criteria:**
- [ ] `--json-output` flag added
- [ ] All stages emit JSON progress
- [ ] Regular output still works without flag
- [ ] JSON format consistent

---

#### Task 2B.3: Create WebSocket Handler
**File:** `web/websocket.py`

```python
"""WebSocket handler for real-time progress updates."""

from fastapi import WebSocket
from typing import Dict, Set
import asyncio

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        # Map project_name -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, project_name: str):
        """Accept a new connection."""
        await websocket.accept()

        if project_name not in self.active_connections:
            self.active_connections[project_name] = set()

        self.active_connections[project_name].add(websocket)

    def disconnect(self, websocket: WebSocket, project_name: str):
        """Remove a connection."""
        if project_name in self.active_connections:
            self.active_connections[project_name].discard(websocket)

    async def broadcast(self, project_name: str, message: dict):
        """Broadcast message to all connections for a project."""

        if project_name not in self.active_connections:
            return

        dead_connections = set()

        for connection in self.active_connections[project_name]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.active_connections[project_name].discard(conn)
```

**Success Criteria:**
- [ ] WebSocket connections managed
- [ ] Broadcasting works to multiple clients
- [ ] Dead connections cleaned up

---

#### Task 2B.4: Create Pipeline Routes
**File:** `web/routers/pipeline.py`

```python
"""Pipeline execution routes."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from web.pipeline_manager import PipelineManager
from web.websocket import ConnectionManager

router = APIRouter()
pipeline_manager = PipelineManager()
connection_manager = ConnectionManager()

@router.post("/projects/{name}/start")
async def start_pipeline(name: str, stages: list[str]):
    """Start pipeline for a project."""

    # Create job
    job = pipeline_manager.create_job(name, stages)

    # Start in background
    await job.start()

    # Monitor progress
    async def progress_callback(data):
        await connection_manager.broadcast(name, data)

    asyncio.create_task(job.monitor(progress_callback))

    return {"status": "started"}

@router.post("/projects/{name}/stop")
async def stop_pipeline(name: str):
    """Stop pipeline for a project."""

    pipeline_manager.stop_job(name)
    return {"status": "stopped"}

@router.websocket("/ws/{project_name}")
async def websocket_endpoint(websocket: WebSocket, project_name: str):
    """WebSocket endpoint for real-time progress."""

    await connection_manager.connect(websocket, project_name)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, project_name)
```

**Success Criteria:**
- [ ] Can start pipeline from API
- [ ] Can stop pipeline from API
- [ ] WebSocket receives progress updates
- [ ] Multiple clients can monitor same job

---

### Phase 2B Exit Criteria

- [ ] Pipeline executes from web interface
- [ ] Real-time progress visible in browser
- [ ] Can start/stop jobs
- [ ] Multiple stages work correctly
- [ ] Error handling in place

---

## Phase 2C: User Interface Implementation ⚪

**Goal:** Complete HTML/CSS/JavaScript frontend

### Deliverables
- All HTML templates
- CSS styling
- JavaScript for interactions
- WebSocket client code

### Tasks

#### Task 2C.1: Create Base Template
**File:** `web/templates/base.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}VFX Ingest Platform{% endblock %}</title>
    <link rel="stylesheet" href="/static/css/styles.css">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header>
        <div class="container">
            <h1>VFX Ingest Platform</h1>
            <nav>
                <a href="/dashboard">Dashboard</a>
                <a href="/projects/new">New Project</a>
                <a href="#" id="shutdown-btn">Shutdown</a>
            </nav>
        </div>
    </header>

    <main class="container">
        {% block content %}{% endblock %}
    </main>

    <footer>
        <div class="container">
            <p>VFX Ingest Platform v1.0</p>
        </div>
    </footer>

    <script src="/static/js/websocket.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

---

#### Task 2C.2: Create Dashboard Template
**File:** `web/templates/dashboard.html`

```html
{% extends "base.html" %}

{% block content %}
<h2>Projects</h2>

<div class="project-grid">
    {% for project in projects %}
    <div class="project-card" data-status="{{ project.status }}">
        <div class="project-thumbnail">
            <img src="/static/img/placeholder.png" alt="{{ project.name }}">
        </div>
        <h3>{{ project.name }}</h3>
        <p class="status">{{ project.status }}</p>
        <div class="actions">
            <a href="/projects/{{ project.name }}" class="btn">Open</a>
        </div>
    </div>
    {% endfor %}

    <div class="project-card new-project">
        <a href="/projects/new">
            <div class="plus-icon">+</div>
            <p>New Project</p>
        </a>
    </div>
</div>
{% endblock %}
```

---

#### Task 2C.3: Create Processing View Template
**File:** `web/templates/processing.html`

```html
{% extends "base.html" %}

{% block content %}
<h2>{{ project.name }} - Processing</h2>

<div class="progress-container">
    <div class="overall-progress">
        <h3>Overall Progress</h3>
        <div class="progress-bar">
            <div class="progress-fill" id="overall-progress" style="width: 0%"></div>
        </div>
        <p id="overall-status">Starting...</p>
    </div>

    <div class="stage-progress">
        <h3>Current Stage: <span id="current-stage">-</span></h3>
        <div class="progress-bar">
            <div class="progress-fill" id="stage-progress" style="width: 0%"></div>
        </div>
        <p id="stage-status">-</p>
    </div>
</div>

<div class="stage-list">
    <h3>Stages</h3>
    <ul id="stages">
        <!-- Populated by JavaScript -->
    </ul>
</div>

<div class="actions">
    <button id="pause-btn" class="btn">Pause</button>
    <button id="stop-btn" class="btn btn-danger">Stop</button>
</div>

<div class="logs">
    <h3>Live Logs <button id="toggle-logs">Show</button></h3>
    <pre id="log-output" style="display: none;"></pre>
</div>
{% endblock %}

{% block extra_js %}
<script>
    // WebSocket connection for real-time updates
    const projectName = "{{ project.name }}";
    const ws = new WebSocket(`ws://${window.location.host}/ws/${projectName}`);

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);

        // Update progress bars
        if (data.progress !== undefined) {
            document.getElementById('stage-progress').style.width = (data.progress * 100) + '%';
        }

        if (data.stage) {
            document.getElementById('current-stage').textContent = data.stage;
        }

        if (data.message) {
            document.getElementById('stage-status').textContent = data.message;

            // Append to logs
            const logs = document.getElementById('log-output');
            logs.textContent += data.message + '\n';
            logs.scrollTop = logs.scrollHeight;
        }
    };

    // Stop button
    document.getElementById('stop-btn').addEventListener('click', async function() {
        if (confirm('Are you sure you want to stop processing? Progress will be lost.')) {
            await fetch(`/projects/${projectName}/stop`, { method: 'POST' });
            window.location.href = `/projects/${projectName}`;
        }
    });
</script>
{% endblock %}
```

---

#### Task 2C.4: Create Stylesheet
**File:** `web/static/css/styles.css`

```css
/* Base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
header {
    background: #2c3e50;
    color: white;
    padding: 20px 0;
}

header h1 {
    display: inline-block;
    margin-right: 40px;
}

nav {
    display: inline-block;
}

nav a {
    color: white;
    text-decoration: none;
    margin-right: 20px;
}

/* Project grid */
.project-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.project-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.project-thumbnail img {
    width: 100%;
    height: 150px;
    object-fit: cover;
    border-radius: 4px;
}

/* Progress bars */
.progress-bar {
    width: 100%;
    height: 30px;
    background: #e0e0e0;
    border-radius: 15px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3498db, #2ecc71);
    transition: width 0.3s ease;
}

/* Buttons */
.btn {
    display: inline-block;
    padding: 10px 20px;
    background: #3498db;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    border: none;
    cursor: pointer;
}

.btn:hover {
    background: #2980b9;
}

.btn-danger {
    background: #e74c3c;
}

.btn-danger:hover {
    background: #c0392b;
}

/* Logs */
#log-output {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 15px;
    border-radius: 4px;
    max-height: 300px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
```

---

### Phase 2C Exit Criteria

- [ ] All templates render correctly
- [ ] Styling consistent and professional
- [ ] Responsive design works on different screen sizes
- [ ] JavaScript interactions functional
- [ ] WebSocket client connects and updates UI

---

## Phase 2D: Startup/Shutdown Management ⚪

**Goal:** One-click platform startup and graceful shutdown

### Deliverables
- `start-platform.sh` / `start-platform.bat` - Platform launcher
- `stop-platform.sh` / `stop-platform.bat` - Platform stopper
- Desktop shortcuts (platform-specific)
- Graceful shutdown handling in web app

### Tasks

#### Task 2D.1: Create Platform Launcher
**File:** `start-platform.sh`

```bash
#!/bin/bash
set -e

echo "=== VFX Ingest Platform ==="

# Check if already running
if docker ps | grep -q vfx-ingest-web; then
    echo "Platform already running!"
    URL="http://localhost:5000"
else
    # Start container
    echo "Starting platform..."
    docker-compose up -d

    # Wait for health check
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    URL="http://localhost:5000"
fi

# Open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "$URL"
fi

echo "✓ Platform running at $URL"
echo ""
echo "To stop: Click 'Shutdown' in web interface"
echo "  or run: ./stop-platform.sh"
```

**Success Criteria:**
- [ ] Detects if already running
- [ ] Starts container if needed
- [ ] Opens browser automatically
- [ ] Platform-specific browser opening works

---

#### Task 2D.2: Create Shutdown Endpoint
**File:** `web/routers/api.py`

```python
"""API routes."""

from fastapi import APIRouter
import os
import signal

router = APIRouter()

@router.post("/shutdown")
async def shutdown():
    """Graceful shutdown of the platform."""

    # Check for running jobs
    from web.pipeline_manager import pipeline_manager

    active_jobs = [
        name for name, job in pipeline_manager.jobs.items()
        if job.status == "running"
    ]

    if active_jobs:
        return {
            "status": "warning",
            "message": f"Active jobs: {', '.join(active_jobs)}",
            "jobs": active_jobs,
        }

    # Shutdown server
    os.kill(os.getpid(), signal.SIGTERM)

    return {"status": "shutting_down"}
```

**Success Criteria:**
- [ ] Shutdown endpoint implemented
- [ ] Warns about active jobs
- [ ] Gracefully stops server

---

### Phase 2D Exit Criteria

- [ ] One-click startup works on all platforms
- [ ] Browser opens automatically
- [ ] Shutdown button functional
- [ ] Active job warnings work
- [ ] Graceful cleanup on shutdown

---

## Phase 2E: Testing & Polish ⚪

**Goal:** End-to-end testing and UX refinements

### Test Cases

#### Test 2E.1: Complete User Flow
1. Run `./start-platform.sh`
2. Browser opens to dashboard
3. Click "New Project"
4. Upload video file
5. Select stages
6. Click "Start Processing"
7. Monitor real-time progress
8. View completed results
9. Click "Shutdown"
10. Verify graceful shutdown

**Success Criteria:**
- [ ] Entire flow completes without errors
- [ ] No console errors in browser
- [ ] Progress updates smooth
- [ ] Results accessible

---

#### Test 2E.2: Multi-Project Management
1. Create project A
2. Start processing project A
3. Create project B (should queue or warn)
4. View both projects in dashboard
5. Check status indicators correct

**Success Criteria:**
- [ ] Multiple projects visible
- [ ] Status accurate for each
- [ ] Can't start two jobs simultaneously

---

#### Test 2E.3: Error Recovery
1. Start pipeline with missing model
2. Verify error displayed in UI
3. Stop failed job
4. Retry with correct configuration

**Success Criteria:**
- [ ] Errors surface in UI
- [ ] Can recover from errors
- [ ] No stale state

---

### Phase 2E Exit Criteria

- [ ] All test cases pass
- [ ] No critical bugs
- [ ] UX smooth and intuitive
- [ ] Error messages helpful
- [ ] Documentation complete

---

## Roadmap 2 Success Criteria

**Ready for production when:**

- [ ] All Phase 2A-2E tasks complete
- [ ] Web interface fully functional
- [ ] Real-time progress updates work
- [ ] One-click startup/shutdown operational
- [ ] All Roadmap 1 functionality accessible via web
- [ ] User testing with 3+ artists successful
- [ ] Documentation complete
- [ ] No critical bugs
- [ ] Performance acceptable

**Future Enhancements (v2.0+):**
- Batch processing queue
- Remote server mode (multi-user)
- Advanced preset management
- Integrated help/tutorials
- Export/import projects
- Mobile monitoring app

---

**Previous:** [Roadmap 1: Docker Migration](ROADMAP-1-DOCKER.md)
**Up:** [Atlas Overview](ATLAS.md)
