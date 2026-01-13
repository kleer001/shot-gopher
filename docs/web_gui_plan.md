# Web GUI Implementation Plan

A local web interface for the VFX Pipeline - drag-and-drop video processing with real-time progress monitoring.

## Overview

The web GUI provides a browser-based interface to the existing pipeline, served from a local Python server. Users can upload videos, configure processing stages, monitor progress, and download results without touching the command line.

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                    User's Browser                        │
│         http://localhost:5000                           │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                 Web Server (FastAPI)                     │
│  ├── Static files (HTML/CSS/JS)                         │
│  ├── REST API endpoints                                 │
│  └── WebSocket for real-time progress                   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│               Existing Pipeline Scripts                  │
│  ├── run_pipeline.py                                    │
│  ├── env_config.py (paths, env settings)                │
│  └── comfyui_utils.py (workflow execution)              │
└─────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Minimal new code** - Reuse existing scripts, don't rewrite pipeline logic
2. **No external services** - Everything runs locally, no cloud dependencies
3. **No build step** - Vanilla HTML/CSS/JS, no npm/webpack
4. **Single entry point** - One command from repo root: `./start_web.py` (auto-launches browser)
5. **Respect existing config** - Use `env_config.py` for all paths
6. **Sensible defaults** - Roto prompt defaults to "person" (covers 80% of use cases)

## MVP Features

### Must Have
- [ ] Video upload: drag-and-drop **and** browse button (not everyone likes drag-and-drop)
- [ ] Stage selection (checkboxes or preset)
- [ ] Roto prompt text input (defaults to "person", editable)
- [ ] "Start Processing" button
- [ ] Progress display (current stage, percentage)
- [ ] "Done" state with output file listing
- [ ] "Open Folder" button to reveal outputs

### Nice to Have (Post-MVP)
- [ ] Thumbnail previews of each pass
- [ ] Log viewer (collapsible)
- [ ] Job history / project list
- [ ] Re-run with different settings
- [ ] Download ZIP of all outputs
- [ ] Side-by-side comparison viewer

## File Structure

```
comfyui_ingest/
├── start_web.py                  # NEW: Root entry point (launches browser)
├── web/                          # NEW: Web UI package
│   ├── __init__.py
│   ├── server.py                 # FastAPI application
│   ├── api.py                    # REST API endpoints
│   ├── websocket.py              # WebSocket handlers
│   ├── pipeline_runner.py        # Pipeline execution wrapper
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── app.js            # Frontend logic
│   └── templates/
│       └── index.html            # Main page
├── scripts/
│   └── ...                       # Existing scripts unchanged
└── requirements.txt              # Updated with web dependencies
```

## API Design

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main HTML page |
| `/api/upload` | POST | Upload video file, returns `project_id` |
| `/api/projects` | GET | List all projects |
| `/api/projects/{id}` | GET | Get project status and details |
| `/api/projects/{id}/start` | POST | Start processing with config |
| `/api/projects/{id}/stop` | POST | Cancel processing |
| `/api/projects/{id}/outputs` | GET | List output files |
| `/api/system/status` | GET | Check ComfyUI, disk space, etc. |

### WebSocket

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client→Server | Establish connection for project |
| `progress` | Server→Client | Stage progress update |
| `log` | Server→Client | Log line (optional) |
| `stage_complete` | Server→Client | Stage finished |
| `pipeline_complete` | Server→Client | All stages done |
| `error` | Server→Client | Error occurred |

### Request/Response Examples

**Upload Video:**
```http
POST /api/upload
Content-Type: multipart/form-data

file: <video_file>
name: "My_Shot"  (optional, defaults to filename)
```

Response:
```json
{
  "project_id": "my_shot_20240115_143022",
  "project_dir": "/path/to/vfx_projects/My_Shot",
  "video_info": {
    "duration": 10.5,
    "fps": 24.0,
    "resolution": [1920, 1080],
    "frame_count": 252
  }
}
```

**Start Processing:**
```http
POST /api/projects/{id}/start
Content-Type: application/json

{
  "stages": ["ingest", "depth", "roto", "cleanplate"],
  "roto_prompt": "person",
  "skip_existing": false
}
```

**Progress WebSocket Message:**
```json
{
  "type": "progress",
  "stage": "roto",
  "stage_index": 2,
  "total_stages": 4,
  "progress": 0.42,
  "frame": 84,
  "total_frames": 200,
  "message": "Processing frame 84..."
}
```

## UI States

### State 1: Ready (Initial)

```
┌─────────────────────────────────────────────────────────┐
│                    VFX Pipeline                         │
│                                                         │
│         ┌─────────────────────────────────┐            │
│         │                                 │            │
│         │      Drop video here            │            │
│         │             or                  │            │
│         │      [Browse Files...]          │            │
│         │                                 │            │
│         │      Supported: mp4, mov, avi   │            │
│         │                                 │            │
│         └─────────────────────────────────┘            │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  Recent Projects:                                       │
│  (none yet)                                            │
│                                                         │
│  System Status: ● ComfyUI running                      │
└─────────────────────────────────────────────────────────┘
```

### State 2: Configure

```
┌─────────────────────────────────────────────────────────┐
│  hero_shot.mp4                              [✕ Cancel] │
│  1920x1080 • 24fps • 252 frames • 10.5s               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Project Name: [hero_shot___________________]          │
│                                                         │
│  Processing Stages:                                     │
│  ☑ Depth Maps                                          │
│  ☑ Segmentation (Roto)                                 │
│      Prompt: [person______________________]            │
│  ☑ Clean Plate                                         │
│  ☐ Camera Solve (COLMAP)                               │
│  ☐ Materials (GS-IR) - Requires COLMAP                 │
│  ☐ Motion Capture - Requires COLMAP                    │
│                                                         │
│  ─────────────── Quick Presets ───────────────         │
│  [Quick Preview]  [Full VFX]  [Everything]             │
│                                                         │
│  ☐ Skip existing outputs                               │
│                                                         │
│                              [▶ Start Processing]      │
└─────────────────────────────────────────────────────────┘
```

### State 3: Processing

```
┌─────────────────────────────────────────────────────────┐
│  hero_shot                                  [■ Cancel] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Stage 2 of 4: Segmentation                            │
│                                                         │
│  ████████████████░░░░░░░░░░░░░░  42%                  │
│                                                         │
│  Frame 84 / 200                                        │
│  Elapsed: 2m 34s • Remaining: ~3m 30s                  │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  ✓ Ingest           200 frames extracted               │
│  ✓ Depth            200 depth maps                     │
│  ◐ Segmentation     84/200 masks...                    │
│  ○ Clean Plate      pending                            │
│                                                         │
│                                           [▼ Show Log] │
└─────────────────────────────────────────────────────────┘
```

### State 4: Complete

```
┌─────────────────────────────────────────────────────────┐
│  hero_shot                          ✓ Complete         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Output Passes:                                         │
│                                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ source  │ │  depth  │ │  roto   │ │ clean   │      │
│  │  [img]  │ │  [img]  │ │  [img]  │ │  [img]  │      │
│  │ 200 fr  │ │ 200 fr  │ │ 200 fr  │ │ 200 fr  │      │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
│                                                         │
│  Total processing time: 6m 04s                         │
│                                                         │
│  [📁 Open Folder]                   [🔄 Run Again]     │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│  [← New Project]                                       │
└─────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Backend)
1. Create `web/` package structure
2. Implement FastAPI server with static file serving
3. Create upload endpoint with video validation
4. Implement project status endpoint
5. Create pipeline runner wrapper (calls `run_pipeline.py`)

### Phase 2: Core UI (Frontend)
1. HTML page with drop zone
2. CSS styling (dark theme, clean layout)
3. JavaScript for drag-and-drop upload
4. Stage selection form
5. Basic progress polling (before WebSocket)

### Phase 3: Real-Time Progress
1. WebSocket server integration
2. Pipeline output parsing for progress
3. Frontend WebSocket client
4. Real-time progress bar updates
5. Log streaming (optional)

### Phase 4: Polish
1. Project history / listing
2. Thumbnail generation for outputs
3. Error handling and display
4. "Open Folder" integration
5. System status checks

## Technical Decisions

### Why FastAPI?
- Async-native (good for WebSocket + long-running tasks)
- Automatic OpenAPI docs at `/docs`
- Built-in WebSocket support
- Minimal boilerplate

### Why Vanilla JS?
- No build step required
- Works offline
- Small bundle size (it's just a few KB)
- Easy to modify

### Why Local Server (not GitHub Pages)?
- Can't execute backend code from static hosting
- Can't access local filesystem
- Can't run GPU processing remotely
- Simpler architecture (no CORS issues)

## Dependencies

**New Python packages:**
```
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6  # For file uploads
websockets>=11.0         # For real-time progress
```

**Optional (for thumbnails):**
```
pillow>=9.0.0            # Already in requirements
```

## Entry Point

Root-level `start_web.py` (in repo root for easy access):
```python
#!/usr/bin/env python3
"""Launch the VFX Pipeline web interface.

Usage:
    ./start_web.py           # Start server and open browser
    ./start_web.py --no-browser  # Start server only
    ./start_web.py --port 8080   # Use custom port
"""

import argparse
import sys
import webbrowser
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import uvicorn
from env_config import check_conda_env_or_warn

def main():
    parser = argparse.ArgumentParser(description="Launch VFX Pipeline web interface")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    check_conda_env_or_warn()

    url = f"http://{args.host}:{args.port}"

    print(f"""
╔════════════════════════════════════════════════════════╗
║           VFX Pipeline Web Interface                   ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║   Server running at: {url:<29} ║
║                                                        ║
║   Press Ctrl+C to stop                                 ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
""")

    # Open browser (unless disabled)
    if not args.no_browser:
        webbrowser.open(url)

    # Start server
    uvicorn.run("web.server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
```

## Integration with Existing Code

The web server will:

1. **Import from `env_config.py`:**
   - `DEFAULT_PROJECTS_DIR` - where to create projects
   - `INSTALL_DIR` - where ComfyUI lives
   - `check_conda_env_or_warn()` - environment validation

2. **Import from `comfyui_utils.py`:**
   - `check_comfyui_running()` - system status
   - `DEFAULT_COMFYUI_URL` - ComfyUI endpoint

3. **Call `run_pipeline.py` via subprocess:**
   - Capture stdout/stderr for progress parsing
   - Pass through all configuration options
   - Handle cancellation via process termination

4. **Use existing project structure:**
   - Same folder layout as CLI
   - Same workflow templates
   - Compatible with `janitor.py` maintenance

## Open Questions

1. **Multi-user support?** - MVP assumes single user. Queue system needed for concurrent processing.

2. **File size limits?** - Large videos (10GB+) may need chunked upload or path-based input.

3. **ComfyUI auto-start?** - Should web server start ComfyUI automatically, or require it pre-running?

4. **Authentication?** - MVP has none. Add basic auth if exposing to network.

## Installation Integration

The web GUI components must be included in the existing installation and update procedures.

### New Installation (install_wizard.py)

Add to the installation wizard's component list:

```python
# In scripts/install_wizard/installers.py or wizard.py

WEB_DEPENDENCIES = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "python-multipart>=0.0.6",
    "websockets>=11.0",
]

def install_web_dependencies():
    """Install web GUI dependencies into conda environment."""
    # pip install within the vfx-pipeline conda env
    ...
```

**Wizard flow addition:**
```
Step N: Web Interface
  Installing web GUI dependencies...
  ✓ fastapi
  ✓ uvicorn
  ✓ python-multipart
  ✓ websockets
```

### Updating Existing Installation (janitor.py)

Add web GUI to the janitor's update and health check routines:

```python
# In scripts/janitor.py

def check_web_dependencies():
    """Verify web GUI dependencies are installed."""
    required = ["fastapi", "uvicorn", "python-multipart", "websockets"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    return missing

def update_web_dependencies():
    """Update web GUI dependencies to latest compatible versions."""
    # pip install --upgrade within conda env
    ...
```

**Janitor health check output:**
```
Web Interface:
  ✓ fastapi 0.109.0
  ✓ uvicorn 0.27.0
  ✓ python-multipart 0.0.6
  ✓ websockets 12.0
  ✓ start_web.py exists
  ✓ web/ package exists
```

**Janitor update command:**
```bash
python scripts/janitor.py -u  # Now also updates web dependencies
```

### requirements.txt Update

Add web dependencies to the main requirements file:

```
# requirements.txt (additions)

# Web GUI
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
websockets>=11.0
```

### Post-Update Validation

After `git pull` or janitor update, validate web components:

```python
def validate_web_installation():
    """Check web GUI is properly installed."""
    checks = [
        ("start_web.py exists", Path("start_web.py").exists()),
        ("web/ package exists", Path("web/__init__.py").exists()),
        ("Dependencies installed", len(check_web_dependencies()) == 0),
    ]
    return all(ok for _, ok in checks)
```

### Backward Compatibility

- Web GUI is **optional** - CLI pipeline works without it
- If web dependencies missing, `start_web.py` prints helpful install instructions
- Janitor `-H` reports web status but doesn't fail if missing

## Success Criteria

MVP is complete when:
- [ ] User can upload video (drag-and-drop OR browse button)
- [ ] User can select stages with roto prompt defaulting to "person"
- [ ] User can start processing
- [ ] User sees progress updates in real-time
- [ ] User sees completion with output file listing
- [ ] User can click to open output folder
- [ ] `./start_web.py` from repo root launches server and opens browser
- [ ] `python scripts/janitor.py -u` installs/updates web dependencies
- [ ] All without touching command line after initial setup
