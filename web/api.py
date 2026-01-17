"""REST API endpoints for VFX Pipeline web interface."""

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from env_config import DEFAULT_PROJECTS_DIR, INSTALL_DIR
from services.config_service import get_config_service

router = APIRouter()

# Store for active jobs (in-memory for MVP)
active_jobs = {}


class ProjectConfig(BaseModel):
    """Configuration for starting a pipeline job."""
    stages: list[str] = ["ingest", "depth", "roto", "cleanplate"]
    roto_prompt: str = "person"
    skip_existing: bool = False


class ProjectInfo(BaseModel):
    """Information about a project."""
    project_id: str
    name: str
    project_dir: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str
    video_info: Optional[dict] = None
    stages: Optional[list[str]] = None
    current_stage: Optional[str] = None
    progress: Optional[float] = None
    error: Optional[str] = None


def get_video_info(video_path: Path) -> dict:
    """Extract video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            return {}

        # Extract info
        duration = float(data.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Parse frame rate
        fps_str = video_stream.get("r_frame_rate", "24/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 24.0
        else:
            fps = float(fps_str)

        frame_count = int(duration * fps) if duration > 0 else 0

        return {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "resolution": [width, height],
            "frame_count": frame_count,
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}


def sanitize_project_name(name: str) -> str:
    """Sanitize project name for filesystem use."""
    # Remove/replace problematic characters
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip("_")
    # Ensure not empty
    return safe_name or "project"


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    """Upload a video file and create a new project."""
    # Validate file type
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate project name and ID
    base_name = name or Path(file.filename).stem
    project_name = sanitize_project_name(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_id = f"{project_name}_{timestamp}"

    # Create project directory
    projects_dir = Path(DEFAULT_PROJECTS_DIR)
    projects_dir.mkdir(parents=True, exist_ok=True)

    project_dir = projects_dir / project_name

    # Handle existing project directory
    if project_dir.exists():
        # Add timestamp to make unique
        project_dir = projects_dir / f"{project_name}_{timestamp}"

    project_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    source_dir = project_dir / "source"
    source_dir.mkdir(exist_ok=True)
    video_path = source_dir / f"input{file_ext}"

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        # Clean up on failure
        shutil.rmtree(project_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Get video info
    video_info = get_video_info(video_path)

    # Store project info
    project_info = ProjectInfo(
        project_id=project_id,
        name=project_name,
        project_dir=str(project_dir),
        status="pending",
        created_at=datetime.now().isoformat(),
        video_info=video_info,
    )
    active_jobs[project_id] = project_info.model_dump()

    return {
        "project_id": project_id,
        "name": project_name,
        "project_dir": str(project_dir),
        "video_info": video_info,
    }


@router.get("/projects")
async def list_projects():
    """List all projects."""
    # Include active jobs
    projects = list(active_jobs.values())

    # Also scan projects directory for existing projects
    projects_dir = Path(DEFAULT_PROJECTS_DIR)
    if projects_dir.exists():
        for proj_dir in sorted(projects_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if proj_dir.is_dir():
                proj_id = proj_dir.name
                if proj_id not in [p["project_id"] for p in projects]:
                    # Check for outputs to determine status
                    try:
                        has_depth = (proj_dir / "depth").is_dir() and any((proj_dir / "depth").iterdir())
                    except (OSError, StopIteration):
                        has_depth = False

                    try:
                        has_roto = (proj_dir / "roto").is_dir() and any((proj_dir / "roto").iterdir())
                    except (OSError, StopIteration):
                        has_roto = False

                    status = "completed" if (has_depth or has_roto) else "pending"

                    projects.append({
                        "project_id": proj_id,
                        "name": proj_id,
                        "project_dir": str(proj_dir),
                        "status": status,
                        "created_at": datetime.fromtimestamp(proj_dir.stat().st_mtime).isoformat(),
                    })

    # Limit to most recent 20
    return {"projects": projects[:20]}


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details and status."""
    if project_id in active_jobs:
        return active_jobs[project_id]

    # Check if project exists on disk
    projects_dir = Path(DEFAULT_PROJECTS_DIR)
    project_dir = projects_dir / project_id

    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Build info from disk
    return {
        "project_id": project_id,
        "name": project_id,
        "project_dir": str(project_dir),
        "status": "completed",
        "created_at": datetime.fromtimestamp(project_dir.stat().st_mtime).isoformat(),
    }


@router.post("/projects/{project_id}/start")
async def start_processing(project_id: str, config: ProjectConfig):
    """Start pipeline processing for a project."""
    from .pipeline_runner import start_pipeline

    if project_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Project not found")

    project = active_jobs[project_id]

    if project["status"] == "processing":
        raise HTTPException(status_code=400, detail="Project is already processing")

    # Update project with config
    project["stages"] = config.stages
    project["status"] = "processing"
    project["progress"] = 0.0
    project["current_stage"] = config.stages[0] if config.stages else None

    # Start pipeline in background
    start_pipeline(
        project_id=project_id,
        project_dir=project["project_dir"],
        stages=config.stages,
        roto_prompt=config.roto_prompt,
        skip_existing=config.skip_existing,
    )

    return {"status": "started", "project_id": project_id}


@router.post("/projects/{project_id}/stop")
async def stop_processing(project_id: str):
    """Stop pipeline processing for a project."""
    from .pipeline_runner import stop_pipeline

    if project_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Project not found")

    stopped = stop_pipeline(project_id)

    if stopped:
        active_jobs[project_id]["status"] = "stopped"
        return {"status": "stopped"}
    else:
        return {"status": "not_running"}


@router.get("/projects/{project_id}/outputs")
async def get_outputs(project_id: str):
    """Get list of output files for a project."""
    if project_id in active_jobs:
        project_dir = Path(active_jobs[project_id]["project_dir"])
    else:
        project_dir = Path(DEFAULT_PROJECTS_DIR) / project_id

    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    outputs = {}

    # Check each output directory (from configuration)
    config_service = get_config_service()
    output_dirs = config_service.get_output_directories()

    for dir_name in output_dirs:
        dir_path = project_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            files = []
            for f in sorted(dir_path.iterdir()):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "path": str(f),
                    })
            if files:
                outputs[dir_name] = {
                    "count": len(files),
                    "files": files[:10],  # First 10 files
                    "total_files": len(files),
                }

    # Also check for source frames
    frames_dir = project_dir / "source" / "frames"
    if frames_dir.exists() and frames_dir.is_dir():
        frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        if frame_files:
            outputs["source"] = {
                "count": len(frame_files),
                "files": [{"name": f.name, "path": str(f)} for f in sorted(frame_files)[:5]],
                "total_files": len(frame_files),
            }

    return {
        "project_id": project_id,
        "project_dir": str(project_dir),
        "outputs": outputs,
    }


@router.get("/system/status")
async def system_status():
    """Check system status (ComfyUI, disk space, etc.)."""
    status = {
        "comfyui": False,
        "disk_space_gb": 0,
        "projects_dir": str(DEFAULT_PROJECTS_DIR),
        "install_dir": str(INSTALL_DIR),
    }

    # Check ComfyUI
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8188/system_stats", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            status["comfyui"] = resp.status == 200
    except Exception:
        status["comfyui"] = False

    # Check disk space
    try:
        projects_dir = Path(DEFAULT_PROJECTS_DIR)
        if projects_dir.exists():
            stat = shutil.disk_usage(projects_dir)
            status["disk_space_gb"] = round(stat.free / (1024**3), 1)
        else:
            # Check parent
            parent = projects_dir.parent
            if parent.exists():
                stat = shutil.disk_usage(parent)
                status["disk_space_gb"] = round(stat.free / (1024**3), 1)
    except Exception:
        pass

    return status


@router.get("/config")
async def get_config():
    """Get pipeline configuration (stages, presets, settings).

    Returns the complete pipeline configuration including:
    - Stage definitions with metadata
    - Preset configurations
    - Supported video formats
    - WebSocket settings
    - UI configuration

    This provides a single source of truth for configuration,
    following the DRY principle.
    """
    config_service = get_config_service()
    return {
        "stages": config_service.get_stages(),
        "presets": config_service.get_presets(),
        "supportedVideoFormats": config_service.get_supported_video_formats(),
        "websocket": config_service.get_websocket_config(),
        "ui": config_service.get_ui_config(),
    }


@router.post("/projects/{project_id}/open-folder")
async def open_folder(project_id: str):
    """Open the project folder in the system file manager."""
    if project_id in active_jobs:
        project_dir = Path(active_jobs[project_id]["project_dir"])
    else:
        project_dir = Path(DEFAULT_PROJECTS_DIR) / project_id

    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Try to open file manager
    try:
        import platform
        system = platform.system()

        if system == "Linux":
            subprocess.Popen(["xdg-open", str(project_dir)])
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", str(project_dir)])
        elif system == "Windows":
            subprocess.Popen(["explorer", str(project_dir)])
        else:
            return {"status": "unsupported", "path": str(project_dir)}

        return {"status": "opened", "path": str(project_dir)}
    except Exception as e:
        return {"status": "error", "error": str(e), "path": str(project_dir)}
