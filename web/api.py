"""REST API endpoints for VFX Pipeline web interface."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from env_config import DEFAULT_PROJECTS_DIR, INSTALL_DIR
from install_wizard.platform import PlatformManager
from vram_analyzer import analyze_and_save, load_vram_analysis
from web.services.config_service import get_config_service
from web.services.project_service import ProjectService
from web.services.pipeline_service import PipelineService
from web.repositories.project_repository import ProjectRepository
from web.repositories.job_repository import JobRepository
from web.models.domain import JobStatus
from web.models.dto import (
    ProjectCreateRequest,
    JobStartRequest,
    SystemStatusResponse,
    ProjectOutputsResponse
)

router = APIRouter()

_project_repo = None
_job_repo = None
_project_service = None
_pipeline_service = None

active_jobs: Dict[str, dict] = {}


def get_project_repo() -> ProjectRepository:
    """Get project repository instance."""
    global _project_repo
    if _project_repo is None:
        _project_repo = ProjectRepository(Path(DEFAULT_PROJECTS_DIR))
    return _project_repo


def get_job_repo() -> JobRepository:
    """Get job repository instance."""
    global _job_repo
    if _job_repo is None:
        _job_repo = JobRepository()
    return _job_repo


def get_project_service(
    project_repo: ProjectRepository = Depends(get_project_repo)
) -> ProjectService:
    """Get project service instance."""
    global _project_service
    if _project_service is None:
        _project_service = ProjectService(project_repo)
    return _project_service


def get_pipeline_service(
    job_repo: JobRepository = Depends(get_job_repo),
    project_repo: ProjectRepository = Depends(get_project_repo)
) -> PipelineService:
    """Get pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService(job_repo, project_repo)
    return _pipeline_service


def get_video_info(video_path: Path) -> dict:
    """Extract video metadata using ffprobe.

    Uses PlatformManager.find_tool to locate ffprobe on Windows
    even if it's not in PATH.
    """
    try:
        ffprobe_path = PlatformManager.find_tool("ffprobe")
        if not ffprobe_path:
            return {}

        cmd = [
            str(ffprobe_path), "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)

        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            return {}

        duration = float(data.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

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
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    safe_name = safe_name.strip("_")
    return safe_name or "project"


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    project_service: ProjectService = Depends(get_project_service),
):
    """Upload a video file and create a new project."""
    config_service = get_config_service()
    allowed_extensions = set(config_service.get_supported_video_formats())
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    base_name = name or Path(file.filename).stem
    project_name = sanitize_project_name(base_name)

    try:
        project_dto = project_service.create_project(
            ProjectCreateRequest(name=project_name, stages=[]),
            Path(DEFAULT_PROJECTS_DIR)
        )
    except ValueError as e:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"{project_name}_{timestamp}"
        project_dto = project_service.create_project(
            ProjectCreateRequest(name=project_name, stages=[]),
            Path(DEFAULT_PROJECTS_DIR)
        )

    video_filename = f"input{file_ext}"
    try:
        video_content = await file.read()
        project_dto = project_service.save_uploaded_video(
            project_name,
            video_filename,
            video_content
        )
    except Exception as e:
        project_service.delete_project(project_name)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    project = get_project_repo().get(project_name)
    video_info = get_video_info(project.video_path) if project.video_path else {}

    vram_analysis = None
    if video_info:
        try:
            vram_analysis = analyze_and_save(
                project_dir=project.path,
                frame_count=video_info.get("frame_count", 0),
                resolution=tuple(video_info.get("resolution", [1920, 1080])),
                fps=video_info.get("fps", 24.0),
            )
        except Exception as e:
            print(f"VRAM analysis failed: {e}")

    return {
        "project_id": project_name,
        "name": project_name,
        "project_dir": str(project.path),
        "video_info": video_info,
        "vram_analysis": vram_analysis,
    }


@router.post("/projects")
async def create_project(
    request: ProjectCreateRequest,
    project_service: ProjectService = Depends(get_project_service),
):
    """Create a new project (without video upload)."""
    try:
        project = project_service.create_project(request, Path(DEFAULT_PROJECTS_DIR))
        return project.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/projects")
async def list_projects(
    project_service: ProjectService = Depends(get_project_service)
):
    """List all projects."""
    response = project_service.list_projects()
    return {"projects": [p.model_dump() for p in response.projects[:20]]}


@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """Get project details and status."""
    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return project.model_dump()


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """Delete a project."""
    job = pipeline_service.get_job_status(project_id)
    if job and job.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete project while pipeline is running. Stop the job first."
        )

    deleted = project_service.delete_project(project_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"status": "deleted", "project_id": project_id}


@router.post("/projects/{project_id}/start")
async def start_processing(
    project_id: str,
    config: JobStartRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """Start pipeline processing for a project."""
    try:
        response = pipeline_service.start_job(project_id, config)
        return response.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/projects/{project_id}/stop")
async def stop_processing(
    project_id: str,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """Stop pipeline processing for a project."""
    stopped = pipeline_service.stop_job(project_id)

    if stopped:
        return {"status": "stopped"}
    else:
        return {"status": "not_running"}


@router.get("/projects/{project_id}/vram")
async def get_vram_analysis(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """Get VRAM analysis for a project."""
    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_entity = get_project_repo().get(project_id)
    analysis = load_vram_analysis(project_entity.path)

    if not analysis:
        return {"project_id": project_id, "analysis": None, "message": "No VRAM analysis available"}

    return {"project_id": project_id, "analysis": analysis}


@router.get("/projects/{project_id}/outputs")
async def get_outputs(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """Get list of output files for a project."""
    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_entity = get_project_repo().get(project_id)
    project_dir = project_entity.path

    outputs = {}
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
                    "files": files[:10],
                    "total_files": len(files),
                }

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

    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8188/system_stats", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            status["comfyui"] = resp.status == 200
    except Exception:
        status["comfyui"] = False

    try:
        projects_dir = Path(DEFAULT_PROJECTS_DIR)
        if projects_dir.exists():
            stat = shutil.disk_usage(projects_dir)
            status["disk_space_gb"] = round(stat.free / (1024**3), 1)
        else:
            parent = projects_dir.parent
            if parent.exists():
                stat = shutil.disk_usage(parent)
                status["disk_space_gb"] = round(stat.free / (1024**3), 1)
    except Exception:
        pass

    return status


@router.post("/system/shutdown")
async def shutdown_system():
    """Shutdown the web server gracefully."""
    import signal
    import os

    def delayed_shutdown():
        """Shutdown after a brief delay to allow response to be sent."""
        import time
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    import threading
    shutdown_thread = threading.Thread(target=delayed_shutdown)
    shutdown_thread.start()

    return {
        "status": "shutdown_initiated",
        "message": "Server is shutting down..."
    }


@router.get("/config")
async def get_config():
    """Get pipeline configuration (stages, presets, settings)."""
    config_service = get_config_service()
    return {
        "stages": config_service.get_stages(),
        "presets": config_service.get_presets(),
        "supportedVideoFormats": config_service.get_supported_video_formats(),
        "websocket": config_service.get_websocket_config(),
        "ui": config_service.get_ui_config(),
    }


@router.post("/projects/{project_id}/open-folder")
async def open_folder(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """Open the project folder in the system file manager."""
    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_entity = get_project_repo().get(project_id)
    project_dir = project_entity.path

    try:
        import platform
        system = platform.system()

        if system == "Linux":
            subprocess.Popen(["xdg-open", str(project_dir)])
        elif system == "Darwin":
            subprocess.Popen(["open", str(project_dir)])
        elif system == "Windows":
            subprocess.Popen(["explorer", str(project_dir)])
        else:
            return {"status": "unsupported", "path": str(project_dir)}

        return {"status": "opened", "path": str(project_dir)}
    except Exception as e:
        return {"status": "error", "error": str(e), "path": str(project_dir)}
