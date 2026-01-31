"""REST API endpoints for VFX Pipeline web interface."""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from env_config import DEFAULT_PROJECTS_DIR, INSTALL_DIR
from vram_analyzer import analyze_and_save, load_vram_analysis
from web.services.config_service import get_config_service
from web.services.project_service import ProjectService
from web.services.pipeline_service import PipelineService
from web.services.video_service import get_video_service
from web.services.system_service import get_system_service
from web.repositories.project_repository import ProjectRepository
from web.repositories.job_repository import JobRepository
from web.models.domain import JobStatus
from web.utils.media import find_video_or_frames, get_dir_size_bytes, get_dir_size_gb
from web.job_state import active_jobs, active_jobs_lock
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
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

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

    project_dto = project_service.create_project_with_unique_name(
        project_name,
        Path(DEFAULT_PROJECTS_DIR)
    )
    project_name = project_dto.name

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
    video_service = get_video_service()
    video_info = video_service.get_video_info(project.video_path) if project.video_path else {}

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
    """List all projects with sizes."""
    import asyncio

    response = project_service.list_projects()
    projects_data = []

    for proj_dto in response.projects[:20]:
        project_entity = get_project_repo().get(proj_dto.name)
        if project_entity:
            size_bytes = await asyncio.to_thread(
                get_dir_size_bytes, project_entity.path
            )
            proj_dict = proj_dto.model_dump()
            proj_dict["size_bytes"] = size_bytes
            projects_data.append(proj_dict)
        else:
            projects_data.append(proj_dto.model_dump())

    return {"projects": projects_data}


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


@router.get("/projects/{project_id}/video-info")
async def get_project_video_info(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """Get video information for a project."""
    import asyncio

    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_entity = get_project_repo().get(project_id)
    source_dir = project_entity.path / "source"
    video_path, has_frames, frame_count = find_video_or_frames(source_dir)

    video_service = get_video_service()
    video_info = {}

    if has_frames:
        frames_dir = source_dir / "frames"
        resolution = await asyncio.to_thread(video_service.get_resolution_from_frames, frames_dir)
        video_info = {
            "source": "frames",
            "frame_count": frame_count,
            "resolution": list(resolution) if resolution else None,
        }
    elif video_path:
        info = await asyncio.to_thread(video_service.get_video_info, video_path)
        if info:
            video_info = {
                "source": "video",
                "filename": video_path.name,
                "frame_count": info.get("frame_count", 0),
                "resolution": info.get("resolution"),
                "fps": info.get("fps"),
                "duration": info.get("duration"),
            }

    return {"project_id": project_id, "video_info": video_info}


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
    print(f"[START] project_id={project_id}, stages={config.stages}")
    try:
        response = pipeline_service.start_job(project_id, config)
        return response.model_dump()
    except ValueError as e:
        print(f"[START ERROR] {e}")
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


@router.get("/projects/{project_id}/job")
async def get_job_status(project_id: str):
    """Get current job status for a project."""
    with active_jobs_lock:
        job = active_jobs.get(project_id)

    if not job:
        return {
            "project_id": project_id,
            "status": "idle",
            "message": "No active job"
        }

    return {
        "project_id": project_id,
        "status": job.get("status", "unknown"),
        "current_stage": job.get("current_stage"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "last_output": job.get("last_output", ""),
    }


@router.get("/projects/{project_id}/vram")
async def get_vram_analysis(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """Get VRAM analysis for a project, generating it on-demand if missing."""
    import asyncio

    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_entity = get_project_repo().get(project_id)
    analysis = load_vram_analysis(project_entity.path)

    if not analysis:
        resolution = (1920, 1080)
        fps = 24.0

        source_dir = project_entity.path / "source"
        video_path, has_frames, frame_count = find_video_or_frames(source_dir)

        video_service = get_video_service()
        if has_frames:
            frames_dir = source_dir / "frames"
            res = await asyncio.to_thread(video_service.get_resolution_from_frames, frames_dir)
            if res:
                resolution = res
        elif video_path:
            video_info = await asyncio.to_thread(video_service.get_video_info, video_path)
            if video_info:
                frame_count = video_info.get("frame_count", 0)
                res = video_info.get("resolution", [1920, 1080])
                if res and len(res) >= 2:
                    resolution = (res[0], res[1])
                fps = video_info.get("fps", 24.0)

        if frame_count > 0:
            try:
                analysis = await asyncio.to_thread(
                    analyze_and_save,
                    project_entity.path,
                    frame_count,
                    resolution,
                    fps,
                )
            except Exception as e:
                print(f"Failed to generate VRAM analysis: {e}")

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
            all_files = list(dir_path.rglob("*"))
            files = []
            for f in sorted(all_files):
                if f.is_file():
                    try:
                        files.append({
                            "name": f.name,
                            "size": f.stat().st_size,
                            "path": str(f),
                        })
                    except (OSError, IOError):
                        pass
            if files:
                output_key = dir_name.split("/")[0]
                outputs[output_key] = {
                    "count": len(files),
                    "files": files[:10],
                    "total_files": len(files),
                }

    return {
        "project_id": project_id,
        "project_dir": str(project_dir),
        "outputs": outputs,
    }


@router.get("/system/status")
async def system_status():
    """Check system status (ComfyUI, disk space, GPU, etc.)."""
    import asyncio
    import platform

    system_service = get_system_service()
    projects_dir = Path(DEFAULT_PROJECTS_DIR)

    status = {
        "comfyui": False,
        "os": platform.system(),
        "disk_free_gb": 0,
        "disk_total_gb": 0,
        "disk_used_percent": 0,
        "projects_size_gb": 0,
        "projects_dir": str(DEFAULT_PROJECTS_DIR),
        "install_dir": str(INSTALL_DIR),
        "gpu_name": "Unknown",
        "gpu_vram_gb": 0,
    }

    try:
        status["comfyui"] = await asyncio.to_thread(system_service.check_comfyui_status)
    except Exception:
        pass

    disk_usage = await asyncio.to_thread(system_service.get_disk_usage, projects_dir)
    if disk_usage:
        status["disk_free_gb"] = disk_usage["free_gb"]
        status["disk_total_gb"] = disk_usage["total_gb"]
        status["disk_used_percent"] = disk_usage["used_percent"]

    if projects_dir.exists():
        try:
            status["projects_size_gb"] = await asyncio.to_thread(get_dir_size_gb, projects_dir)
        except Exception:
            pass

    try:
        gpu_info = await asyncio.to_thread(system_service.get_gpu_info)
        status["gpu_name"] = gpu_info["name"]
        status["gpu_vram_gb"] = gpu_info["vram_gb"]
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
