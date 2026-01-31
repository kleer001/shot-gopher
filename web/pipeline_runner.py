"""Pipeline execution wrapper for web interface."""

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from web.utils.media import find_video_or_frames

active_processes: Dict[str, subprocess.Popen] = {}


def get_run_pipeline_path() -> Path:
    """Get path to run_pipeline.py."""
    return Path(__file__).parent.parent / "scripts" / "run_pipeline.py"


def parse_progress_line(line: str, current_stage: str, stages: list) -> Optional[dict]:
    """Parse a line of pipeline output for progress information.

    Returns dict with the line as message, plus stage detection.
    """
    result = {"message": line.strip()[:200]}

    # Detect stage changes: "=== Stage: depth ===" or "=== ingest ==="
    if "===" in line:
        for stage in stages:
            if stage.lower() in line.lower():
                result["current_stage"] = stage
                result["stage_index"] = stages.index(stage)
                result["total_stages"] = len(stages)
                break

    return result


def update_job_repo_status(project_id: str, status: str, error: str = None):
    """Update job status in the job repository."""
    from .api import get_job_repo
    from .models.domain import JobStatus
    from datetime import datetime

    job_repo = get_job_repo()
    job = job_repo.get(project_id)

    if job:
        if status == "completed":
            job.status = JobStatus.COMPLETE
            job.completed_at = datetime.now()
            job.progress = 1.0
        elif status == "failed":
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = error
        elif status == "cancelled":
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

        job_repo.save(job)


def run_pipeline_thread(
    project_id: str,
    project_dir: str,
    stages: list,
    roto_prompt: str,
    skip_existing: bool,
    stage_options: dict = None,
):
    """Run pipeline in a background thread."""
    from .job_state import active_jobs, active_jobs_lock
    from .websocket import update_progress

    with active_jobs_lock:
        active_jobs[project_id] = {
            "status": "running",
            "current_stage": stages[0] if stages else None,
            "progress": 0.0,
            "error": None,
        }

    script_path = get_run_pipeline_path()
    source_dir = Path(project_dir) / "source"

    input_video, has_frames, _ = find_video_or_frames(source_dir)

    if not input_video and not has_frames:
        error_msg = f"No input video or frames found in {source_dir}"
        with active_jobs_lock:
            if project_id in active_jobs:
                active_jobs[project_id]["status"] = "failed"
                active_jobs[project_id]["error"] = error_msg
                active_jobs[project_id]["last_output"] = error_msg
        update_job_repo_status(project_id, "failed", error_msg)
        update_progress(project_id, {
            "status": "failed",
            "error": error_msg,
        })
        return

    # Build command - use project dir if we have frames, video path otherwise
    if has_frames:
        # Pass the project directory - run_pipeline.py can work with existing frames
        cmd = [
            sys.executable,
            str(script_path),
            str(project_dir),
            "--stages", ",".join(stages),
            "--no-auto-comfyui",
        ]
    else:
        cmd = [
            sys.executable,
            str(script_path),
            str(input_video),
            "--name", Path(project_dir).name,
            "--projects-dir", str(Path(project_dir).parent),
            "--stages", ",".join(stages),
            "--no-auto-comfyui",
        ]

    if roto_prompt and "roto" in stages:
        cmd.extend(["--prompt", roto_prompt])

    if skip_existing:
        cmd.append("--skip-existing")

    if stage_options:
        opts = stage_options.get("ingest", {})
        if opts.get("fps"):
            cmd.extend(["--fps", str(opts["fps"])])

        opts = stage_options.get("roto", {})
        if opts.get("separate_instances"):
            cmd.append("--separate-instances")

        opts = stage_options.get("colmap", {})
        if opts.get("quality"):
            cmd.extend(["-q", opts["quality"]])
        if opts.get("dense"):
            cmd.append("-d")
        if opts.get("mesh"):
            cmd.append("-m")
        if opts.get("no_masks"):
            cmd.append("-M")

        opts = stage_options.get("gsir", {})
        if opts.get("iterations"):
            cmd.extend(["-i", str(opts["iterations"])])

        opts = stage_options.get("camera", {})
        if opts.get("rotation_order"):
            cmd.extend(["--rotation-order", opts["rotation_order"]])

    # Start process
    try:
        env = os.environ.copy()
        # Ensure unbuffered output
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(Path(__file__).parent.parent),
        )

        active_processes[project_id] = process

        current_stage = stages[0] if stages else None
        stage_index = 0

        # Read output line by line
        for line in iter(process.stdout.readline, ""):
            if not line:
                break

            line = line.rstrip()
            if not line:
                continue

            # Skip carriage-return spinner lines (they overwrite themselves)
            if line.startswith("\r") or line.startswith("    \r"):
                clean_line = line.lstrip("\r").strip()
                if clean_line:
                    with active_jobs_lock:
                        if project_id in active_jobs:
                            active_jobs[project_id]["last_output"] = clean_line
                continue

            # Parse progress
            progress_info = parse_progress_line(line, current_stage, stages)

            if progress_info:
                # Update current stage tracking
                if "current_stage" in progress_info:
                    current_stage = progress_info["current_stage"]
                    stage_index = progress_info.get("stage_index", stage_index)

                # Build update
                update = {
                    "stage": current_stage,
                    "stage_index": stage_index,
                    "total_stages": len(stages),
                    **progress_info,
                }

                # Update stored state
                with active_jobs_lock:
                    if project_id in active_jobs:
                        active_jobs[project_id]["current_stage"] = current_stage
                        active_jobs[project_id]["last_output"] = line[:100]
                        if "progress" in progress_info:
                            active_jobs[project_id]["progress"] = progress_info["progress"]

                # Send WebSocket update
                update_progress(project_id, update)
            else:
                # Store any output line for display
                with active_jobs_lock:
                    if project_id in active_jobs:
                        active_jobs[project_id]["last_output"] = line[:100]

        # Wait for process to complete
        process.wait()

        # Update final status
        with active_jobs_lock:
            if project_id in active_jobs:
                if process.returncode == 0:
                    active_jobs[project_id]["status"] = "completed"
                    active_jobs[project_id]["progress"] = 1.0
                else:
                    active_jobs[project_id]["status"] = "failed"
                    active_jobs[project_id]["error"] = f"Pipeline exited with code {process.returncode}"

        if process.returncode == 0:
            update_job_repo_status(project_id, "completed")
            update_progress(project_id, {
                "status": "completed",
                "progress": 1.0,
                "stage": stages[-1] if stages else None,
                "stage_index": len(stages) - 1,
                "total_stages": len(stages),
            })
        else:
            error_msg = f"Pipeline exited with code {process.returncode}"
            update_job_repo_status(project_id, "failed", error_msg)
            update_progress(project_id, {
                "status": "failed",
                "error": error_msg,
            })

    except Exception as e:
        with active_jobs_lock:
            if project_id in active_jobs:
                active_jobs[project_id]["status"] = "failed"
                active_jobs[project_id]["error"] = str(e)
        update_job_repo_status(project_id, "failed", str(e))
        update_progress(project_id, {
            "status": "failed",
            "error": str(e),
        })
    finally:
        # Clean up
        if project_id in active_processes:
            del active_processes[project_id]


def start_pipeline(
    project_id: str,
    project_dir: str,
    stages: list,
    roto_prompt: str = "person",
    skip_existing: bool = False,
    stage_options: dict = None,
):
    """Start pipeline processing in a background thread."""
    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(project_id, project_dir, stages, roto_prompt, skip_existing, stage_options),
        daemon=True,
    )
    thread.start()


def stop_pipeline(project_id: str) -> bool:
    """Stop a running pipeline.

    Returns True if process was stopped, False if not running.
    """
    from .job_state import active_jobs, active_jobs_lock

    if project_id not in active_processes:
        return False

    process = active_processes[project_id]

    try:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        with active_jobs_lock:
            if project_id in active_jobs:
                active_jobs[project_id]["status"] = "cancelled"

        update_job_repo_status(project_id, "cancelled")

        if project_id in active_processes:
            del active_processes[project_id]

        return True
    except Exception as e:
        print(f"Error stopping pipeline: {e}")
        return False
