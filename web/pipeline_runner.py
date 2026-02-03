"""Pipeline execution wrapper for web interface."""

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from web.utils.media import find_video_or_frames

active_processes: Dict[str, subprocess.Popen] = {}


def get_run_pipeline_path() -> Path:
    """Get path to run_pipeline.py."""
    return Path(__file__).parent.parent / "scripts" / "run_pipeline.py"


ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*[mGKHF]|\x1b\].*?\x07')

HTTP_METHODS = r'(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)'
HTTP_LOG_PATTERN = re.compile(
    rf'(INFO:\s+\d+\.\d+\.\d+\.\d+:\d+\s+-\s+"{HTTP_METHODS})|'
    rf'(INFO:[a-z._]+:\d+\.\d+\.\d+\.\d+:\d+\s+-\s+"{HTTP_METHODS})'
)

FRAME_PATTERNS = [
    re.compile(r'frame\s+(\d+)\s*/\s*(\d+)', re.IGNORECASE),
    re.compile(r'(\d+)\s*/\s*(\d+)\s*frames', re.IGNORECASE),
    re.compile(r'\[(\d+)/(\d+)\]'),
    re.compile(r'Processing\s+(\d+)\s+of\s+(\d+)', re.IGNORECASE),
    re.compile(r'\|\s*(\d+)/(\d+)\s*[\[\s]'),
]

PROGRESS_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return ANSI_ESCAPE_PATTERN.sub('', text)


def is_http_traffic(line: str) -> bool:
    """Check if line is HTTP traffic log (should be hidden from UI)."""
    return bool(HTTP_LOG_PATTERN.search(line))


def read_output_lines(stdout):
    """Read output handling both \\n and \\r as line delimiters.

    Yields lines as they come, treating \\r (carriage return) as a line break.
    This allows capturing tqdm-style progress bars that use \\r for updates.

    Uses character-by-character reading to ensure immediate processing
    of output rather than blocking for a buffer to fill.
    """
    buffer = ""
    while True:
        char = stdout.read(1)
        if not char:
            if buffer.strip():
                yield buffer
            break

        if char in '\r\n':
            if buffer.strip():
                yield buffer
            buffer = ""
        else:
            buffer += char


def parse_progress_line(line: str, current_stage: str, stages: list) -> dict:
    """Parse a line of pipeline output for progress information.

    Returns dict with the line as message, plus stage/frame/progress detection.
    """
    result = {"message": strip_ansi_codes(line.strip()[:200])}

    for pattern in FRAME_PATTERNS:
        match = pattern.search(line)
        if match:
            current_frame = int(match.group(1))
            total_frames = int(match.group(2))
            if total_frames > 0:
                result["frame"] = current_frame
                result["total_frames"] = total_frames
                result["progress"] = current_frame / total_frames
            break

    if "progress" not in result:
        match = PROGRESS_PATTERN.search(line)
        if match:
            percent = float(match.group(1))
            result["progress"] = percent / 100.0

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
    roto_start_frame: int | None,
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
            "last_output": "",
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

    if roto_start_frame is not None and "roto" in stages:
        cmd.extend(["--start-frame", str(roto_start_frame)])

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

        opts = stage_options.get("cleanplate", {})
        if opts.get("method") == "median":
            cmd.append("--cleanplate-median")

        opts = stage_options.get("gsir", {})
        if opts.get("iterations"):
            cmd.extend(["-i", str(opts["iterations"])])

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

        for line in read_output_lines(process.stdout):
            clean_line = strip_ansi_codes(line.strip())
            if not clean_line:
                continue

            if is_http_traffic(clean_line):
                continue

            progress_info = parse_progress_line(clean_line, current_stage, stages)

            if "current_stage" in progress_info:
                current_stage = progress_info["current_stage"]
                stage_index = progress_info.get("stage_index", stage_index)

            with active_jobs_lock:
                if project_id in active_jobs:
                    active_jobs[project_id]["current_stage"] = current_stage
                    active_jobs[project_id]["last_output"] = clean_line[:200]
                    if "progress" in progress_info:
                        active_jobs[project_id]["progress"] = progress_info["progress"]

            update = {
                "stage": current_stage,
                "stage_index": stage_index,
                "total_stages": len(stages),
                **progress_info,
            }
            update_progress(project_id, update)

        # Wait for process to complete
        process.wait()

        # Update final status (but don't overwrite "cancelled" status)
        with active_jobs_lock:
            if project_id in active_jobs:
                current_status = active_jobs[project_id].get("status")
                if current_status == "cancelled":
                    pass
                elif process.returncode == 0:
                    active_jobs[project_id]["status"] = "completed"
                    active_jobs[project_id]["progress"] = 1.0
                else:
                    active_jobs[project_id]["status"] = "failed"
                    active_jobs[project_id]["error"] = f"Pipeline exited with code {process.returncode}"

        with active_jobs_lock:
            final_status = active_jobs.get(project_id, {}).get("status")

        if final_status == "cancelled":
            pass
        elif process.returncode == 0:
            update_job_repo_status(project_id, "completed")
            update_progress(project_id, {
                "status": "completed",
                "progress": 1.0,
                "stage": stages[-1] if stages else None,
                "stage_index": max(0, len(stages) - 1),
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
    roto_start_frame: int | None = None,
    skip_existing: bool = False,
    stage_options: dict = None,
):
    """Start pipeline processing in a background thread."""
    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(project_id, project_dir, stages, roto_prompt, roto_start_frame, skip_existing, stage_options),
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
