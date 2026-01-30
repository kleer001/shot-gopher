"""Pipeline execution wrapper for web interface.

Runs run_pipeline.py as a subprocess and parses output for progress updates.
"""

import asyncio
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Store active processes
active_processes: Dict[str, subprocess.Popen] = {}


def get_run_pipeline_path() -> Path:
    """Get path to run_pipeline.py."""
    return Path(__file__).parent.parent / "scripts" / "run_pipeline.py"


def parse_progress_line(line: str, current_stage: str, stages: list) -> Optional[dict]:
    """Parse a line of pipeline output for progress information.

    Returns dict with progress info or None if not a progress line.
    """
    # Patterns to match pipeline output
    patterns = {
        # Stage start: "=== Stage: depth ==="
        "stage_start": r"===\s*(?:Stage:?\s*)?(\w+)\s*===",
        # Frame progress: "Processing frame 42/200" or "Frame 42 of 200"
        "frame_progress": r"(?:Processing\s+)?[Ff]rame\s+(\d+)\s*(?:/|of)\s*(\d+)",
        # Percentage: "Progress: 42%" or "42% complete"
        "percentage": r"(\d+(?:\.\d+)?)\s*%",
        # ComfyUI progress: "ComfyUI: 42/100"
        "comfyui_progress": r"ComfyUI:\s*(\d+)/(\d+)",
        # Stage complete
        "stage_complete": r"(?:Completed|Finished|Done):\s*(\w+)",
        # COLMAP specific
        "colmap_progress": r"Registered\s+(\d+)\s*/\s*(\d+)\s+images",
        # FFmpeg progress: "frame=  142 fps=..." or "[FFmpeg] Extracting frame 42/200"
        "ffmpeg_progress": r"\[FFmpeg\]\s*Extracting\s+frame\s+(\d+)\s*/\s*(\d+)",
        # ComfyUI file-based progress: "[ComfyUI] depth frame 42/200"
        "comfyui_file_progress": r"\[ComfyUI\]\s*(\w+)\s+frame\s+(\d+)/(\d+)",
        # Mocap/ECON keyframe progress: "[1/5] Processing frame_0001.png"
        "bracket_progress": r"\[(\d+)/(\d+)\]",
        # GS-IR training: "Iteration 1000/30000"
        "gsir_progress": r"[Ii]teration\s+(\d+)\s*/\s*(\d+)",
    }

    result = {}

    # Check for stage start
    match = re.search(patterns["stage_start"], line, re.IGNORECASE)
    if match:
        stage = match.group(1).lower()
        if stage in stages:
            result["current_stage"] = stage
            result["stage_index"] = stages.index(stage)
            result["total_stages"] = len(stages)

    # Check for frame progress
    match = re.search(patterns["frame_progress"], line)
    if match:
        frame = int(match.group(1))
        total = int(match.group(2))
        result["frame"] = frame
        result["total_frames"] = total
        result["progress"] = frame / total if total > 0 else 0

    # Check for ComfyUI progress
    match = re.search(patterns["comfyui_progress"], line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        result["frame"] = current
        result["total_frames"] = total
        result["progress"] = current / total if total > 0 else 0

    # Check for COLMAP progress
    match = re.search(patterns["colmap_progress"], line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        result["frame"] = current
        result["total_frames"] = total
        result["progress"] = current / total if total > 0 else 0

    # Check for FFmpeg progress
    match = re.search(patterns["ffmpeg_progress"], line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        result["frame"] = current
        result["total_frames"] = total
        result["progress"] = current / total if total > 0 else 0

    # Check for ComfyUI file-based progress
    match = re.search(patterns["comfyui_file_progress"], line)
    if match:
        stage = match.group(1).lower()
        current = int(match.group(2))
        total = int(match.group(3))
        result["frame"] = current
        result["total_frames"] = total
        result["progress"] = current / total if total > 0 else 0
        result["current_stage"] = stage  # Update stage if detected

    # Check for bracket progress [1/5] style
    match = re.search(patterns["bracket_progress"], line)
    if match and "progress" not in result:
        current = int(match.group(1))
        total = int(match.group(2))
        result["frame"] = current
        result["total_frames"] = total
        result["progress"] = current / total if total > 0 else 0

    # Check for GS-IR training progress
    match = re.search(patterns["gsir_progress"], line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        result["frame"] = current
        result["total_frames"] = total
        result["progress"] = current / total if total > 0 else 0

    # Check for percentage
    match = re.search(patterns["percentage"], line)
    if match and "progress" not in result:
        result["progress"] = float(match.group(1)) / 100.0

    # Add message if we found something
    if result:
        result["message"] = line.strip()[:200]  # Truncate long messages

    return result if result else None


def run_pipeline_thread(
    project_id: str,
    project_dir: str,
    stages: list,
    roto_prompt: str,
    skip_existing: bool,
):
    """Run pipeline in a background thread."""
    from .api import active_jobs, active_jobs_lock
    from .websocket import update_progress

    with active_jobs_lock:
        active_jobs[project_id] = {
            "status": "running",
            "current_stage": stages[0] if stages else None,
            "progress": 0.0,
            "error": None,
        }

    # Build command
    script_path = get_run_pipeline_path()
    video_path = Path(project_dir) / "source"

    # Find the input video
    input_video = None
    for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"]:
        candidate = video_path / f"input{ext}"
        if candidate.exists():
            input_video = candidate
            break

    if not input_video:
        with active_jobs_lock:
            if project_id in active_jobs:
                active_jobs[project_id]["status"] = "failed"
                active_jobs[project_id]["error"] = "No input video found"
        update_progress(project_id, {
            "status": "failed",
            "error": "No input video found",
        })
        return

    cmd = [
        sys.executable,
        str(script_path),
        str(input_video),
        "--name", Path(project_dir).name,
        "--projects-dir", str(Path(project_dir).parent),
        "--stages", ",".join(stages),
        "--no-auto-comfyui",  # Web server already manages ComfyUI
    ]

    if roto_prompt and "roto" in stages:
        cmd.extend(["--prompt", roto_prompt])

    if skip_existing:
        cmd.append("--skip-existing")

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
                        if "progress" in progress_info:
                            active_jobs[project_id]["progress"] = progress_info["progress"]

                # Send WebSocket update
                update_progress(project_id, update)

            # Also send raw log line (for log viewer)
            # update_progress(project_id, {"type": "log", "line": line})

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
            update_progress(project_id, {
                "status": "completed",
                "progress": 1.0,
                "stage": stages[-1] if stages else None,
                "stage_index": len(stages) - 1,
                "total_stages": len(stages),
            })
        else:
            update_progress(project_id, {
                "status": "failed",
                "error": f"Pipeline exited with code {process.returncode}",
            })

    except Exception as e:
        with active_jobs_lock:
            if project_id in active_jobs:
                active_jobs[project_id]["status"] = "failed"
                active_jobs[project_id]["error"] = str(e)
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
):
    """Start pipeline processing in a background thread."""
    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(project_id, project_dir, stages, roto_prompt, skip_existing),
        daemon=True,
    )
    thread.start()


def stop_pipeline(project_id: str) -> bool:
    """Stop a running pipeline.

    Returns True if process was stopped, False if not running.
    """
    from .api import active_jobs, active_jobs_lock

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

        if project_id in active_processes:
            del active_processes[project_id]

        return True
    except Exception as e:
        print(f"Error stopping pipeline: {e}")
        return False
