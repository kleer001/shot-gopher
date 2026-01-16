"""ComfyUI process management for automatic startup.

Provides functions to start/stop ComfyUI automatically.
Used by both run_pipeline.py and the web interface.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from env_config import INSTALL_DIR
from comfyui_utils import DEFAULT_COMFYUI_URL, check_comfyui_running

# ComfyUI installation path
COMFYUI_DIR = INSTALL_DIR / "ComfyUI"

# Global process handle
_comfyui_process: Optional[subprocess.Popen] = None

# Video Depth Anything model configuration
# Using Small model (~6.8GB VRAM) instead of Large (~23.6GB VRAM)
VIDEO_DEPTH_MODEL_REPO = "depth-anything/Video-Depth-Anything-Small"
VIDEO_DEPTH_MODEL_NAME = "video_depth_anything_vits.pth"
VIDEO_DEPTH_MODEL_DIR = "videodepthanything"


def _predownload_depth_model(comfyui_path: Path) -> bool:
    """Ensure Video Depth Anything model is available before ComfyUI starts.

    This is a fallback for when the model wasn't downloaded via the install wizard.
    Downloads BEFORE ComfyUI starts to avoid BrokenPipeError during workflow
    execution (HuggingFace's tqdm progress bars conflict with ComfyUI's stderr).

    The install wizard is the primary way to download this model.

    Args:
        comfyui_path: Path to ComfyUI installation

    Returns:
        True if model is available (downloaded or already exists)
    """
    import shutil

    # Check if model already exists in ComfyUI models folder
    models_dir = comfyui_path / "models" / VIDEO_DEPTH_MODEL_DIR
    model_path = models_dir / VIDEO_DEPTH_MODEL_NAME
    if model_path.exists():
        return True

    print(f"  Pre-downloading {VIDEO_DEPTH_MODEL_NAME} from HuggingFace...")

    try:
        # Import here to avoid import errors if huggingface_hub not installed
        from huggingface_hub import snapshot_download

        # Download to HF cache (with progress bar - safe before ComfyUI starts)
        cache_dir = snapshot_download(
            repo_id=VIDEO_DEPTH_MODEL_REPO,
            allow_patterns=["*.pth"],
        )

        # Find the downloaded .pth file and copy to ComfyUI models folder
        cache_path = Path(cache_dir)
        pth_files = list(cache_path.glob("*.pth"))

        if pth_files:
            # Ensure models directory exists
            models_dir.mkdir(parents=True, exist_ok=True)

            # Find the correct model file
            src_file = None
            for f in pth_files:
                if f.name == VIDEO_DEPTH_MODEL_NAME:
                    src_file = f
                    break
            if src_file is None:
                src_file = pth_files[0]

            shutil.copy2(src_file, model_path)
            print(f"  Model ready: {VIDEO_DEPTH_MODEL_NAME}")
            return True
        else:
            print(f"  Warning: No .pth file found in {VIDEO_DEPTH_MODEL_REPO}", file=sys.stderr)
            return False

    except ImportError:
        print("  Warning: huggingface_hub not installed, skipping pre-download", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Warning: Could not pre-download model: {e}", file=sys.stderr)
        return False


def get_comfyui_path() -> Optional[Path]:
    """Get the path to ComfyUI installation."""
    if COMFYUI_DIR.exists() and (COMFYUI_DIR / "main.py").exists():
        return COMFYUI_DIR
    return None


def is_comfyui_running(url: str = DEFAULT_COMFYUI_URL) -> bool:
    """Check if ComfyUI is already running."""
    return check_comfyui_running(url)


def start_comfyui(
    url: str = DEFAULT_COMFYUI_URL,
    timeout: int = 60,
) -> bool:
    """Start ComfyUI server if not already running.

    Args:
        url: Expected ComfyUI URL
        timeout: Maximum seconds to wait for startup

    Returns:
        True if ComfyUI is running (either started or already was)
    """
    global _comfyui_process

    # Find ComfyUI installation first
    comfyui_path = get_comfyui_path()
    if not comfyui_path:
        print(f"ComfyUI not found at {COMFYUI_DIR}", file=sys.stderr)
        print("Run the install wizard to install ComfyUI", file=sys.stderr)
        return False

    # Pre-download Depth Anything model to avoid HuggingFace download during workflow
    # (HuggingFace's tqdm progress bars cause BrokenPipeError in ComfyUI)
    _predownload_depth_model(comfyui_path)

    # Check if already running
    if is_comfyui_running(url):
        print("ComfyUI already running")
        return True

    print(f"Starting ComfyUI from {comfyui_path}...")

    # Build command
    # Set output-directory to repo parent to allow saving to project directories
    # (ComfyUI security blocks saving outside its output folder)
    output_base = COMFYUI_DIR.parent.parent.parent  # .vfx_pipeline -> comfyui_ingest -> parent
    cmd = [
        sys.executable,
        "main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--output-directory", str(output_base),
    ]

    # Set environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Disable HuggingFace progress bars to prevent BrokenPipeError
    # when tqdm tries to write to stderr during model downloads
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        # Start ComfyUI as subprocess
        # Don't pipe stdout/stderr - let them flow to terminal to avoid
        # "Broken pipe" errors when tqdm/progress bars write output
        _comfyui_process = subprocess.Popen(
            cmd,
            cwd=str(comfyui_path),
            stdout=None,  # Inherit from parent (shows in terminal)
            stderr=None,  # Inherit from parent (shows in terminal)
            env=env,
        )

        # Wait for ComfyUI to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if is_comfyui_running(url):
                print(f"ComfyUI started successfully (PID: {_comfyui_process.pid})")
                return True

            # Check if process died
            if _comfyui_process.poll() is not None:
                print("ComfyUI process exited unexpectedly", file=sys.stderr)
                print(f"Exit code: {_comfyui_process.returncode}", file=sys.stderr)
                return False

            time.sleep(1)

        print(f"Timeout waiting for ComfyUI to start ({timeout}s)", file=sys.stderr)
        stop_comfyui()
        return False

    except Exception as e:
        print(f"Failed to start ComfyUI: {e}", file=sys.stderr)
        return False


def stop_comfyui():
    """Stop the ComfyUI process if we started it."""
    global _comfyui_process

    if _comfyui_process is None:
        return

    print("Stopping ComfyUI...")

    try:
        _comfyui_process.terminate()
        try:
            _comfyui_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _comfyui_process.kill()
            _comfyui_process.wait()
        print("ComfyUI stopped")
    except Exception as e:
        print(f"Error stopping ComfyUI: {e}", file=sys.stderr)
    finally:
        _comfyui_process = None


def get_comfyui_status() -> dict:
    """Get ComfyUI status information."""
    comfyui_path = get_comfyui_path()
    running = is_comfyui_running()

    return {
        "installed": comfyui_path is not None,
        "path": str(comfyui_path) if comfyui_path else None,
        "running": running,
        "managed": _comfyui_process is not None,
        "pid": _comfyui_process.pid if _comfyui_process else None,
    }


def ensure_comfyui(url: str = DEFAULT_COMFYUI_URL, timeout: int = 60) -> bool:
    """Ensure ComfyUI is running, starting it if necessary.

    This is a convenience function for scripts that need ComfyUI.
    Returns True if ComfyUI is available, False otherwise.
    """
    return start_comfyui(url=url, timeout=timeout)


def kill_all_comfyui_processes() -> int:
    """Kill ALL ComfyUI processes system-wide to free GPU memory.

    This is useful before starting a fresh pipeline run to ensure
    no stale ComfyUI processes are hogging VRAM.

    Returns:
        Number of processes killed
    """
    global _comfyui_process

    # ANSI escape codes for bold red text
    BOLD_RED = "\033[1;91m"
    RESET = "\033[0m"

    killed = 0
    pids_to_kill = []

    # Check for managed process
    if _comfyui_process is not None:
        pids_to_kill.append(str(_comfyui_process.pid))

    # Find any other ComfyUI processes
    # Try multiple patterns since ComfyUI can be started different ways
    patterns = [
        "ComfyUI.*main.py",           # If ComfyUI is in the path
        "main.py.*--port.*8188",      # ComfyUI's default port
        "python.*main.py.*--listen",  # ComfyUI uses --listen flag
    ]

    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                for pid in result.stdout.strip().split('\n'):
                    pid = pid.strip()
                    if pid and pid not in pids_to_kill:
                        pids_to_kill.append(pid)
        except FileNotFoundError:
            break  # pgrep not available, no point trying other patterns
        except Exception:
            pass

    # If nothing to kill, we're good
    if not pids_to_kill:
        print("  → No stale ComfyUI processes found")
        return 0

    # Found stale processes - print warning and kill them
    print(f"{BOLD_RED}  ⚠ FOUND STALE COMFYUI PROCESS(ES). KILLING THEM. SORRY!{RESET}")

    # Stop managed process first
    if _comfyui_process is not None:
        stop_comfyui()
        killed += 1

    # Kill the rest
    for pid in pids_to_kill:
        if _comfyui_process and pid == str(_comfyui_process.pid):
            continue  # Already handled above
        try:
            subprocess.run(["kill", pid], capture_output=True)
            killed += 1
            print(f"  → Killed ComfyUI process (PID: {pid})")
        except Exception:
            pass

    # Give processes time to die gracefully
    if killed > 0:
        time.sleep(2)

        # Force kill any survivors
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    for pid in result.stdout.strip().split('\n'):
                        pid = pid.strip()
                        if pid:
                            try:
                                subprocess.run(["kill", "-9", pid], capture_output=True)
                                print(f"  → Force killed ComfyUI process (PID: {pid})")
                            except Exception:
                                pass
            except Exception:
                pass

    print(f"  → Killed {killed} ComfyUI process(es)")
    return killed
