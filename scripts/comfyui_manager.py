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

# Depth Anything V3 model configuration
# HuggingFace repos for each model variant
DA3_MODEL_REPOS = {
    "da3_small.safetensors": "depth-anything/DA3-SMALL",
    "da3_base.safetensors": "depth-anything/DA3-BASE",
    "da3_large.safetensors": "depth-anything/DA3-LARGE-1.1",
    "da3_giant.safetensors": "depth-anything/DA3-GIANT-1.1",
    "da3mono_large.safetensors": "depth-anything/DA3MONO-LARGE",
    "da3metric_large.safetensors": "depth-anything/DA3METRIC-LARGE",
    "da3nested_giant_large.safetensors": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
}

# Default model used in our workflows
DEFAULT_DA3_MODEL = "da3metric_large.safetensors"


def _predownload_depth_model(comfyui_path: Path, model_name: str = DEFAULT_DA3_MODEL) -> bool:
    """Pre-download Depth Anything V3 model to avoid HuggingFace download during workflow.

    Downloads BEFORE ComfyUI starts, so tqdm progress bars work normally.
    This avoids the BrokenPipeError that occurs when HuggingFace downloads
    happen during ComfyUI workflow execution (when stderr is wrapped).

    Args:
        comfyui_path: Path to ComfyUI installation
        model_name: Model filename (e.g., "da3metric_large.safetensors")

    Returns:
        True if model is available (downloaded or already exists)
    """
    if model_name not in DA3_MODEL_REPOS:
        print(f"  Unknown model: {model_name}", file=sys.stderr)
        return False

    models_dir = comfyui_path / "models" / "depthanything3"
    model_path = models_dir / model_name

    # Check if already downloaded
    if model_path.exists():
        return True

    repo_id = DA3_MODEL_REPOS[model_name]
    print(f"  Downloading {model_name} from HuggingFace ({repo_id})...")

    # Ensure models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Import here to avoid import errors if huggingface_hub not installed
        from huggingface_hub import hf_hub_download

        # Download with progress bars enabled (safe before ComfyUI starts)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_name,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  Model downloaded: {model_name}")
        return True

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
    cmd = [
        sys.executable,
        "main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
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
