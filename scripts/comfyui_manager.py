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

from env_config import INSTALL_DIR, is_in_container, is_windows
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


def _get_docker_compose_cmd() -> Optional[list]:
    """Get the docker compose command (plugin or standalone)."""
    import shutil
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return ["docker", "compose"]
    except Exception:
        pass

    if shutil.which("docker-compose"):
        return ["docker-compose"]

    return None


def _check_docker_available() -> bool:
    """Check if Docker is available and vfx-ingest image exists."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "vfx-ingest:latest"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def start_comfyui_docker(url: str = DEFAULT_COMFYUI_URL, timeout: int = 90) -> bool:
    """Start ComfyUI via Docker container.

    Args:
        url: Expected ComfyUI URL
        timeout: Maximum seconds to wait for startup

    Returns:
        True if ComfyUI is running
    """
    if is_comfyui_running(url):
        print("ComfyUI already running")
        return True

    compose_cmd = _get_docker_compose_cmd()
    if not compose_cmd:
        print("Docker Compose not available", file=sys.stderr)
        return False

    if not _check_docker_available():
        print("Docker image vfx-ingest:latest not found", file=sys.stderr)
        print("Build it first: docker compose build", file=sys.stderr)
        return False

    repo_root = _get_repo_root()
    models_dir = repo_root / ".vfx_pipeline" / "models"
    projects_dir = repo_root.parent / "vfx_projects"

    env = os.environ.copy()
    env["VFX_MODELS_DIR"] = str(models_dir)
    env["VFX_PROJECTS_DIR"] = str(projects_dir)
    env["HOST_UID"] = str(os.getuid()) if hasattr(os, 'getuid') else "0"
    env["HOST_GID"] = str(os.getgid()) if hasattr(os, 'getgid') else "0"

    print(f"Starting ComfyUI via Docker...")

    try:
        cmd = compose_cmd + ["run", "-d", "--rm", "--service-ports", "vfx-ingest", "interactive"]
        subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            check=True,
            capture_output=True
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            if is_comfyui_running(url):
                print(f"ComfyUI (Docker) ready on {url}")
                return True
            time.sleep(2)

        print(f"Timeout waiting for Docker ComfyUI ({timeout}s)", file=sys.stderr)
        return False

    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker ComfyUI: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr.decode(), file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error starting Docker ComfyUI: {e}", file=sys.stderr)
        return False


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
    # Container-aware: use COMFYUI_OUTPUT_DIR environment variable if in container
    if is_in_container():
        output_base = Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/workspace"))
        listen_addr = "0.0.0.0"  # Must listen on all interfaces in container
    else:
        output_base = COMFYUI_DIR.parent.parent.parent  # .vfx_pipeline -> shot-gopher -> parent
        listen_addr = "127.0.0.1"  # Local only for security

    cmd = [
        sys.executable,
        "main.py",
        "--listen", listen_addr,
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

    Tries local installation first, falls back to Docker if local unavailable.
    Returns True if ComfyUI is available, False otherwise.
    """
    if is_comfyui_running(url):
        print("ComfyUI already running")
        return True

    if get_comfyui_path():
        return start_comfyui(url=url, timeout=timeout)

    if _check_docker_available():
        print("Local ComfyUI not found, trying Docker...")
        return start_comfyui_docker(url=url, timeout=timeout)

    print("No ComfyUI installation found (local or Docker)", file=sys.stderr)
    print("Install via: python scripts/install_wizard.py", file=sys.stderr)
    print("Or Docker:   python scripts/install_wizard.py --docker", file=sys.stderr)
    return False


def _find_comfyui_pids_unix() -> list:
    """Find ComfyUI process PIDs on Unix systems using pgrep."""
    pids = []
    patterns = [
        "ComfyUI.*main.py",
        "main.py.*--port.*8188",
        "python.*main.py.*--listen",
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
                    if pid and pid not in pids:
                        pids.append(pid)
        except FileNotFoundError:
            break
        except Exception:
            pass

    return pids


def _find_comfyui_pids_windows() -> list:
    """Find ComfyUI process PIDs on Windows using PowerShell (preferred) or wmic (fallback)."""
    pids = []
    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0

    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*main.py*' -and $_.CommandLine -like '*--listen*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            creationflags=creation_flags
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and line.isdigit():
                    pids.append(line)
    except Exception:
        pass

    if pids:
        return pids

    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             "CommandLine like '%main.py%' and CommandLine like '%--listen%'",
             "get", "ProcessId"],
            capture_output=True,
            text=True,
            creationflags=creation_flags
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and line.isdigit():
                    pids.append(line)
    except Exception:
        pass

    if not pids:
        try:
            result = subprocess.run(
                ["wmic", "process", "where",
                 "CommandLine like '%ComfyUI%main.py%'",
                 "get", "ProcessId"],
                capture_output=True,
                text=True,
                creationflags=creation_flags
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    line = line.strip()
                    if line and line.isdigit():
                        pids.append(line)
        except Exception:
            pass

    return pids


def _kill_pid_unix(pid: str, force: bool = False) -> bool:
    """Kill a process by PID on Unix systems."""
    try:
        cmd = ["kill", "-9", pid] if force else ["kill", pid]
        subprocess.run(cmd, capture_output=True)
        return True
    except Exception:
        return False


def _kill_pid_windows(pid: str, force: bool = False) -> bool:
    """Kill a process by PID on Windows systems."""
    try:
        cmd = ["taskkill", "/F", "/PID", pid] if force else ["taskkill", "/PID", pid]
        subprocess.run(
            cmd,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        return True
    except Exception:
        return False


def kill_all_comfyui_processes() -> int:
    """Kill ALL ComfyUI processes system-wide to free GPU memory.

    This is useful before starting a fresh pipeline run to ensure
    no stale ComfyUI processes are hogging VRAM.

    Cross-platform: Uses pgrep/kill on Unix, wmic/taskkill on Windows.

    Returns:
        Number of processes killed
    """
    global _comfyui_process

    BOLD_RED = "\033[1;91m"
    RESET = "\033[0m"

    killed = 0
    pids_to_kill = []

    if _comfyui_process is not None:
        pids_to_kill.append(str(_comfyui_process.pid))

    if is_windows():
        pids_to_kill.extend(_find_comfyui_pids_windows())
        kill_fn = _kill_pid_windows
        find_fn = _find_comfyui_pids_windows
    else:
        pids_to_kill.extend(_find_comfyui_pids_unix())
        kill_fn = _kill_pid_unix
        find_fn = _find_comfyui_pids_unix

    pids_to_kill = list(dict.fromkeys(pids_to_kill))

    if not pids_to_kill:
        print("  → No stale ComfyUI processes found")
        return 0

    print(f"{BOLD_RED}  ⚠ FOUND STALE COMFYUI PROCESS(ES). KILLING THEM. SORRY!{RESET}")

    if _comfyui_process is not None:
        stop_comfyui()
        killed += 1

    for pid in pids_to_kill:
        if _comfyui_process and pid == str(_comfyui_process.pid):
            continue
        if kill_fn(pid, force=False):
            killed += 1
            print(f"  → Killed ComfyUI process (PID: {pid})")

    if killed > 0:
        time.sleep(2)

        remaining_pids = find_fn()
        for pid in remaining_pids:
            if kill_fn(pid, force=True):
                print(f"  → Force killed ComfyUI process (PID: {pid})")

    print(f"  → Killed {killed} ComfyUI process(es)")
    return killed


def prepare_comfyui_for_processing(
    url: str = DEFAULT_COMFYUI_URL,
    auto_start: bool = True
) -> bool:
    """Prepare ComfyUI for processing - cleanup stale processes and start if needed.

    Args:
        url: ComfyUI API URL
        auto_start: If True, start ComfyUI if not running. If False, just check.

    Returns:
        True if ComfyUI is ready for processing
    """
    if auto_start:
        print("\n[GPU Cleanup]")
        kill_all_comfyui_processes()

        print("\n[ComfyUI] Starting ComfyUI...")
        if not ensure_comfyui(url=url):
            print("Error: Failed to start ComfyUI", file=sys.stderr)
            print("Install ComfyUI with the install wizard or start it manually", file=sys.stderr)
            return False
        return True

    if not is_comfyui_running(url):
        print(f"\nError: ComfyUI not running at {url}", file=sys.stderr)
        print("Start ComfyUI first or enable auto-start", file=sys.stderr)
        return False

    return True
