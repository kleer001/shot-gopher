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


def get_comfyui_path() -> Optional[Path]:
    """Get the path to ComfyUI installation."""
    if COMFYUI_DIR.exists() and (COMFYUI_DIR / "main.py").exists():
        return COMFYUI_DIR
    return None


def is_comfyui_running(url: str = DEFAULT_COMFYUI_URL) -> bool:
    """Check if ComfyUI is already running."""
    return check_comfyui_running(url)


def _check_comfyui_needs_patch(comfyui_path: Path) -> tuple[bool, str]:
    """Check if ComfyUI's logger.py needs the BrokenPipeError patch.

    Returns:
        (needs_patch, reason) tuple
    """
    logger_path = comfyui_path / "app" / "logger.py"

    if not logger_path.exists():
        return False, "logger.py not found"

    content = logger_path.read_text()

    # Check if already has error handling (patched or fixed upstream)
    if "except (OSError, ValueError)" in content:
        return False, "already patched"

    if "except OSError" in content or "except BrokenPipeError" in content:
        return False, "already fixed upstream"

    # Check for the vulnerable pattern: flush without try/except
    vulnerable_pattern = "def flush(self):\n        super().flush()"
    if vulnerable_pattern in content:
        return True, "vulnerable pattern found"

    # Check alternate indentation
    vulnerable_pattern_alt = "def flush(self):\n    super().flush()"
    if vulnerable_pattern_alt in content:
        return True, "vulnerable pattern found (4-space indent)"

    return False, "flush method not found or different structure"


def _patch_comfyui_logger(comfyui_path: Path) -> bool:
    """Patch ComfyUI's logger.py to handle flush errors gracefully.

    Only patches if the vulnerable code pattern is detected.
    Does nothing if already patched or fixed upstream.

    This fixes BrokenPipeError that can occur when tqdm/progress bars
    try to write to stderr and the flush fails.

    See: https://github.com/Comfy-Org/ComfyUI/pull/11629

    Returns:
        True if patch was applied this run, False otherwise
    """
    logger_path = comfyui_path / "app" / "logger.py"

    # Check if patching is needed
    needs_patch, reason = _check_comfyui_needs_patch(comfyui_path)

    if not needs_patch:
        # No action needed - either already fixed or not applicable
        return False

    try:
        content = logger_path.read_text()

        # Apply patch based on detected pattern
        patched = False
        for old_pattern, indent in [
            ("def flush(self):\n        super().flush()", "        "),
            ("def flush(self):\n    super().flush()", "    "),
        ]:
            if old_pattern in content:
                new_pattern = f"""def flush(self):
{indent}try:
{indent}    super().flush()
{indent}except (OSError, ValueError):
{indent}    pass  # Ignore flush errors (BrokenPipe, etc.)"""
                content = content.replace(old_pattern, new_pattern)
                patched = True
                break

        if patched:
            logger_path.write_text(content)
            print("  Patched ComfyUI logger.py to handle flush errors")
            return True

        return False

    except Exception as e:
        print(f"  Warning: Could not patch ComfyUI logger: {e}", file=sys.stderr)
        return False


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

    # Find ComfyUI installation first (needed for patching)
    comfyui_path = get_comfyui_path()
    if not comfyui_path:
        print(f"ComfyUI not found at {COMFYUI_DIR}", file=sys.stderr)
        print("Run the install wizard to install ComfyUI", file=sys.stderr)
        return False

    # Check and apply logger patch if needed (fixes BrokenPipeError)
    needs_patch, reason = _check_comfyui_needs_patch(comfyui_path)
    patch_applied = False

    if needs_patch:
        print(f"  ComfyUI logger needs patch: {reason}")
        patch_applied = _patch_comfyui_logger(comfyui_path)

    # Check if already running
    if is_comfyui_running(url):
        print("ComfyUI already running")
        if patch_applied:
            print("  Warning: Logger patch applied - restart ComfyUI to take effect!")
            print("  Run: pkill -f 'ComfyUI/main.py' && restart the pipeline")
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
