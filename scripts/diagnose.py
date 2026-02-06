"""Diagnostic script for the VFX pipeline.

Reports platform info, resolved tool paths, GPU status, conda environments,
and PyTorch CUDA availability. Useful for debugging installation issues.

Usage:
    python scripts/diagnose.py
    python scripts/install_wizard.py --diagnose
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


ESSENTIAL_TOOLS: List[str] = [
    "nvidia-smi",
    "nvcc",
    "conda",
    "python",
    "git",
    "ffmpeg",
    "ffprobe",
    "colmap",
    "7z",
    "blender",
    "aria2c",
]


class Colors:
    """Terminal colors."""

    HEADER = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def _header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")


def _ok(text: str) -> None:
    print(f"  {Colors.GREEN}OK{Colors.END} {text}")


def _warn(text: str) -> None:
    print(f"  {Colors.YELLOW}!!{Colors.END} {text}")


def _fail(text: str) -> None:
    print(f"  {Colors.RED}XX{Colors.END} {text}")


def _info(text: str) -> None:
    print(f"  {Colors.CYAN}>>{Colors.END} {text}")


def _run(cmd: List[str], timeout: int = 10) -> Tuple[bool, str]:
    """Run a command, return (success, stdout+stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, (result.stdout + result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def report_platform() -> None:
    """Report platform and environment info."""
    _header("Platform")
    _info(f"OS:           {platform.system()} {platform.release()}")
    _info(f"Architecture: {platform.machine()}")
    _info(f"Python:       {sys.version}")
    _info(f"Executable:   {sys.executable}")
    _info(f"Repo root:    {REPO_ROOT}")

    if platform.system() == "Linux":
        try:
            with open("/proc/version", "r", encoding="utf-8") as f:
                version_info = f.read().lower()
            if "microsoft" in version_info or "wsl" in version_info:
                _info("Environment:  WSL2")
            else:
                _info("Environment:  Native Linux")
        except FileNotFoundError:
            _info("Environment:  Linux (unknown)")


def _resolve_tool(tool_name: str) -> Tuple[Optional[str], str]:
    """Find a tool and report how it was discovered.

    Returns:
        (resolved_path_or_None, discovery_method)
        discovery_method is one of: "PATH", "repo-local", "known location", "not found"
    """
    from install_wizard.platform import PlatformManager

    local_paths = PlatformManager._get_local_tool_paths(tool_name)
    for path in local_paths:
        if path.exists():
            return str(path), "repo-local"

    path_result = shutil.which(tool_name)
    if path_result:
        return path_result, "PATH"

    if sys.platform == "win32":
        search_paths = PlatformManager._get_windows_tool_paths(tool_name)
    else:
        search_paths = PlatformManager._get_unix_tool_paths(tool_name)

    for path in search_paths:
        if path.exists():
            return str(path), "known location"

    return None, "not found"


def _get_version(tool_path: str, tool_name: str) -> str:
    """Try to get a tool's version string."""
    version_flags: Dict[str, List[str]] = {
        "nvidia-smi": [],
        "nvcc": ["--version"],
        "conda": ["--version"],
        "python": ["--version"],
        "git": ["--version"],
        "ffmpeg": ["-version"],
        "ffprobe": ["-version"],
        "colmap": ["--version"],
        "7z": [],
        "blender": ["--version"],
        "aria2c": ["--version"],
    }

    flags = version_flags.get(tool_name, ["--version"])
    success, output = _run([tool_path] + flags)
    if not success or not output:
        return ""

    first_line = output.splitlines()[0]
    return first_line[:80]


def report_tools() -> None:
    """Report location and discovery method for each essential tool."""
    _header("Tool Paths")

    max_name = max(len(t) for t in ESSENTIAL_TOOLS)

    for tool_name in ESSENTIAL_TOOLS:
        resolved, method = _resolve_tool(tool_name)
        label = tool_name.ljust(max_name)

        if resolved:
            version = _get_version(resolved, tool_name)
            version_suffix = f"  ({version})" if version else ""
            _ok(f"{label}  [{method}] {resolved}{version_suffix}")
        else:
            _fail(f"{label}  [not found]")


def report_gpu() -> None:
    """Report GPU info via nvidia-smi if available."""
    _header("GPU")

    resolved, _ = _resolve_tool("nvidia-smi")
    if not resolved:
        _warn("nvidia-smi not found - cannot detect NVIDIA GPU")
        return

    success, output = _run([
        resolved,
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not success:
        _fail(f"nvidia-smi found at {resolved} but failed to query GPU")
        return

    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            _ok(f"{parts[0]}  (driver {parts[1]}, {parts[2]} MB VRAM)")
        else:
            _ok(line.strip())


def report_conda_envs() -> None:
    """Report conda environments."""
    _header("Conda Environments")

    conda_path, _ = _resolve_tool("conda")
    if not conda_path:
        _warn("conda not found")
        return

    success, output = _run([conda_path, "env", "list"], timeout=15)
    if not success:
        _fail("Failed to list conda environments")
        return

    pipeline_envs = {"vfx-pipeline", "gvhmr", "colmap", "videomama", "gs-ir"}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        env_name = parts[0]
        env_path = parts[-1] if len(parts) > 1 else ""
        if env_name in pipeline_envs:
            _ok(f"{env_name.ljust(15)} {env_path}")
        elif env_name == "base":
            _info(f"{env_name.ljust(15)} {env_path}")


def report_pytorch() -> None:
    """Report PyTorch installation and CUDA status."""
    _header("PyTorch / CUDA")

    try:
        import torch

        _ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                _ok(f"CUDA device {i}: {name}")
        else:
            _warn("torch.cuda.is_available() = False (CPU-only build)")
            _info("Reinstall with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        _warn("PyTorch not installed in current environment")


def report_env_vars() -> None:
    """Report pipeline-relevant environment variables."""
    _header("Environment Variables")

    relevant_vars = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "CONDA_PREFIX",
        "VFX_MODELS_DIR",
        "VFX_PROJECTS_DIR",
        "COMFYUI_PORT",
        "PATH",
    ]

    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            display = value[:120] + "..." if len(value) > 120 else value
            _info(f"{var}={display}")


def main() -> None:
    """Run all diagnostic reports."""
    print(f"\n{Colors.BOLD}VFX Pipeline Diagnostics{Colors.END}")
    print(f"{'─' * 40}")

    report_platform()
    report_tools()
    report_gpu()
    report_conda_envs()
    report_pytorch()
    report_env_vars()

    print()


if __name__ == "__main__":
    main()
