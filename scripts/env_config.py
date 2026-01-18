#!/usr/bin/env python3
"""Centralized conda environment configuration for VFX Pipeline.

This module provides:
- Single source of truth for environment name and settings
- Functions to check if the correct environment is active
- Helpers to guide users to activate the environment

Usage in production scripts:
    from env_config import require_conda_env, CONDA_ENV_NAME

    # At the start of your script:
    require_conda_env()  # Exits with helpful message if wrong env

    # Or check manually:
    if not is_conda_env_active():
        print(get_activation_instructions())
"""

import os
import sys
from pathlib import Path


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# The canonical conda environment name for the VFX Pipeline
CONDA_ENV_NAME = "vfx-pipeline"

# Python version requirement
PYTHON_VERSION = "3.10"

# Repo root directory (comfyui_ingest/)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Installation directory for tools (ComfyUI, WHAM, ECON, models)
# This stays inside/near the repo as it's tooling, not project data
# Allow environment variable override for container support
INSTALL_DIR = Path(os.environ.get(
    "VFX_INSTALL_DIR",
    str(_REPO_ROOT / ".vfx_pipeline")
))

# Default location for VFX projects (sibling to repo, not inside it)
# Allow environment variable override for container support
DEFAULT_PROJECTS_DIR = Path(os.environ.get(
    "VFX_PROJECTS_DIR",
    str(_REPO_ROOT.parent / "vfx_projects")
))

# Path to the generated activation script
ACTIVATION_SCRIPT = INSTALL_DIR / "activate.sh"


# =============================================================================
# CONTAINER DETECTION
# =============================================================================

def is_in_container() -> bool:
    """Detect if running inside a container (Docker, Kubernetes, etc.).

    Returns:
        True if in container, False otherwise.
    """
    # Check for Docker-specific file
    if os.path.exists("/.dockerenv"):
        return True

    # Check environment variable (set in Dockerfile)
    if os.environ.get("CONTAINER") == "true":
        return True

    # Check for Kubernetes
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True

    # Check cgroup for container indicators (Linux containers)
    try:
        with open("/proc/1/cgroup", "rt") as f:
            content = f.read()
            return "docker" in content or "kubepods" in content
    except Exception:
        pass

    return False


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def get_active_conda_env() -> str | None:
    """Get the currently active conda environment name.

    Returns:
        Environment name if in a conda environment, None otherwise.
    """
    return os.environ.get("CONDA_DEFAULT_ENV")


def is_conda_env_active(env_name: str = None) -> bool:
    """Check if the specified conda environment is currently active.

    Args:
        env_name: Environment name to check. Defaults to CONDA_ENV_NAME.

    Returns:
        True if the specified environment is active.
    """
    if env_name is None:
        env_name = CONDA_ENV_NAME
    return get_active_conda_env() == env_name


def is_any_conda_env_active() -> bool:
    """Check if any conda environment is currently active.

    Returns:
        True if any conda environment is active.
    """
    return get_active_conda_env() is not None


def get_conda_prefix() -> Path | None:
    """Get the path to the active conda environment.

    Returns:
        Path to conda environment prefix, or None if not in conda.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    return Path(prefix) if prefix else None


# =============================================================================
# ACTIVATION HELPERS
# =============================================================================

def get_activation_command() -> str:
    """Get the shell command to activate the VFX Pipeline environment.

    Returns:
        Shell command string.
    """
    return f"conda activate {CONDA_ENV_NAME}"


def get_activation_script_path() -> Path | None:
    """Get path to the activation script if it exists.

    Returns:
        Path to activate.sh if it exists, None otherwise.
    """
    if ACTIVATION_SCRIPT.exists():
        return ACTIVATION_SCRIPT
    return None


def get_activation_instructions() -> str:
    """Get user-friendly instructions for activating the environment.

    Returns:
        Multi-line string with activation instructions.
    """
    active_env = get_active_conda_env()

    lines = [
        "",
        "=" * 60,
        f"VFX Pipeline requires the '{CONDA_ENV_NAME}' conda environment",
        "=" * 60,
        "",
    ]

    if active_env:
        lines.append(f"Currently active: '{active_env}'")
        lines.append(f"Required:         '{CONDA_ENV_NAME}'")
        lines.append("")
    else:
        lines.append("No conda environment is currently active.")
        lines.append("")

    lines.append("To activate the correct environment, run:")
    lines.append("")
    lines.append(f"    {get_activation_command()}")
    lines.append("")

    # Check for the generated activation script
    if ACTIVATION_SCRIPT.exists():
        lines.append("Or use the generated activation script:")
        lines.append("")
        lines.append(f"    source {ACTIVATION_SCRIPT}")
        lines.append("")

    lines.append("Then re-run this script.")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# ENFORCEMENT
# =============================================================================

def require_conda_env(env_name: str = None, exit_on_fail: bool = True) -> bool:
    """Require that the correct conda environment is active.

    This function should be called at the start of production scripts
    to ensure they run in the correct environment.

    Args:
        env_name: Environment name to require. Defaults to CONDA_ENV_NAME.
        exit_on_fail: If True, exit the process with an error message.
                      If False, just return False.

    Returns:
        True if environment is correct, False otherwise (only if exit_on_fail=False).

    Exits:
        With code 1 if environment is wrong and exit_on_fail=True.
    """
    # Skip conda checks inside containers (dependencies managed by Docker)
    if is_in_container():
        return True

    if env_name is None:
        env_name = CONDA_ENV_NAME

    if is_conda_env_active(env_name):
        return True

    if exit_on_fail:
        print(get_activation_instructions(), file=sys.stderr)
        sys.exit(1)

    return False


def check_conda_env_or_warn(env_name: str = None) -> bool:
    """Check environment and print a warning if wrong, but continue execution.

    Use this for scripts that can partially work outside the environment
    but may have reduced functionality.

    Args:
        env_name: Environment name to check. Defaults to CONDA_ENV_NAME.

    Returns:
        True if environment is correct, False otherwise.
    """
    # Skip conda checks inside containers (dependencies managed by Docker)
    if is_in_container():
        print("Running in container environment - skipping conda checks")
        return True

    if env_name is None:
        env_name = CONDA_ENV_NAME

    if is_conda_env_active(env_name):
        return True

    active = get_active_conda_env()
    if active:
        print(f"Warning: Running in '{active}' environment, "
              f"expected '{env_name}'", file=sys.stderr)
    else:
        print(f"Warning: No conda environment active, "
              f"expected '{env_name}'", file=sys.stderr)
    print(f"Some features may not work. Run: {get_activation_command()}",
          file=sys.stderr)
    print("", file=sys.stderr)

    return False


# =============================================================================
# SCRIPT EXECUTION HELPERS
# =============================================================================

def get_env_python() -> Path | None:
    """Get the path to the Python executable in the VFX Pipeline environment.

    Returns:
        Path to Python if environment exists, None otherwise.
    """
    # Check if we're in the right environment
    prefix = get_conda_prefix()
    if prefix and is_conda_env_active():
        return prefix / "bin" / "python"

    # Try to find it in the expected location
    import subprocess
    try:
        result = subprocess.run(
            ["conda", "run", "-n", CONDA_ENV_NAME, "which", "python"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def run_in_env(command: list[str], **kwargs) -> "subprocess.CompletedProcess":
    """Run a command in the VFX Pipeline conda environment.

    If already in the correct environment, runs directly.
    Otherwise, uses 'conda run' to execute in the environment.

    Args:
        command: Command and arguments as a list.
        **kwargs: Additional arguments passed to subprocess.run.

    Returns:
        CompletedProcess instance.
    """
    import subprocess

    if is_conda_env_active():
        # Already in correct environment, run directly
        return subprocess.run(command, **kwargs)
    else:
        # Use conda run to execute in the environment
        conda_cmd = ["conda", "run", "-n", CONDA_ENV_NAME, "--no-capture-output"]
        conda_cmd.extend(command)
        return subprocess.run(conda_cmd, **kwargs)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI to check environment status and show activation instructions."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and manage VFX Pipeline conda environment"
    )
    parser.add_argument(
        "--check", "-c", action="store_true",
        help="Check if correct environment is active (exit 0 if yes, 1 if no)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress output (useful for scripting)"
    )
    parser.add_argument(
        "--show-env", action="store_true",
        help="Show the required environment name"
    )
    parser.add_argument(
        "--show-activate", action="store_true",
        help="Show the activation command"
    )

    args = parser.parse_args()

    # Handle specific queries
    if args.show_env:
        print(CONDA_ENV_NAME)
        return 0

    if args.show_activate:
        print(get_activation_command())
        return 0

    # Default: show status
    active = is_conda_env_active()

    if args.check:
        if not args.quiet:
            if active:
                print(f"Environment '{CONDA_ENV_NAME}' is active")
            else:
                current = get_active_conda_env() or "none"
                print(f"Wrong environment: '{current}' (need '{CONDA_ENV_NAME}')")
        return 0 if active else 1

    # Full status display
    print(f"VFX Pipeline Environment Configuration")
    print("=" * 40)
    print(f"Required environment: {CONDA_ENV_NAME}")
    print(f"Current environment:  {get_active_conda_env() or 'none'}")
    print(f"Status:              {'ACTIVE' if active else 'NOT ACTIVE'}")
    print()

    if not active:
        print("To activate:")
        print(f"  {get_activation_command()}")
        if ACTIVATION_SCRIPT.exists():
            print(f"  source {ACTIVATION_SCRIPT}")

    return 0 if active else 1


if __name__ == "__main__":
    sys.exit(main())
