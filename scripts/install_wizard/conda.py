"""Conda environment management for the installation wizard.

This module handles conda environment detection, creation, and package
installation within conda environments.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Import centralized environment configuration
from env_config import CONDA_ENV_NAME, CONDA_ENV_PREFIX, CONDA_ENVS_DIR, PYTHON_VERSION

from .utils import print_error, print_success, print_warning, run_command


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


class CondaEnvironmentManager:
    """Manages conda environment creation and activation.

    All environments use prefix mode (-p) under CONDA_ENVS_DIR so they
    are sandboxed inside the repo. The env_name property is derived from
    the prefix path for display purposes.
    """

    def __init__(self, env_prefix: Path = CONDA_ENV_PREFIX):
        self.env_prefix = env_prefix
        self.env_name = env_prefix.name
        self.conda_exe = None
        self.python_version = PYTHON_VERSION

    def detect_conda(self) -> bool:
        """Check if conda is installed and available."""
        # Try conda command directly (works if in PATH)
        success, output = run_command(["conda", "--version"], check=False, capture=True)
        if success:
            self.conda_exe = "conda"
            return True

        # Try mamba
        success, output = run_command(["mamba", "--version"], check=False, capture=True)
        if success:
            self.conda_exe = "mamba"
            return True

        # Check CONDA_EXE environment variable (set by conda init)
        conda_exe_env = os.environ.get('CONDA_EXE')
        if conda_exe_env and Path(conda_exe_env).exists():
            success, output = run_command([conda_exe_env, "--version"], check=False, capture=True)
            if success:
                self.conda_exe = conda_exe_env
                return True

        home = Path.home()

        if _is_windows():
            localappdata = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
            programdata = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
            common_paths = [
                home / "miniconda3" / "Scripts" / "conda.exe",
                home / "Miniconda3" / "Scripts" / "conda.exe",
                home / "anaconda3" / "Scripts" / "conda.exe",
                home / "Anaconda3" / "Scripts" / "conda.exe",
                home / "mambaforge" / "Scripts" / "conda.exe",
                home / ".conda" / "Scripts" / "conda.exe",
                localappdata / "miniconda3" / "Scripts" / "conda.exe",
                localappdata / "Continuum" / "miniconda3" / "Scripts" / "conda.exe",
                localappdata / "Continuum" / "anaconda3" / "Scripts" / "conda.exe",
                programdata / "miniconda3" / "Scripts" / "conda.exe",
                programdata / "Anaconda3" / "Scripts" / "conda.exe",
                Path("C:/tools/miniconda3/Scripts/conda.exe"),
                Path("C:/tools/Anaconda3/Scripts/conda.exe"),
                home / "scoop" / "apps" / "miniconda3" / "current" / "Scripts" / "conda.exe",
            ]
        else:
            common_paths = [
                home / "miniconda3" / "bin" / "conda",
                home / "miniconda" / "bin" / "conda",
                home / "anaconda3" / "bin" / "conda",
                home / "anaconda" / "bin" / "conda",
                home / "mambaforge" / "bin" / "conda",
                home / ".conda" / "bin" / "conda",
                Path("/opt/conda/bin/conda"),
                Path("/opt/miniconda3/bin/conda"),
                Path("/opt/anaconda3/bin/conda"),
            ]

        for conda_path in common_paths:
            if conda_path.exists():
                success, output = run_command([str(conda_path), "--version"], check=False, capture=True)
                if success:
                    self.conda_exe = str(conda_path)
                    return True

        # Check if we're in an active conda environment (CONDA_PREFIX is set)
        # In this case, we can use pip directly even if conda binary isn't found
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix and Path(conda_prefix).exists():
            # We're in an active conda env, mark as detected but note pip-only mode
            self.conda_exe = "pip-only"
            return True

        return False

    def get_current_env(self) -> Optional[str]:
        """Get name of currently active conda environment."""
        env = os.environ.get('CONDA_DEFAULT_ENV')
        return env

    def list_environments(self) -> List[str]:
        """List all conda environments."""
        if not self.conda_exe:
            return []

        success, output = run_command(
            [self.conda_exe, "env", "list"],
            check=False,
            capture=True
        )
        if not success:
            return []

        # Parse output
        envs = []
        for line in output.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                # Format: "envname    /path/to/env"
                parts = line.split()
                if parts:
                    envs.append(parts[0])
        return envs

    def environment_exists(self, env_name_or_prefix: str | Path | None = None) -> bool:
        """Check if a conda environment exists.

        For prefix-based environments, checks for conda-meta/ directory.
        Falls back to listing envs for named environments.

        Args:
            env_name_or_prefix: Path to prefix dir, env name string, or
                                None to check self.env_prefix.
        """
        if env_name_or_prefix is None:
            env_name_or_prefix = self.env_prefix

        if isinstance(env_name_or_prefix, Path):
            return (env_name_or_prefix / "conda-meta").is_dir()

        prefix_path = CONDA_ENVS_DIR / env_name_or_prefix
        if (prefix_path / "conda-meta").is_dir():
            return True

        return env_name_or_prefix in self.list_environments()

    def accept_tos_if_needed(self) -> None:
        """Accept conda channel TOS if required (conda >= 26)."""
        if not self.conda_exe or self.conda_exe == "pip-only":
            return

        channels = [
            "https://repo.anaconda.com/pkgs/main",
            "https://repo.anaconda.com/pkgs/r",
            "https://repo.anaconda.com/pkgs/msys2",
        ]
        for channel in channels:
            run_command(
                [self.conda_exe, "tos", "accept", "--override-channels", "--channel", channel],
                check=False,
                capture=True,
            )

    def create_environment(self, python_version: Optional[str] = None) -> bool:
        """Create new conda environment using prefix mode.

        Args:
            python_version: Python version (e.g., "3.10"), uses default if None

        Returns:
            True if successful
        """
        if not self.conda_exe:
            print_error("Conda not available")
            return False

        self.accept_tos_if_needed()

        py_ver = python_version or self.python_version
        self.env_prefix.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nCreating conda environment '{self.env_name}' at {self.env_prefix}...")

        success, _ = run_command([
            self.conda_exe, "create",
            "-p", str(self.env_prefix),
            f"python={py_ver}",
            "-y"
        ])

        if success:
            print_success(f"Environment '{self.env_name}' created at {self.env_prefix}")
        else:
            print_error(f"Failed to create environment '{self.env_name}'")

        return success

    def get_activation_command(self) -> str:
        """Get command to activate the environment."""
        return f"conda activate {self.env_prefix}"

    def get_python_executable(self) -> Optional[Path]:
        """Get path to Python executable in the environment."""
        if not self.conda_exe:
            return None

        if _is_windows():
            where_cmd = ["where", "python"]
        else:
            where_cmd = ["which", "python"]

        success, output = run_command(
            [self.conda_exe, "run", "-p", str(self.env_prefix)] + where_cmd,
            check=False,
            capture=True
        )

        if success and output.strip():
            lines = output.strip().splitlines()
            if lines:
                return Path(lines[0])
        return None

    def install_package_conda(self, package: str, channel: Optional[str] = None) -> bool:
        """Install package via conda in the environment.

        Args:
            package: Package name (e.g., "pytorch")
            channel: Conda channel (e.g., "pytorch", "conda-forge")

        Returns:
            True if successful
        """
        if not self.conda_exe:
            return False

        if self.conda_exe == "pip-only":
            print_warning(f"  Conda binary not available, cannot install {package} via conda")
            return False

        cmd = [self.conda_exe, "install", "-p", str(self.env_prefix), package, "-y"]
        if channel:
            cmd.extend(["-c", channel])

        print(f"  Installing {package} via conda (this may take a few minutes)...")
        success, _ = run_command(cmd, timeout=600)
        return success

    def install_package_pip(self, package: str) -> bool:
        """Install package via pip in the environment.

        Args:
            package: Package name or pip install spec

        Returns:
            True if successful
        """
        if not self.conda_exe:
            return False

        print(f"  Installing {package} via pip...")

        if self.conda_exe == "pip-only":
            success, _ = run_command(["pip", "install", package])
        else:
            success, _ = run_command([
                self.conda_exe, "run", "-p", str(self.env_prefix),
                "pip", "install", package
            ])
        return success

    def check_setup(self) -> Tuple[bool, str]:
        """Check conda setup and recommend action.

        Returns:
            (is_ready, message)
        """
        if not self.detect_conda():
            return False, "Conda not installed. Please install Miniconda or Anaconda first."

        # If in pip-only mode (active conda env but no conda binary)
        if self.conda_exe == "pip-only":
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            return True, f"Using active conda environment (pip-only mode): {conda_prefix}"

        current_env = self.get_current_env()

        if self.environment_exists():
            if current_env == self.env_name:
                return True, f"Using existing environment '{self.env_name}'"
            else:
                return True, f"Environment '{self.env_name}' exists but not activated. Will use it."

        return True, f"Will create new environment '{self.env_name}' at {self.env_prefix}"
