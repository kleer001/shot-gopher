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
from env_config import CONDA_ENV_NAME, PYTHON_VERSION

from .utils import print_error, print_success, print_warning, run_command


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


class CondaEnvironmentManager:
    """Manages conda environment creation and activation."""

    def __init__(self, env_name: str = CONDA_ENV_NAME):
        self.env_name = env_name
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

    def environment_exists(self, env_name: str) -> bool:
        """Check if a conda environment exists."""
        return env_name in self.list_environments()

    def create_environment(self, python_version: Optional[str] = None) -> bool:
        """Create new conda environment.

        Args:
            python_version: Python version (e.g., "3.10"), uses default if None

        Returns:
            True if successful
        """
        if not self.conda_exe:
            print_error("Conda not available")
            return False

        py_ver = python_version or self.python_version
        print(f"\nCreating conda environment '{self.env_name}' with Python {py_ver}...")

        success, _ = run_command([
            self.conda_exe, "create",
            "-n", self.env_name,
            f"python={py_ver}",
            "-y"
        ])

        if success:
            print_success(f"Environment '{self.env_name}' created")
        else:
            print_error(f"Failed to create environment '{self.env_name}'")

        return success

    def get_activation_command(self) -> str:
        """Get command to activate the environment."""
        return f"conda activate {self.env_name}"

    def get_python_executable(self) -> Optional[Path]:
        """Get path to Python executable in the environment."""
        if not self.conda_exe:
            return None

        if _is_windows():
            where_cmd = ["where", "python"]
        else:
            where_cmd = ["which", "python"]

        success, output = run_command(
            [self.conda_exe, "run", "-n", self.env_name] + where_cmd,
            check=False,
            capture=True
        )

        if success and output.strip():
            return Path(output.strip().splitlines()[0])
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

        # Can't use conda install in pip-only mode
        if self.conda_exe == "pip-only":
            print_warning(f"  Conda binary not available, cannot install {package} via conda")
            return False

        cmd = [self.conda_exe, "install", "-n", self.env_name, package, "-y"]
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

        # If we're in pip-only mode (active conda env but no conda binary),
        # run pip directly
        if self.conda_exe == "pip-only":
            success, _ = run_command(["pip", "install", package])
        else:
            success, _ = run_command([
                self.conda_exe, "run", "-n", self.env_name,
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

        # Check if vfx-pipeline exists
        if self.environment_exists(self.env_name):
            if current_env == self.env_name:
                return True, f"Using existing environment '{self.env_name}'"
            else:
                return True, f"Environment '{self.env_name}' exists but not activated. Will use it."

        # Need to create environment
        return True, f"Will create new environment '{self.env_name}'"
