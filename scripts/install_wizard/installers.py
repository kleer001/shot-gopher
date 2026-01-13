"""Component installers for the installation wizard.

This module provides installer classes for different types of components
including Python packages and Git repositories.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from env_config import INSTALL_DIR

from .utils import check_python_package, print_error, print_success, print_warning, run_command

if TYPE_CHECKING:
    from .conda import CondaEnvironmentManager


class ComponentInstaller:
    """Base class for component installers."""

    def __init__(self, name: str, size_gb: float = 0.0):
        self.name = name
        self.installed = False
        self.size_gb = size_gb  # Estimated disk space in GB

    def check(self) -> bool:
        """Check if component is installed."""
        raise NotImplementedError

    def install(self) -> bool:
        """Install component."""
        raise NotImplementedError

    def validate(self) -> bool:
        """Validate installation."""
        return self.check()


class PythonPackageInstaller(ComponentInstaller):
    """Installer for Python packages via pip."""

    def __init__(self, name: str, package: str, import_name: Optional[str] = None, size_gb: float = 0.0):
        super().__init__(name, size_gb)
        self.package = package
        self.import_name = import_name or package
        self.conda_manager: Optional['CondaEnvironmentManager'] = None

    def set_conda_manager(self, conda_manager: 'CondaEnvironmentManager'):
        """Set the conda manager for environment-aware installation."""
        self.conda_manager = conda_manager

    def check(self) -> bool:
        """Check if package is installed in the conda environment."""
        if self.conda_manager and self.conda_manager.conda_exe:
            # Check within the conda environment
            success, output = run_command([
                self.conda_manager.conda_exe, "run", "-n", self.conda_manager.env_name,
                "python", "-c", f"import {self.import_name}"
            ], check=False, capture=True)
            self.installed = success
        else:
            self.installed = check_python_package(self.package, self.import_name)
        return self.installed

    def install(self) -> bool:
        print(f"\nInstalling {self.name}...")

        # Use conda manager if available to install into the environment
        if self.conda_manager and self.conda_manager.conda_exe:
            success = self.conda_manager.install_package_pip(self.package)
        else:
            # Fallback to system pip (may fail on externally-managed environments)
            print_warning("No conda environment configured, using system pip")
            success, _ = run_command([sys.executable, "-m", "pip", "install", self.package])

        if success:
            print_success(f"{self.name} installed")
            self.installed = True
        else:
            print_error(f"Failed to install {self.name}")
        return success


class GitRepoInstaller(ComponentInstaller):
    """Installer for Git repositories."""

    def __init__(self, name: str, repo_url: str, install_dir: Optional[Path] = None, size_gb: float = 0.0, extra_packages: list = None):
        super().__init__(name, size_gb)
        self.repo_url = repo_url
        self.install_dir = install_dir or INSTALL_DIR / name.lower()
        self.conda_manager: Optional['CondaEnvironmentManager'] = None
        self.extra_packages = extra_packages or []

    def set_conda_manager(self, conda_manager: 'CondaEnvironmentManager'):
        """Set the conda manager for environment-aware installation."""
        self.conda_manager = conda_manager

    def check(self) -> bool:
        self.installed = self.install_dir.exists() and (self.install_dir / ".git").exists()
        return self.installed

    def install(self) -> bool:
        print(f"\nInstalling {self.name} from {self.repo_url}...")

        # Create parent directory
        self.install_dir.parent.mkdir(parents=True, exist_ok=True)

        # Clone repository
        success, _ = run_command(["git", "clone", self.repo_url, str(self.install_dir)])
        if not success:
            print_error(f"Failed to clone {self.name}")
            return False

        # Install dependencies from requirements.txt if exists
        requirements_txt = self.install_dir / "requirements.txt"
        if requirements_txt.exists():
            print(f"  Installing {self.name} dependencies...")
            if self.conda_manager and self.conda_manager.conda_exe:
                success, _ = run_command([
                    self.conda_manager.conda_exe, "run", "-n", self.conda_manager.env_name,
                    "pip", "install", "-r", str(requirements_txt)
                ])
            else:
                print_warning("No conda environment configured, using system pip")
                success, _ = run_command(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_txt)]
                )
            if not success:
                print_warning(f"requirements.txt install failed for {self.name}")

        # Run pip install if setup.py exists
        setup_py = self.install_dir / "setup.py"
        if setup_py.exists():
            print(f"  Installing {self.name} package...")
            # Use conda manager if available to install into the environment
            if self.conda_manager and self.conda_manager.conda_exe:
                success, _ = run_command([
                    self.conda_manager.conda_exe, "run", "-n", self.conda_manager.env_name,
                    "pip", "install", "-e", str(self.install_dir)
                ])
            else:
                print_warning("No conda environment configured, using system pip")
                success, _ = run_command(
                    [sys.executable, "-m", "pip", "install", "-e", str(self.install_dir)]
                )
            if not success:
                print_warning(f"pip install failed for {self.name}")

        # Install extra packages (missing from requirements.txt)
        if self.extra_packages:
            print(f"  Installing extra packages for {self.name}...")
            for pkg in self.extra_packages:
                if self.conda_manager and self.conda_manager.conda_exe:
                    success, _ = run_command([
                        self.conda_manager.conda_exe, "run", "-n", self.conda_manager.env_name,
                        "pip", "install", pkg
                    ])
                else:
                    success, _ = run_command([sys.executable, "-m", "pip", "install", pkg])
                if not success:
                    print_warning(f"Failed to install {pkg}")

        self.installed = True
        print_success(f"{self.name} cloned to {self.install_dir}")
        return True
