"""VFX Pipeline Installation Wizard.

This package provides an interactive installation wizard for setting up
all dependencies for the VFX pipeline including:
- Core pipeline (ComfyUI workflows, COLMAP, etc.)
- Dynamic scene segmentation (SAM3)
- Human motion capture (GVHMR)

Usage:
    python -m scripts.install_wizard
    python scripts/install_wizard.py
    python scripts/install_wizard.py --component mocap_core
    python scripts/install_wizard.py --check-only
"""

from .cli import main
from .conda import CondaEnvironmentManager
from .config import ConfigurationGenerator
from .downloader import CheckpointDownloader
from .installers import ComponentInstaller, GitRepoInstaller, PythonPackageInstaller
from .progress import ProgressBarManager
from .state import InstallationStateManager
from .utils import print_success, print_warning, print_error, print_info, run_command
from .validator import InstallationValidator
from .wizard import InstallationWizard

__all__ = [
    'main',
    'CondaEnvironmentManager',
    'ConfigurationGenerator',
    'CheckpointDownloader',
    'ComponentInstaller',
    'GitRepoInstaller',
    'PythonPackageInstaller',
    'ProgressBarManager',
    'InstallationStateManager',
    'InstallationValidator',
    'InstallationWizard',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'run_command',
]
