"""VFX Pipeline Installation Wizard.

This package provides an interactive installation wizard for setting up
all dependencies for the VFX pipeline including:
- Core pipeline (ComfyUI workflows, COLMAP, etc.)
- Dynamic scene segmentation (SAM3)
- Human motion capture (WHAM, ECON)

Usage:
    python -m scripts.install_wizard
    python scripts/install_wizard.py
    python scripts/install_wizard.py --component mocap
    python scripts/install_wizard.py --check-only
"""

from .cli import main
from .conda import CondaEnvironmentManager
from .config import ConfigurationGenerator
from .downloader import CheckpointDownloader
from .installers import ComponentInstaller, GitRepoInstaller, PythonPackageInstaller
from .progress import ProgressBarManager
from .state import InstallationStateManager
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
]
