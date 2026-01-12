#!/usr/bin/env python3
"""Interactive installation wizard for the VFX pipeline.

Guides users through installing all dependencies for:
- Core pipeline (ComfyUI workflows, COLMAP, etc.)
- Dynamic scene segmentation (SAM3)
- Human motion capture (WHAM, ECON)

Usage:
    python scripts/install_wizard.py
    python scripts/install_wizard.py --component mocap
    python scripts/install_wizard.py --check-only
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Global TTY file handle for reading input when piped
_tty_handle = None


def tty_input(prompt: str = "") -> str:
    """Read input from TTY, even when stdin is piped.

    This allows the script to work when run via: curl ... | bash
    """
    global _tty_handle

    if sys.stdin.isatty():
        # Normal interactive mode
        return input(prompt)

    # stdin is a pipe, read from /dev/tty instead
    if _tty_handle is None:
        try:
            _tty_handle = open('/dev/tty', 'r')
        except OSError:
            # No TTY available (non-interactive), raise EOFError
            raise EOFError("No TTY available for input")

    if prompt:
        print(prompt, end='', flush=True)
    return _tty_handle.readline().rstrip('\n')


class Colors:
    """Terminal colors for pretty output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Ask user yes/no question."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = tty_input(f"{question} [{default_str}]: ").strip().lower()
        if not response:
            return default
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please answer yes or no.")


def run_command(cmd: List[str], check: bool = True, capture: bool = False) -> Tuple[bool, str]:
    """Run shell command and return success status and output."""
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout + result.stderr
        else:
            result = subprocess.run(cmd, check=check, timeout=30)
            return result.returncode == 0, ""
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def check_python_package(package: str, import_name: Optional[str] = None) -> bool:
    """Check if Python package is installed."""
    import_name = import_name or package
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_command_available(command: str) -> bool:
    """Check if command-line tool is available."""
    success, _ = run_command(["which", command], check=False, capture=True)
    return success


def check_gpu_available() -> Tuple[bool, str]:
    """Check if NVIDIA GPU is available."""
    success, output = run_command(["nvidia-smi"], check=False, capture=True)
    if not success:
        return False, "No NVIDIA GPU detected (nvidia-smi failed)"

    # Parse VRAM
    try:
        import re
        vram_match = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', output)
        if vram_match:
            total_vram = int(vram_match.group(2))
            return True, f"{total_vram}MB total VRAM"
    except (ValueError, AttributeError, ImportError):
        pass

    return True, "GPU detected"


def get_disk_space(path: Path = Path.cwd()) -> Tuple[float, float]:
    """Get available and total disk space in GB.

    Args:
        path: Path to check (default: home directory)

    Returns:
        Tuple of (available_gb, total_gb)
    """
    import shutil
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        return available_gb, total_gb
    except (OSError, AttributeError):
        return 0.0, 0.0


def format_size_gb(size_gb: float) -> str:
    """Format size in GB to human-readable string."""
    if size_gb < 1:
        return f"{size_gb * 1024:.0f} MB"
    elif size_gb < 10:
        return f"{size_gb:.1f} GB"
    else:
        return f"{size_gb:.0f} GB"


class CondaEnvironmentManager:
    """Manages conda environment creation and activation."""

    def __init__(self, env_name: str = "vfx-pipeline"):
        self.env_name = env_name
        self.conda_exe = None
        self.python_version = "3.10"

    def detect_conda(self) -> bool:
        """Check if conda is installed and available."""
        # Try conda command
        success, output = run_command(["conda", "--version"], check=False, capture=True)
        if success:
            self.conda_exe = "conda"
            return True

        # Try mamba
        success, output = run_command(["mamba", "--version"], check=False, capture=True)
        if success:
            self.conda_exe = "mamba"
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
        for line in output.split('\n'):
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

        success, output = run_command(
            [self.conda_exe, "run", "-n", self.env_name, "which", "python"],
            check=False,
            capture=True
        )

        if success and output.strip():
            return Path(output.strip())
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

        cmd = [self.conda_exe, "install", "-n", self.env_name, package, "-y"]
        if channel:
            cmd.extend(["-c", channel])

        print(f"  Installing {package} via conda...")
        success, _ = run_command(cmd)
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

        current_env = self.get_current_env()

        # Check if vfx-pipeline exists
        if self.environment_exists(self.env_name):
            if current_env == self.env_name:
                return True, f"Using existing environment '{self.env_name}'"
            else:
                return True, f"Environment '{self.env_name}' exists but not activated. Will use it."

        # Need to create environment
        return True, f"Will create new environment '{self.env_name}'"


class InstallationStateManager:
    """Manages installation state for resume/recovery."""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or Path.cwd() / ".vfx_pipeline" / "install_state.json"
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load installation state from file."""
        if not self.state_file.exists():
            return self._create_initial_state()

        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print_warning(f"Could not load state from {self.state_file}, creating new state")
            return self._create_initial_state()

    def _create_initial_state(self) -> Dict:
        """Create initial state structure."""
        return {
            "version": "1.0",
            "environment": None,
            "last_updated": None,
            "components": {},
            "checkpoints": {}
        }

    def save_state(self):
        """Save installation state to file."""
        self.state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write (write to temp, then rename)
        temp_file = self.state_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.state, indent=2, fp=f)
            temp_file.replace(self.state_file)
        except IOError as e:
            print_warning(f"Could not save state: {e}")

    def set_environment(self, env_name: str):
        """Set the conda environment name."""
        self.state["environment"] = env_name
        self.save_state()

    def mark_component_started(self, comp_id: str):
        """Mark component installation as started."""
        self.state["components"][comp_id] = {
            "status": "in_progress",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": None
        }
        self.save_state()

    def mark_component_completed(self, comp_id: str):
        """Mark component installation as completed."""
        if comp_id in self.state["components"]:
            self.state["components"][comp_id]["status"] = "completed"
            self.state["components"][comp_id]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            self.state["components"][comp_id] = {
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": None
            }
        self.save_state()

    def mark_component_failed(self, comp_id: str, error: str):
        """Mark component installation as failed."""
        self.state["components"][comp_id] = {
            "status": "failed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": error
        }
        self.save_state()

    def get_component_status(self, comp_id: str) -> Optional[str]:
        """Get status of a component.

        Returns:
            "completed", "in_progress", "failed", or None
        """
        if comp_id not in self.state["components"]:
            return None
        return self.state["components"][comp_id].get("status")

    def get_incomplete_components(self) -> List[str]:
        """Get list of components that are not completed."""
        incomplete = []
        for comp_id, info in self.state["components"].items():
            if info.get("status") != "completed":
                incomplete.append(comp_id)
        return incomplete

    def can_resume(self) -> bool:
        """Check if there's a resumable installation."""
        return len(self.get_incomplete_components()) > 0

    def mark_checkpoint_downloaded(self, comp_id: str, path: Path):
        """Mark checkpoint as downloaded."""
        self.state["checkpoints"][comp_id] = {
            "downloaded": True,
            "path": str(path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_state()

    def is_checkpoint_downloaded(self, comp_id: str) -> bool:
        """Check if checkpoint is already downloaded."""
        return self.state["checkpoints"].get(comp_id, {}).get("downloaded", False)

    def clear_state(self):
        """Clear installation state (for fresh start)."""
        self.state = self._create_initial_state()
        self.save_state()


class ProgressBarManager:
    """Manages progress bars with ncurses support and fallback.

    Note: Full ncurses implementation deferred for maintainability.
    Currently uses simple inline progress bars.
    """

    def __init__(self):
        self.use_tqdm = False

        # Check if tqdm is available for better progress bars
        try:
            __import__('tqdm')
            self.use_tqdm = True
        except ImportError:
            pass

    def create_progress_bar(self, total: int, desc: str = ""):
        """Create a progress bar.

        Args:
            total: Total number of items
            desc: Description

        Returns:
            Progress bar object or None
        """
        if self.use_tqdm:
            import tqdm
            return tqdm.tqdm(total=total, desc=desc, unit='B', unit_scale=True)
        return None


class CheckpointDownloader:
    """Handles automatic checkpoint downloading."""

    # Checkpoint metadata
    CHECKPOINTS = {
        'wham': {
            'name': 'WHAM Checkpoints',
            'use_gdown': True,  # Use gdown for Google Drive downloads
            'files': [
                {
                    'url': 'https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB',
                    'filename': 'wham_vit_w_3dpw.pth.tar',
                    'size_mb': 1200,
                    'sha256': None  # TODO: Add checksums
                }
            ],
            'dest_dir_rel': 'WHAM/checkpoints',
            'instructions': '''WHAM checkpoints are hosted on Google Drive.
If automatic download fails, manually download from:
  https://drive.google.com/file/d/1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB/view
Or run the fetch_demo_data.sh script from the WHAM repository:
  cd .vfx_pipeline/WHAM && bash fetch_demo_data.sh'''
        },
        'econ': {
            'name': 'ECON Checkpoints',
            'requires_auth': True,
            'auth_type': 'basic',
            'auth_file': 'SMPL.login.dat',
            'files': [
                {
                    'url': 'https://download.is.tue.mpg.de/download.php?domain=icon&sfile=econ_data.zip&resume=1',
                    'filename': 'econ_data.zip',
                    'size_mb': 2500,
                    'sha256': None,
                    'extract': True
                }
            ],
            'dest_dir_rel': 'ECON/data',
            'instructions': '''ECON checkpoints require registration (same as SMPL-X):
1. Register at https://icon.is.tue.mpg.de/
2. Wait for approval email (usually within 24 hours)
3. Create SMPL.login.dat in repository root with:
   Line 1: your email
   Line 2: your password
4. Re-run the wizard to download models

Alternatively, run the fetch_data.sh script from the ECON repository.'''
        },
        'smplx': {
            'name': 'SMPL-X Models',
            'requires_auth': True,
            'auth_type': 'basic',
            'auth_file': 'SMPL.login.dat',
            'files': [
                {
                    'url': 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip',
                    'filename': 'models_smplx_v1_1.zip',
                    'size_mb': 830,
                    'sha256': None,
                    'extract': True
                }
            ],
            'dest_dir_rel': 'smplx_models',  # Relative to install_dir
            'use_home_dir': False,
            'instructions': '''SMPL-X models require registration:
1. Register at https://smpl-x.is.tue.mpg.de/
2. Wait for approval email (usually within 24 hours)
3. Create SMPL.login.dat in repository root with:
   Line 1: your email
   Line 2: your password
4. Re-run the wizard to download models'''
        },
        'sam3': {
            'name': 'SAM3 Model',
            'requires_auth': True,
            'auth_type': 'bearer',
            'auth_file': 'HF_TOKEN.dat',
            'files': [
                {
                    'url': 'https://huggingface.co/facebook/sam3/resolve/main/model.safetensors',
                    'filename': 'sam3_model.safetensors',
                    'size_mb': 2400,
                    'sha256': None
                }
            ],
            'dest_dir_rel': 'ComfyUI/models/sam',
            'instructions': '''SAM3 model requires HuggingFace access:
1. Visit https://huggingface.co/facebook/sam3
2. Click "Access repository" and accept the license
3. Get your HuggingFace token from https://huggingface.co/settings/tokens
4. Create HF_TOKEN.dat in repository root with your token
5. Re-run the wizard to download the model'''
        }
    }

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.cwd() / ".vfx_pipeline"

    def download_file(self, url: str, dest: Path, expected_size_mb: Optional[int] = None) -> bool:
        """Download file with progress tracking.

        Args:
            url: URL to download from
            dest: Destination file path
            expected_size_mb: Expected file size in MB (for validation)

        Returns:
            True if successful
        """
        try:
            import requests
        except ImportError:
            print_warning("requests library not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
            import requests

        try:
            print(f"  Downloading from {url}...")
            print(f"  -> {dest}")

            # Ensure directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Stream download with progress
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress indicator
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            progress_msg = f"\r  Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                            print(progress_msg, end='', flush=True)

            print()  # New line after progress
            print_success(f"Downloaded {dest.name}")
            return True

        except requests.exceptions.RequestException as e:
            print_error(f"Download failed: {e}")
            # Clean up partial download
            if dest.exists():
                dest.unlink()
            return False

    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file checksum.

        Args:
            file_path: Path to file
            expected_sha256: Expected SHA256 hash

        Returns:
            True if checksum matches
        """
        if not expected_sha256:
            return True  # Skip verification if no checksum provided

        print(f"  Verifying checksum for {file_path.name}...")
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256_hash.update(chunk)

            actual = sha256_hash.hexdigest()
            if actual == expected_sha256:
                print_success("Checksum verified")
                return True
            else:
                print_error(f"Checksum mismatch: expected {expected_sha256}, got {actual}")
                return False

        except IOError as e:
            print_error(f"Could not read file for checksum: {e}")
            return False

    def _install_gdown(self) -> Optional['module']:
        """Install gdown handling PEP 668 externally-managed environments.

        Tries multiple installation methods in order:
        1. pipx (preferred for CLI tools)
        2. pip with --user flag
        3. pip with --break-system-packages (last resort)

        Returns:
            The gdown module if successful, None otherwise
        """
        install_methods = [
            # Method 1: Try pipx (best for externally-managed environments)
            {
                'name': 'pipx',
                'cmd': ['pipx', 'install', 'gdown'],
                'check_cmd': ['pipx', 'list'],
            },
            # Method 2: Try pip with --user flag
            {
                'name': 'pip --user',
                'cmd': [sys.executable, '-m', 'pip', 'install', '--user', 'gdown'],
                'check_cmd': None,
            },
            # Method 3: Try pip with --break-system-packages (last resort)
            {
                'name': 'pip --break-system-packages',
                'cmd': [sys.executable, '-m', 'pip', 'install', '--break-system-packages', 'gdown'],
                'check_cmd': None,
            },
        ]

        for method in install_methods:
            # Check if the tool is available (for pipx)
            if method['check_cmd']:
                check_result = subprocess.run(
                    method['check_cmd'],
                    capture_output=True,
                    text=True
                )
                if check_result.returncode != 0:
                    continue

            print_info(f"Trying to install gdown via {method['name']}...")
            result = subprocess.run(
                method['cmd'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print_success(f"Installed gdown via {method['name']}")
                try:
                    import gdown
                    return gdown
                except ImportError:
                    # pipx installs to a different location, need to use subprocess
                    # Check if gdown CLI is available
                    gdown_check = subprocess.run(
                        ['gdown', '--version'],
                        capture_output=True,
                        text=True
                    )
                    if gdown_check.returncode == 0:
                        # gdown CLI is available, use wrapper class
                        return self._create_gdown_cli_wrapper()
                    continue
            else:
                # Check for PEP 668 error and try next method
                if 'externally-managed-environment' in result.stderr:
                    continue
                # Other errors, log and try next method
                print_warning(f"Failed with {method['name']}: {result.stderr[:100]}")

        # All methods failed - provide manual instructions
        print_error("Could not install gdown automatically.")
        print_info("To install gdown manually, try one of these options:")
        print("  1. Using pipx (recommended): pipx install gdown")
        print("  2. Using pip with user flag: pip install --user gdown")
        print("  3. In a virtual environment: python -m venv venv && source venv/bin/activate && pip install gdown")
        return None

    def _create_gdown_cli_wrapper(self):
        """Create a wrapper object that mimics gdown module using CLI."""
        class GdownCLIWrapper:
            @staticmethod
            def download(url, output, quiet=False):
                cmd = ['gdown', url, '-O', output]
                if quiet:
                    cmd.append('--quiet')
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return output
                return None
        return GdownCLIWrapper()

    def _download_gdrive_wget(self, file_id: str, dest: Path) -> bool:
        """Download from Google Drive using wget (fallback method).

        Uses the confirmation cookie trick for large files.

        Args:
            file_id: Google Drive file ID
            dest: Destination file path

        Returns:
            True if successful
        """
        # Check if wget is available
        wget_check = subprocess.run(['which', 'wget'], capture_output=True)
        if wget_check.returncode != 0:
            return False

        print_info("Trying wget fallback for Google Drive...")

        # Google Drive download URL with confirmation bypass
        # For large files, we need to handle the virus scan warning
        base_url = "https://drive.google.com/uc?export=download"
        confirm_url = f"{base_url}&id={file_id}&confirm=t"

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Use wget with appropriate flags for Google Drive
            cmd = [
                'wget',
                '--no-check-certificate',
                '-q', '--show-progress',
                '-O', str(dest),
                confirm_url
            ]

            result = subprocess.run(cmd, capture_output=False)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via wget")
                return True

            # If file is too small, might be HTML error page
            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"wget fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _download_gdrive_curl(self, file_id: str, dest: Path) -> bool:
        """Download from Google Drive using curl (fallback method).

        Args:
            file_id: Google Drive file ID
            dest: Destination file path

        Returns:
            True if successful
        """
        # Check if curl is available
        curl_check = subprocess.run(['which', 'curl'], capture_output=True)
        if curl_check.returncode != 0:
            return False

        print_info("Trying curl fallback for Google Drive...")

        # Google Drive download URL with confirmation bypass
        confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Use curl with appropriate flags
            cmd = [
                'curl',
                '-L',  # Follow redirects
                '-o', str(dest),
                '--progress-bar',
                confirm_url
            ]

            result = subprocess.run(cmd)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via curl")
                return True

            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"curl fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _download_with_wget_auth(self, url: str, dest: Path, username: str, password: str) -> bool:
        """Download file using wget with HTTP basic auth (fallback).

        Args:
            url: URL to download
            dest: Destination file path
            username: HTTP basic auth username
            password: HTTP basic auth password

        Returns:
            True if successful
        """
        wget_check = subprocess.run(['which', 'wget'], capture_output=True)
        if wget_check.returncode != 0:
            return False

        print_info("Trying wget fallback with authentication...")

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'wget',
                '--no-check-certificate',
                '-q', '--show-progress',
                f'--user={username}',
                f'--password={password}',
                '-O', str(dest),
                url
            ]

            result = subprocess.run(cmd)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via wget")
                return True

            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"wget auth fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _download_with_curl_auth(self, url: str, dest: Path, username: str, password: str) -> bool:
        """Download file using curl with HTTP basic auth (fallback).

        Args:
            url: URL to download
            dest: Destination file path
            username: HTTP basic auth username
            password: HTTP basic auth password

        Returns:
            True if successful
        """
        curl_check = subprocess.run(['which', 'curl'], capture_output=True)
        if curl_check.returncode != 0:
            return False

        print_info("Trying curl fallback with authentication...")

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'curl',
                '-L',
                '-u', f'{username}:{password}',
                '-o', str(dest),
                '--progress-bar',
                url
            ]

            result = subprocess.run(cmd)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via curl")
                return True

            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"curl auth fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _extract_gdrive_file_id(self, url: str) -> Optional[str]:
        """Extract Google Drive file ID from various URL formats.

        Args:
            url: Google Drive URL

        Returns:
            File ID or None
        """
        import re

        # Format: https://drive.google.com/uc?id=FILE_ID
        match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)

        # Format: https://drive.google.com/file/d/FILE_ID/view
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)

        return None

    def download_file_gdown(self, url: str, dest: Path, expected_size_mb: Optional[int] = None) -> bool:
        """Download file from Google Drive using gdown with wget/curl fallbacks.

        Args:
            url: Google Drive URL (format: https://drive.google.com/uc?id=FILE_ID)
            dest: Destination file path
            expected_size_mb: Expected file size in MB (for info only)

        Returns:
            True if successful
        """
        print(f"  Downloading from Google Drive...")
        print(f"  -> {dest}")
        if expected_size_mb:
            print(f"  Expected size: ~{expected_size_mb} MB")

        # Extract file ID for fallback methods
        file_id = self._extract_gdrive_file_id(url)

        # Method 1: Try gdown (preferred)
        gdown = None
        try:
            import gdown
        except ImportError:
            print_warning("gdown library not found, installing...")
            gdown = self._install_gdown()

        if gdown is not None:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                output = gdown.download(url, str(dest), quiet=False)

                if output is not None and dest.exists() and dest.stat().st_size > 1000:
                    print_success(f"Downloaded {dest.name}")
                    return True

                print_warning("gdown download failed, trying fallbacks...")
                if dest.exists():
                    dest.unlink()

            except Exception as e:
                print_warning(f"gdown failed: {e}")
                if dest.exists():
                    dest.unlink()

        # Method 2: Try wget fallback
        if file_id:
            if self._download_gdrive_wget(file_id, dest):
                return True

            # Method 3: Try curl fallback
            if self._download_gdrive_curl(file_id, dest):
                return True

        print_error("All download methods failed")
        return False

    def read_smpl_credentials(self, repo_root: Path) -> Optional[Tuple[str, str]]:
        """Read SMPL-X credentials from SMPL.login.dat file.

        Args:
            repo_root: Repository root directory

        Returns:
            Tuple of (username, password) or None if file not found
        """
        cred_file = repo_root / "SMPL.login.dat"

        if not cred_file.exists():
            return None

        try:
            with open(cred_file, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    username = lines[0].strip()
                    password = lines[1].strip()
                    return (username, password)
                else:
                    print_error("SMPL.login.dat must contain username on line 1 and password on line 2")
                    return None
        except IOError as e:
            print_error(f"Could not read SMPL.login.dat: {e}")
            return None

    def read_hf_token(self, repo_root: Path) -> Optional[str]:
        """Read HuggingFace token from HF_TOKEN.dat file.

        Args:
            repo_root: Repository root directory

        Returns:
            HuggingFace token or None if file not found
        """
        token_file = repo_root / "HF_TOKEN.dat"

        if not token_file.exists():
            return None

        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
                if token:
                    return token
                else:
                    print_error("HF_TOKEN.dat is empty")
                    return None
        except IOError as e:
            print_error(f"Could not read HF_TOKEN.dat: {e}")
            return None

    def download_file_with_auth(
        self,
        url: str,
        dest: Path,
        auth: Optional[Tuple[str, str]] = None,
        token: Optional[str] = None,
        expected_size_mb: Optional[int] = None
    ) -> bool:
        """Download file with optional HTTP authentication.

        Args:
            url: URL to download from
            dest: Destination file path
            auth: Optional (username, password) tuple for basic auth
            token: Optional bearer token for token-based auth
            expected_size_mb: Expected file size in MB

        Returns:
            True if successful
        """
        try:
            import requests
        except ImportError:
            print_warning("requests library not found, installing...")
            # Try pip with --user to handle PEP 668 externally-managed environments
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "requests"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                # Fall back to --break-system-packages if --user fails
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "requests"],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print_error(f"Failed to install requests: {result.stderr}")
                    return False
            import requests

        try:
            print(f"  Downloading from {url}...")
            print(f"  -> {dest}")

            # Ensure directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Create session with auth if provided
            session = requests.Session()
            if auth:
                session.auth = auth
            elif token:
                session.headers.update({'Authorization': f'Bearer {token}'})

            # Stream download with progress
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Check content-type to catch HTML error pages early
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                print_error("Server returned HTML instead of the expected file")
                print_info("This usually means:")
                print("  - Your credentials may be invalid or expired")
                print("  - You may not have been approved for access yet")
                print("  - The download URL may have changed")
                return False

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress indicator
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            progress_msg = f"\r  Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                            print(progress_msg, end='', flush=True)

            print()  # New line after progress
            print_success(f"Downloaded {dest.name}")
            return True

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                if token:
                    print_error("Authentication failed - check your HF_TOKEN.dat")
                else:
                    print_error("Authentication failed - check your credentials")
            elif e.response.status_code == 403:
                if token:
                    print_error("Access denied - have you accepted the model license on HuggingFace?")
                else:
                    print_error("Access denied - have you been approved for access?")
            else:
                print_error(f"Download failed: {e}")
            # Clean up partial download
            if dest.exists():
                dest.unlink()
            # Don't try fallbacks for auth errors
            return False
        except requests.exceptions.RequestException as e:
            print_warning(f"requests download failed: {e}")
            # Clean up partial download
            if dest.exists():
                dest.unlink()

            # Try wget/curl fallbacks for basic auth
            if auth:
                username, password = auth
                print_info("Trying alternative download methods...")

                if self._download_with_wget_auth(url, dest, username, password):
                    return True

                if self._download_with_curl_auth(url, dest, username, password):
                    return True

            print_error("All download methods failed")
            return False

    def _validate_zip_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate that a file is actually a zip archive.

        Args:
            file_path: Path to the file to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        import zipfile

        # Check file exists and has content
        if not file_path.exists():
            return False, "File does not exist"

        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty"

        # Check magic bytes (PK signature for zip files)
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)

            # Zip files start with PK\x03\x04 (regular) or PK\x05\x06 (empty archive)
            if not (magic[:2] == b'PK'):
                # Check if it's an HTML response (common error)
                with open(file_path, 'rb') as f:
                    start = f.read(500)

                if b'<!DOCTYPE' in start or b'<html' in start.lower() or b'<HTML' in start:
                    # Try to extract error message from HTML
                    try:
                        text = start.decode('utf-8', errors='ignore')
                        return False, f"Server returned HTML instead of zip file. This usually means:\n" \
                                      f"  - Authentication failed or session expired\n" \
                                      f"  - You haven't been approved for access yet\n" \
                                      f"  - The download link has changed"
                    except Exception:
                        pass
                    return False, "Server returned HTML instead of zip file"

                return False, f"File is not a zip archive (magic bytes: {magic.hex()})"

            # Verify it's a valid zip structure
            if not zipfile.is_zipfile(file_path):
                return False, "File has zip signature but is not a valid zip archive (possibly truncated)"

            return True, ""

        except IOError as e:
            return False, f"Could not read file: {e}"

    def extract_zip(self, zip_path: Path, dest_dir: Path) -> bool:
        """Extract zip file.

        Args:
            zip_path: Path to zip file
            dest_dir: Destination directory

        Returns:
            True if successful
        """
        import zipfile

        # Validate the zip file first
        is_valid, error_msg = self._validate_zip_file(zip_path)
        if not is_valid:
            print_error(f"Invalid zip file: {error_msg}")
            # Show file size for debugging
            if zip_path.exists():
                size_kb = zip_path.stat().st_size / 1024
                print_info(f"Downloaded file size: {size_kb:.1f} KB")
                if size_kb < 100:
                    # Small file - likely an error page, show first few lines
                    try:
                        with open(zip_path, 'r', errors='ignore') as f:
                            preview = f.read(500)
                        print_info(f"File preview:\n{preview[:300]}...")
                    except Exception:
                        pass
            return False

        try:
            print(f"  Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            print_success(f"Extracted to {dest_dir}")
            return True
        except zipfile.BadZipFile as e:
            print_error(f"Extraction failed - corrupted zip file: {e}")
            return False
        except Exception as e:
            print_error(f"Extraction failed: {e}")
            return False

    def download_checkpoint(
        self,
        comp_id: str,
        state_manager: Optional['InstallationStateManager'] = None,
        repo_root: Optional[Path] = None
    ) -> bool:
        """Download checkpoints for a component.

        Args:
            comp_id: Component ID (e.g., 'wham', 'econ', 'smplx')
            state_manager: Optional state manager for tracking
            repo_root: Repository root for finding credentials

        Returns:
            True if successful or already downloaded
        """
        if comp_id not in self.CHECKPOINTS:
            print_warning(f"No checkpoints defined for {comp_id}")
            return True  # Not an error, just skip

        # Check if already downloaded
        if state_manager and state_manager.is_checkpoint_downloaded(comp_id):
            print_success(f"{self.CHECKPOINTS[comp_id]['name']} already downloaded")
            return True

        checkpoint_info = self.CHECKPOINTS[comp_id]
        print(f"\n{Colors.BOLD}Downloading {checkpoint_info['name']}...{Colors.ENDC}")

        # Handle skip_download flag (for components without available checkpoints)
        if checkpoint_info.get('skip_download'):
            print_warning(f"Automatic download not available for {checkpoint_info['name']}")
            print_info(checkpoint_info['instructions'])
            return True  # Not a failure, just manual setup required

        # Handle authentication if required
        auth = None
        token = None
        use_gdown = checkpoint_info.get('use_gdown', False)
        if checkpoint_info.get('requires_auth'):
            if not repo_root:
                repo_root = Path.cwd()  # Fallback to current directory

            auth_type = checkpoint_info.get('auth_type', 'basic')
            auth_file = checkpoint_info.get('auth_file', '')

            if auth_type == 'basic':
                auth = self.read_smpl_credentials(repo_root)
                if not auth:
                    print_error(f"{checkpoint_info['name']} requires authentication")
                    print_info(checkpoint_info['instructions'])
                    return False
                print_success(f"Loaded credentials from {auth_file}")
            elif auth_type == 'bearer':
                token = self.read_hf_token(repo_root)
                if not token:
                    print_error(f"{checkpoint_info['name']} requires HuggingFace token")
                    print_info(checkpoint_info['instructions'])
                    return False
                print_success(f"Loaded token from {auth_file}")

        # Determine destination directory
        if checkpoint_info.get('use_home_dir'):
            dest_dir = Path.cwd() / checkpoint_info['dest_dir_rel']
        else:
            dest_dir = self.base_dir / checkpoint_info['dest_dir_rel']

        dest_dir.mkdir(parents=True, exist_ok=True)

        success = True
        for file_info in checkpoint_info['files']:
            dest_path = dest_dir / file_info['filename']

            # Skip if already exists (unless it needs extraction)
            if dest_path.exists() and not file_info.get('extract'):
                print_success(f"{file_info['filename']} already exists")
                continue

            # Download with appropriate method
            if use_gdown:
                # Use gdown for Google Drive downloads
                if not self.download_file_gdown(
                    file_info['url'],
                    dest_path,
                    expected_size_mb=file_info.get('size_mb')
                ):
                    success = False
                    print_info(checkpoint_info['instructions'])
                    break
            elif auth or token:
                # Use authenticated download
                if not self.download_file_with_auth(
                    file_info['url'],
                    dest_path,
                    auth=auth,
                    token=token,
                    expected_size_mb=file_info.get('size_mb')
                ):
                    success = False
                    break
            else:
                # Standard download
                if not self.download_file(file_info['url'], dest_path, file_info.get('size_mb')):
                    success = False
                    break

            # Verify checksum if provided
            if file_info.get('sha256'):
                if not self.verify_checksum(dest_path, file_info['sha256']):
                    success = False
                    dest_path.unlink()  # Remove corrupted file
                    break

            # Extract if needed
            if file_info.get('extract'):
                if not self.extract_zip(dest_path, dest_dir):
                    success = False
                    break
                # Optionally remove zip after extraction
                dest_path.unlink()
                print_info(f"Removed {dest_path.name} after extraction")

        if success and state_manager:
            state_manager.mark_checkpoint_downloaded(comp_id, dest_dir)

        if not success:
            print_error(f"Failed to download {checkpoint_info['name']}")
            print_info(checkpoint_info['instructions'])

        return success

    def download_all_checkpoints(
        self,
        component_ids: List[str],
        state_manager: Optional['InstallationStateManager'] = None,
        repo_root: Optional[Path] = None
    ) -> bool:
        """Download checkpoints for multiple components.

        Args:
            component_ids: List of component IDs
            state_manager: Optional state manager
            repo_root: Repository root for credentials

        Returns:
            True if all downloads successful
        """
        print_header("Downloading Checkpoints")

        # Auto-detect repo root if not provided
        if not repo_root:
            repo_root = Path(__file__).parent.parent.resolve()

        success = True
        for comp_id in component_ids:
            if comp_id in self.CHECKPOINTS:
                if not self.download_checkpoint(comp_id, state_manager, repo_root):
                    success = False

        return success


class InstallationValidator:
    """Validates installation with smoke tests."""

    def __init__(self, conda_manager: 'CondaEnvironmentManager', install_dir: Optional[Path] = None):
        self.conda_manager = conda_manager
        self.install_dir = install_dir

    def validate_python_imports(self) -> Dict[str, bool]:
        """Test importing key Python packages.

        Returns:
            Dict mapping package name to success status
        """
        packages_to_test = [
            ('numpy', 'numpy'),
            ('cv2', 'opencv-python'),
            ('PIL', 'Pillow'),
            ('torch', 'PyTorch'),
            ('smplx', 'SMPL-X'),
            ('trimesh', 'trimesh'),
        ]

        results = {}
        for import_name, display_name in packages_to_test:
            try:
                __import__(import_name)
                results[display_name] = True
            except ImportError:
                results[display_name] = False

        return results

    def validate_pytorch_cuda(self) -> Tuple[bool, str]:
        """Check if PyTorch can access CUDA.

        Returns:
            (success, message)
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                return True, f"CUDA available: {device_count} device(s), {device_name}"
            else:
                return False, "CUDA not available (CPU-only mode)"
        except ImportError:
            return False, "PyTorch not installed"
        except Exception as e:
            return False, f"Error checking CUDA: {e}"

    def validate_colmap(self) -> Tuple[bool, str]:
        """Check if COLMAP is accessible.

        Returns:
            (success, message)
        """
        success, output = run_command(["colmap", "--version"], check=False, capture=True)
        if success and output:
            version = output.strip().split('\n')[0] if output else "unknown"
            return True, f"COLMAP {version}"
        return False, "COLMAP not found"

    def validate_checkpoint_files(self, base_dir: Optional[Path] = None) -> Dict[str, bool]:
        """Check if checkpoint files exist.

        Args:
            base_dir: Base directory for checkpoints

        Returns:
            Dict mapping component to checkpoint status
        """
        base_dir = base_dir or self.install_dir
        if not base_dir:
            return {}

        results = {}

        # WHAM: Check for the main checkpoint file
        wham_ckpt = base_dir / "WHAM" / "checkpoints" / "wham_vit_w_3dpw.pth.tar"
        results['wham'] = wham_ckpt.exists()

        # ECON: Check for extracted data from econ_data.zip
        econ_data_dir = base_dir / "ECON" / "data"
        if econ_data_dir.exists():
            # Check for any model/data files in the extracted directory
            has_data = any(econ_data_dir.glob("**/*.pkl")) or \
                       any(econ_data_dir.glob("**/*.pth")) or \
                       any(econ_data_dir.glob("**/smpl_related"))
            results['econ'] = has_data
        else:
            results['econ'] = False

        return results

    def validate_smplx_models(self) -> Tuple[bool, str]:
        """Check if SMPL-X models are installed.

        Returns:
            (success, message)
        """
        smplx_dir = self.base_dir / "smplx_models"
        if not smplx_dir.exists():
            return False, "SMPL-X directory not found (.vfx_pipeline/smplx_models/)"

        # Look for model files
        model_files = list(smplx_dir.glob("SMPLX_*.pkl"))
        if model_files:
            return True, f"Found {len(model_files)} SMPL-X model file(s)"
        return False, "No SMPL-X model files found"

    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all validation tests.

        Returns:
            Dict with test categories and results
        """
        results = {
            'python_packages': self.validate_python_imports(),
            'pytorch_cuda': self.validate_pytorch_cuda(),
            'colmap': self.validate_colmap(),
            'checkpoints': self.validate_checkpoint_files(),
            'smplx_models': self.validate_smplx_models(),
        }
        return results

    def print_validation_report(self, results: Dict[str, Dict]):
        """Print formatted validation report.

        Args:
            results: Results from run_all_tests()
        """
        print_header("Installation Validation")

        # Python packages
        print("\n📦 Python Packages:")
        for pkg, status in results['python_packages'].items():
            if status:
                print_success(f"{pkg}")
            else:
                print_error(f"{pkg} - not found")

        # PyTorch CUDA
        print("\n🎮 GPU Support:")
        cuda_success, cuda_msg = results['pytorch_cuda']
        if cuda_success:
            print_success(cuda_msg)
        else:
            print_warning(cuda_msg)

        # COLMAP
        print("\n📐 COLMAP:")
        colmap_success, colmap_msg = results['colmap']
        if colmap_success:
            print_success(colmap_msg)
        else:
            print_warning(colmap_msg)

        # Checkpoints
        print("\n🎯 Motion Capture Checkpoints:")
        for comp, status in results['checkpoints'].items():
            if status:
                print_success(f"{comp.upper()} checkpoint found")
            else:
                print_info(f"{comp.upper()} checkpoint not found (install with wizard)")

        # SMPL-X models
        print("\n🧍 SMPL-X Models:")
        smplx_success, smplx_msg = results['smplx_models']
        if smplx_success:
            print_success(smplx_msg)
        else:
            print_warning(smplx_msg)
            print_info("Register at https://smpl-x.is.tue.mpg.de/ to download")

    def validate_and_report(self):
        """Run validation tests and print report."""
        results = self.run_all_tests()
        self.print_validation_report(results)


class ConfigurationGenerator:
    """Generates configuration files and activation scripts."""

    def __init__(self, conda_manager: 'CondaEnvironmentManager', base_dir: Optional[Path] = None):
        self.conda_manager = conda_manager
        self.base_dir = base_dir or Path.cwd() / ".vfx_pipeline"

    def generate_config_dict(self) -> Dict:
        """Generate configuration dictionary.

        Returns:
            Configuration dict
        """
        python_exe = self.conda_manager.get_python_executable()

        config = {
            "version": "1.0",
            "environment": self.conda_manager.env_name,
            "paths": {
                "base": str(self.base_dir),
                "wham": str(self.base_dir / "WHAM"),
                "econ": str(self.base_dir / "ECON"),
                "smplx_models": str(self.base_dir / "smplx_models"),
            },
            "python": str(python_exe) if python_exe else None,
            "conda_activate": self.conda_manager.get_activation_command(),
        }

        return config

    def write_config_file(self) -> Path:
        """Write configuration to JSON file.

        Returns:
            Path to config file
        """
        config = self.generate_config_dict()
        config_file = self.base_dir / "config.json"

        self.base_dir.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(config, indent=2, fp=f)

        print_success(f"Configuration saved to {config_file}")
        return config_file

    def generate_activation_script(self) -> str:
        """Generate shell activation script.

        Returns:
            Activation script content
        """
        script = f"""#!/bin/bash
# VFX Pipeline Environment Activation Script
# Generated by installation wizard

# Activate conda environment
{self.conda_manager.get_activation_command()}

# Set up Python path
export PYTHONPATH="${{PYTHONPATH}}:{self.base_dir / "WHAM"}:{self.base_dir / "ECON"}"

# Set up environment variables
export VFX_PIPELINE_BASE="{self.base_dir}"
export WHAM_DIR="{self.base_dir / "WHAM"}"
export ECON_DIR="{self.base_dir / "ECON"}"
export SMPLX_MODEL_DIR="{self.base_dir / "smplx_models"}"

echo "✓ VFX Pipeline environment activated"
echo "  Environment: {self.conda_manager.env_name}"
echo "  Base directory: {self.base_dir}"
echo ""
echo "Available commands:"
echo "  python scripts/run_pipeline.py --help"
echo "  python scripts/run_mocap.py --help"
"""
        return script

    def write_activation_script(self) -> Path:
        """Write activation script to file.

        Returns:
            Path to activation script
        """
        script = self.generate_activation_script()
        script_file = self.base_dir / "activate.sh"

        with open(script_file, 'w') as f:
            f.write(script)

        # Make executable
        script_file.chmod(0o755)

        print_success(f"Activation script saved to {script_file}")
        print_info(f"Source with: source {script_file}")
        return script_file

    def generate_all(self):
        """Generate all configuration files."""
        print_header("Generating Configuration")

        self.write_config_file()
        self.write_activation_script()

        print()
        print_info("Configuration complete!")
        print_info("To activate the environment:")
        print(f"  {Colors.OKBLUE}{self.conda_manager.get_activation_command()}{Colors.ENDC}")
        print("  or")
        print(f"  {Colors.OKBLUE}source {self.base_dir / 'activate.sh'}{Colors.ENDC}")


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

    def __init__(self, name: str, repo_url: str, install_dir: Optional[Path] = None, size_gb: float = 0.0):
        super().__init__(name, size_gb)
        self.repo_url = repo_url
        self.install_dir = install_dir or Path.cwd() / ".vfx_pipeline" / name.lower()
        self.conda_manager: Optional['CondaEnvironmentManager'] = None

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

        self.installed = True
        print_success(f"{self.name} cloned to {self.install_dir}")
        return True


class InstallationWizard:
    """Main installation wizard."""

    def __init__(self):
        self.components = {}
        # Get repository root (parent of scripts directory)
        self.repo_root = Path(__file__).parent.parent.resolve()
        self.install_dir = self.repo_root / ".vfx_pipeline"

        self.conda_manager = CondaEnvironmentManager()
        self.state_manager = InstallationStateManager(self.install_dir / "install_state.json")
        self.checkpoint_downloader = CheckpointDownloader(self.install_dir)
        self.validator = InstallationValidator(self.conda_manager, self.install_dir)
        self.config_generator = ConfigurationGenerator(self.conda_manager, self.install_dir)
        self.setup_components()

    def setup_components(self):
        """Define all installable components."""

        # Core dependencies
        self.components['core'] = {
            'name': 'Core Pipeline',
            'required': True,
            'installers': [
                PythonPackageInstaller('NumPy', 'numpy', size_gb=0.1),
                PythonPackageInstaller('OpenCV', 'opencv-python', 'cv2', size_gb=0.3),
                PythonPackageInstaller('Pillow', 'pillow', 'PIL', size_gb=0.05),
            ]
        }

        # PyTorch (special handling for CUDA)
        self.components['pytorch'] = {
            'name': 'PyTorch',
            'required': True,
            'installers': [
                PythonPackageInstaller('PyTorch', 'torch', size_gb=6.0),  # With CUDA
            ]
        }

        # COLMAP
        self.components['colmap'] = {
            'name': 'COLMAP',
            'required': False,
            'installers': [],  # System install, check only
            'size_gb': 0.5,  # If installed via conda
        }

        # Motion capture dependencies
        self.components['mocap_core'] = {
            'name': 'Motion Capture Core',
            'required': False,
            'installers': [
                PythonPackageInstaller('SMPL-X', 'smplx', size_gb=0.1),
                PythonPackageInstaller('Trimesh', 'trimesh', size_gb=0.05),
            ]
        }

        # WHAM (code ~0.1GB + checkpoints ~2.5GB)
        self.components['wham'] = {
            'name': 'WHAM',
            'required': False,
            'installers': [
                GitRepoInstaller(
                    'WHAM',
                    'https://github.com/yohanshin/WHAM.git',
                    self.install_dir / "WHAM",
                    size_gb=3.0  # Code + checkpoints
                )
            ]
        }

        # ECON (code ~0.2GB + dependencies ~1GB + checkpoints ~4GB + SMPL-X models ~0.5GB)
        self.components['econ'] = {
            'name': 'ECON',
            'required': False,
            'installers': [
                GitRepoInstaller(
                    'ECON',
                    'https://github.com/YuliangXiu/ECON.git',
                    self.install_dir / "ECON",
                    size_gb=6.0  # Code + dependencies + checkpoints + models
                )
            ]
        }

        # ComfyUI and custom nodes
        comfyui_dir = self.install_dir / "ComfyUI"
        self.components['comfyui'] = {
            'name': 'ComfyUI',
            'required': False,
            'installers': [
                GitRepoInstaller(
                    'ComfyUI',
                    'https://github.com/comfyanonymous/ComfyUI.git',
                    comfyui_dir,
                    size_gb=2.0
                ),
                GitRepoInstaller(
                    'ComfyUI-VideoHelperSuite',
                    'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git',
                    comfyui_dir / "custom_nodes" / "ComfyUI-VideoHelperSuite",
                    size_gb=0.1
                ),
                GitRepoInstaller(
                    'ComfyUI-DepthAnythingV3',
                    'https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3.git',
                    comfyui_dir / "custom_nodes" / "ComfyUI-DepthAnythingV3",
                    size_gb=0.5
                ),
                GitRepoInstaller(
                    'ComfyUI-SAM2',
                    'https://github.com/neverbiasu/ComfyUI-SAM2.git',
                    comfyui_dir / "custom_nodes" / "ComfyUI-SAM2",
                    size_gb=1.0
                )
            ]
        }

    def setup_conda_environment(self) -> bool:
        """Set up conda environment."""
        print_header("Conda Environment Setup")

        # Check conda
        is_ready, message = self.conda_manager.check_setup()
        if not is_ready:
            print_error(message)
            print_info("Install from: https://docs.conda.io/en/latest/miniconda.html")
            return False

        print_info(message)

        # Create environment if needed
        if not self.conda_manager.environment_exists(self.conda_manager.env_name):
            print_info(f"Creating dedicated environment '{self.conda_manager.env_name}'...")
            if not self.conda_manager.create_environment():
                return False
            self.state_manager.set_environment(self.conda_manager.env_name)
        else:
            print_success(f"Environment '{self.conda_manager.env_name}' already exists")

        # Show activation command
        current_env = self.conda_manager.get_current_env()
        if current_env != self.conda_manager.env_name:
            print("\n" + "="*60)
            print_info("After installation, activate the environment:")
            print(f"  {Colors.OKBLUE}{self.conda_manager.get_activation_command()}{Colors.ENDC}")
            print("="*60 + "\n")

        return True

    def check_system_requirements(self):
        """Check system requirements."""
        print_header("System Requirements Check")

        # Python version
        py_version = sys.version_info
        if py_version >= (3, 8):
            print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        else:
            print_error(f"Python {py_version.major}.{py_version.minor} (3.8+ required)")
            return False

        # Conda
        if self.conda_manager.detect_conda():
            print_success(f"Conda available ({self.conda_manager.conda_exe})")
        else:
            print_error("Conda not found (required for environment management)")
            print_info("Install from: https://docs.conda.io/en/latest/miniconda.html")
            return False

        # Git
        if check_command_available("git"):
            print_success("Git available")
        else:
            print_error("Git not found (required for cloning repositories)")
            return False

        # Disk space
        available_gb, total_gb = get_disk_space()
        if available_gb > 0:
            used_pct = ((total_gb - available_gb) / total_gb) * 100
            if available_gb >= 50:
                print_success(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
            elif available_gb >= 20:
                print_warning(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
                print_info("Full installation requires ~40 GB")
            else:
                print_error(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
                print_info("Insufficient space - full installation requires ~40 GB")
                return False
        else:
            print_warning("Could not check disk space")

        # GPU
        has_gpu, gpu_info = check_gpu_available()
        if has_gpu:
            print_success(f"GPU: {gpu_info}")
        else:
            print_warning("No GPU detected - CPU-only mode (slower)")
            print_info("Motion capture requires NVIDIA GPU with 12GB+ VRAM")

        # ffmpeg
        if check_command_available("ffmpeg"):
            print_success("ffmpeg available")
        else:
            print_warning("ffmpeg not found (required for video ingestion)")
            print_info("Install: sudo apt install ffmpeg")

        # COLMAP
        if check_command_available("colmap"):
            print_success("COLMAP available")
        else:
            print_warning("COLMAP not found (optional, for 3D reconstruction)")
            print_info("Install: sudo apt install colmap")

        return True

    def check_all_components(self) -> Dict[str, bool]:
        """Check status of all components."""
        status = {}

        for comp_id, comp_info in self.components.items():
            all_installed = True
            for installer in comp_info['installers']:
                # Set conda manager for environment-aware checking
                if hasattr(installer, 'set_conda_manager'):
                    installer.set_conda_manager(self.conda_manager)
                if not installer.check():
                    all_installed = False
            status[comp_id] = all_installed

        return status

    def print_status(self, status: Dict[str, bool]):
        """Print installation status."""
        print_header("Installation Status")

        for comp_id, comp_info in self.components.items():
            is_installed = status.get(comp_id, False)
            required = " (required)" if comp_info['required'] else ""

            if is_installed:
                print_success(f"{comp_info['name']}{required}")
            else:
                if comp_info['required']:
                    print_error(f"{comp_info['name']}{required}")
                else:
                    print_warning(f"{comp_info['name']} - not installed")

    def calculate_space_needed(self, component_ids: List[str]) -> float:
        """Calculate total disk space needed for components.

        Args:
            component_ids: List of component IDs to install

        Returns:
            Total disk space needed in GB
        """
        total_gb = 0.0
        for comp_id in component_ids:
            comp_info = self.components.get(comp_id, {})
            for installer in comp_info.get('installers', []):
                total_gb += installer.size_gb
            # Add component-level size (for things like COLMAP)
            total_gb += comp_info.get('size_gb', 0.0)
        return total_gb

    def show_space_estimate(self, component_ids: List[str]):
        """Show disk space estimate for installation.

        Args:
            component_ids: List of component IDs to install
        """
        print_header("Disk Space Estimate")

        # Calculate per-component
        breakdown = []
        total_gb = 0.0

        for comp_id in component_ids:
            comp_info = self.components.get(comp_id, {})
            comp_size = sum(inst.size_gb for inst in comp_info.get('installers', []))
            comp_size += comp_info.get('size_gb', 0.0)

            if comp_size > 0:
                breakdown.append((comp_info['name'], comp_size))
                total_gb += comp_size

        # Sort by size (largest first)
        breakdown.sort(key=lambda x: x[1], reverse=True)

        # Print breakdown
        for name, size_gb in breakdown:
            print(f"  {name:30s} {format_size_gb(size_gb):>10s}")

        print("  " + "-" * 42)
        print(f"  {'Total':30s} {format_size_gb(total_gb):>10s}")

        # Additional space for working data
        working_space = 10.0  # ~10 GB per project
        print(f"\n  {'Working space (per project)':30s} ~{format_size_gb(working_space):>9s}")
        print(f"  {'Recommended total':30s} ~{format_size_gb(total_gb + working_space):>9s}")

        # Check available space
        available_gb, _ = get_disk_space()
        if available_gb > 0:
            print(f"\n  Available disk space: {format_size_gb(available_gb)}")

            if available_gb >= total_gb + working_space:
                print_success("Sufficient disk space available")
            elif available_gb >= total_gb:
                print_warning("Sufficient for installation, but limited working space")
            else:
                print_error(f"Insufficient disk space (need {format_size_gb(total_gb)})")
                return False

        print()
        return True

    def install_component(self, comp_id: str) -> bool:
        """Install a component."""
        comp_info = self.components[comp_id]

        print(f"\n{Colors.BOLD}Installing {comp_info['name']}...{Colors.ENDC}")

        # Check if already completed
        status = self.state_manager.get_component_status(comp_id)
        if status == "completed":
            print_success(f"{comp_info['name']} already installed (from previous run)")
            return True

        # Mark as started
        self.state_manager.mark_component_started(comp_id)

        success = True
        try:
            for installer in comp_info['installers']:
                # Set conda manager for environment-aware installation
                if hasattr(installer, 'set_conda_manager'):
                    installer.set_conda_manager(self.conda_manager)

                if not installer.check():
                    if not installer.install():
                        success = False
                        break
                else:
                    print_success(f"{installer.name} already installed")

            if success:
                self.state_manager.mark_component_completed(comp_id)
            else:
                self.state_manager.mark_component_failed(comp_id, "Installation failed")

        except Exception as e:
            self.state_manager.mark_component_failed(comp_id, str(e))
            print_error(f"Error installing {comp_info['name']}: {e}")
            success = False

        return success

    def setup_credentials(self, repo_root: Path) -> None:
        """Prompt user to set up credentials for authenticated downloads.

        Sets up:
        - HF_TOKEN.dat for HuggingFace (SAM3, etc.)
        - SMPL.login.dat for SMPL-X models (motion capture)
        """
        print_header("Credentials Setup")
        print("Some components require authentication to download:")
        print("  - SAM3 segmentation model (HuggingFace)")
        print("  - SMPL-X body models (smpl-x.is.tue.mpg.de)")
        print("")

        # Check existing credentials
        hf_token_file = repo_root / "HF_TOKEN.dat"
        smpl_creds_file = repo_root / "SMPL.login.dat"

        hf_exists = hf_token_file.exists()
        smpl_exists = smpl_creds_file.exists()

        if hf_exists and smpl_exists:
            print_success("All credential files already exist")
            if ask_yes_no("Update credentials?", default=False):
                hf_exists = False
                smpl_exists = False
            else:
                return

        # HuggingFace token setup
        if not hf_exists:
            print(f"\n{Colors.BOLD}HuggingFace Token Setup{Colors.ENDC}")
            print("Required for: SAM3 segmentation model")
            print("Steps:")
            print("  1. Request access at https://huggingface.co/facebook/sam3")
            print("  2. Get token from https://huggingface.co/settings/tokens")
            print("")

            if ask_yes_no("Set up HuggingFace token now?", default=True):
                token = tty_input("Enter your HuggingFace token (hf_...): ").strip()
                if token:
                    if token.startswith("hf_") or len(token) > 20:
                        with open(hf_token_file, 'w') as f:
                            f.write(token + '\n')
                        hf_token_file.chmod(0o600)
                        print_success(f"Token saved to {hf_token_file}")
                    else:
                        print_warning("Token looks invalid (should start with 'hf_')")
                        if ask_yes_no("Save anyway?", default=False):
                            with open(hf_token_file, 'w') as f:
                                f.write(token + '\n')
                            hf_token_file.chmod(0o600)
                            print_success(f"Token saved to {hf_token_file}")
                else:
                    print_info("Skipped - you can add HF_TOKEN.dat later")
            else:
                print_info("Skipped - you can add HF_TOKEN.dat later")

        # SMPL-X credentials setup
        if not smpl_exists:
            print(f"\n{Colors.BOLD}SMPL-X Credentials Setup{Colors.ENDC}")
            print("Required for: Motion capture (WHAM, ECON)")
            print("Steps:")
            print("  1. Register at https://smpl-x.is.tue.mpg.de/")
            print("  2. Wait for approval email (usually within 24 hours)")
            print("")

            if ask_yes_no("Set up SMPL-X credentials now?", default=True):
                email = tty_input("Enter your SMPL-X email: ").strip()
                if email and '@' in email:
                    password = tty_input("Enter your SMPL-X password: ").strip()
                    if password:
                        with open(smpl_creds_file, 'w') as f:
                            f.write(email + '\n')
                            f.write(password + '\n')
                        # Set restrictive permissions
                        smpl_creds_file.chmod(0o600)
                        print_success(f"Credentials saved to {smpl_creds_file}")
                    else:
                        print_info("Skipped - you can add SMPL.login.dat later")
                else:
                    print_info("Skipped - you can add SMPL.login.dat later")
            else:
                print_info("Skipped - you can add SMPL.login.dat later")

        print("")

    def interactive_install(self, component: Optional[str] = None, resume: bool = False):
        """Interactive installation flow."""
        print_header("VFX Pipeline Installation Wizard")

        # Check for resumable installation
        if not resume and self.state_manager.can_resume():
            incomplete = self.state_manager.get_incomplete_components()
            print_warning("Found incomplete installation from previous run:")
            for comp_id in incomplete:
                status = self.state_manager.get_component_status(comp_id)
                print(f"  - {self.components.get(comp_id, {}).get('name', comp_id)}: {status}")

            if ask_yes_no("\nResume previous installation?", default=True):
                resume = True
            else:
                if ask_yes_no("Start fresh (clear previous state)?", default=False):
                    self.state_manager.clear_state()

        # System requirements
        if not self.check_system_requirements():
            print_error("\nSystem requirements not met. Please install missing components.")
            return False

        # Conda environment setup
        if not self.setup_conda_environment():
            return False

        # Check current status
        status = self.check_all_components()
        self.print_status(status)

        # Set up credentials for authenticated downloads
        self.setup_credentials(self.repo_root)

        # Determine what to install
        if component:
            # Specific component requested
            if component not in self.components:
                print_error(f"Unknown component: {component}")
                print_info(f"Available: {', '.join(self.components.keys())}")
                return False

            to_install = [component]
        else:
            # Interactive selection
            print("\n" + "="*60)
            print("What would you like to install?")
            print("="*60)
            print("1. Core pipeline only (COLMAP, segmentation)")
            print("2. Core + ComfyUI (workflows ready to use)")
            print("3. Full stack (Core + ComfyUI + Motion capture)")
            print("4. Custom selection")
            print("5. Nothing (check only)")

            while True:
                choice = tty_input("\nChoice [1-5]: ").strip()
                if choice == '1':
                    to_install = ['core', 'pytorch']
                    break
                elif choice == '2':
                    to_install = ['core', 'pytorch', 'comfyui']
                    break
                elif choice == '3':
                    to_install = ['core', 'pytorch', 'comfyui', 'mocap_core', 'wham', 'econ']
                    break
                elif choice == '4':
                    to_install = []
                    for comp_id, comp_info in self.components.items():
                        if ask_yes_no(f"Install {comp_info['name']}?", default=False):
                            to_install.append(comp_id)
                    break
                elif choice == '5':
                    return True
                else:
                    print("Invalid choice")

        # Show disk space estimate
        if to_install:
            if not self.show_space_estimate(to_install):
                print_error("\nInsufficient disk space for installation")
                return False

            # Confirm installation
            if not ask_yes_no("\nProceed with installation?", default=True):
                print_info("Installation cancelled")
                return True

        # Install components
        print_header("Installing Components")

        for comp_id in to_install:
            if not status.get(comp_id, False):
                if not self.install_component(comp_id):
                    print_error(f"Failed to install {self.components[comp_id]['name']}")
                    if self.components[comp_id]['required']:
                        return False

        # Download checkpoints for motion capture components
        mocap_components = [cid for cid in to_install if cid in ['wham', 'econ']]
        if mocap_components:
            if ask_yes_no("\nDownload checkpoints for motion capture components?", default=True):
                self.checkpoint_downloader.download_all_checkpoints(mocap_components, self.state_manager)

        # Final status
        final_status = self.check_all_components()
        self.print_status(final_status)

        # Generate configuration files
        if to_install:
            self.config_generator.generate_all()

        # Run validation tests
        if to_install and ask_yes_no("\nRun installation validation tests?", default=True):
            self.validator.validate_and_report()

        # Post-installation instructions
        self.print_post_install_instructions(final_status)

        return True

    def print_post_install_instructions(self, status: Dict[str, bool]):
        """Print post-installation instructions."""
        print_header("Next Steps")

        # SMPL-X models (manual download required - registration needed)
        if status.get('mocap_core', False):
            print("\n📦 SMPL-X Body Models (Manual Download Required):")
            print("  1. Register at https://smpl-x.is.tue.mpg.de/")
            print("  2. Download SMPL-X models")
            print("  3. Place in .vfx_pipeline/smplx_models/")
            print("     mkdir -p .vfx_pipeline/smplx_models && cp SMPLX_*.pkl .vfx_pipeline/smplx_models/")

        # Checkpoints status
        has_mocap = status.get('wham', False) or status.get('econ', False)
        if has_mocap:
            print("\n📦 Motion Capture Checkpoints:")
            if status.get('wham', False):
                if self.state_manager.is_checkpoint_downloaded('wham'):
                    print("  ✓ WHAM checkpoints downloaded")
                else:
                    print("  ⚠ WHAM checkpoints not downloaded - run wizard again or visit:")
                    print("    https://github.com/yohanshin/WHAM")

            if status.get('econ', False):
                if self.state_manager.is_checkpoint_downloaded('econ'):
                    print("  ✓ ECON checkpoints downloaded")
                else:
                    print("  ⚠ ECON checkpoints not downloaded - run wizard again or visit:")
                    print("    https://github.com/YuliangXiu/ECON")

        # ComfyUI
        if status.get('comfyui', False):
            comfyui_path = self.install_dir / "ComfyUI"
            print("\n🎨 ComfyUI:")
            print(f"  ✓ Installed at {comfyui_path}")
            print("  ✓ Custom nodes installed")
            print("\n  Start server:")
            print(f"    cd {comfyui_path}")
            print("    python main.py --listen")
        else:
            print("\n🎨 ComfyUI (Optional):")
            print("  Not installed. Run wizard again to add ComfyUI support.")

        # Testing
        print("\n✅ Test Installation:")
        print("  python scripts/run_pipeline.py --help")
        print("  python scripts/run_mocap.py --check")

        # Documentation
        print("\n📖 Documentation:")
        print("  README.md - Pipeline overview and usage")
        print("  TESTING.md - Testing and validation guide")
        print("  IMPLEMENTATION_NOTES.md - Developer notes")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive installation wizard for VFX pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--component", "-C",
        type=str,
        choices=['core', 'pytorch', 'colmap', 'mocap_core', 'wham', 'econ', 'comfyui'],
        help="Install specific component"
    )
    parser.add_argument(
        "--check-only", "-c",
        action="store_true",
        help="Check installation status only (don't install)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run validation tests on existing installation"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume previous interrupted installation"
    )

    args = parser.parse_args()

    wizard = InstallationWizard()

    if args.check_only:
        wizard.check_system_requirements()
        status = wizard.check_all_components()
        wizard.print_status(status)
        sys.exit(0)

    if args.validate:
        wizard.validator.validate_and_report()
        sys.exit(0)

    success = wizard.interactive_install(component=args.component, resume=args.resume)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
