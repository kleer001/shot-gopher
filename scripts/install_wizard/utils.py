"""Utility functions for the installation wizard.

This module provides terminal output formatting, user input helpers,
and system check utilities used throughout the wizard.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
