"""Utility functions for the installation wizard.

This module provides terminal output formatting, user input helpers,
and system check utilities used throughout the wizard.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Global TTY file handle for reading input when piped (Unix only)
_tty_handle = None


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def tty_input(prompt: str = "") -> str:
    """Read input from TTY, even when stdin is piped.

    This allows the script to work when run via: curl ... | bash (Unix)
    or when stdin is redirected on Windows.

    On Windows, uses msvcrt for direct console input when stdin is piped.
    On Unix, uses /dev/tty for direct terminal access.
    """
    global _tty_handle

    if sys.stdin.isatty():
        return input(prompt)

    if _is_windows():
        try:
            import msvcrt
            if prompt:
                print(prompt, end='', flush=True)
            chars = []
            while True:
                char = msvcrt.getwch()
                if char in ('\r', '\n'):
                    print()
                    break
                if char == '\x08':
                    if chars:
                        chars.pop()
                        print('\b \b', end='', flush=True)
                else:
                    chars.append(char)
                    print(char, end='', flush=True)
            return ''.join(chars)
        except ImportError:
            raise EOFError("No TTY available for input on Windows")
    else:
        if _tty_handle is None:
            try:
                _tty_handle = open('/dev/tty', 'r', encoding='utf-8')
            except OSError:
                raise EOFError("No TTY available for input")

        if prompt:
            print(prompt, end='', flush=True)
        return _tty_handle.readline().rstrip('\n')


BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


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
    print(f"{Colors.OKGREEN}OK {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}! {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}X {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}> {text}{Colors.ENDC}")


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


def run_command(
    cmd: List[str],
    check: bool = True,
    capture: bool = False,
    timeout: int = 600,
    stream: bool = False,
    shell: bool = False,
    cwd: str = None
) -> Tuple[bool, str]:
    """Run shell command and return success status and output.

    Args:
        cmd: Command and arguments
        check: Raise on non-zero exit (only if not capturing)
        capture: Capture output instead of showing it
        timeout: Timeout in seconds (default 600 = 10 minutes for conda installs)
        stream: Stream output line by line (for long-running commands)
        shell: Use shell execution (required for Windows .bat files)
        cwd: Working directory for the command
    """
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=shell, cwd=cwd)
            return result.returncode == 0, result.stdout + result.stderr
        elif stream:
            import sys
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=shell,
                cwd=cwd
            )
            output_lines = []
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(f"    {line.rstrip()}")
                    sys.stdout.flush()
                    output_lines.append(line)
            process.wait()
            return process.returncode == 0, ''.join(output_lines)
        else:
            result = subprocess.run(cmd, check=check, timeout=timeout, shell=shell, cwd=cwd)
            return result.returncode == 0, ""
    except subprocess.TimeoutExpired:
        print_warning(f"Command timed out after {timeout}s")
        return False, ""
    except (subprocess.CalledProcessError, FileNotFoundError):
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
    """Check if command-line tool is available.

    Uses shutil.which() for cross-platform compatibility (Windows/Linux/macOS).
    """
    return shutil.which(command) is not None


def _find_nvidia_smi() -> str:
    """Find nvidia-smi executable, checking PATH then known Windows locations."""
    path = shutil.which("nvidia-smi")
    if path:
        return path
    from .platform import PlatformManager
    found = PlatformManager.find_tool("nvidia-smi")
    return str(found) if found else "nvidia-smi"


def check_gpu_available() -> Tuple[bool, str]:
    """Check if NVIDIA GPU is available."""
    nvidia_smi = _find_nvidia_smi()
    success, output = run_command([nvidia_smi], check=False, capture=True)
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
