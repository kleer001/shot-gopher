"""Platform detection and OS-specific instructions.

Provides platform detection and OS-specific installation instructions
for system dependencies across Linux, macOS, Windows, and WSL2.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from env_config import INSTALL_DIR

from .utils import BROWSER_USER_AGENT

# Repo-local tools directory (sandboxed, no home directory pollution)
TOOLS_DIR = INSTALL_DIR / "tools"


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


class PlatformManager:
    """Handles platform detection and OS-specific instructions."""

    @staticmethod
    def detect_platform() -> Tuple[str, str, str]:
        """Detect operating system and package manager.

        Returns:
            Tuple of (os_name, environment, package_manager) where:
            - os_name: 'linux', 'macos', 'windows'
            - environment: 'native', 'wsl2'
            - package_manager: 'apt', 'yum', 'brew', 'choco', 'unknown'
        """
        system = platform.system().lower()

        if system == "linux":
            try:
                with open("/proc/version", "r", encoding='utf-8') as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        return "linux", "wsl2", PlatformManager._detect_linux_package_manager()
            except FileNotFoundError:
                pass
            return "linux", "native", PlatformManager._detect_linux_package_manager()

        elif system == "darwin":
            has_brew = shutil.which("brew") is not None
            return "macos", "native", "brew" if has_brew else "unknown"

        elif system == "windows":
            pkg_manager = PlatformManager._detect_windows_package_manager()
            return "windows", "native", pkg_manager

        return system, "unknown", "unknown"

    @staticmethod
    def _detect_linux_package_manager() -> str:
        """Detect Linux package manager."""
        managers = [
            ("apt", ["apt", "--version"]),
            ("apt-get", ["apt-get", "--version"]),
            ("yum", ["yum", "--version"]),
            ("dnf", ["dnf", "--version"]),
            ("pacman", ["pacman", "--version"]),
            ("zypper", ["zypper", "--version"]),
        ]

        for name, cmd in managers:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=2)
                if result.returncode == 0:
                    if name == "apt-get":
                        return "apt"
                    return name
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return "unknown"

    @staticmethod
    def _detect_windows_package_manager() -> str:
        """Detect Windows package manager in priority order."""
        if shutil.which("winget"):
            return "winget"
        if shutil.which("choco"):
            return "choco"
        if shutil.which("scoop"):
            return "scoop"
        return "unknown"

    @staticmethod
    def find_tool(tool_name: str) -> Optional[Path]:
        """Find a tool executable with cross-platform path search.

        Search order (sandboxed tools take priority):
        1. Repo-local tools directory (.vfx_pipeline/tools/)
        2. System PATH
        3. Platform-specific standard locations

        Args:
            tool_name: Name of the tool (e.g., 'colmap', 'ffmpeg', '7z')

        Returns:
            Path to the executable if found, None otherwise.
        """
        # 1. Check repo-local tools directory FIRST (sandboxed)
        local_paths = PlatformManager._get_local_tool_paths(tool_name)
        for path in local_paths:
            if path.exists():
                return path

        # 2. Check system PATH
        path_result = shutil.which(tool_name)
        if path_result:
            return Path(path_result)

        # 3. Check platform-specific standard locations (fallback)
        if _is_windows():
            search_paths = PlatformManager._get_windows_tool_paths(tool_name)
        else:
            search_paths = PlatformManager._get_unix_tool_paths(tool_name)

        for path in search_paths:
            if path.exists():
                return path

        return None

    @staticmethod
    def _get_local_tool_paths(tool_name: str) -> List[Path]:
        """Get repo-local tool paths (sandboxed, highest priority)."""
        tool_dir = TOOLS_DIR / tool_name

        if _is_windows():
            paths = [
                tool_dir / f"{tool_name}.exe",
                tool_dir / f"{tool_name.upper()}.bat",
                tool_dir / "bin" / f"{tool_name}.exe",
                tool_dir / f"{tool_name.upper()}.exe",
            ]
            if tool_name == "blender":
                paths.insert(0, tool_dir / "blender.exe")
            return paths
        elif platform.system() == "Darwin":
            paths = [
                tool_dir / tool_name,
                tool_dir / "bin" / tool_name,
            ]
            if tool_name == "blender":
                paths.insert(0, tool_dir / "Blender.app" / "Contents" / "MacOS" / "Blender")
            return paths
        else:
            paths = [
                tool_dir / tool_name,
                tool_dir / "bin" / tool_name,
            ]
            if tool_name == "blender":
                paths.insert(0, tool_dir / "blender")
            return paths

    @staticmethod
    def _get_windows_tool_paths(tool_name: str) -> List[Path]:
        """Get Windows system-wide search paths for a tool.

        NOTE: Only searches system directories (Program Files, etc.).
        User home directories are NOT searched - all user tools should
        be installed to the repo-local .vfx_pipeline/tools/ directory.
        """
        programfiles = Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        programfiles_x86 = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))

        tool_paths: Dict[str, List[Path]] = {
            "colmap": [
                programfiles / "COLMAP" / "COLMAP.bat",
                programfiles_x86 / "COLMAP" / "COLMAP.bat",
                Path("C:/COLMAP/COLMAP.bat"),
            ],
            "ffmpeg": [
                programfiles / "FFmpeg" / "bin" / "ffmpeg.exe",
                programfiles_x86 / "FFmpeg" / "bin" / "ffmpeg.exe",
                Path("C:/ffmpeg/bin/ffmpeg.exe"),
            ],
            "ffprobe": [
                programfiles / "FFmpeg" / "bin" / "ffprobe.exe",
                programfiles_x86 / "FFmpeg" / "bin" / "ffprobe.exe",
                Path("C:/ffmpeg/bin/ffprobe.exe"),
            ],
            "7z": [
                programfiles / "7-Zip" / "7z.exe",
                programfiles_x86 / "7-Zip" / "7z.exe",
            ],
            "nvidia-smi": [
                programfiles / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
                Path("C:/Windows/System32/nvidia-smi.exe"),
            ],
            "nvcc": [
                programfiles / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v12.1" / "bin" / "nvcc.exe",
                programfiles / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v12.0" / "bin" / "nvcc.exe",
                programfiles / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v11.8" / "bin" / "nvcc.exe",
                programfiles / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v11.7" / "bin" / "nvcc.exe",
                Path("C:/CUDA/bin/nvcc.exe"),
            ],
            "aria2c": [],
            "blender": [
                programfiles / "Blender Foundation" / "Blender 4.2" / "blender.exe",
                programfiles / "Blender Foundation" / "Blender 4.1" / "blender.exe",
                programfiles / "Blender Foundation" / "Blender 4.0" / "blender.exe",
                programfiles / "Blender Foundation" / "Blender" / "blender.exe",
            ],
        }

        return tool_paths.get(tool_name, [])

    @staticmethod
    def _get_unix_tool_paths(tool_name: str) -> List[Path]:
        """Get Unix system-wide search paths for a tool.

        NOTE: Only searches system directories (/usr/bin, /usr/local/bin).
        User home directories are NOT searched - all user tools should
        be installed to the repo-local .vfx_pipeline/tools/ directory.
        """
        tool_paths: Dict[str, List[Path]] = {
            "colmap": [
                Path("/usr/local/bin/colmap"),
                Path("/usr/bin/colmap"),
            ],
            "ffmpeg": [
                Path("/usr/local/bin/ffmpeg"),
                Path("/usr/bin/ffmpeg"),
            ],
            "ffprobe": [
                Path("/usr/local/bin/ffprobe"),
                Path("/usr/bin/ffprobe"),
            ],
            "7z": [
                Path("/usr/bin/7z"),
                Path("/usr/bin/7za"),
                Path("/usr/bin/7zr"),
            ],
            "aria2c": [
                Path("/usr/bin/aria2c"),
                Path("/usr/local/bin/aria2c"),
            ],
            "blender": [
                Path("/usr/bin/blender"),
                Path("/usr/local/bin/blender"),
                Path("/opt/blender/blender"),
                Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            ],
        }

        return tool_paths.get(tool_name, [])

    @staticmethod
    def run_tool(
        tool_path: Path,
        args: List[str],
        **subprocess_kwargs
    ) -> subprocess.CompletedProcess:
        """Run an external tool with proper handling for Windows .bat files.

        On Windows, .bat files need shell=True or explicit cmd /c invocation.

        Args:
            tool_path: Path to the tool executable
            args: Arguments to pass to the tool
            **subprocess_kwargs: Additional args for subprocess.run

        Returns:
            CompletedProcess result
        """
        cmd = [str(tool_path)] + args

        if _is_windows() and str(tool_path).lower().endswith('.bat'):
            subprocess_kwargs['shell'] = True

        return subprocess.run(cmd, **subprocess_kwargs)

    @staticmethod
    def get_system_package_install_cmd(
        package: str,
        os_name: str,
        pkg_manager: str
    ) -> Optional[str]:
        """Get command to install a system package.

        Args:
            package: Package name (may differ per OS)
            os_name: Operating system ('linux', 'macos', 'windows')
            pkg_manager: Package manager ('apt', 'yum', 'brew', etc.)

        Returns:
            Installation command string or None if not available
        """
        commands = {
            ("linux", "apt"): f"sudo apt update && sudo apt install -y {package}",
            ("linux", "yum"): f"sudo yum install -y {package}",
            ("linux", "dnf"): f"sudo dnf install -y {package}",
            ("linux", "pacman"): f"sudo pacman -S {package}",
            ("linux", "zypper"): f"sudo zypper install {package}",
            ("macos", "brew"): f"brew install {package}",
            ("windows", "winget"): f"winget install {package}",
            ("windows", "choco"): f"choco install {package} -y",
            ("windows", "scoop"): f"scoop install {package}",
        }

        return commands.get((os_name, pkg_manager))

    @staticmethod
    def get_missing_dependency_instructions(
        dependency: str,
        os_name: str,
        environment: str,
        pkg_manager: str
    ) -> str:
        """Get detailed installation instructions for missing dependency.

        Args:
            dependency: Name of missing dependency (ffmpeg, git, colmap, etc.)
            os_name: Operating system
            environment: Environment (native, wsl2)
            pkg_manager: Package manager

        Returns:
            Multi-line installation instructions
        """
        pkg_map = {
            "ffmpeg": {
                "apt": "ffmpeg",
                "yum": "ffmpeg",
                "dnf": "ffmpeg",
                "brew": "ffmpeg",
                "choco": "ffmpeg",
            },
            "git": {
                "apt": "git",
                "yum": "git",
                "dnf": "git",
                "brew": "git",
                "choco": "git",
            },
            "colmap": {
                "apt": "colmap",
                "yum": "colmap",  # May need EPEL
                "dnf": "colmap",
                "brew": "colmap",
                "choco": None,  # Not available
            },
        }

        package_name = pkg_map.get(dependency, {}).get(pkg_manager)

        if os_name == "linux":
            if pkg_manager == "apt":
                if package_name:
                    return f"""
Install {dependency} on Ubuntu/Debian:

    sudo apt update
    sudo apt install -y {package_name}
"""
                else:
                    return f"{dependency} not available via apt. Please install from source."

            elif pkg_manager in ("yum", "dnf"):
                if package_name:
                    return f"""
Install {dependency} on RHEL/CentOS/Fedora:

    sudo {pkg_manager} install -y {package_name}

Note: May require EPEL repository:
    sudo {pkg_manager} install -y epel-release
"""
                else:
                    return f"{dependency} not available via {pkg_manager}. Install from source."

            elif pkg_manager == "pacman":
                return f"""
Install {dependency} on Arch Linux:

    sudo pacman -S {package_name or dependency}
"""
            else:
                return f"""
{dependency} package manager not detected. Install manually:
    - Ubuntu/Debian: sudo apt install {dependency}
    - RHEL/CentOS: sudo yum install {dependency}
    - Arch: sudo pacman -S {dependency}
"""

        elif os_name == "macos":
            if pkg_manager == "brew":
                if package_name:
                    return f"""
Install {dependency} on macOS:

    brew install {package_name}
"""
                else:
                    return f"""
{dependency} is not available via Homebrew.
Please check the official {dependency} website for macOS installation.
"""
            else:
                return f"""
Homebrew not found. Install Homebrew first:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Then install {dependency}:

    brew install {package_name or dependency}
"""

        elif os_name == "windows":
            if environment == "wsl2":
                return f"""
You are in WSL2. Use Linux package manager:

    sudo apt update
    sudo apt install -y {pkg_map.get(dependency, {}).get('apt', dependency)}
"""
            else:
                if pkg_manager == "choco" and package_name:
                    return f"""
Install {dependency} on Windows via Chocolatey:

    choco install {package_name} -y
"""
                else:
                    if dependency == "colmap":
                        return """
COLMAP is not available via Chocolatey on Windows.

Options:
1. Use WSL2 (recommended for GPU workloads):
   - Install Ubuntu in WSL2
   - sudo apt install colmap

2. Use conda:
   - conda install -c conda-forge colmap

3. Build from source (advanced):
   - https://colmap.github.io/install.html
"""
                    else:
                        return f"""
Install Chocolatey first (run PowerShell as Administrator):

    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

Then install {dependency}:

    choco install {dependency} -y
"""

        return f"Installation instructions not available for {dependency} on {os_name}"

    @staticmethod
    def get_wizard_recommendation(
        os_name: str,
        environment: str,
        has_gpu: bool
    ) -> str:
        """Get recommendation for which wizard to use.

        Args:
            os_name: Operating system
            environment: Environment (native, wsl2)
            has_gpu: Whether NVIDIA GPU is available

        Returns:
            Recommendation text
        """
        if os_name == "macos":
            return """
[macOS Detected]

Run: python scripts/install_wizard.py

Notes:
  - All features available except GPU-accelerated processing
  - macOS uses CPU fallback for ML models
"""

        elif os_name == "linux" and environment == "native" and has_gpu:
            return """
[Linux + GPU Detected]

Run: python scripts/install_wizard.py

Benefits:
  + Direct filesystem access
  + Full GPU acceleration
  + Optimal performance
"""

        elif os_name == "linux" and environment == "wsl2" and has_gpu:
            return """
[WSL2 + GPU Detected]

Run: python scripts/install_wizard.py

Notes:
  - GPU passthrough works via NVIDIA WSL2 drivers
  - Install from within WSL2 environment
"""

        elif not has_gpu:
            return """
[!] No NVIDIA GPU Detected

Run: python scripts/install_wizard.py

Notes:
  - Motion capture requires NVIDIA GPU (12GB+ VRAM)
  - Without GPU, only roto workflows are available
"""

        else:
            return """
Run: python scripts/install_wizard.py
"""

    # =========================================================================
    # SANDBOXED TOOL INSTALLATION
    # =========================================================================

    # Tool download URLs - GitHub releases and official builds
    TOOL_DOWNLOADS: Dict[str, Dict[str, str]] = {
        "colmap": {
            "windows": "https://github.com/colmap/colmap/releases/download/3.9.1/COLMAP-3.9.1-windows-cuda.zip",
            "linux": "https://github.com/colmap/colmap/releases/download/3.9.1/COLMAP-3.9.1-linux-no-cuda.tar.gz",
        },
        "ffmpeg": {
            "windows": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
            "linux": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
        },
        "blender": {
            "windows": "https://download.blender.org/release/Blender4.2/blender-4.2.5-windows-x64.zip",
            "linux": "https://download.blender.org/release/Blender4.2/blender-4.2.5-linux-x64.tar.xz",
            "macos_arm": "https://download.blender.org/release/Blender4.2/blender-4.2.5-macos-arm64.dmg",
            "macos_intel": "https://download.blender.org/release/Blender4.2/blender-4.2.5-macos-x64.dmg",
        },
    }

    @staticmethod
    def get_tools_dir() -> Path:
        """Get the repo-local tools directory."""
        return TOOLS_DIR

    @staticmethod
    def _get_platform_key(tool_name: str) -> str:
        """Get the platform key for tool downloads.

        Handles macOS ARM vs Intel distinction for tools that provide
        separate builds.

        Args:
            tool_name: Name of the tool

        Returns:
            Platform key string ('windows', 'linux', 'macos_arm', 'macos_intel')
        """
        if _is_windows():
            return "windows"
        elif platform.system() == "Darwin":
            tool_config = PlatformManager.TOOL_DOWNLOADS.get(tool_name, {})
            if "macos_arm" in tool_config or "macos_intel" in tool_config:
                is_arm = platform.machine() == "arm64"
                return "macos_arm" if is_arm else "macos_intel"
            return "macos" if "macos" in tool_config else "linux"
        else:
            return "linux"

    @staticmethod
    def _extract_dmg(dmg_path: Path, dest_dir: Path, app_name: str) -> bool:
        """Extract .app from a macOS DMG file.

        Uses hdiutil to mount the DMG, copies the .app bundle, then unmounts.

        Args:
            dmg_path: Path to the DMG file
            dest_dir: Destination directory for the .app
            app_name: Name of the .app bundle (e.g., 'Blender.app')

        Returns:
            True if extraction succeeded
        """
        import shutil
        import tempfile

        mount_point = Path(tempfile.gettempdir()) / f"dmg_mount_{dmg_path.stem}"
        mounted = False

        try:
            mount_point.mkdir(parents=True, exist_ok=True)

            print(f"    Mounting DMG...")
            result = subprocess.run(
                ["hdiutil", "attach", str(dmg_path), "-mountpoint", str(mount_point), "-nobrowse"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"    Error mounting DMG: {result.stderr}")
                return False

            mounted = True

            app_src = mount_point / app_name
            if not app_src.exists():
                for item in mount_point.iterdir():
                    if item.name.endswith(".app"):
                        app_src = item
                        break

            if not app_src.exists():
                print(f"    Error: Could not find .app in DMG")
                return False

            print(f"    Copying {app_src.name}...")
            app_dest = dest_dir / app_src.name
            if app_dest.exists():
                shutil.rmtree(app_dest)
            shutil.copytree(app_src, app_dest, symlinks=True)

            return True

        except subprocess.TimeoutExpired:
            print(f"    Error: DMG operation timed out")
            return False
        except Exception as e:
            print(f"    Error extracting DMG: {e}")
            return False
        finally:
            if mounted:
                print(f"    Unmounting DMG...")
                try:
                    subprocess.run(
                        ["hdiutil", "detach", str(mount_point)],
                        capture_output=True,
                        timeout=60
                    )
                except Exception:
                    pass
            if mount_point.exists() and not any(mount_point.iterdir()):
                try:
                    mount_point.rmdir()
                except Exception:
                    pass

    @staticmethod
    def install_tool(tool_name: str, force: bool = False) -> Optional[Path]:
        """Download and install a tool to the repo-local tools directory.

        Tools are installed to .vfx_pipeline/tools/<tool_name>/ to keep
        everything sandboxed within the repo directory. No files are
        placed in the user's home directory.

        Args:
            tool_name: Name of the tool to install ('colmap', 'ffmpeg', 'blender')
            force: Re-download even if already installed

        Returns:
            Path to the installed executable, or None if installation failed
        """
        import tempfile
        import zipfile
        import tarfile

        if tool_name not in PlatformManager.TOOL_DOWNLOADS:
            print(f"    Error: No download URL configured for {tool_name}")
            return None

        platform_key = PlatformManager._get_platform_key(tool_name)
        url = PlatformManager.TOOL_DOWNLOADS[tool_name].get(platform_key)

        if not url:
            print(f"    Error: {tool_name} not available for {platform_key}")
            return None

        tool_dir = TOOLS_DIR / tool_name

        if not force:
            existing = PlatformManager.find_tool(tool_name)
            if existing and str(TOOLS_DIR) in str(existing):
                print(f"    {tool_name} already installed at {existing}")
                return existing

        print(f"    Downloading {tool_name}...")
        print(f"    URL: {url}")

        TOOLS_DIR.mkdir(parents=True, exist_ok=True)

        tmp_path = None
        try:
            import urllib.request
            import urllib.parse

            url_path = urllib.parse.urlparse(url).path
            suffix = Path(url_path).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = Path(tmp.name)

            request = urllib.request.Request(
                url,
                headers={"User-Agent": BROWSER_USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                last_print_len = 0

                with open(tmp_path, "wb") as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)

                        mb_downloaded = downloaded / (1024 * 1024)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_total = total_size / (1024 * 1024)
                            msg = f"\r    Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                        else:
                            msg = f"\r    Downloaded: {mb_downloaded:.1f} MB"
                        padding = " " * max(0, last_print_len - len(msg))
                        print(msg + padding, end='', flush=True)
                        last_print_len = len(msg)

                print()
            print(f"    Downloaded to {tmp_path}")

            if tool_dir.exists():
                import shutil
                shutil.rmtree(tool_dir)
            tool_dir.mkdir(parents=True, exist_ok=True)

            print(f"    Extracting to {tool_dir}...")

            if url_path.endswith('.zip'):
                with zipfile.ZipFile(tmp_path, 'r') as zf:
                    zf.extractall(tool_dir)
                PlatformManager._flatten_single_subdir(tool_dir)
            elif url_path.endswith('.tar.gz') or url_path.endswith('.tar.xz'):
                with tarfile.open(tmp_path, 'r:*') as tf:
                    tf.extractall(tool_dir)
                PlatformManager._flatten_single_subdir(tool_dir)
            elif url_path.endswith('.dmg'):
                app_names = {
                    "blender": "Blender.app",
                }
                app_name = app_names.get(tool_name, f"{tool_name.title()}.app")
                if not PlatformManager._extract_dmg(tmp_path, tool_dir, app_name):
                    return None

            installed_path = PlatformManager.find_tool(tool_name)
            if installed_path:
                print(f"    Successfully installed {tool_name} at {installed_path}")
                return installed_path
            else:
                print(f"    Warning: {tool_name} extracted but executable not found")
                return None

        except Exception as e:
            print()
            print(f"    Error installing {tool_name}: {e}")
            return None
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _flatten_single_subdir(tool_dir: Path) -> None:
        """If extraction created a single subdirectory, flatten it.

        Many archives extract to a single directory like 'COLMAP-3.9.1-windows/'.
        This flattens that so executables are directly accessible.
        """
        import shutil

        subdirs = [d for d in tool_dir.iterdir() if d.is_dir()]
        files = [f for f in tool_dir.iterdir() if f.is_file()]

        if len(subdirs) == 1 and len(files) == 0:
            single_dir = subdirs[0]
            for item in single_dir.iterdir():
                shutil.move(str(item), str(tool_dir / item.name))
            single_dir.rmdir()
