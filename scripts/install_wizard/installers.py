"""Component installers for the installation wizard.

This module provides installer classes for different types of components
including Python packages and Git repositories.
"""

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from env_config import INSTALL_DIR

from .utils import check_python_package, print_error, print_info, print_success, print_warning, run_command

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


class CondaPackageInstaller(ComponentInstaller):
    """Installer for packages via conda (for system tools like COLMAP)."""

    def __init__(self, name: str, package: str, channel: str = "conda-forge", command: str = None, size_gb: float = 0.0):
        super().__init__(name, size_gb)
        self.package = package
        self.channel = channel
        self.command = command or package  # Command to check availability
        self.conda_manager: Optional['CondaEnvironmentManager'] = None

    def set_conda_manager(self, conda_manager: 'CondaEnvironmentManager'):
        """Set the conda manager for environment-aware installation."""
        self.conda_manager = conda_manager

    def check(self) -> bool:
        """Check if package command is available in the conda environment."""
        if self.conda_manager and self.conda_manager.conda_exe:
            success, _ = run_command([
                self.conda_manager.conda_exe, "run", "-n", self.conda_manager.env_name,
                self.command, "--version"
            ], check=False, capture=True)
            self.installed = success
        else:
            # Check system-wide
            success, _ = run_command([self.command, "--version"], check=False, capture=True)
            self.installed = success
        return self.installed

    def install(self) -> bool:
        print(f"\nInstalling {self.name} via conda...")

        if not self.conda_manager or not self.conda_manager.conda_exe:
            print_error("Conda not available for installation")
            print_warning(f"Install manually: conda install -c {self.channel} {self.package}")
            return False

        success = self.conda_manager.install_package_conda(self.package, self.channel)

        if success:
            print_success(f"{self.name} installed")
            self.installed = True
        else:
            print_error(f"Failed to install {self.name}")
        return success


class SystemPackageInstaller(ComponentInstaller):
    """Installer for system packages (apt on Linux)."""

    def __init__(self, name: str, apt_package: str, command: str = None, size_gb: float = 0.0):
        super().__init__(name, size_gb)
        self.apt_package = apt_package
        self.command = command or apt_package

    def check(self) -> bool:
        """Check if command is available system-wide."""
        import shutil
        # Use shutil.which() - more reliable than running --version
        self.installed = shutil.which(self.command) is not None
        return self.installed

    def install(self) -> bool:
        import platform
        system = platform.system()

        if system == "Linux":
            print(f"\nInstalling {self.name} via apt...")
            print_warning("This requires sudo access. You may be prompted for your password.")
            success, _ = run_command(["sudo", "apt-get", "install", "-y", self.apt_package], stream=True)
            if success:
                print_success(f"{self.name} installed")
                self.installed = True
                return True
            else:
                print_error(f"apt install failed for {self.name}")

        # Fallback: provide manual instructions
        print_warning(f"\nCould not auto-install {self.name}. Install manually:")
        if system == "Linux":
            print(f"  sudo apt install {self.apt_package}")
        elif system == "Darwin":
            print(f"  brew install {self.apt_package}")
        else:
            print(f"  See: https://colmap.github.io/install.html")
        return False


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

        # Run install.py if exists (for GPU extensions like SAM3)
        install_py = self.install_dir / "install.py"
        if install_py.exists():
            print(f"  Running {self.name} install script...")
            if self.conda_manager and self.conda_manager.conda_exe:
                success, output = run_command([
                    self.conda_manager.conda_exe, "run", "-n", self.conda_manager.env_name,
                    "python", str(install_py)
                ], capture=True)
            else:
                success, output = run_command(
                    [sys.executable, str(install_py)], capture=True
                )
            if success:
                print_success(f"{self.name} install script completed")
            else:
                print_warning(f"install.py failed for {self.name} (GPU acceleration may be slower)")

        self.installed = True
        print_success(f"{self.name} cloned to {self.install_dir}")
        return True


class ComfyUICustomNodesInstaller(ComponentInstaller):
    """Installer for ComfyUI custom nodes required by the VFX pipeline.

    These nodes can be updated via ComfyUI Manager or git pull.
    """

    CUSTOM_NODES = [
        {
            "name": "ComfyUI-VideoHelperSuite",
            "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        },
        {
            "name": "ComfyUI-Video-Depth-Anything",
            "url": "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git",
        },
        {
            "name": "ComfyUI-SAM3",
            "url": "https://github.com/PozzettiAndrea/ComfyUI-SAM3.git",
        },
        {
            "name": "ComfyUI_ProPainter_Nodes",
            "url": "https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git",
        },
    ]

    def __init__(self, install_dir: Optional[Path] = None, size_gb: float = 0.5):
        super().__init__("ComfyUI Custom Nodes", size_gb)
        self.install_dir = install_dir or INSTALL_DIR / "ComfyUI" / "custom_nodes"

    def check(self) -> bool:
        if not self.install_dir.exists():
            return False
        installed_count = sum(
            1 for node in self.CUSTOM_NODES
            if (self.install_dir / node["name"]).exists()
        )
        self.installed = installed_count == len(self.CUSTOM_NODES)
        return self.installed

    def install(self) -> bool:
        print(f"\nInstalling ComfyUI custom nodes to {self.install_dir}...")

        self.install_dir.mkdir(parents=True, exist_ok=True)

        all_success = True
        for node in self.CUSTOM_NODES:
            node_dir = self.install_dir / node["name"]

            if node_dir.exists():
                print_info(f"  {node['name']} already exists, skipping")
                continue

            print(f"  Cloning {node['name']}...")
            success, _ = run_command([
                "git", "clone", node["url"], str(node_dir)
            ], check=False)

            if success:
                print_success(f"  {node['name']} installed")
            else:
                print_error(f"  Failed to clone {node['name']}")
                all_success = False

        if all_success:
            self.installed = True
            print_success(f"All ComfyUI custom nodes installed to {self.install_dir}")
        else:
            print_warning("Some nodes failed to install")

        return all_success

    def get_node_list(self) -> list:
        return [node["name"] for node in self.CUSTOM_NODES]


class GSIRInstaller(ComponentInstaller):
    """Installer for GS-IR (Gaussian Splatting Inverse Rendering).

    GS-IR requires special handling due to CUDA submodules that need to be built.
    """

    def __init__(self, install_dir: Optional[Path] = None, size_gb: float = 2.0):
        super().__init__("GS-IR", size_gb)
        self.install_dir = install_dir or INSTALL_DIR / "GS-IR"
        self.conda_manager: Optional['CondaEnvironmentManager'] = None

    def set_conda_manager(self, conda_manager: 'CondaEnvironmentManager'):
        self.conda_manager = conda_manager

    def check(self) -> bool:
        if not self.install_dir.exists():
            return False
        train_py = self.install_dir / "train.py"
        gsir_module = self.install_dir / "gs-ir"
        self.installed = train_py.exists() and gsir_module.exists()
        return self.installed

    def _run_pip(self, args: list) -> bool:
        if self.conda_manager and self.conda_manager.conda_exe:
            cmd = [
                self.conda_manager.conda_exe, "run", "-n", self.conda_manager.env_name,
                "pip"
            ] + args
        else:
            cmd = [sys.executable, "-m", "pip"] + args
        success, _ = run_command(cmd, check=False)
        return success

    def install(self) -> bool:
        print(f"\nInstalling GS-IR from https://github.com/lzhnb/GS-IR.git...")

        self.install_dir.parent.mkdir(parents=True, exist_ok=True)

        if self.install_dir.exists():
            print_info("GS-IR directory exists, updating...")
            success, _ = run_command(["git", "-C", str(self.install_dir), "pull"])
            success2, _ = run_command(["git", "-C", str(self.install_dir), "submodule", "update", "--init", "--recursive"])
            if not success or not success2:
                print_warning("Failed to update GS-IR, continuing with existing version")
        else:
            success, _ = run_command([
                "git", "clone", "--recursive",
                "https://github.com/lzhnb/GS-IR.git",
                str(self.install_dir)
            ])
            if not success:
                print_error("Failed to clone GS-IR")
                return False

        print("  Installing kornia...")
        if not self._run_pip(["install", "kornia"]):
            print_warning("Failed to install kornia")

        print("  Installing nvdiffrast...")
        if not self._run_pip(["install", "--no-build-isolation", "git+https://github.com/NVlabs/nvdiffrast.git"]):
            print_warning("Failed to install nvdiffrast")

        if not shutil.which("nvcc"):
            print_warning("CUDA compiler (nvcc) not found - CUDA extensions may fail to build")
            print_info("Ensure CUDA toolkit is installed and nvcc is in PATH")

        diff_gauss = self.install_dir / "submodules" / "diff-gaussian-rasterization"
        simple_knn = self.install_dir / "submodules" / "simple-knn"

        print("  Building diff-gaussian-rasterization...")
        if diff_gauss.exists():
            if not self._run_pip(["install", "--no-build-isolation", str(diff_gauss)]):
                print_error("Failed to build diff-gaussian-rasterization")
                return False
        else:
            print_error(f"Submodule not found: {diff_gauss}")
            return False

        print("  Building simple-knn...")
        if simple_knn.exists():
            if not self._run_pip(["install", "--no-build-isolation", str(simple_knn)]):
                print_error("Failed to build simple-knn")
                return False
        else:
            print_error(f"Submodule not found: {simple_knn}")
            return False

        gsir_module = self.install_dir / "gs-ir"
        print("  Installing gs-ir module...")
        if gsir_module.exists():
            if not self._run_pip(["install", "-e", str(gsir_module)]):
                print_warning("Failed to install gs-ir module (may work without it)")
        else:
            print_warning("gs-ir module directory not found")

        self.installed = True
        print_success(f"GS-IR installed to {self.install_dir}")
        return True


class VideoMaMaInstaller(ComponentInstaller):
    """Installer for VideoMaMa (diffusion-based video matting).

    VideoMaMa requires:
    - Separate conda environment (videomama)
    - Stable Video Diffusion base model (~10GB)
    - VideoMaMa checkpoint (~1.5GB)
    - PyTorch with CUDA support

    Total disk space: ~12GB
    """

    def __init__(self, size_gb: float = 12.0):
        super().__init__("VideoMaMa", size_gb)
        self.tools_dir = INSTALL_DIR / "tools" / "VideoMaMa"
        self.models_dir = INSTALL_DIR / "models" / "VideoMaMa"
        self.svd_dir = self.models_dir / "stable-video-diffusion-img2vid-xt"
        self.checkpoint_dir = self.models_dir / "checkpoints" / "VideoMaMa"
        self.env_name = "videomama"

    def check(self) -> bool:
        repo_exists = self.tools_dir.exists() and (self.tools_dir / ".git").exists()
        svd_exists = self.svd_dir.exists() and bool(list(self.svd_dir.glob("*.safetensors")))
        checkpoint_exists = self.checkpoint_dir.exists() and bool(list(self.checkpoint_dir.glob("*")))

        conda_exe = self._find_conda()
        env_exists = False
        if conda_exe:
            success, output = run_command(
                [conda_exe, "env", "list"], check=False, capture=True
            )
            env_exists = success and self.env_name in output

        self.installed = repo_exists and svd_exists and checkpoint_exists and env_exists
        return self.installed

    def _find_conda(self) -> Optional[str]:
        import os
        for cmd in ["conda", "mamba"]:
            if shutil.which(cmd):
                return cmd
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe and Path(conda_exe).exists():
            return str(conda_exe)
        return None

    def install(self) -> bool:
        print(f"\nInstalling VideoMaMa (diffusion-based video matting)...")
        print_info("This will create a separate conda environment and download ~12GB of models")

        repo_root = INSTALL_DIR.parent
        install_script = repo_root / "scripts" / "video_mama_install.py"

        if not install_script.exists():
            print_error(f"VideoMaMa install script not found: {install_script}")
            return False

        print_info("Running VideoMaMa installation script...")
        success, _ = run_command(
            [sys.executable, str(install_script)],
            check=False,
            capture=False
        )

        if success:
            self.installed = True
            print_success("VideoMaMa installed successfully")
        else:
            print_error("VideoMaMa installation failed")

        return success
