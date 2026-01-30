"""Docker-specific installation components for the VFX pipeline.

This module provides Docker-specific functionality for the install wizard:
- DockerManager: Docker and NVIDIA Container Toolkit checks/installation
- DockerStateManager: Docker-specific state tracking
- DockerModelDownloader: Full model downloads for Docker workflow
- DockerWizard: Main Docker installation orchestrator

The Docker wizard downloads the same models as the conda wizard, to
<repo>/.vfx_pipeline/models/ for container mounting.

Usage:
    from install_wizard.docker import DockerWizard

    wizard = DockerWizard()
    wizard.interactive_install()
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .downloader import CheckpointDownloader
from .installers import ComfyUICustomNodesInstaller
from .platform import PlatformManager
from .utils import (
    Colors,
    ask_yes_no,
    check_command_available,
    format_size_gb,
    get_disk_space,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    tty_input,
)


class DockerStateManager:
    """Manages Docker installation state for resume/recovery capability."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load installation state from file."""
        if not self.state_file.exists():
            return self._create_initial_state()

        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)
            initial_state = self._create_initial_state()
            for key, value in initial_state.items():
                if key not in loaded_state:
                    loaded_state[key] = value
            return loaded_state
        except (json.JSONDecodeError, IOError):
            print_warning(f"Could not load state from {self.state_file}, creating new state")
            return self._create_initial_state()

    def _create_initial_state(self) -> Dict:
        """Create initial state structure."""
        return {
            "version": "1.0",
            "platform": None,
            "environment": None,
            "last_updated": None,
            "docker_installed": False,
            "nvidia_runtime_installed": False,
            "checkpoints": {},
            "image_built": False,
            "test_completed": False,
        }

    def save_state(self):
        """Save installation state to file."""
        self.state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        temp_file = self.state_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.state, indent=2, fp=f)
            temp_file.replace(self.state_file)
        except IOError as e:
            print_warning(f"Could not save state: {e}")

    def mark_checkpoint_downloaded(self, checkpoint_id: str, dest_dir: Path):
        """Mark a checkpoint as downloaded (compatible with CheckpointDownloader)."""
        self.state["checkpoints"][checkpoint_id] = {
            "downloaded": True,
            "path": str(dest_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_state()

    def is_checkpoint_downloaded(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint is downloaded."""
        return self.state["checkpoints"].get(checkpoint_id, {}).get("downloaded", False)

    def can_resume(self) -> bool:
        """Check if there's a resumable installation."""
        return (
            not self.state["docker_installed"] or
            not self.state["nvidia_runtime_installed"] or
            len(self.state["checkpoints"]) < 4 or
            not self.state["image_built"]
        )

    def clear_state(self):
        """Clear installation state for fresh start."""
        self.state = self._create_initial_state()
        self.save_state()


class DockerManager:
    """Manages Docker-specific operations."""

    REQUIRED_CUDA_VERSION = (12, 6)

    @staticmethod
    def get_driver_cuda_version() -> Tuple[Optional[Tuple[int, int]], str]:
        """Get the maximum CUDA version supported by the installed NVIDIA driver.

        Returns:
            Tuple of ((major, minor) version or None, message)
        """
        if not shutil.which("nvidia-smi"):
            return None, "nvidia-smi not found"

        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return None, "nvidia-smi failed to execute"

            import re
            cuda_match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
            if cuda_match:
                major = int(cuda_match.group(1))
                minor = int(cuda_match.group(2))
                return (major, minor), f"CUDA {major}.{minor}"
            return None, "Could not parse CUDA version from nvidia-smi"
        except Exception as e:
            return None, f"Error checking CUDA version: {e}"

    @classmethod
    def check_cuda_version_compatible(cls) -> Tuple[bool, str]:
        """Check if driver supports the required CUDA version for Docker image.

        Returns:
            Tuple of (compatible, message)
        """
        cuda_version, msg = cls.get_driver_cuda_version()
        if cuda_version is None:
            return False, msg

        required = cls.REQUIRED_CUDA_VERSION
        if cuda_version >= required:
            return True, f"Driver supports CUDA {cuda_version[0]}.{cuda_version[1]} (required: {required[0]}.{required[1]})"
        else:
            return False, (
                f"Driver only supports CUDA {cuda_version[0]}.{cuda_version[1]}, "
                f"but Docker image requires CUDA {required[0]}.{required[1]}. "
                f"Please update your NVIDIA driver (nvidia-driver-550 or newer)."
            )

    @staticmethod
    def check_nvidia_driver() -> Tuple[bool, str]:
        """Check if NVIDIA driver is installed.

        Returns:
            Tuple of (success, message)
        """
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    import re
                    vram_match = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', result.stdout)
                    if vram_match:
                        total_vram = int(vram_match.group(2))
                        return True, f"NVIDIA GPU with {total_vram}MB VRAM"
                    return True, "NVIDIA GPU detected"
                return False, "nvidia-smi found but failed to execute"
            except Exception as e:
                return False, f"Error checking GPU: {e}"
        return False, "nvidia-smi not found"

    @staticmethod
    def check_docker_installed() -> Tuple[bool, str]:
        """Check if Docker is installed and running.

        Returns:
            Tuple of (success, message)
        """
        if not shutil.which("docker"):
            return False, "Docker not found in PATH"

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, "Docker is running"
            else:
                return False, "Docker found but not running (try: sudo systemctl start docker)"
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out"
        except Exception as e:
            return False, f"Error checking Docker: {e}"

    @staticmethod
    def check_docker_compose() -> bool:
        """Check if docker-compose is available."""
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

        return shutil.which("docker-compose") is not None

    @staticmethod
    def check_nvidia_docker() -> Tuple[bool, str]:
        """Check if NVIDIA Docker runtime is available.

        Returns:
            Tuple of (success, message)
        """
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all",
                 "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return True, "GPU access verified in Docker containers"
            else:
                return False, "GPU test failed - check NVIDIA Container Toolkit installation"
        except subprocess.TimeoutExpired:
            return False, "GPU test timed out"
        except Exception as e:
            return False, f"GPU test failed: {e}"

    @staticmethod
    def get_docker_install_instructions(platform_name: str, environment: str) -> str:
        """Get Docker installation instructions for platform."""
        if platform_name == "linux" and environment == "native":
            return """
Install Docker on Linux:

    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    newgrp docker

Or follow official guide: https://docs.docker.com/engine/install/
"""
        elif platform_name == "linux" and environment == "wsl2":
            return """
Install Docker Desktop for Windows with WSL2 backend:

1. Download: https://www.docker.com/products/docker-desktop
2. Install with WSL2 backend enabled
3. In Docker Desktop settings:
   - Enable "Use the WSL 2 based engine"
   - Under "Resources > WSL Integration", enable your Ubuntu distro
4. Restart Docker Desktop

Then verify in WSL2:
    docker info
"""
        elif platform_name == "windows" and environment == "native":
            return """
Install Docker Desktop for Windows:

1. Download: https://www.docker.com/products/docker-desktop
2. Install Docker Desktop
3. During installation, choose:
   - "Use WSL 2 instead of Hyper-V" (Recommended)
   - OR Hyper-V (requires Windows Pro/Enterprise)
4. Restart computer when prompted
5. Open Docker Desktop and complete setup
6. Verify: docker --version

Warning: GPU Support on Windows
Docker Desktop on Windows cannot access NVIDIA GPUs directly.

For GPU-accelerated workloads:
  1. Install WSL2 (Ubuntu) - https://aka.ms/wsl
  2. Run this wizard inside WSL2 Ubuntu
  3. GPU will be accessible via NVIDIA Container Toolkit

Alternative: Use conda-based installation (no Docker)
  python scripts/install_wizard.py
"""
        else:
            return "Visit https://docs.docker.com/get-docker/"

    @staticmethod
    def get_nvidia_runtime_install_instructions(platform_name: str, environment: str) -> str:
        """Get NVIDIA Container Toolkit installation instructions."""
        if platform_name == "linux" and environment == "native":
            return """
Install NVIDIA Container Toolkit:

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\
        sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg --yes

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
        sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

Or follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
"""
        elif platform_name == "linux" and environment == "wsl2":
            return """
NVIDIA CUDA on WSL2:

Follow NVIDIA's official guide:
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

Key steps:
1. Install NVIDIA Driver on Windows (version 450.80.02 or later)
2. Install NVIDIA Container Toolkit in WSL2 (same as Linux native)

Verify in WSL2:
    nvidia-smi
"""
        else:
            return "See: https://github.com/NVIDIA/nvidia-docker"


class DockerCheckpointDownloader(CheckpointDownloader):
    """Checkpoint downloader configured for Docker's flat directory structure.

    Docker mounts models from <repo>/.vfx_pipeline/models/ into the container,
    using a flat structure rather than the nested ComfyUI layout.

    Inherits checkpoint source configs (URLs, auth, files) from base class
    and only overrides dest_dir_rel for Docker's directory layout.
    """

    DOCKER_DEST_DIRS: Dict[str, str] = {
        'sam3': 'sam3',
        'video_depth_anything': 'videodepthanything',
        'wham': 'wham',
        'smplx': 'smplx',
    }

    def __init__(self, models_dir: Path):
        super().__init__(base_dir=models_dir)
        self.CHECKPOINTS = self._build_docker_checkpoints()

    def _build_docker_checkpoints(self) -> Dict:
        """Build Docker checkpoints by copying base configs with Docker dest_dir_rel."""
        import copy
        docker_checkpoints = {}
        for checkpoint_id, dest_dir in self.DOCKER_DEST_DIRS.items():
            if checkpoint_id in CheckpointDownloader.CHECKPOINTS:
                config = copy.deepcopy(CheckpointDownloader.CHECKPOINTS[checkpoint_id])
                config['dest_dir_rel'] = dest_dir
                docker_checkpoints[checkpoint_id] = config
        return docker_checkpoints

    def get_total_size_gb(self, checkpoint_ids: List[str]) -> float:
        """Calculate total download size in GB."""
        total_mb = 0
        for cid in checkpoint_ids:
            if cid in self.CHECKPOINTS:
                for f in self.CHECKPOINTS[cid].get('files', []):
                    total_mb += f.get('size_mb', 0)
        return total_mb / 1024


class DockerWizard:
    """Main Docker installation wizard orchestrator."""

    def __init__(self):
        self.repo_root = Path(__file__).parent.parent.parent
        self.models_dir = self.repo_root / ".vfx_pipeline" / "models"
        self.projects_dir = self.repo_root.parent / "vfx_projects"
        self.state_file = self.models_dir / "docker_install_state.json"

        self.state_manager = DockerStateManager(self.state_file)
        self.docker_manager = DockerManager()
        self.checkpoint_downloader = DockerCheckpointDownloader(self.models_dir)
        self.custom_nodes_installer = ComfyUICustomNodesInstaller()

        platform_manager = PlatformManager()
        self.platform_name, self.environment, _ = platform_manager.detect_platform()

    def check_system_requirements(self) -> bool:
        """Check all system requirements."""
        print_header("System Requirements Check")

        all_checks_passed = True

        print_info(f"Platform: {self.platform_name} ({self.environment})")

        if self.platform_name == "macos":
            print_error("macOS is not supported for Docker-based pipeline")
            print()
            print("  Reason: Docker on macOS cannot access NVIDIA GPUs")
            print("  Solution: Use local conda installation instead")
            print()
            print("  Run: python scripts/install_wizard.py")
            return False

        if self.environment == "unsupported":
            print_error(f"Unsupported platform: {self.platform_name}")
            return False

        has_gpu, gpu_msg = self.docker_manager.check_nvidia_driver()
        if has_gpu:
            print_success(f"NVIDIA driver: {gpu_msg}")

            cuda_ok, cuda_msg = self.docker_manager.check_cuda_version_compatible()
            if cuda_ok:
                print_success(f"CUDA version: {cuda_msg}")
            else:
                print_error(f"CUDA version: {cuda_msg}")
                print()
                print("  Your NVIDIA driver is too old for the Docker image.")
                print("  Update your driver with:")
                print()
                print("    sudo apt update && sudo apt install nvidia-driver-550")
                print("    sudo reboot")
                print()
                all_checks_passed = False
        else:
            print_error(f"NVIDIA driver not found: {gpu_msg}")
            print_info("Install from: https://www.nvidia.com/Download/index.aspx")
            all_checks_passed = False

        docker_ok, docker_msg = self.docker_manager.check_docker_installed()
        if docker_ok:
            print_success(f"Docker: {docker_msg}")
        else:
            print_error(f"Docker: {docker_msg}")
            print()
            print(self.docker_manager.get_docker_install_instructions(self.platform_name, self.environment))
            all_checks_passed = False
            return all_checks_passed

        if self.docker_manager.check_docker_compose():
            print_success("docker-compose available")
        else:
            print_warning("docker-compose not found (will use docker build)")

        nvidia_ok, nvidia_msg = self.docker_manager.check_nvidia_docker()
        if nvidia_ok:
            print_success(f"NVIDIA Container Toolkit: {nvidia_msg}")
        else:
            print_warning(f"NVIDIA Container Toolkit: {nvidia_msg}")
            all_checks_passed = False

        available_gb, total_gb = get_disk_space(self.models_dir)
        if available_gb > 0:
            used_pct = ((total_gb - available_gb) / total_gb) * 100
            if available_gb >= 30:
                print_success(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
            elif available_gb >= 20:
                print_warning(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
                print_info("Full installation requires ~25 GB")
            else:
                print_error(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
                print_info("Insufficient space - installation requires ~25 GB")
                all_checks_passed = False
        else:
            print_warning("Could not check disk space")

        if check_command_available("git"):
            print_success("git available")
        else:
            print_error("git not found (required)")
            all_checks_passed = False

        return all_checks_passed

    def install_nvidia_container_toolkit(self) -> bool:
        """Install NVIDIA Container Toolkit with guided steps."""
        print_info("Starting NVIDIA Container Toolkit installation...")

        repo_dir = Path("/etc/apt/sources.list.d")
        if repo_dir.exists():
            conflicting = []
            for filename in ["libnvidia-container.list", "nvidia-docker.list", "nvidia-container-runtime.list"]:
                filepath = repo_dir / filename
                if filepath.exists():
                    conflicting.append(str(filepath))

            if conflicting:
                print_warning("Found conflicting repository files:")
                for f in conflicting:
                    print(f"    - {f}")
                print()
                if ask_yes_no("Remove these files to prevent conflicts?", default=True):
                    for f in conflicting:
                        try:
                            subprocess.run(["sudo", "rm", "-f", f], check=True)
                            print_success(f"Removed {f}")
                        except subprocess.CalledProcessError:
                            print_error(f"Failed to remove {f}")
                            return False

        try:
            print_info("Creating keyrings directory...")
            subprocess.run(["sudo", "mkdir", "-p", "/etc/apt/keyrings"], check=True)

            print_info("Adding NVIDIA GPG key...")
            gpg_cmd = (
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
                "sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg --yes"
            )
            subprocess.run(gpg_cmd, shell=True, check=True)

            print_info("Adding NVIDIA Container Toolkit repository...")
            repo_cmd = (
                "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
                "sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
                "sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null"
            )
            subprocess.run(repo_cmd, shell=True, check=True)

            print_info("Updating package lists...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)

            print_info("Installing nvidia-container-toolkit...")
            subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"], check=True)

            print_info("Configuring Docker runtime...")
            subprocess.run(["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"], check=True)

            print_info("Restarting Docker...")
            try:
                subprocess.run(["sudo", "systemctl", "restart", "docker"], check=True)
            except subprocess.CalledProcessError:
                subprocess.run(["sudo", "service", "docker", "restart"], check=False)

            time.sleep(3)

            print_success("NVIDIA Container Toolkit installed successfully")
            self.state_manager.state["nvidia_runtime_installed"] = True
            self.state_manager.save_state()
            return True

        except subprocess.CalledProcessError as e:
            print_error(f"Installation failed: {e}")
            return False
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            return False

    def install_nvidia_toolkit_if_needed(self) -> bool:
        """Install NVIDIA Container Toolkit if not present."""
        nvidia_ok, _ = self.docker_manager.check_nvidia_docker()
        if nvidia_ok:
            return True

        print()
        print_info("NVIDIA Container Toolkit not detected - installing automatically...")
        print()

        if self.environment == "wsl2":
            print("Detected: WSL2 environment")
            print("Note: GPU support in WSL2 requires Windows NVIDIA driver")
        else:
            print("Detected: Native Linux")

        print()

        if not self.install_nvidia_container_toolkit():
            print_error("NVIDIA Container Toolkit installation failed")
            print()
            print("Manual installation instructions:")
            print("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
            print()
            return False

        print_info("Verifying installation...")
        nvidia_ok, nvidia_msg = self.docker_manager.check_nvidia_docker()
        if nvidia_ok:
            print_success(nvidia_msg)
            return True
        else:
            print_error(f"GPU test failed: {nvidia_msg}")
            print()
            print("Troubleshooting steps:")
            print("  1. Check Docker daemon configuration:")
            print("     cat /etc/docker/daemon.json")
            print()
            print("  2. Ensure it contains the nvidia runtime:")
            print('     {"runtimes": {"nvidia": {"path": "nvidia-container-runtime"}}}')
            print()
            print("  3. Restart Docker:")
            print("     sudo systemctl restart docker")
            print()
            return False

    def setup_credentials(self) -> None:
        """Prompt user to set up credentials for authenticated downloads.

        Sets up SMPL.login.dat for SMPL-X body models (required for mocap).
        Only prompts if credentials don't exist and models aren't downloaded.
        """
        smpl_creds_file = self.repo_root / "SMPL.login.dat"

        if smpl_creds_file.exists():
            print_success("SMPL-X credentials file found")
            return

        smplx_dir = self.models_dir / "smplx"
        if smplx_dir.exists() and any(smplx_dir.iterdir()):
            print_success("SMPL-X models already downloaded")
            return

        print_header("SMPL-X Credentials Required")
        print("SMPL-X body models require registration for download.")
        print("These models are required for the motion capture (mocap) stage.")
        print()
        print("Registration: https://smpl-x.is.tue.mpg.de/register.php")
        print()
        print("If you have registered, enter your credentials below.")
        print("Press Enter to skip (mocap stage will not be available).")
        print()

        email = tty_input("SMPL-X email (or Enter to skip): ").strip()
        if not email or '@' not in email:
            print_info("Skipped - SMPL-X models will not be downloaded")
            return

        password = tty_input("SMPL-X password: ").strip()
        if not password:
            print_info("Password required - SMPL-X models will not be downloaded")
            return

        with open(smpl_creds_file, 'w') as f:
            f.write(email + '\n')
            f.write(password + '\n')
        smpl_creds_file.chmod(0o600)
        print_success(f"Credentials saved to {smpl_creds_file}")

    def build_docker_image(self, force_rebuild: bool = False) -> bool:
        """Build the Docker image."""
        if self.state_manager.state.get("image_built") and not force_rebuild:
            print_success("Docker image already built")
            return True

        print_header("Building Docker Image")
        print_info("First build takes 10-15 minutes (cached afterwards)...")

        try:
            if self.docker_manager.check_docker_compose():
                print_info("Using docker compose...")
                subprocess.run(
                    ["docker", "compose", "build"],
                    cwd=self.repo_root,
                    check=True
                )
            else:
                print_info("Using docker build...")
                subprocess.run(
                    ["docker", "build", "-t", "vfx-ingest:latest", "."],
                    cwd=self.repo_root,
                    check=True
                )

            print_success("Docker image built successfully")
            self.state_manager.state["image_built"] = True
            self.state_manager.save_state()
            return True

        except subprocess.CalledProcessError:
            print_error("Docker image build failed")
            return False

    def run_test_pipeline(self, force_rerun: bool = False) -> bool:
        """Download test video and run test pipeline."""
        if self.state_manager.state.get("test_completed") and not force_rerun:
            print_success("Test pipeline already completed")
            return True

        print_header("Running Test Pipeline")

        test_script = self.repo_root / "tests" / "fixtures" / "download_football.sh"
        if not test_script.exists():
            print_warning("Test video download script not found, skipping test")
            return True

        print_info("Downloading test video (Football CIF, ~2MB)...")
        try:
            subprocess.run(["bash", str(test_script)], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print_warning("Test video download failed, skipping test")
            return True

        test_video = self.repo_root / "tests" / "fixtures" / "football_short.mp4"
        projects_dir = self.projects_dir
        projects_dir.mkdir(parents=True, exist_ok=True)

        if test_video.exists():
            shutil.copy2(test_video, projects_dir / "football_short.mp4")
            print_success(f"Test video copied to {projects_dir}/")
        else:
            print_warning("Test video not found, skipping test")
            return True

        print_info("Running depth analysis test (~1 minute)...")

        run_script = self.repo_root / "scripts" / "run_docker.sh"
        try:
            subprocess.run(
                [
                    "bash", str(run_script),
                    "--name", "FootballTest",
                    "--stages", "depth",
                    "/workspace/projects/football_short.mp4"
                ],
                cwd=self.repo_root,
                check=True
            )

            output_dir = projects_dir / "FootballTest" / "depth"
            if output_dir.exists() and list(output_dir.glob("*.png")):
                print_success("Test pipeline completed successfully!")
                print_info(f"Output: {output_dir}")
                self.state_manager.state["test_completed"] = True
                self.state_manager.save_state()
                return True
            else:
                print_warning("Test completed but no output found")
                return False

        except subprocess.CalledProcessError:
            print_error("Test pipeline failed")
            return False

    def print_post_install_instructions(self, downloaded_smplx: bool):
        """Print post-installation usage instructions."""
        print_header("Next Steps")

        print_info("Quick start:")
        print(f"  1. Copy your video to {self.projects_dir}/")
        print("  2. Run:")
        print("     bash scripts/run_docker.sh --name MyProject /workspace/projects/video.mp4")
        print()

        print_info("Available stages:")
        print("  --stages depth              # Depth maps only")
        print("  --stages segmentation       # Segmentation only")
        print("  --stages mocap              # Motion capture only")
        print("  --stages all                # All stages")
        print()

        print_info("Documentation:")
        print("  - QUICKSTART.md - Quick start guide")
        print("  - docs/README-DOCKER.md - Full Docker documentation")
        print()

        if not downloaded_smplx:
            smplx_dir = self.models_dir / "smplx"
            print_warning("SMPL-X models not downloaded (required for mocap stage)")
            print("  SMPL-X requires registration:")
            print("  1. Register at https://smpl-x.is.tue.mpg.de/")
            print("  2. Create SMPL.login.dat with your credentials")
            print("  3. Re-run: python scripts/install_wizard.py --docker")
            print(f"  Or manually extract to: {smplx_dir}/")
            print()

    def interactive_install(
        self,
        check_only: bool = False,
        run_test: bool = False,
        yolo: bool = False,
        resume: bool = False
    ):
        """Full installation flow.

        The wizard proceeds automatically with full installation.
        Only prompts for SMPL-X credentials if not present.

        Args:
            check_only: Only check prerequisites, don't install
            run_test: Run the test pipeline after installation
            yolo: Ignored (kept for backward compatibility)
            resume: Ignored (kept for backward compatibility)
        """
        print_header("VFX Ingest Platform - Docker Installation Wizard")

        if not self.check_system_requirements():
            if check_only:
                print_error("\nPrerequisites not satisfied")
                sys.exit(1)
            else:
                print_error("\nPlease install missing components and run this wizard again")
                sys.exit(1)

        if check_only:
            print_success("\nAll prerequisites satisfied!")
            sys.exit(0)

        self.state_manager.state["platform"] = self.platform_name
        self.state_manager.state["environment"] = self.environment
        self.state_manager.state["docker_installed"] = True
        self.state_manager.save_state()

        nvidia_ok, _ = self.docker_manager.check_nvidia_docker()
        if not nvidia_ok and not self.state_manager.state.get("nvidia_runtime_installed"):
            if not self.install_nvidia_toolkit_if_needed():
                print_error("NVIDIA Container Toolkit not available")
                print_info("Install it manually and run this wizard again")
                sys.exit(1)
            print_success("NVIDIA Container Toolkit ready")

        self.setup_credentials()

        checkpoints_to_download = ['sam3', 'video_depth_anything', 'wham']
        download_smplx = False

        smpl_creds_file = self.repo_root / "SMPL.login.dat"
        if smpl_creds_file.exists():
            checkpoints_to_download.append('smplx')
            download_smplx = True

        print_header("Installation Summary")
        model_size = self.checkpoint_downloader.get_total_size_gb(checkpoints_to_download)
        image_size = 8.0
        total_size = model_size + image_size
        working_space = 10.0

        print(f"  {'Models':30s} {format_size_gb(model_size):>10s}")
        print(f"  {'Docker image':30s} {format_size_gb(image_size):>10s}")
        print("  " + "-" * 42)
        print(f"  {'Total':30s} {format_size_gb(total_size):>10s}")
        print(f"  {'Working space (per project)':30s} ~{format_size_gb(working_space):>9s}")

        available_gb, _ = get_disk_space(self.models_dir)
        if available_gb > 0:
            print(f"\n  Available disk space: {format_size_gb(available_gb)}")

            if available_gb >= total_size + working_space:
                print_success("Sufficient disk space available")
            elif available_gb >= total_size:
                print_warning("Sufficient for installation, but limited working space")
            else:
                print_error(f"Insufficient disk space (need {format_size_gb(total_size)})")
                sys.exit(1)

        print_header("Downloading Models")
        print(f"Total download size: ~{format_size_gb(model_size)}")
        print()

        for checkpoint_id in checkpoints_to_download:
            if not self.checkpoint_downloader.download_checkpoint(
                checkpoint_id,
                self.state_manager,
                self.repo_root
            ):
                if checkpoint_id == 'smplx':
                    print_warning("SMPL-X download failed - mocap stage will not be available")
                    download_smplx = False
                else:
                    print_error(f"Failed to download {checkpoint_id}")
                    sys.exit(1)

        print_header("Installing ComfyUI Custom Nodes")
        if not self.custom_nodes_installer.check():
            if not self.custom_nodes_installer.install():
                print_warning("Some ComfyUI custom nodes failed to install")
                print_info("You can install them manually via ComfyUI Manager")
        else:
            print_success("ComfyUI custom nodes already installed")

        if not self.build_docker_image():
            print_error("Docker image build failed")
            sys.exit(1)

        if run_test:
            self.run_test_pipeline()

        self.print_post_install_instructions(download_smplx)

        print_header("Installation Complete!")
        print_success("Docker-based VFX pipeline is ready to use!")
