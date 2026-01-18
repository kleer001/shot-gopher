#!/usr/bin/env python3
"""Interactive installation wizard for Docker-based VFX pipeline.

Rebuilt from scratch based on the original install_wizard architecture,
incorporating quality of life features and robust fallback mechanisms.

Automates Docker setup for Linux and Windows/WSL2:
- Platform detection and validation
- Docker installation (guided)
- NVIDIA Container Toolkit installation (automated with fallbacks)
- Model downloads (with retry and fallback mechanisms)
- Docker image build
- Test pipeline execution
- Resume capability for interrupted installations

Usage:
    python scripts/install_wizard_docker.py
    python scripts/install_wizard_docker.py --check-only
    python scripts/install_wizard_docker.py --skip-test
    python scripts/install_wizard_docker.py --yolo  # Non-interactive full install
    python scripts/install_wizard_docker.py --resume # Resume interrupted install
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import utilities from the original install_wizard package
sys.path.insert(0, str(Path(__file__).parent))
from install_wizard.utils import (
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
    """Manages installation state for resume/recovery capability."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
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
            "platform": None,
            "environment": None,
            "last_updated": None,
            "docker_installed": False,
            "nvidia_runtime_installed": False,
            "models_downloaded": {},
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

    def mark_model_downloaded(self, model_name: str):
        """Mark a model as downloaded."""
        self.state["models_downloaded"][model_name] = {
            "downloaded": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_state()

    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is downloaded."""
        return self.state["models_downloaded"].get(model_name, {}).get("downloaded", False)

    def can_resume(self) -> bool:
        """Check if there's a resumable installation."""
        return (
            not self.state["docker_installed"] or
            not self.state["nvidia_runtime_installed"] or
            len(self.state["models_downloaded"]) < 4 or  # 4 required models
            not self.state["image_built"]
        )

    def clear_state(self):
        """Clear installation state for fresh start."""
        self.state = self._create_initial_state()
        self.save_state()


class DockerManager:
    """Manages Docker-specific operations."""

    @staticmethod
    def detect_platform() -> Tuple[str, str]:
        """Detect operating system and environment.

        Returns:
            Tuple of (platform, environment) where:
            - platform: 'linux', 'windows', 'macos'
            - environment: 'native', 'wsl2', 'unsupported'
        """
        system = platform.system().lower()

        if system == "linux":
            try:
                with open("/proc/version", "r") as f:
                    version = f.read().lower()
                    if "microsoft" in version or "wsl" in version:
                        return "linux", "wsl2"
            except FileNotFoundError:
                pass
            return "linux", "native"
        elif system == "darwin":
            return "macos", "unsupported"
        elif system == "windows":
            return "windows", "native"
        else:
            return system, "unsupported"

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


class ModelDownloader:
    """Handles model downloads with fallback mechanisms."""

    MODELS = {
        'sam3': {
            'name': 'SAM3 (Segment Anything Model 3)',
            'method': 'huggingface',
            'repo_id': '1038lab/sam3',
            'size_gb': 3.2,
        },
        'videodepthanything': {
            'name': 'Video Depth Anything',
            'method': 'huggingface',
            'repo_id': 'depth-anything/Video-Depth-Anything-Small',
            'size_gb': 0.12,
        },
        'wham': {
            'name': 'WHAM (4D Human MoCap)',
            'method': 'huggingface',
            'repo_id': 'yohanshin/WHAM',
            'filename': 'wham_vit_w_3dpw.pth.tar',
            'size_gb': 1.2,
        },
        'matanyone': {
            'name': 'MatAnyone (Matte Refinement)',
            'method': 'direct',
            'url': 'https://github.com/FuouM/ComfyUI-MatAnyone/releases/download/v1.0/matanyone.pth',
            'filename': 'matanyone.pth',
            'size_gb': 0.14,
        },
    }

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def _ensure_huggingface_hub(self) -> bool:
        """Ensure huggingface_hub is installed with fallback mechanisms."""
        try:
            import huggingface_hub
            return True
        except ImportError:
            print_info("Installing huggingface_hub...")

            install_methods = [
                [sys.executable, '-m', 'pip', 'install', 'huggingface_hub'],
                [sys.executable, '-m', 'pip', 'install', '--user', 'huggingface_hub'],
                [sys.executable, '-m', 'pip', 'install', '--break-system-packages', 'huggingface_hub'],
            ]

            for method in install_methods:
                result = subprocess.run(method, capture_output=True, text=True)
                if result.returncode == 0:
                    try:
                        import huggingface_hub
                        print_success("huggingface_hub installed")
                        return True
                    except ImportError:
                        continue

            print_error("Failed to install huggingface_hub")
            return False

    def _download_from_huggingface(
        self,
        repo_id: str,
        dest_dir: Path,
        filename: Optional[str] = None
    ) -> bool:
        """Download model from HuggingFace with retry logic."""
        if not self._ensure_huggingface_hub():
            return False

        try:
            from huggingface_hub import snapshot_download, hf_hub_download

            dest_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Downloading from HuggingFace: {repo_id}")

            if filename:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(dest_dir),
                    local_dir_use_symlinks=False,
                )
            else:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(dest_dir),
                    local_dir_use_symlinks=False,
                )

            print_success(f"Downloaded {repo_id}")
            return True

        except Exception as e:
            print_error(f"HuggingFace download failed: {e}")
            return False

    def _download_direct(self, url: str, dest: Path) -> bool:
        """Download file directly with wget/curl fallback."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading from {url}")
        print(f"  -> {dest}")

        methods = [
            (['wget', '-O', str(dest), url], "wget"),
            (['curl', '-L', '-o', str(dest), url], "curl"),
        ]

        for cmd, name in methods:
            if not shutil.which(cmd[0]):
                continue

            try:
                result = subprocess.run(cmd, timeout=600)
                if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                    print_success(f"Downloaded via {name}")
                    return True
                if dest.exists():
                    dest.unlink()
            except subprocess.TimeoutExpired:
                print_warning(f"{name} timed out")
                if dest.exists():
                    dest.unlink()
            except Exception as e:
                print_warning(f"{name} failed: {e}")
                if dest.exists():
                    dest.unlink()

        print_error("All download methods failed")
        return False

    def download_model(
        self,
        model_id: str,
        state_manager: Optional[DockerStateManager] = None
    ) -> bool:
        """Download a specific model with progress tracking."""
        if model_id not in self.MODELS:
            print_warning(f"Unknown model: {model_id}")
            return False

        model_info = self.MODELS[model_id]
        dest_dir = self.models_dir / model_id

        if state_manager and state_manager.is_model_downloaded(model_id):
            if dest_dir.exists() and any(dest_dir.iterdir()):
                print_success(f"{model_info['name']} already downloaded")
                return True
            else:
                print_warning(f"{model_info['name']} marked downloaded but files missing, re-downloading...")

        print(f"\n{Colors.BOLD}Downloading {model_info['name']}...{Colors.ENDC}")
        print(f"  Size: ~{format_size_gb(model_info['size_gb'])}")

        success = False
        if model_info['method'] == 'huggingface':
            success = self._download_from_huggingface(
                model_info['repo_id'],
                dest_dir,
                model_info.get('filename')
            )
        elif model_info['method'] == 'direct':
            success = self._download_direct(
                model_info['url'],
                dest_dir / model_info['filename']
            )

        if success and state_manager:
            state_manager.mark_model_downloaded(model_id)

        return success

    def download_all(
        self,
        models: Optional[List[str]] = None,
        state_manager: Optional[DockerStateManager] = None
    ) -> bool:
        """Download all specified models (or all available models)."""
        if models is None:
            models = list(self.MODELS.keys())

        print_header("Downloading Models")

        total_size = sum(self.MODELS[m]['size_gb'] for m in models if m in self.MODELS)
        print(f"Total download size: ~{format_size_gb(total_size)}")
        print()

        success = True
        for model_id in models:
            if not self.download_model(model_id, state_manager):
                print_error(f"Failed to download {model_id}")
                success = False

        return success


class DockerWizard:
    """Main Docker installation wizard orchestrator."""

    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.models_dir = Path.home() / ".vfx_pipeline" / "models"
        self.state_file = self.models_dir / "docker_install_state.json"

        self.state_manager = DockerStateManager(self.state_file)
        self.docker_manager = DockerManager()
        self.model_downloader = ModelDownloader(self.models_dir)

        self.platform_name, self.environment = self.docker_manager.detect_platform()

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

    def offer_nvidia_toolkit_installation(self) -> bool:
        """Offer to install NVIDIA Container Toolkit if not present."""
        nvidia_ok, _ = self.docker_manager.check_nvidia_docker()
        if nvidia_ok:
            return True

        print()
        print("The NVIDIA Container Toolkit is required for GPU access in Docker containers.")
        print()

        if self.environment == "wsl2":
            print("Detected: WSL2 environment")
            print()
            print("Installation will:")
            print("  - Add NVIDIA package repository")
            print("  - Install nvidia-container-toolkit")
            print("  - Configure Docker to use NVIDIA runtime")
            print("  - Restart Docker service")
            print()
            print("Note: GPU support in WSL2 requires Windows NVIDIA driver")
        else:
            print("Detected: Native Linux")
            print()
            print("Installation will:")
            print("  - Clean up any conflicting repository configurations")
            print("  - Add NVIDIA package repository with GPG key")
            print("  - Install nvidia-container-toolkit package")
            print("  - Configure Docker daemon for NVIDIA runtime")
            print("  - Restart Docker service")

        print()
        if not ask_yes_no("Would you like to install NVIDIA Container Toolkit now?", default=True):
            print_warning("Skipping NVIDIA Container Toolkit installation")
            print()
            print("You can install it manually later. See:")
            print("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
            print()
            return False

        if not self.install_nvidia_container_toolkit():
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

    def build_docker_image(self) -> bool:
        """Build the Docker image."""
        if self.state_manager.state.get("image_built"):
            print_info("Docker image already built")
            if not ask_yes_no("Rebuild image?", default=False):
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

    def run_test_pipeline(self) -> bool:
        """Download test video and run test pipeline."""
        if self.state_manager.state.get("test_completed"):
            print_info("Test already completed")
            if not ask_yes_no("Run test again?", default=False):
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
        projects_dir = Path.home() / "VFX-Projects"
        projects_dir.mkdir(parents=True, exist_ok=True)

        if test_video.exists():
            shutil.copy2(test_video, projects_dir / "football_short.mp4")
            print_success("Test video copied to ~/VFX-Projects/")
        else:
            print_warning("Test video not found, skipping test")
            return True

        print_info("Running depth analysis test (~1 minute)...")

        run_script = self.repo_root / "scripts" / "run_docker.sh"
        try:
            subprocess.run(
                [
                    "bash", str(run_script),
                    "--input", "/workspace/projects/football_short.mp4",
                    "--name", "FootballTest",
                    "--stages", "depth"
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

    def interactive_install(
        self,
        check_only: bool = False,
        skip_test: bool = False,
        yolo: bool = False,
        resume: bool = False
    ):
        """Interactive installation flow."""
        print_header("VFX Ingest Platform - Docker Installation Wizard")

        if yolo:
            print_info("YOLO mode: Full install with auto-yes")
            print()

        if not resume and self.state_manager.can_resume():
            print_warning("Found incomplete installation from previous run")
            if ask_yes_no("Resume previous installation?", default=True):
                resume = True
            else:
                if ask_yes_no("Start fresh (clear previous state)?", default=False):
                    self.state_manager.clear_state()

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
            if yolo or self.offer_nvidia_toolkit_installation():
                print_success("NVIDIA Container Toolkit ready")
            else:
                print_error("NVIDIA Container Toolkit not available")
                print_info("Install it manually and run this wizard again")
                sys.exit(1)

        print_header("Component Selection")

        models_to_download = ['sam3', 'videodepthanything', 'wham', 'matanyone']
        download_models = True
        build_image = True
        run_test = not skip_test

        if not yolo:
            if not ask_yes_no("Download required models (~5GB)?", default=True):
                download_models = False

            if not ask_yes_no("Build Docker image?", default=True):
                build_image = False

            if download_models and build_image:
                if not ask_yes_no("Run test pipeline after installation?", default=True):
                    run_test = False

        if download_models or build_image:
            print_header("Disk Space Estimate")

            total_size = 0.0
            if download_models:
                model_size = sum(
                    self.model_downloader.MODELS[m]['size_gb']
                    for m in models_to_download
                )
                total_size += model_size
                print(f"  {'Models':30s} {format_size_gb(model_size):>10s}")

            if build_image:
                image_size = 8.0
                total_size += image_size
                print(f"  {'Docker image':30s} {format_size_gb(image_size):>10s}")

            working_space = 10.0
            print("  " + "-" * 42)
            print(f"  {'Total':30s} {format_size_gb(total_size):>10s}")
            print(f"  {'Working space (per project)':30s} ~{format_size_gb(working_space):>9s}")
            print(f"  {'Recommended total':30s} ~{format_size_gb(total_size + working_space):>9s}")

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

            print()
            if not yolo:
                if not ask_yes_no("Proceed with installation?", default=True):
                    print_info("Installation cancelled")
                    sys.exit(0)

        if download_models:
            if not self.model_downloader.download_all(models_to_download, self.state_manager):
                print_error("Model download failed")
                sys.exit(1)

        if build_image:
            if not self.build_docker_image():
                print_error("Docker image build failed")
                sys.exit(1)

        if run_test:
            self.run_test_pipeline()

        self.print_post_install_instructions()

        print_header("Installation Complete!")
        print_success("Docker-based VFX pipeline is ready to use!")

    def print_post_install_instructions(self):
        """Print post-installation usage instructions."""
        print_header("Next Steps")

        print_info("Quick start:")
        print("  1. Copy your video to ~/VFX-Projects/")
        print("  2. Run:")
        print("     ./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name MyProject")
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

        smplx_dir = self.models_dir / "smplx"
        if not (smplx_dir.exists() and any(smplx_dir.iterdir())):
            print_warning("SMPL-X models not downloaded (optional, required for mocap stage)")
            print("  SMPL-X requires registration:")
            print("  1. Register at https://smpl-x.is.tue.mpg.de/")
            print("  2. Download SMPL-X models")
            print(f"  3. Extract to {smplx_dir}/")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Docker-based VFX Pipeline Installation Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/install_wizard_docker.py                  # Interactive install
    python scripts/install_wizard_docker.py --check-only     # Check prerequisites only
    python scripts/install_wizard_docker.py --skip-test      # Skip test pipeline
    python scripts/install_wizard_docker.py --yolo           # Non-interactive full install
    python scripts/install_wizard_docker.py --resume         # Resume interrupted install
"""
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites without installing"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test pipeline execution"
    )
    parser.add_argument(
        "--yolo", "-y",
        action="store_true",
        help="Non-interactive full install with auto-yes"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume previous interrupted installation"
    )

    args = parser.parse_args()

    wizard = DockerWizard()
    wizard.interactive_install(
        check_only=args.check_only,
        skip_test=args.skip_test,
        yolo=args.yolo,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
