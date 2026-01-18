#!/usr/bin/env python3
"""Interactive installation wizard for Docker-based VFX pipeline.

Automates Docker setup for Linux and Windows/WSL2:
- Platform detection and validation
- Docker installation (guided)
- NVIDIA Container Toolkit installation (guided)
- Model downloads
- Docker image build
- Test pipeline execution

Usage:
    python scripts/install_wizard_docker.py
    python scripts/install_wizard_docker.py --check-only
    python scripts/install_wizard_docker.py --skip-test
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
NC = "\033[0m"  # No Color


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{BOLD}{BLUE}{'='*60}{NC}")
    print(f"{BOLD}{BLUE}{text}{NC}")
    print(f"{BOLD}{BLUE}{'='*60}{NC}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {text}{NC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{NC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}✗ {text}{NC}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{BLUE}→ {text}{NC}")


def run_command(
    cmd: list[str],
    check: bool = True,
    capture_output: bool = True,
    timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    """Run command and return result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return e
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        sys.exit(1)


def detect_platform() -> Tuple[str, str]:
    """Detect operating system and environment.

    Returns:
        Tuple of (platform, environment) where:
        - platform: 'linux', 'windows', 'macos'
        - environment: 'native', 'wsl2', 'unsupported'
    """
    system = platform.system().lower()

    if system == "linux":
        # Check if running in WSL2
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
        # Should not reach here if running in WSL
        return "windows", "native"
    else:
        return system, "unsupported"


def check_nvidia_driver() -> bool:
    """Check if NVIDIA driver is installed."""
    if shutil.which("nvidia-smi"):
        try:
            result = run_command(["nvidia-smi"], check=False)
            return result.returncode == 0
        except Exception:
            return False
    return False


def check_docker_installed() -> bool:
    """Check if Docker is installed and running."""
    if not shutil.which("docker"):
        return False

    try:
        result = run_command(["docker", "info"], check=False, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def check_docker_compose_installed() -> bool:
    """Check if docker-compose is available."""
    # Check docker compose (new style)
    try:
        result = run_command(["docker", "compose", "version"], check=False)
        if result.returncode == 0:
            return True
    except Exception:
        pass

    # Check docker-compose (old style)
    if shutil.which("docker-compose"):
        return True

    return False


def check_nvidia_docker() -> bool:
    """Check if NVIDIA Docker runtime is available."""
    try:
        result = run_command(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
            check=False,
            timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def get_install_instructions_docker(platform_name: str, environment: str) -> str:
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
Install Docker Desktop for Windows:

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


def get_install_instructions_nvidia_runtime(platform_name: str, environment: str) -> str:
    """Get NVIDIA Container Toolkit installation instructions."""
    if platform_name == "linux" and environment == "native":
        return """
Install NVIDIA Container Toolkit:

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \\
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
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
2. No need to install CUDA toolkit in WSL2 - uses Windows driver

Verify in WSL2:
    nvidia-smi
"""
    else:
        return "See: https://github.com/NVIDIA/nvidia-docker"


def download_models(models_dir: Path) -> bool:
    """Download required models."""
    print_info("Downloading models (15-20GB, may take 10-20 minutes)...")

    download_script = Path(__file__).parent / "download_models.sh"

    if not download_script.exists():
        print_error(f"Download script not found: {download_script}")
        return False

    try:
        # Set environment variable for model directory
        env = os.environ.copy()
        env["VFX_MODELS_DIR"] = str(models_dir)

        result = subprocess.run(
            [str(download_script)],
            env=env,
            check=True,
            text=True
        )

        print_success("Models downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Model download failed")
        return False


def verify_models(models_dir: Path) -> bool:
    """Verify downloaded models."""
    verify_script = Path(__file__).parent / "verify_models.py"

    if not verify_script.exists():
        print_warning("Verify script not found, skipping verification")
        return True

    try:
        env = os.environ.copy()
        env["VFX_MODELS_DIR"] = str(models_dir)

        result = subprocess.run(
            [sys.executable, str(verify_script)],
            env=env,
            capture_output=True,
            text=True,
            check=False
        )

        # Print output
        print(result.stdout)

        if result.returncode == 0:
            print_success("Model verification passed")
            return True
        else:
            print_warning("Some models may be missing (see above)")
            return False
    except Exception as e:
        print_warning(f"Model verification failed: {e}")
        return False


def build_docker_image() -> bool:
    """Build the Docker image."""
    print_info("Building Docker image (first build takes 10-15 minutes)...")

    repo_root = Path(__file__).parent.parent

    try:
        # Use docker compose if available
        result = run_command(
            ["docker", "compose", "version"],
            check=False,
            capture_output=True
        )

        if result.returncode == 0:
            # Use docker compose
            subprocess.run(
                ["docker", "compose", "build"],
                cwd=repo_root,
                check=True
            )
        else:
            # Fallback to docker build
            subprocess.run(
                ["docker", "build", "-t", "vfx-ingest:latest", "."],
                cwd=repo_root,
                check=True
            )

        print_success("Docker image built successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Docker image build failed")
        return False


def run_test_pipeline(repo_root: Path) -> bool:
    """Download test video and run test pipeline."""
    print_info("Running test pipeline...")

    # Download test video
    test_script = repo_root / "tests" / "fixtures" / "download_football.sh"

    if not test_script.exists():
        print_warning("Test video download script not found, skipping test")
        return True

    print_info("Downloading test video (Football CIF, ~2MB)...")
    try:
        subprocess.run([str(test_script)], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print_warning("Test video download failed, skipping test")
        return True

    # Copy to projects directory
    test_video = repo_root / "tests" / "fixtures" / "football_short.mp4"
    projects_dir = Path.home() / "VFX-Projects"
    projects_dir.mkdir(parents=True, exist_ok=True)

    if test_video.exists():
        shutil.copy2(test_video, projects_dir / "football_short.mp4")
        print_success("Test video copied to ~/VFX-Projects/")
    else:
        print_warning("Test video not found, skipping test")
        return True

    # Run test pipeline
    print_info("Running depth analysis test (takes ~1 minute)...")

    run_script = repo_root / "scripts" / "run_docker.sh"

    try:
        subprocess.run(
            [
                str(run_script),
                "--input", "/workspace/projects/football_short.mp4",
                "--name", "FootballTest",
                "--stages", "depth"
            ],
            cwd=repo_root,
            check=True
        )

        # Check output
        output_dir = projects_dir / "FootballTest" / "depth"
        if output_dir.exists() and list(output_dir.glob("*.png")):
            print_success("Test pipeline completed successfully!")
            print_info(f"Output: {output_dir}")
            return True
        else:
            print_warning("Test completed but no output found")
            return False
    except subprocess.CalledProcessError:
        print_error("Test pipeline failed")
        return False


def main():
    """Main installation wizard."""
    parser = argparse.ArgumentParser(
        description="Docker-based VFX Pipeline Installation Wizard"
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
        "--models-dir",
        type=Path,
        default=Path.home() / ".vfx_pipeline" / "models",
        help="Models directory (default: ~/.vfx_pipeline/models)"
    )

    args = parser.parse_args()

    print_header("VFX Ingest Platform - Docker Installation Wizard")

    # Detect platform
    platform_name, environment = detect_platform()

    print_info(f"Platform: {platform_name} ({environment})")

    # Check if platform is supported
    if platform_name == "macos":
        print_error("macOS is not supported for Docker-based pipeline")
        print()
        print("  Reason: Docker on macOS cannot access NVIDIA GPUs")
        print("  Solution: Use local conda installation instead")
        print()
        print("  Run: python scripts/install_wizard.py")
        sys.exit(1)

    if environment == "unsupported":
        print_error(f"Unsupported platform: {platform_name}")
        sys.exit(1)

    # Check prerequisites
    print_header("Checking Prerequisites")

    all_checks_passed = True

    # Check NVIDIA driver
    print_info("Checking NVIDIA driver...")
    if check_nvidia_driver():
        print_success("NVIDIA driver installed")
    else:
        print_error("NVIDIA driver not found")
        print()
        print("  Install NVIDIA driver for your GPU:")
        print("  https://www.nvidia.com/Download/index.aspx")
        print()
        all_checks_passed = False

    # Check Docker
    print_info("Checking Docker...")
    if check_docker_installed():
        print_success("Docker installed and running")
    else:
        print_error("Docker not installed or not running")
        print()
        print(get_install_instructions_docker(platform_name, environment))
        all_checks_passed = False

    # Check docker-compose
    print_info("Checking docker-compose...")
    if check_docker_compose_installed():
        print_success("docker-compose available")
    else:
        print_warning("docker-compose not found (will use docker build)")

    # Check NVIDIA Docker runtime
    if check_docker_installed():
        print_info("Checking NVIDIA Container Toolkit...")
        if check_nvidia_docker():
            print_success("NVIDIA Docker runtime working")
        else:
            print_error("NVIDIA Docker runtime not available")
            print()
            print(get_install_instructions_nvidia_runtime(platform_name, environment))
            all_checks_passed = False

    if args.check_only:
        if all_checks_passed:
            print()
            print_success("All prerequisites satisfied!")
            sys.exit(0)
        else:
            print()
            print_error("Some prerequisites missing (see above)")
            sys.exit(1)

    if not all_checks_passed:
        print()
        print_error("Prerequisites not satisfied")
        print_info("Install missing components and run this wizard again")
        sys.exit(1)

    print()
    print_success("All prerequisites satisfied!")

    # Download models
    print_header("Downloading Models")

    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if models already exist
    required_models = ["sam3", "videodepthanything", "wham"]
    existing = [m for m in required_models if (models_dir / m).exists()]

    if len(existing) == len(required_models):
        print_info("Models already downloaded")
        verify_models(models_dir)
    else:
        if not download_models(models_dir):
            print_error("Model download failed")
            sys.exit(1)

        verify_models(models_dir)

        # Check for SMPL-X (optional, requires registration)
        smplx_dir = models_dir / "smplx"
        if not smplx_dir.exists() or not list(smplx_dir.glob("*.npz")):
            print()
            print_warning("SMPL-X models not found (optional, required for mocap stage)")
            print("  SMPL-X can be downloaded automatically with credentials:")
            print()
            print("  1. Register at https://smpl-x.is.tue.mpg.de/")
            print("  2. Create SMPL.login.dat in repository root with:")
            print("     Line 1: your email")
            print("     Line 2: your password")
            print("  3. Run: python3 scripts/install_wizard.py --component mocap")
            print()
            print("  The wizard will handle authentication and download automatically.")

    # Build Docker image
    print_header("Building Docker Image")

    if not build_docker_image():
        print_error("Docker image build failed")
        sys.exit(1)

    # Run test pipeline
    if not args.skip_test:
        print_header("Running Test Pipeline")

        repo_root = Path(__file__).parent.parent
        run_test_pipeline(repo_root)

    # Success!
    print_header("Installation Complete!")

    print_success("Docker-based VFX pipeline is ready to use!")
    print()
    print_info("Quick start:")
    print("  1. Copy your video to ~/VFX-Projects/")
    print("  2. Run: ./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name MyProject")
    print()
    print_info("Documentation:")
    print("  - QUICKSTART.md - Quick start guide")
    print("  - docs/README-DOCKER.md - Full Docker documentation")
    print()


if __name__ == "__main__":
    main()
