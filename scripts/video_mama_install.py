#!/usr/bin/env python3
"""VideoMaMa installation script.

Installs VideoMaMa for high-quality video matting from coarse segmentation masks.
Creates a separate conda environment and downloads required models.

DISK SPACE REQUIREMENT: ~12GB total
  - Stable Video Diffusion base model: ~10GB
  - VideoMaMa checkpoint: ~1.5GB
  - VideoMaMa code + dependencies: ~500MB

Usage:
    python scripts/video_mama_install.py
    python scripts/video_mama_install.py --check  # Check installation status
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from env_config import INSTALL_DIR

VIDEOMAMA_ENV_NAME = "videomama"
VIDEOMAMA_PYTHON_VERSION = "3.10"
VIDEOMAMA_TOOLS_DIR = INSTALL_DIR / "tools" / "VideoMaMa"
VIDEOMAMA_MODELS_DIR = INSTALL_DIR / "models" / "VideoMaMa"
SVD_MODEL_DIR = VIDEOMAMA_MODELS_DIR / "stable-video-diffusion-img2vid-xt"
VIDEOMAMA_CHECKPOINT_DIR = VIDEOMAMA_MODELS_DIR / "checkpoints"

VIDEOMAMA_REPO_URL = "https://github.com/cvlab-kaist/VideoMaMa.git"
SVD_HF_REPO = "stabilityai/stable-video-diffusion-img2vid-xt"
VIDEOMAMA_HF_REPO = "SammyLim/VideoMaMa"


def print_header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def print_success(msg: str) -> None:
    print(f"  [OK] {msg}")


def print_error(msg: str) -> None:
    print(f"  [ERROR] {msg}", file=sys.stderr)


def print_warning(msg: str) -> None:
    print(f"  [WARN] {msg}")


def print_info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def run_command(
    cmd: list,
    check: bool = True,
    capture: bool = False,
    cwd: Path = None
) -> tuple[bool, str]:
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            cwd=cwd
        )
        output = result.stdout if capture else ""
        return True, output
    except subprocess.CalledProcessError as e:
        output = e.stdout or "" if capture else ""
        return False, output
    except FileNotFoundError:
        return False, ""


def find_conda() -> str | None:
    """Find conda executable."""
    for cmd in ["conda", "mamba"]:
        success, output = run_command([cmd, "--version"], check=False, capture=True)
        if success:
            return cmd

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    return None


def env_exists(conda_exe: str, env_name: str) -> bool:
    """Check if conda environment exists."""
    success, output = run_command([conda_exe, "env", "list"], check=False, capture=True)
    if not success:
        return False
    return env_name in output


def create_conda_env(conda_exe: str) -> bool:
    """Create VideoMaMa conda environment."""
    if env_exists(conda_exe, VIDEOMAMA_ENV_NAME):
        print_success(f"Conda environment '{VIDEOMAMA_ENV_NAME}' already exists")
        return True

    print_info(f"Creating conda environment '{VIDEOMAMA_ENV_NAME}'...")
    success, _ = run_command([
        conda_exe, "create", "-n", VIDEOMAMA_ENV_NAME,
        f"python={VIDEOMAMA_PYTHON_VERSION}", "-y"
    ])

    if success:
        print_success(f"Created environment '{VIDEOMAMA_ENV_NAME}'")
    else:
        print_error(f"Failed to create environment '{VIDEOMAMA_ENV_NAME}'")

    return success


def run_in_env(conda_exe: str, cmd: list, cwd: Path = None) -> bool:
    """Run command in VideoMaMa conda environment."""
    full_cmd = [conda_exe, "run", "-n", VIDEOMAMA_ENV_NAME] + cmd
    success, _ = run_command(full_cmd, check=False, cwd=cwd)
    return success


def clone_videomama() -> bool:
    """Clone VideoMaMa repository."""
    if VIDEOMAMA_TOOLS_DIR.exists():
        if (VIDEOMAMA_TOOLS_DIR / ".git").exists():
            print_success("VideoMaMa repository already cloned")
            return True
        else:
            print_warning("VideoMaMa directory exists but is not a git repo, removing...")
            shutil.rmtree(VIDEOMAMA_TOOLS_DIR)

    VIDEOMAMA_TOOLS_DIR.parent.mkdir(parents=True, exist_ok=True)

    print_info("Cloning VideoMaMa repository...")
    success, _ = run_command([
        "git", "clone", VIDEOMAMA_REPO_URL, str(VIDEOMAMA_TOOLS_DIR)
    ])

    if success:
        print_success("Cloned VideoMaMa repository")
    else:
        print_error("Failed to clone VideoMaMa repository")

    return success


def install_dependencies(conda_exe: str) -> bool:
    """Install VideoMaMa dependencies."""
    print_info("Installing PyTorch...")
    if not run_in_env(conda_exe, [
        "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]):
        print_error("Failed to install PyTorch")
        return False

    print_info("Installing VideoMaMa dependencies...")
    requirements_file = VIDEOMAMA_TOOLS_DIR / "requirements.txt"
    if requirements_file.exists():
        if not run_in_env(conda_exe, ["pip", "install", "-r", str(requirements_file)]):
            print_error("Failed to install requirements.txt")
            return False

    print_info("Installing additional dependencies...")
    deps = [
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "opencv-python",
        "einops",
        "omegaconf",
        "safetensors",
    ]
    if not run_in_env(conda_exe, ["pip", "install"] + deps):
        print_error("Failed to install additional dependencies")
        return False

    print_success("Installed all dependencies")
    return True


def download_svd_model(conda_exe: str) -> bool:
    """Download Stable Video Diffusion base model from HuggingFace."""
    if SVD_MODEL_DIR.exists() and list(SVD_MODEL_DIR.glob("*.safetensors")):
        print_success("SVD base model already downloaded")
        return True

    SVD_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print_info(f"Downloading Stable Video Diffusion model (~10GB)...")
    print_info("This may take a while depending on your connection...")
    print()

    download_script = f"""
import os
import sys
import time

# Disable XET storage backend (causes CAS service errors)
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import snapshot_download

print("Starting download from HuggingFace...", flush=True)
print("(XET disabled to avoid CAS service errors)", flush=True)
print("-" * 50, flush=True)

max_retries = 3
for attempt in range(max_retries):
    try:
        snapshot_download(
            repo_id='{SVD_HF_REPO}',
            local_dir='{SVD_MODEL_DIR}',
            resume_download=True,
        )
        print("-" * 50, flush=True)
        print("Download complete!", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Attempt {{attempt + 1}}/{{max_retries}} failed: {{e}}", flush=True)
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 30
            print(f"Retrying in {{wait_time}} seconds...", flush=True)
            time.sleep(wait_time)
        else:
            print("All retry attempts failed.", flush=True)
            sys.exit(1)
"""

    script_path = VIDEOMAMA_MODELS_DIR / "_download_svd.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(download_script)

    try:
        full_cmd = [conda_exe, "run", "-n", VIDEOMAMA_ENV_NAME, "python", "-u", str(script_path)]
        result = subprocess.run(full_cmd, check=False)
        success = result.returncode == 0
        if success:
            print()
            print_success("Downloaded SVD base model")
        else:
            print_error("Failed to download SVD base model")
        return success
    finally:
        if script_path.exists():
            script_path.unlink()


def download_videomama_checkpoint(conda_exe: str) -> bool:
    """Download VideoMaMa checkpoint from HuggingFace."""
    checkpoint_path = VIDEOMAMA_CHECKPOINT_DIR / "VideoMaMa"
    if checkpoint_path.exists() and list(checkpoint_path.glob("*")):
        print_success("VideoMaMa checkpoint already downloaded")
        return True

    VIDEOMAMA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print_info("Downloading VideoMaMa checkpoint (~1.5GB)...")
    print()

    download_script = f"""
import os
import sys
import time

# Disable XET storage backend (causes CAS service errors)
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import snapshot_download

print("Starting download from HuggingFace...", flush=True)
print("(XET disabled to avoid CAS service errors)", flush=True)
print("-" * 50, flush=True)

max_retries = 3
for attempt in range(max_retries):
    try:
        snapshot_download(
            repo_id='{VIDEOMAMA_HF_REPO}',
            local_dir='{checkpoint_path}',
            resume_download=True,
        )
        print("-" * 50, flush=True)
        print("Download complete!", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Attempt {{attempt + 1}}/{{max_retries}} failed: {{e}}", flush=True)
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 30
            print(f"Retrying in {{wait_time}} seconds...", flush=True)
            time.sleep(wait_time)
        else:
            print("All retry attempts failed.", flush=True)
            sys.exit(1)
"""

    script_path = VIDEOMAMA_MODELS_DIR / "_download_checkpoint.py"
    script_path.write_text(download_script)

    try:
        full_cmd = [conda_exe, "run", "-n", VIDEOMAMA_ENV_NAME, "python", "-u", str(script_path)]
        result = subprocess.run(full_cmd, check=False)
        success = result.returncode == 0
        if success:
            print()
            print_success("Downloaded VideoMaMa checkpoint")
        else:
            print_error("Failed to download VideoMaMa checkpoint")
        return success
    finally:
        if script_path.exists():
            script_path.unlink()


def check_installation() -> dict:
    """Check VideoMaMa installation status."""
    status = {
        "conda_found": False,
        "env_exists": False,
        "repo_cloned": False,
        "svd_model": False,
        "videomama_checkpoint": False,
        "ready": False,
    }

    conda_exe = find_conda()
    if conda_exe:
        status["conda_found"] = True
        status["env_exists"] = env_exists(conda_exe, VIDEOMAMA_ENV_NAME)

    status["repo_cloned"] = (
        VIDEOMAMA_TOOLS_DIR.exists() and
        (VIDEOMAMA_TOOLS_DIR / ".git").exists()
    )

    status["svd_model"] = (
        SVD_MODEL_DIR.exists() and
        bool(list(SVD_MODEL_DIR.glob("*.safetensors")))
    )

    checkpoint_path = VIDEOMAMA_CHECKPOINT_DIR / "VideoMaMa"
    status["videomama_checkpoint"] = (
        checkpoint_path.exists() and
        bool(list(checkpoint_path.glob("*")))
    )

    status["ready"] = all([
        status["conda_found"],
        status["env_exists"],
        status["repo_cloned"],
        status["svd_model"],
        status["videomama_checkpoint"],
    ])

    return status


def print_status(status: dict) -> None:
    """Print installation status."""
    print_header("VideoMaMa Installation Status")

    def status_icon(ok: bool) -> str:
        return "[OK]" if ok else "[  ]"

    print(f"  {status_icon(status['conda_found'])} Conda available")
    print(f"  {status_icon(status['env_exists'])} Conda environment '{VIDEOMAMA_ENV_NAME}'")
    print(f"  {status_icon(status['repo_cloned'])} VideoMaMa repository cloned")
    print(f"  {status_icon(status['svd_model'])} SVD base model (~10GB)")
    print(f"  {status_icon(status['videomama_checkpoint'])} VideoMaMa checkpoint (~1.5GB)")

    print()
    if status["ready"]:
        print_success("VideoMaMa is ready to use!")
        print_info(f"Run: python scripts/video_mama.py <project_dir>")
    else:
        print_warning("VideoMaMa is not fully installed")
        print_info("Run: python scripts/video_mama_install.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install VideoMaMa for video matting")
    parser.add_argument("--check", action="store_true", help="Check installation status only")
    args = parser.parse_args()

    if args.check:
        status = check_installation()
        print_status(status)
        return 0 if status["ready"] else 1

    start_time = time.time()

    print_header("VideoMaMa Installation")
    print_info("This will install VideoMaMa for high-quality video matting")
    print_info(f"Installation directory: {INSTALL_DIR}")
    print_warning("Disk space required: ~12GB (SVD model + checkpoint)")
    print()

    conda_exe = find_conda()
    if not conda_exe:
        print_error("Conda not found. Please install Miniconda or Anaconda first.")
        return 1
    print_success(f"Found conda: {conda_exe}")

    if not create_conda_env(conda_exe):
        return 1

    if not clone_videomama():
        return 1

    if not install_dependencies(conda_exe):
        return 1

    if not download_svd_model(conda_exe):
        return 1

    if not download_videomama_checkpoint(conda_exe):
        return 1

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print_header("Installation Complete")
    print_success("VideoMaMa is ready to use!")
    print()
    print_info(f"Total installation time: {minutes}m {seconds}s")
    print()
    print_info("Usage:")
    print("  python scripts/video_mama.py <project_dir>")
    print()
    print_info("This will process roto/person/ masks and output to roto/person_mama/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
