#!/usr/bin/env python3
"""Diagnose and fix PyTorch CUDA installation.

Run from the vfx-pipeline conda environment:
    conda activate vfx-pipeline
    python scripts/fix_pytorch_cuda.py
"""

import platform
import shutil
import subprocess
import sys


def run(cmd, capture=True):
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("PyTorch CUDA Diagnostic")
    print("=" * 60)

    # 1. Platform
    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python:   {sys.executable}")
    print(f"Version:  {sys.version}")

    # 2. Conda environment
    import os
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "(none)")
    conda_prefix = os.environ.get("CONDA_PREFIX", "(none)")
    print(f"Conda env:    {conda_env}")
    print(f"Conda prefix: {conda_prefix}")

    if conda_env != "vfx-pipeline":
        print("\n[!] WARNING: Not in vfx-pipeline environment.")
        print("    Run: conda activate vfx-pipeline")
        print("    Then re-run this script.")
        return

    # 3. nvidia-smi
    print(f"\n{'─' * 40}")
    print("GPU Detection")
    print(f"{'─' * 40}")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        search_paths = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
        ]
        for p in search_paths:
            if shutil.os.path.exists(p):
                nvidia_smi = p
                break

    if nvidia_smi:
        print(f"nvidia-smi: {nvidia_smi}")
        ok, output = run([nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        if ok:
            print(f"GPU:        {output}")
        else:
            print(f"nvidia-smi failed: {output}")
    else:
        print("nvidia-smi: NOT FOUND")
        print("[!] No NVIDIA driver detected. CUDA PyTorch requires an NVIDIA GPU + driver.")
        return

    # 4. Current PyTorch
    print(f"\n{'─' * 40}")
    print("Current PyTorch Installation")
    print(f"{'─' * 40}")

    try:
        import torch
        print(f"torch version:    {torch.__version__}")
        print(f"CUDA available:   {torch.cuda.is_available()}")
        print(f"CUDA built with:  {torch.version.cuda or 'NONE (CPU-only build)'}")

        if torch.cuda.is_available():
            print(f"CUDA device:      {torch.cuda.get_device_name(0)}")
            print(f"CUDA version:     {torch.version.cuda}")
            print("\n[OK] PyTorch CUDA is working. No fix needed.")
            return
        else:
            print("\n[!] PyTorch is CPU-only. Needs reinstall with CUDA.")
    except ImportError:
        print("torch: NOT INSTALLED")

    # 5. Fix it
    print(f"\n{'─' * 40}")
    print("Fixing: Reinstalling PyTorch with CUDA 12.1")
    print(f"{'─' * 40}")

    pip_cmd = [
        sys.executable, "-m", "pip", "install",
        "--force-reinstall",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ]

    print(f"Running: {' '.join(pip_cmd)}")
    print("(This will download ~2.5 GB, may take a few minutes)\n")

    result = subprocess.run(pip_cmd, timeout=600)

    if result.returncode != 0:
        print(f"\n[FAIL] pip install failed with exit code {result.returncode}")
        print("Try manually:")
        print(f"  conda activate vfx-pipeline")
        print(f"  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return

    # 6. Verify
    print(f"\n{'─' * 40}")
    print("Verifying...")
    print(f"{'─' * 40}")

    ok, output = run([
        sys.executable, "-c",
        "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
        "print(f'CUDA version: {torch.version.cuda}'); "
        "print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
    ])

    print(output)

    if "CUDA available: True" in output:
        print("\n[OK] PyTorch CUDA is now working!")
    else:
        print("\n[FAIL] CUDA still not available after reinstall.")
        print("Possible causes:")
        print("  - NVIDIA driver too old (need 525+ for CUDA 12.1)")
        print("  - Check driver version with: nvidia-smi")


if __name__ == "__main__":
    main()
