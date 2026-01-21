# Windows IT Setup (Admin Required)

Run these as Administrator. Users handle everything else.

## 1. NVIDIA Driver + CUDA

```powershell
# Verify after install
nvidia-smi
nvcc --version
```

Download: https://www.nvidia.com/download/index.aspx
CUDA: https://developer.nvidia.com/cuda-toolkit-archive (11.8 or 12.1)

**Why:** PyTorch and ComfyUI require CUDA for GPU-accelerated inference. Without the driver and CUDA toolkit, all ML operations fall back to CPU, making image generation impractically slow (minutes vs seconds per image).

**What it does:** `nvidia-smi` confirms the GPU driver is installed and shows GPU status. `nvcc --version` confirms the CUDA compiler toolkit is available for building GPU-accelerated Python packages.

## 2. PowerShell Execution Policy

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

**Why:** Windows blocks all PowerShell scripts by default. The bootstrap installer and conda activation scripts won't run without this policy change.

**What it does:** Allows locally-created scripts to run without a signature, while still requiring downloaded scripts to be signed by a trusted publisher. `LocalMachine` scope applies to all users on this computer.

## 3. Long Paths

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

**Why:** Windows has a legacy 260-character path limit. Python packages (especially ML dependencies like PyTorch) create deeply nested directories that exceed this limit, causing installation failures.

**What it does:** Enables the Windows 10+ long path feature via the registry, allowing paths up to 32,767 characters. A reboot may be required for this to take effect.

## Done
