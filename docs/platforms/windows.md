# Windows Support

This document covers Windows compatibility, prerequisites, and troubleshooting for the VFX Pipeline.

**Support Status:** Native Windows support is available via Conda.

---

## Prerequisites

### Required Software

| Software | Purpose | Download |
|----------|---------|----------|
| **Git** | Version control | https://git-scm.com/download/win |
| **Miniconda** | Python environment | https://docs.conda.io/en/latest/miniconda.html |
| **NVIDIA Drivers** | GPU support | https://www.nvidia.com/drivers |
| **CUDA Toolkit** | GPU computation | https://developer.nvidia.com/cuda-toolkit |
| **Visual Studio Build Tools** | C++ compilation | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

### Optional Software

| Software | Purpose | Download |
|----------|---------|----------|
| **COLMAP** | Camera tracking | https://colmap.github.io/install.html |
| **7-Zip** | Archive extraction | https://www.7-zip.org/ |

**Note:** FFmpeg and COLMAP are auto-installed to `.vfx_pipeline/tools/` when the wizard runs.

### IT Admin Setup

If you don't have administrator access, give this one-pager to your IT department:
- [IT Setup: Native](../windows-it-native.md) - For native conda installation

---

## Installation

**PowerShell (recommended):**
```powershell
git clone https://github.com/kleer001/shot-gopher.git
cd shot-gopher
python scripts/install_wizard.py
```

**Command Prompt:**
```batch
git clone https://github.com/kleer001/shot-gopher.git
cd shot-gopher
python scripts/install_wizard.py
```

The install wizard detects Windows and generates appropriate scripts.

---

## Troubleshooting

### Quick Diagnostics

Run this in PowerShell to check your environment:

```powershell
# Check if conda is available
Get-Command conda -ErrorAction SilentlyContinue

# Check PowerShell execution policy
Get-ExecutionPolicy

# Check long paths enabled
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction SilentlyContinue

# Check auto-installed tools
ls .vfx_pipeline\tools\ -ErrorAction SilentlyContinue
```

### Common Issues

#### "irm is not recognized" Error

**Symptom:** Running the bootstrap command shows `'irm' is not recognized as an internal or external command`.

**Cause:** You're running the command in Command Prompt (cmd.exe) instead of PowerShell. The `irm` (Invoke-RestMethod) and `iex` (Invoke-Expression) commands are PowerShell-only.

**Solution:** Open PowerShell instead:
- Press `Win+X` → select "Windows PowerShell" or "Terminal"
- Or press `Win+R`, type `powershell`, press Enter
- Then run the bootstrap command

**Alternative for Command Prompt users:**
```batch
powershell -Command "irm https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.ps1 | iex"
```

#### Running from the Wrong Terminal

**Symptom:** Scripts behave unexpectedly or console input doesn't work.

**Cause:** Running from Git Bash instead of PowerShell or CMD.

**Solution:** Always use **PowerShell** or **CMD**, not Git Bash.

#### Using the Wrong Activation Script

| Shell | Activation Command |
|-------|-------------------|
| PowerShell | `. .\.vfx_pipeline\activate.ps1` |
| CMD | `.vfx_pipeline\activate.bat` |
| Git Bash | `source .vfx_pipeline/activate.sh` |

#### Conda Not Initialized

**Symptom:** `conda activate` shows "conda is not recognized".

**Solution:**
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
# Close and reopen PowerShell
```

### Error Reference

| Error | Fix |
|-------|-----|
| "irm is not recognized" | Use PowerShell, not Command Prompt (see above) |
| "running scripts is disabled" | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| "system cannot find the path" (long paths) | Ask IT to enable long paths |
| "CUDA out of memory" | Close other GPU apps, reduce batch size |
| "No module named 'torch'" | Reactivate conda environment |

---

## Known Limitations

### Performance
Linux typically has better PyTorch/CUDA performance. WSL2 provides near-native Linux performance on Windows.

### Remote Desktop
COLMAP GPU mode may fail over RDP. Run locally or use CPU mode: `--quality slow`

### File Locking
Windows locks files in use. If updates fail, close all Python processes:
```powershell
taskkill /F /IM python.exe
```

### Antivirus Interference
Large model downloads may be flagged. Add the repository directory to exclusions:
- `C:\path\to\shot-gopher`

### Case-Insensitive Filesystem
Windows treats `GVHMR` and `gvhmr` as the same. Use consistent casing when cloning.

---

## Getting Help

1. Check [GitHub Issues](https://github.com/kleer001/shot-gopher/issues)
2. Include diagnostics in reports:
   ```powershell
   python -c "import sys; print(sys.platform, sys.version)"
   conda info
   nvidia-smi
   ```
