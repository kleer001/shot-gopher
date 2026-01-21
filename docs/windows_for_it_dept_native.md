# Windows Native Setup for IT Department

This document lists **only the items requiring administrator privileges**. Everything else is handled automatically by the install wizard or can be installed by the user.

For Docker/WSL2 installation, see [windows_for_it_dept_docker.md](windows_for_it_dept_docker.md).

## What IT Needs to Do

| Item | Why Admin Required |
|------|-------------------|
| NVIDIA GPU Driver | System-level hardware driver |
| CUDA Toolkit | Installs to Program Files |
| PowerShell Execution Policy | System security setting |
| Long Paths Registry Key | System registry modification |

**Everything else is automatic:** Git, Miniconda, FFmpeg, COLMAP, 7-Zip, and all Python dependencies are either user-installable or auto-installed by the wizard to a sandboxed directory.

---

## 1. NVIDIA GPU Driver (REQUIRED)

Download and install the latest driver for the user's graphics card.

**Download:** https://www.nvidia.com/download/index.aspx

**Verification:**
```powershell
nvidia-smi
```

---

## 2. CUDA Toolkit (REQUIRED)

Required for GPU-accelerated machine learning.

**Download:** https://developer.nvidia.com/cuda-toolkit-archive

**Recommended:** CUDA 11.8 or CUDA 12.1

**Installation Notes:**
- Select "Add to PATH" during installation
- Default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

**Verification:**
```powershell
nvcc --version
```

---

## 3. PowerShell Execution Policy (REQUIRED)

Without this, users cannot run the activation script.

**Error if not set:** "cannot be loaded because running scripts is disabled on this system"

```powershell
# Run as Administrator (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

**Verification:**
```powershell
Get-ExecutionPolicy -List
# LocalMachine should show "RemoteSigned"
```

---

## 4. Enable Long Paths (REQUIRED)

Without this, installation fails with "path too long" errors.

**Option A: Group Policy (Domain environments)**
1. Open `gpedit.msc`
2. Computer Configuration > Administrative Templates > System > Filesystem
3. Enable "Enable Win32 long paths"

**Option B: Registry (Standalone machines)**
```powershell
# Run as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

**Verification:**
```powershell
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"
# Should return 1
```

---

## Optional: Developer Mode

Allows creating symbolic links without admin. Not strictly required.

Settings > Privacy & Security > For Developers > Enable "Developer Mode"

---

## Optional: Antivirus Exclusion

If users report slow downloads or disappearing files, add an exclusion for the repository directory:
- `C:\path\to\comfyui_ingest`

---

## What Users Do (No Admin Required)

After IT completes the above, users run:

```powershell
# Install Git (user install, no admin)
winget install Git.Git --scope user

# Install Miniconda (user install, no admin)
winget install Anaconda.Miniconda3 --scope user

# Clone and run wizard
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python scripts/install_wizard.py
```

The wizard automatically:
- Creates conda environment
- Auto-installs FFmpeg, COLMAP to `.vfx_pipeline/tools/`
- Installs all Python dependencies
- Downloads AI models

All files stay within the repository directory - no system pollution.

---

## Support

- Troubleshooting: [windows-troubleshooting.md](windows-troubleshooting.md)
- Issues: https://github.com/kleer001/comfyui_ingest/issues
