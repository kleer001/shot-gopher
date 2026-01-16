# Windows Compatibility Roadmap

This document outlines the plan for adding Windows support to the VFX Pipeline. Currently, the pipeline is Linux-only (tested on Ubuntu 20.04+). This roadmap describes the changes needed for Windows compatibility.

**Status:** Planning phase - not yet implemented

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Python Cross-Platform Fixes](#phase-1-python-cross-platform-fixes)
3. [Phase 2: Parallel Shell Scripts](#phase-2-parallel-shell-scripts)
4. [Phase 3: External Tool Handling](#phase-3-external-tool-handling)
5. [Phase 4: Prerequisites for Windows Users](#phase-4-prerequisites-for-windows-users)
6. [Phase 5: Testing](#phase-5-testing)
7. [Known Limitations](#known-limitations)

---

## Architecture Overview

### Design Principles

1. **OS Detection at Install Time** - Detect the operating system during installation and write it to an environment configuration file. All subsequent scripts read from this file rather than re-detecting.

2. **Parallel Shell Scripts** - Generate platform-specific scripts side-by-side:
   - `activate.sh` (Bash for Linux/macOS)
   - `activate.bat` (Batch for Windows CMD)
   - `activate.ps1` (PowerShell for Windows - recommended)

3. **Cross-Platform Python** - Use Python's cross-platform libraries (`pathlib`, `shutil.which()`, `sys.platform`) instead of shell-specific commands.

### Environment Configuration File

A new `.vfx_pipeline/env_config.json` will store platform information:

```json
{
  "platform": "windows",
  "python_executable": "C:\\Users\\Artist\\miniconda3\\envs\\vfx-pipeline\\python.exe",
  "path_separator": ";",
  "shell_type": "powershell",
  "conda_executable": "C:\\Users\\Artist\\miniconda3\\Scripts\\conda.exe"
}
```

This file is written once during installation and read by all scripts.

---

## Phase 1: Python Cross-Platform Fixes

### 1.1 Replace `which` Command with `shutil.which()`

The Unix `which` command doesn't exist on Windows. Python's `shutil.which()` is cross-platform.

| File | Current Issue | Fix |
|------|---------------|-----|
| `scripts/install_wizard/utils.py:148` | `subprocess.run(['which', command])` | `shutil.which(command)` |
| `scripts/install_wizard/downloader.py:393` | `subprocess.run(['which', 'wget'])` | `shutil.which('wget')` |
| `scripts/install_wizard/downloader.py:444` | `subprocess.run(['which', 'curl'])` | `shutil.which('curl')` |
| `scripts/install_wizard/downloader.py:493` | `subprocess.run(['which', 'aria2c'])` | `shutil.which('aria2c')` |
| `scripts/install_wizard/downloader.py:540` | `subprocess.run(['which', '7z'])` | `shutil.which('7z')` |

### 1.2 Fix `/dev/tty` Console Input

The `/dev/tty` Unix device doesn't exist on Windows. Need platform-specific handling.

| File | Current Issue | Fix |
|------|---------------|-----|
| `scripts/install_wizard/utils.py:27-31` | Opens `/dev/tty` directly | Platform check: use `msvcrt` module on Windows |

**Implementation approach:**

```python
import sys

def get_user_input(prompt):
    if sys.platform == 'win32':
        # Windows: use msvcrt for direct console input
        import msvcrt
        print(prompt, end='', flush=True)
        return msvcrt.getwche()
    else:
        # Unix: use /dev/tty for piped stdin scenarios
        with open('/dev/tty', 'r') as tty:
            return tty.readline().strip()
```

### 1.3 Add Windows Conda Path Detection

Current conda detection only checks Unix paths. Add Windows-specific locations.

| File | Current Paths | Additional Windows Paths |
|------|---------------|-------------------------|
| `scripts/install_wizard/conda.py:49-58` | `~/miniconda3/bin/conda`, `/opt/conda/bin/conda` | See list below |

**Windows conda search paths:**

```
%USERPROFILE%\miniconda3\Scripts\conda.exe
%USERPROFILE%\Miniconda3\Scripts\conda.exe
%USERPROFILE%\anaconda3\Scripts\conda.exe
%USERPROFILE%\Anaconda3\Scripts\conda.exe
%USERPROFILE%\mambaforge\Scripts\conda.exe
%USERPROFILE%\.conda\Scripts\conda.exe
%LOCALAPPDATA%\miniconda3\Scripts\conda.exe
%LOCALAPPDATA%\Continuum\miniconda3\Scripts\conda.exe
%LOCALAPPDATA%\Continuum\anaconda3\Scripts\conda.exe
%PROGRAMDATA%\miniconda3\Scripts\conda.exe
%PROGRAMDATA%\Anaconda3\Scripts\conda.exe
C:\tools\miniconda3\Scripts\conda.exe
C:\tools\Anaconda3\Scripts\conda.exe
```

### 1.4 Standardize Path Handling

Convert remaining `os.path` usage to `pathlib.Path` for consistency.

| File | Lines | Change |
|------|-------|--------|
| `scripts/export_camera.py` | 433-447 | Convert to `pathlib.Path` |
| `scripts/houdini/*.py` | various | Convert to `pathlib.Path` |

---

## Phase 2: Parallel Shell Scripts

### 2.1 Script Generation Strategy

The install wizard will generate **both** Unix and Windows scripts regardless of the current platform. This ensures:
- Configurations are portable across systems
- Teams can share project setups
- Remote/cloud execution flexibility

### 2.2 Activation Scripts

Three parallel scripts will be generated:

#### `activate.sh` (Bash - Linux/macOS)

```bash
#!/bin/bash
export VFX_PIPELINE_BASE="/path/to/install"
export PYTHONPATH="${PYTHONPATH}:${VFX_PIPELINE_BASE}/WHAM:${VFX_PIPELINE_BASE}/ECON"
source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate vfx-pipeline
```

#### `activate.bat` (Batch - Windows CMD)

```batch
@echo off
set VFX_PIPELINE_BASE=C:\path\to\install
set PYTHONPATH=%PYTHONPATH%;%VFX_PIPELINE_BASE%\WHAM;%VFX_PIPELINE_BASE%\ECON
call "%CONDA_PREFIX%\Scripts\activate.bat" vfx-pipeline
```

#### `activate.ps1` (PowerShell - Windows, recommended)

```powershell
$env:VFX_PIPELINE_BASE = "C:\path\to\install"
$env:PYTHONPATH = "$env:PYTHONPATH;$env:VFX_PIPELINE_BASE\WHAM;$env:VFX_PIPELINE_BASE\ECON"
& "$env:CONDA_PREFIX\Scripts\conda.exe" activate vfx-pipeline
```

### 2.3 Key Syntax Differences

| Concept | Bash | Batch | PowerShell |
|---------|------|-------|------------|
| Set variable | `export VAR=value` | `set VAR=value` | `$env:VAR = "value"` |
| Read variable | `$VAR` or `${VAR}` | `%VAR%` | `$env:VAR` |
| Path separator | `:` | `;` | `;` |
| Run script | `source script.sh` | `call script.bat` | `. .\script.ps1` |
| Comment | `# comment` | `REM comment` | `# comment` |
| Conditional | `if [ cond ]; then` | `if cond (` | `if (cond) {` |

### 2.4 Update Script Generator

| File | Change |
|------|--------|
| `scripts/install_wizard/config.py` | Modify `ConfigurationGenerator.generate_activation_script()` to produce all three script variants |

---

## Phase 3: External Tool Handling

### 3.1 Tool Path Detection

External tools may be installed in various locations. The pipeline will search exhaustively.

#### COLMAP Search Paths

**Linux:**
- `PATH` lookup via `shutil.which('colmap')`
- `/usr/local/bin/colmap`
- `/usr/bin/colmap`

**Windows:**
- `PATH` lookup via `shutil.which('colmap')`
- `C:\Program Files\COLMAP\COLMAP.bat`
- `C:\Program Files (x86)\COLMAP\COLMAP.bat`
- `C:\COLMAP\COLMAP.bat`
- `%LOCALAPPDATA%\COLMAP\COLMAP.bat`
- `%USERPROFILE%\COLMAP\COLMAP.bat`
- Chocolatey: `C:\ProgramData\chocolatey\bin\colmap.exe`
- Scoop: `%USERPROFILE%\scoop\apps\colmap\current\COLMAP.bat`
- Custom environment variable: `%COLMAP_PATH%`

#### FFmpeg/FFprobe Search Paths

**Linux:**
- `PATH` lookup (usually pre-installed or via package manager)

**Windows:**
- `PATH` lookup via `shutil.which('ffmpeg')`
- `C:\Program Files\FFmpeg\bin\ffmpeg.exe`
- `C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe`
- `C:\ffmpeg\bin\ffmpeg.exe`
- `%USERPROFILE%\ffmpeg\bin\ffmpeg.exe`
- Chocolatey: `C:\ProgramData\chocolatey\bin\ffmpeg.exe`
- Scoop: `%USERPROFILE%\scoop\apps\ffmpeg\current\bin\ffmpeg.exe`
- Custom environment variable: `%FFMPEG_PATH%`

#### 7-Zip Search Paths

**Linux:**
- `PATH` lookup: `7z`, `7za`, `7zr`
- `/usr/bin/7z`

**Windows:**
- `PATH` lookup via `shutil.which('7z')`
- `C:\Program Files\7-Zip\7z.exe`
- `C:\Program Files (x86)\7-Zip\7z.exe`
- Chocolatey: `C:\ProgramData\chocolatey\bin\7z.exe`
- Scoop: `%USERPROFILE%\scoop\apps\7zip\current\7z.exe`
- Custom environment variable: `%SEVENZIP_PATH%`

#### Download Tools (wget, curl, aria2c)

**Linux:**
- All typically available via package manager

**Windows:**
- `curl.exe` is built into Windows 10/11 (System32)
- `wget` - rare, not recommended
- `aria2c` - must be manually installed
- Fallback: Use Python's `urllib` for downloads when no tools available

### 3.2 GPU Detection and CUDA Toolkit

GPU detection via `nvidia-smi` works on both platforms when CUDA is installed.

**Windows CUDA Requirements:**
- NVIDIA GPU drivers (includes `nvidia-smi.exe`)
- CUDA Toolkit (required for PyTorch GPU support)
  - Download: https://developer.nvidia.com/cuda-toolkit
  - Typical install: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

**Detection locations for `nvidia-smi`:**

```
PATH lookup (preferred)
C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
C:\Windows\System32\nvidia-smi.exe
%CUDA_PATH%\bin\nvidia-smi.exe
```

**Note:** The CUDA Toolkit is effectively a dependency for Windows users who want GPU acceleration. The install wizard should check for it and provide guidance if missing.

### 3.3 Automatic Tool Installation (Windows)

When a required tool is not found, the install wizard will offer to install it automatically (with user confirmation). This keeps the installation experience smooth while respecting user control.

#### Package Manager Strategy

The wizard will detect available package managers in this priority order:

1. **winget** (Windows Package Manager) - Built into Windows 10/11, preferred
2. **Chocolatey** - Popular third-party manager, widely used
3. **Scoop** - User-level installs, no admin required
4. **Manual download** - Fallback with guided instructions

#### Installation Flow

```
Checking for FFmpeg... NOT FOUND

FFmpeg is required for video processing.
Would you like to install it automatically? [y/n]: y

Detected package manager: winget
Installing FFmpeg via winget...
> winget install ffmpeg
✓ FFmpeg installed successfully

Verifying installation...
✓ ffmpeg found at C:\Users\Artist\AppData\Local\Microsoft\WinGet\Packages\...
```

#### Tool Installation Commands

| Tool | winget | Chocolatey | Scoop |
|------|--------|------------|-------|
| **FFmpeg** | `winget install ffmpeg` | `choco install ffmpeg` | `scoop install ffmpeg` |
| **7-Zip** | `winget install 7zip.7zip` | `choco install 7zip` | `scoop install 7zip` |
| **Git** | `winget install Git.Git` | `choco install git` | `scoop install git` |
| **COLMAP** | N/A (manual) | `choco install colmap` | N/A (manual) |
| **aria2** | `winget install aria2.aria2` | `choco install aria2` | `scoop install aria2` |
| **Miniconda** | `winget install Anaconda.Miniconda3` | `choco install miniconda3` | `scoop install miniconda3` |

#### Tools Requiring Manual Installation

Some tools cannot be reliably auto-installed:

| Tool | Reason | Wizard Behavior |
|------|--------|-----------------|
| **CUDA Toolkit** | Large download, requires specific version matching GPU | Provide download link, verify after user installs |
| **NVIDIA Drivers** | Hardware-specific, best from NVIDIA directly | Detect GPU, provide direct download link |
| **Visual Studio Build Tools** | Large, complex installer with options | Provide download link and required components list |
| **COLMAP** (via winget) | Not available in winget | Offer Chocolatey or provide manual download link |

#### Implementation Details

```python
class WindowsToolInstaller:
    """Handles automatic tool installation on Windows."""

    def __init__(self):
        self.package_manager = self._detect_package_manager()

    def _detect_package_manager(self) -> str:
        """Detect available package manager in priority order."""
        if shutil.which('winget'):
            return 'winget'
        elif shutil.which('choco'):
            return 'chocolatey'
        elif shutil.which('scoop'):
            return 'scoop'
        return 'manual'

    def install_tool(self, tool_name: str) -> bool:
        """
        Prompt user and install tool if confirmed.
        Returns True if tool is available after attempt.
        """
        if not self._prompt_user(tool_name):
            return False

        commands = INSTALL_COMMANDS[self.package_manager].get(tool_name)
        if not commands:
            self._show_manual_instructions(tool_name)
            return self._wait_for_manual_install(tool_name)

        return self._run_install(commands)
```

#### User Prompts

The wizard will use clear, consistent prompts:

```
┌─────────────────────────────────────────────────────────────┐
│  Missing Dependency: FFmpeg                                 │
├─────────────────────────────────────────────────────────────┤
│  FFmpeg is required for video frame extraction.             │
│                                                             │
│  Install automatically using winget?                        │
│                                                             │
│  [Y] Yes, install now                                       │
│  [N] No, I'll install it manually                           │
│  [S] Skip (some features won't work)                        │
└─────────────────────────────────────────────────────────────┘
```

#### Error Handling

If automatic installation fails:

1. Show the error message from the package manager
2. Offer to retry with a different package manager (if available)
3. Provide manual installation instructions as fallback
4. Allow user to skip (with warning about reduced functionality)

#### Linux Comparison

On Linux, the wizard already handles package installation via apt/yum/dnf. The Windows implementation follows the same pattern:

| Aspect | Linux | Windows |
|--------|-------|---------|
| Package manager detection | apt, yum, dnf, pacman | winget, choco, scoop |
| User confirmation | Required | Required |
| Fallback | Manual instructions | Manual instructions |
| Admin requirements | sudo for system packages | Varies by manager |

---

## Phase 4: Prerequisites for Windows Users

### Required Software

| Software | Purpose | Download |
|----------|---------|----------|
| **Git** | Version control | https://git-scm.com/download/win |
| **Miniconda** | Python environment | https://docs.conda.io/en/latest/miniconda.html |
| **NVIDIA Drivers** | GPU support | https://www.nvidia.com/drivers |
| **CUDA Toolkit** | GPU computation | https://developer.nvidia.com/cuda-toolkit |
| **FFmpeg** | Video processing | https://ffmpeg.org/download.html#build-windows |
| **Visual Studio Build Tools** | C++ compilation | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

### Optional Software

| Software | Purpose | Download |
|----------|---------|----------|
| **COLMAP** | Camera tracking | https://colmap.github.io/install.html |
| **7-Zip** | Archive extraction | https://www.7-zip.org/ |
| **aria2** | Fast downloads | https://aria2.github.io/ |

### Windows-Specific Notes

1. **PowerShell Execution Policy**

   To run `.ps1` scripts, you may need to adjust the execution policy:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Symlinks Require Elevated Privileges**

   Creating symlinks on Windows requires either:
   - Running as Administrator, OR
   - Enabling Developer Mode (Settings > Update & Security > For developers)

   The installer will fall back to copying files if symlink creation fails.

3. **Path Length Limitations**

   Windows has a 260-character path limit by default. Enable long paths:
   - Run `regedit`
   - Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`

4. **Antivirus Interference**

   Some antivirus software may flag or slow down ML model downloads. Consider adding exclusions for:
   - The installation directory
   - The conda environments directory

### Installation on Windows

Once Windows support is implemented, installation will be:

**Option A: PowerShell (recommended)**
```powershell
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python scripts/install_wizard.py
```

**Option B: Command Prompt**
```batch
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python scripts/install_wizard.py
```

The install wizard will detect Windows and generate appropriate scripts.

---

## Phase 5: Testing

### Test Fixture Updates

| File | Issue | Fix |
|------|-------|-----|
| `tests/test_run_gsir.py:43` | Hardcoded `/home/user/GS-IR` | Use `tempfile` or `Path.home()` |
| Various tests | Unix-style paths in fixtures | Use `pathlib.Path` for cross-platform |

### CI/CD Additions

- Add Windows runner to GitHub Actions (`windows-latest`)
- Run core test suite on Windows
- Test script generation produces valid `.bat` and `.ps1` files

### Manual Testing Checklist

- [ ] Install wizard completes on Windows 10/11
- [ ] Conda environment activates via `activate.bat`
- [ ] Conda environment activates via `activate.ps1`
- [ ] GPU detection works with CUDA Toolkit installed
- [ ] COLMAP integration works (if installed)
- [ ] FFmpeg/ffprobe detection works
- [ ] Full pipeline run completes on test footage

---

## Known Limitations

Even with Windows support, some limitations may remain:

1. **Performance** - Linux typically has better PyTorch/CUDA performance due to driver maturity

2. **WSL Alternative** - Windows users may get better results using WSL2 (Windows Subsystem for Linux) with Ubuntu, which provides a full Linux environment

3. **Some External Tools** - Certain optional components may have limited Windows support:
   - WHAM/ECON - Primarily tested on Linux
   - GS-IR - May require additional Windows-specific configuration

4. **Path Issues** - Some third-party libraries may have hardcoded Unix paths that cause issues

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Replace `which` with `shutil.which()` | Low | High |
| 2 | Fix `/dev/tty` console input | Low | High |
| 3 | Add Windows conda detection paths | Low | High |
| 4 | Create `activate.bat` and `activate.ps1` | Medium | High |
| 5 | Update script generator for multi-platform | Medium | High |
| 6 | Exhaustive tool path detection | Medium | Medium |
| 7 | Automatic tool installation (winget/choco/scoop) | Medium | High |
| 8 | Update documentation | Low | Medium |
| 9 | Standardize pathlib usage | Low | Low |
| 10 | CI testing on Windows | Medium | Medium |

---

## Contributing

If you'd like to help with Windows compatibility:

1. Test the current codebase on Windows and report issues
2. Submit PRs for specific fixes outlined above
3. Help test on various Windows configurations (10, 11, different GPU setups)

File issues at: https://github.com/kleer001/comfyui_ingest/issues
