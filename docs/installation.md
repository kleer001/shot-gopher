# Installation Wizard Documentation

Interactive installation and setup tool for the VFX pipeline.

## Overview

The installation wizard provides a guided, menu-driven interface for installing and configuring all pipeline components. It handles:

- Conda environment management (automatic)
- Dependency installation (PyTorch, COLMAP, etc.)
- Git repository cloning (GVHMR, ComfyUI)
- Checkpoint downloading (automatic with progress bars)
- Installation validation (smoke tests)
- Configuration file generation
- Resume capability for interrupted installations

## Quick Start

### Interactive Installation

```bash
python scripts/install_wizard.py
```

Follow the menu prompts to select components.

### One-Liner Bootstrap

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.sh | bash
```

This clones the repository and runs the wizard automatically.

## Prerequisites: Model Access Credentials

Before running the wizard, you should set up credentials for downloading licensed models.

### SAM3 Model (No Credentials Required)

SAM3 downloads automatically from the public `1038lab/sam3` HuggingFace repository. No token or registration needed.

### SMPL-X Credentials

Required for: Motion capture pipeline

| Site | URL | Purpose |
|------|-----|---------|
| SMPL-X | https://smpl-x.is.tue.mpg.de/ | Parametric body model (skeleton, mesh topology, UVs) |

**SMPL-X** provides the deformable body mesh that gets animated by GVHMR motion data.

Save credentials to `SMPL.login.dat`:

```
your.email@example.com
your_password_here
```

**Template files**: `HF_TOKEN.dat.template` and `SMPL.login.dat.template`

The wizard will prompt for credentials if files don't exist.

## Command Line Options

### Short Options

```bash
-C COMPONENT   # Install specific component
-c             # Check installation status only
-v             # Validate existing installation
-r             # Resume interrupted installation
```

### Long Options

```bash
--component COMPONENT    # Install specific component
--check-only            # Check installation status only
--validate              # Validate existing installation
--resume                # Resume interrupted installation
```

## Usage Examples

### Check Installation Status

View what's installed without making changes:

```bash
python scripts/install_wizard.py -c
```

Output:
```
[Core Dependencies]
  ✓ Python 3.10.12
  ✓ CUDA 12.1
  ✗ PyTorch with CUDA

[Motion Capture]
  ✗ GVHMR
```

### Validate Installation

Run smoke tests on installed components:

```bash
python scripts/install_wizard.py -v
```

Tests include:
- Python imports (torch, numpy, opencv, etc.)
- CUDA availability
- Checkpoint file existence
- Git repository status

### Resume Installation

If installation was interrupted (network error, system crash):

```bash
python scripts/install_wizard.py -r
```

The wizard detects incomplete components and offers to resume.

### Install Specific Component

Skip the menu and install directly:

```bash
python scripts/install_wizard.py -C gvhmr
python scripts/install_wizard.py -C comfyui
python scripts/install_wizard.py -C pytorch
```

Valid components:
- `core` - Python dependencies only
- `pytorch` - PyTorch with CUDA
- `colmap` - COLMAP (built from source)
- `mocap_core` - All motion capture tools
- `gvhmr` - GVHMR motion capture
- `comfyui` - ComfyUI and custom nodes

## Installation Menus

### Main Menu

```
What would you like to install?
1. Core pipeline only (COLMAP, roto)
2. Core + ComfyUI (workflows ready to use)
3. Full stack (Core + ComfyUI + Motion capture)
4. Custom selection
5. Nothing (check only)
```

**Option 1** installs:
- PyTorch with CUDA
- COLMAP (built from source)
- Python dependencies

**Option 2** adds:
- ComfyUI
- VideoHelperSuite custom node
- DepthAnythingV3 custom node
- SAM3 custom node

**Option 3** adds:
- GVHMR (motion capture)
- All checkpoints (automatic download)

**Option 4** shows a detailed component checklist.

**Option 5** runs checks without installing anything (same as `-c`).

### Custom Component Selection

Choose individual components:

```
[Core Components]
  [ ] PyTorch with CUDA
  [ ] COLMAP (Structure-from-Motion)
  [ ] Python dependencies

[ComfyUI]
  [ ] ComfyUI base
  [ ] VideoHelperSuite
  [ ] DepthAnythingV3
  [ ] SAM3

[Motion Capture]
  [ ] GVHMR (motion capture)
```

Use spacebar to toggle, Enter to confirm.

## Installation Process

### 1. System Requirements Check

Before installation:

- Checks Python version (3.8+)
- Detects GPU and CUDA
- Verifies disk space (50GB+ recommended)
- Checks for conda installation

### 2. Conda Environment

Automatically creates and activates `vfx-pipeline` environment:

```bash
conda create -n vfx-pipeline python=3.10 -y
conda activate vfx-pipeline
```

If environment exists, wizard asks to recreate or use existing.

### 3. Component Installation

Each component follows this pattern:

1. **Clone repository** (if git-based)
2. **Install dependencies** (pip install from requirements.txt)
3. **Download checkpoints** (automatic with progress bars)
4. **Validate** (import tests, file checks)
5. **Update state** (mark as completed)

Progress shown in real-time:

```
[GVHMR]
  → Cloning repository...
  → Installing dependencies...
  → Downloading checkpoints...
    Progress: 45.2% (542.1/1200.0 MB)
  → Validating installation...
  ✓ Installation complete
```

### 4. Checkpoint Downloads

Checkpoints are downloaded automatically:

**GVHMR:**
- GVHMR model checkpoints (~4.0 GB)

Downloads include:
- Progress bars with percentage and MB transferred
- Resume capability (if interrupted)
- Checksum verification (when available)

### 5. Post-Installation

After installation completes:

1. **Validation tests** run automatically
2. **Configuration files** generated:
   - `.vfx_pipeline/config.json` - Component paths
   - `.vfx_pipeline/activate.sh` - Environment activation script
3. **Installation state** saved to `.vfx_pipeline/install_state.json`

## Configuration Files

### config.json

Generated configuration with all paths:

```json
{
  "conda_env": "vfx-pipeline",
  "python_executable": "/home/user/miniconda3/envs/vfx-pipeline/bin/python",
  "install_dir": "/home/user/shot-gopher/.vfx_pipeline",
  "gvhmr_dir": "/home/user/shot-gopher/.vfx_pipeline/GVHMR",
  "comfyui_dir": "/home/user/shot-gopher/.vfx_pipeline/ComfyUI",
  "cuda_available": true,
  "cuda_version": "12.1"
}
```

### activate.sh

Environment activation script:

```bash
#!/bin/bash
# VFX Pipeline Environment Activation Script

# Activate conda environment
conda activate vfx-pipeline

# Set up Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/.vfx_pipeline/GVHMR"

# Set up environment variables
export VFX_PIPELINE_BASE="/path/to/.vfx_pipeline"
export GVHMR_DIR="/path/to/.vfx_pipeline/GVHMR"
export SMPLX_MODEL_DIR="${VFX_PIPELINE_BASE}/smplx_models"

echo "✓ VFX Pipeline environment activated"
```

Use with:

```bash
source .vfx_pipeline/activate.sh
```

### install_state.json

Tracks installation progress:

```json
{
  "components": {
    "pytorch": "completed",
    "colmap": "completed",
    "gvhmr": "in_progress",
    "comfyui": "completed"
  },
  "last_updated": "2026-01-12 14:30:00",
  "python_version": "3.10.12",
  "cuda_version": "12.1"
}
```

Status values:
- `pending` - Not started
- `in_progress` - Currently installing
- `completed` - Successfully installed
- `failed` - Installation failed

## Resume Capability

If installation is interrupted:

1. State is saved after each component
2. Next run detects incomplete installation
3. Wizard offers to resume from where it stopped

```bash
python scripts/install_wizard.py -r
```

Output:
```
Previous installation detected (incomplete)

Completed: pytorch, colmap, comfyui
In progress: gvhmr

Resume installation? [Y/n]: y
```

## Validation Tests

The wizard runs smoke tests to verify installation:

### Import Tests

```python
import torch
import numpy
import cv2
import trimesh
import pyrender
# ... etc
```

### CUDA Test

```python
import torch
assert torch.cuda.is_available()
print(f"CUDA devices: {torch.cuda.device_count()}")
```

### Checkpoint Tests

Verifies checkpoint files exist and have correct size:

```
[Checkpoint Validation]
  ✓ GVHMR checkpoint found (4.0 GB)
```

### Repository Tests

Checks git repositories are clean:

```
[Repository Status]
  ✓ GVHMR: Clean working directory
  ✓ ComfyUI: Clean working directory
```

## Troubleshooting

### "Conda not found"

Install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Restart shell after installation.

### "CUDA not available"

Check NVIDIA driver:

```bash
nvidia-smi
```

If not installed, install NVIDIA drivers for your GPU.

Verify CUDA version matches PyTorch requirements (11.8 or 12.1).

### "Insufficient disk space"

Wizard requires 50GB+ free space. Check with:

```bash
df -h .
```

Clean up space or choose a different installation directory.

### "Checkpoint download failed"

URLs may be outdated. Manually download and place in correct directories:

- GVHMR: `.vfx_pipeline/GVHMR/checkpoints/`

### "Import errors after installation"

Activate the environment:

```bash
source .vfx_pipeline/activate.sh
```

Verify environment:

```bash
python -c "import torch; print(torch.__version__)"
```

## Advanced Usage

### Custom Installation Directory

By default, wizard installs to `<repo>/.vfx_pipeline/`.

To customize, edit the wizard before running (not recommended).

### Offline Installation

1. Pre-download checkpoints
2. Clone repositories manually
3. Run wizard with `--check-only` to see what's missing
4. Place files in expected locations

### Manual Component Installation

Each component can be installed separately:

```bash
# PyTorch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# GVHMR
cd .vfx_pipeline
git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR
pip install -r requirements.txt
```

Then run `python scripts/install_wizard.py -v` to validate.

## Related Tools

- **[Maintenance](reference/maintenance.md)** - Maintenance and updates after installation
- **[CLI Reference](reference/cli.md)** - Use the installed components

## See Also

- Main documentation: [README.md](README.md)
- Testing guide: [testing.md](testing.md)
- Troubleshooting: [troubleshooting.md](troubleshooting.md)
