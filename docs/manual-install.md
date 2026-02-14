# Manual Installation Guide

Step-by-step instructions for installing ShotGopher without using the automated wizard.

## Prerequisites

### All Platforms

- Git
- Python 3.10+
- FFmpeg
- NVIDIA GPU with CUDA support (12GB+ VRAM recommended)
- 50GB+ free disk space

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y git python3 python3-pip ffmpeg
```

### macOS

```bash
brew install git python@3.10 ffmpeg
```

### Windows

Install via winget, chocolatey, or manual download:

```powershell
winget install Git.Git
winget install Python.Python.3.10
winget install Gyan.FFmpeg
```

Or with Chocolatey:

```powershell
choco install git python310 ffmpeg -y
```

## Step 1: Clone the Repository

```bash
git clone https://github.com/kleer001/shot-gopher.git
cd shot-gopher
```

## Step 2: Create Python Environment

### Option A: Conda (Recommended)

```bash
conda create -n vfx-pipeline python=3.10 -y
conda activate vfx-pipeline
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

## Step 3: Install PyTorch with CUDA

Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command for your CUDA version.

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Verify:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Install ComfyUI

```bash
mkdir -p .vfx_pipeline
cd .vfx_pipeline

git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
cd ..
```

### Install ComfyUI Custom Nodes

```bash
cd ComfyUI/custom_nodes

# VideoHelperSuite
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
pip install -r ComfyUI-VideoHelperSuite/requirements.txt

# Video Depth Anything
git clone https://github.com/kijai/ComfyUI-DepthAnythingV2.git
pip install -r ComfyUI-DepthAnythingV2/requirements.txt

# SAM3 (Segment Anything 3)
git clone https://github.com/1038lab/ComfyUI-SAM3.git
pip install -r ComfyUI-SAM3/requirements.txt

# ProPainter (clean plates)
git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git

cd ../..
```

## Step 6: Download Model Checkpoints

### SAM3 Models

```bash
mkdir -p ComfyUI/models/sam3
cd ComfyUI/models/sam3

# Download from HuggingFace (no auth required)
wget https://huggingface.co/1038lab/sam3/resolve/main/sam3_hiera_large.pt
cd ../../..
```

### Video Depth Anything

```bash
mkdir -p ComfyUI/models/depth
cd ComfyUI/models/depth

# Download Video Depth Anything model
wget https://huggingface.co/depth-anything/video-depth-anything-large/resolve/main/video_depth_anything_vitl.pth
cd ../../..
```

## Step 7: Install COLMAP (Optional - for camera solving)

### Linux

```bash
sudo apt install -y colmap
```

### macOS

```bash
brew install colmap
```

### Windows

Download from [COLMAP releases](https://github.com/colmap/colmap/releases) and add to PATH, or use conda:

```bash
conda install -c conda-forge colmap
```

## Step 8: Install Motion Capture Tools (Optional)

Required only for the `mocap` stage.

### GVHMR

```bash
cd .vfx_pipeline
git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR
pip install -r requirements.txt

# Download checkpoints from GVHMR project page
cd ..
```

### SMPL-X Models

1. Register at https://smpl-x.is.tue.mpg.de/
2. Download SMPL-X models
3. Extract to `.vfx_pipeline/smplx_models/`

## Step 9: Create Configuration

Create `.vfx_pipeline/config.json`:

```json
{
  "install_dir": ".vfx_pipeline",
  "comfyui_dir": ".vfx_pipeline/ComfyUI",
  "gvhmr_dir": ".vfx_pipeline/GVHMR",
  "cuda_available": true
}
```

## Step 10: Verify Installation

```bash
python scripts/install_wizard.py --check-only
```

This shows what's installed without making changes.

## Directory Structure

After manual installation, your directory should look like:

```
shot-gopher/
├── .vfx_pipeline/
│   ├── ComfyUI/
│   │   ├── custom_nodes/
│   │   │   ├── ComfyUI-VideoHelperSuite/
│   │   │   ├── ComfyUI-DepthAnythingV2/
│   │   │   ├── ComfyUI-SAM3/
│   │   │   └── ComfyUI-Advanced-ControlNet/
│   │   └── models/
│   │       ├── sam3/
│   │       └── depth/
│   ├── GVHMR/
│   │   └── checkpoints/
│   └── config.json
├── scripts/
├── workflow_templates/
└── requirements.txt
```

## Running the Pipeline

After installation:

```bash
# Activate environment
conda activate vfx-pipeline  # or source .venv/bin/activate

# Run pipeline
python scripts/run_pipeline.py your_video.mp4 --name MyProject
```

## Troubleshooting

### CUDA not detected

```bash
nvidia-smi  # Check driver
nvcc --version  # Check CUDA toolkit
python -c "import torch; print(torch.cuda.is_available())"
```

### Import errors

Ensure you're in the correct environment:

```bash
which python  # Should point to your venv/conda
pip list | grep torch  # Verify torch is installed
```

### ComfyUI won't start

Check custom node dependencies:

```bash
cd .vfx_pipeline/ComfyUI
python main.py --cpu  # Test without GPU first
```

## See Also

- [Installation Wizard](installation.md) - Automated installation
- [Troubleshooting](troubleshooting.md) - Common issues
- [First Project](first-project.md) - Running your first project
