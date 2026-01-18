# VFX Pipeline v0.01

An automated VFX pipeline built on ComfyUI for production-ready outputs from raw footage. Combines modern ML models with traditional computer vision to generate depth maps, segmentation masks, clean plates, camera solves, and 3D reconstructions with minimal manual intervention.

## Overview

This pipeline automates first-pass VFX prep work that traditionally requires manual labor. Ingest a movie file, get production-ready outputs following industry conventions (PNG sequences, etc.). Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

**Target workflow:** Run the pipeline overnight, come back to usable first-pass outputs ready for VFX compositing and matchmove.

## Capabilities

- **Frame extraction** - Convert video files to PNG frame sequences
- **Depth estimation** - Monocular depth maps with temporal consistency (Depth Anything V3)
- **Segmentation/Rotoscoping** - Text-prompted video segmentation for dynamic object masking (SAM3)
- **Matte refinement** - Alpha matte generation for human subjects (MatAnyone)
- **Clean plate generation** - Automated inpainting to remove objects from footage (ProPainter)
- **Camera tracking** - Structure-from-Motion camera solves with bundle adjustment (COLMAP)
- **3D reconstruction** - Dense point clouds and mesh generation from multi-view footage
- **Scene material decomposition** - Extract PBR material properties from multi-view footage via GS-IR (outputs EXR format)
  - Albedo maps (diffuse color without lighting)
  - Roughness maps (surface specularity)
  - Metallic maps (metallic vs dielectric)
  - Normal maps (surface orientation)
  - Environment lighting (HDR environment map)
- **Camera export** - Export to Alembic/JSON for Nuke, Maya, Houdini, Blender, After Effects
- **Human motion capture** - World-grounded skeleton tracking and clothed mesh reconstruction (WHAM + ECON, experimental)
- **Batch processing** - Automated multi-stage pipeline orchestration
- **Web interface** - Browser-based GUI for drag-and-drop operation

## Tools & Dependencies

### Core Pipeline
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based workflow engine for ML inference
- [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3) - Monocular depth estimation
- [Segment Anything Model 2/3](https://github.com/facebookresearch/segment-anything-2) - Text-prompted video segmentation
- [MatAnyone](https://github.com/Shine-Light-Tech/MatAnyone) - Video matting for human alpha mattes
- [ProPainter](https://github.com/sczhou/ProPainter) - Video inpainting for clean plates
- [COLMAP](https://colmap.github.io/) - Structure-from-Motion and Multi-View Stereo
- [FFmpeg](https://ffmpeg.org/) - Video/image processing

### Optional Components
- [GS-IR](https://github.com/lzhnb/GS-IR) - Gaussian Splatting for PBR material decomposition
- [WHAM](https://github.com/yohanshin/WHAM) - World-grounded human motion tracking
- [ECON](https://github.com/YuliangXiu/ECON) - Clothed human reconstruction from monocular video
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) - Parametric body model for motion capture

### ComfyUI Custom Nodes
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) - Frame I/O handling
- [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3) - Depth estimation node

### Python Dependencies
- PyTorch - Deep learning framework
- NumPy, OpenCV, Pillow - Image processing
- trimesh, smplx - 3D geometry (motion capture only)

## Getting Started

### Option 1: Docker (Recommended)

Best for: Quick setup, consistent environment, isolation from system packages.

**Requirements:** Linux or WSL2, NVIDIA GPU, Docker installed.

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.sh | bash
```

The installer will:
- Check prerequisites (NVIDIA driver, Docker)
- Offer to install NVIDIA Container Toolkit if missing
- Download models (~15-20GB)
- Build the Docker image
- Run a test pipeline

**Run the pipeline:**
```bash
cp footage.mp4 ~/VFX-Projects/
./scripts/run_docker.sh --input /workspace/projects/footage.mp4 --name MyProject
```

### Option 2: Local Conda Installation

Best for: Development, customization, macOS (CPU-only), systems without Docker.

**Requirements:** Linux or macOS, NVIDIA GPU (optional on macOS), Conda/Miniconda.

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap.sh | bash
```

Or manually:
```bash
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python scripts/install_wizard.py
```

**Run the pipeline:**
```bash
python scripts/run_pipeline.py footage.mp4 -s ingest,depth,roto,cleanplate,colmap,camera
```

### Which Should I Use?

| Factor | Docker | Local Conda |
|--------|--------|-------------|
| **Setup time** | ~20 min | ~30-60 min |
| **Isolation** | Full container isolation | Conda environment |
| **Disk usage** | Higher (Docker layers) | Lower |
| **Customization** | Rebuild image needed | Direct file editing |
| **WSL2 support** | Yes | Limited |
| **macOS support** | No (no GPU passthrough) | Yes (CPU-only) |
| **Windows native** | No | No |

**Basic usage (local installation):**
```bash
# Process footage with full VFX pipeline
python scripts/run_pipeline.py footage.mp4 -s ingest,depth,roto,cleanplate,colmap,camera

# Web interface
./start_web.py
```

## Documentation

Complete documentation available in [docs/](docs/):
- [Installation Guide](docs/install_wizard.md) - Detailed setup instructions
- [Pipeline Reference](docs/run_pipeline.md) - Command-line usage and options
- [Component Scripts](docs/component_scripts.md) - Individual tool documentation
- [Maintenance](docs/janitor.md) - System health and updates
- [Windows Compatibility](docs/windows-compatibility.md) - Roadmap for Windows support

## Project Structure

Output follows VFX production conventions:
```
../vfx_projects/Shot_Name/
├── source/frames/      # Input frames (frame_0001.png, ...)
├── depth/              # Depth maps
├── roto/               # Segmentation masks
├── matte/              # Refined alpha mattes
├── cleanplate/         # Inpainted backgrounds
├── camera/             # Camera data (Alembic, JSON, point clouds, meshes)
└── colmap/             # COLMAP reconstruction data
```

**Note on frame numbering:** Frame sequences start at 0001 rather than the VFX industry standard of 1001. Unfortunately, ComfyUI's SaveImage node and WHAM's output constraints make custom start frame numbering infeasible. We apologize for this deviation from convention.

## System Requirements

**Platform:** Linux (tested on Ubuntu 20.04+), WSL2 on Windows
**Python:** 3.10 or newer (local install only)

**For Docker installation:**
- Docker with docker-compose
- NVIDIA Container Toolkit (installer can set this up automatically)
- NVIDIA GPU with CUDA support

**For local Conda installation:**
- Git, FFmpeg
- NVIDIA GPU with CUDA support
- Conda or Miniconda

**Note:** macOS supports local Conda installation (CPU-only, no GPU acceleration). Native Windows is not supported—use WSL2 with Docker instead. See [Windows Compatibility](docs/windows-compatibility.md) for details.

## Installation Requirements

### Download Sizes (Approximate)

**Core Pipeline:**
- ComfyUI: 2.0 GB
- PyTorch (with CUDA): 6.0 GB
- Custom nodes (VideoHelperSuite, SAM3, ProPainter, etc.): 5.3 GB
- Model checkpoints (Depth Anything V3, SAM2): 1.0 GB

**Core Total: ~14 GB**

**Optional Components:**
- WHAM (motion capture): 3.0 GB
- COLMAP (camera tracking): 0.5 GB
- ECON (clothed reconstruction): ~2.0 GB
- GS-IR (material decomposition): ~1.5 GB

**Full Installation Total: ~21 GB**

### Model Access Requirements

**SMPL-X Models** (required for motion capture):
- Registration required at https://smpl-x.is.tue.mpg.de/
- Free academic/research license
- Approval typically within 24-48 hours
- Provides parametric body models for human reconstruction

### GPU Memory (VRAM) Requirements

**Per-Component VRAM Usage:**
- Depth Anything V3: ~7 GB (Small model)
- SAM3 (segmentation): ~4 GB
- ProPainter (clean plates): ~6 GB
- MatAnyone (matte refinement): 9+ GB
- COLMAP: CPU-based (minimal GPU usage)
- GS-IR (material decomposition): 12+ GB
- WHAM/ECON (motion capture): 12+ GB

**Minimum Recommendation: 9 GB VRAM** (covers core pipeline including MatAnyone)
**Comfortable Recommendation: 12 GB VRAM** (supports all features including motion capture and material decomposition)
**Optimal: 24 GB VRAM** (allows higher batch sizes and parallel processing)

Note: NVIDIA GPU with CUDA support required for all ML models.

## Tool Limitations by Shot Type

Different components perform best under specific conditions:

| Shot Type | Depth (DA3) | Roto (SAM3) | Clean Plate | Camera (COLMAP) | Material (GS-IR) | MoCap (WHAM/ECON) |
|-----------|-------------|-------------|-------------|-----------------|------------------|-------------------|
| **Static camera** | ✓ | ✓ | ✓ | ✗ | ✗ | ⚠ |
| **Moving camera** | ✓ | ✓ | ⚠ | ✓ | ✓ | ✓ |
| **Handheld/shaky** | ✓ | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| **Fast motion** | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| **Low texture** | ✓ | ✓ | ✓ | ✗ | ⚠ | ✓ |
| **Full body person** | ✓ | ✓ | ✓ | ✓ | N/A | ✓ |
| **Partial body/occluded** | ✓ | ⚠ | ⚠ | ✓ | N/A | ⚠ |
| **Multiple people** | ✓ | ⚠ | ⚠ | ✓ | N/A | ✗ |
| **In-focus background** | ✓ | ✓ | ✓ | ✓ | ✓ | N/A |
| **Shallow DOF/bokeh** | ⚠ | ✓ | ⚠ | ⚠ | ⚠ | ✓ |
| **High contrast lighting** | ✓ | ✓ | ✓ | ✓ | ⚠ | ✓ |
| **150+ frames** | ✓ | ⚠ | ✓ | ✓ | ⚠ | ⚠ |

**Legend:**
- ✓ Works well
- ⚠ Limited/challenging
- ✗ Not suitable/fails
- N/A Not applicable

## License

See individual component licenses. This pipeline integrates multiple open-source projects with varying licenses.
