# VFX Pipeline v0.01

An automated VFX pipeline built on ComfyUI for production-ready outputs from raw footage. Combines modern ML models with traditional computer vision to generate depth maps, segmentation masks, clean plates, camera solves, and 3D reconstructions with minimal manual intervention.

## Overview

This pipeline automates first-pass VFX prep work that traditionally requires manual labor. Ingest a movie file, get production-ready outputs following industry conventions (PNG sequences, etc.). Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

**Target workflow:** Run the pipeline overnight, come back to usable first-pass outputs ready for VFX compositing and matchmove.

<details>
<summary><strong>Capabilities</strong></summary>

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

</details>

<details>
<summary><strong>Tools & Dependencies</strong></summary>

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

</details>

## Getting Started

### Linux

Use Docker for NVIDIA GPU support and isolated environment.


```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/scripts/bootstrap_docker.sh | bash
```

**Prerequisites:** NVIDIA GPU with driver, Docker with nvidia-container-toolkit

**Run:** `bash scripts/run_docker.sh --name MyProject /workspace/projects/video.mp4`

---

### Windows

Native Windows unsupported. Use WSL2 + Docker:

1. Install WSL2: `wsl --install` or visit https://aka.ms/wsl
2. Install Docker Desktop with WSL2 backend enabled
3. Run from WSL2 terminal:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/scripts/bootstrap_docker.sh | bash
   ```

**Prerequisites:** Windows 10 2004+ or Windows 11, NVIDIA GPU with driver

---

### macOS

Use Conda for GPU access (Docker can't access Metal/AMD GPUs on macOS).

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/scripts/bootstrap_conda.sh | bash
```

**Prerequisites:** macOS 11+, Apple Silicon recommended (Intel Macs slower)

**Run:** `python scripts/run_pipeline.py video.mp4 -s ingest,depth,roto,cleanplate,colmap,camera`

---

### After Cloning

If you've already cloned the repo, run the wizard directly:
```bash
python scripts/install_wizard.py --docker  # Docker
python scripts/install_wizard.py           # Conda
```

## Running Your First Project

After installation, you're ready to process your first video. The pipeline supports both Docker and Conda environments with similar workflows.

**Quick start examples:**

```bash
# Docker
bash scripts/run_docker.sh --name MyProject --stages all video.mp4

# Conda
python scripts/run_pipeline.py video.mp4 -s ingest,depth,roto,cleanplate,colmap,camera
```

**See [Your First Project Guide](docs/your_first_project.md) for complete walkthrough, examples, and troubleshooting.**

## Documentation

Complete documentation available in [docs/](docs/):
- [Your First Project](docs/your_first_project.md) - Complete walkthrough for running your first pipeline
- [Docker Guide](docs/README-DOCKER.md) - Complete Docker setup, usage, and troubleshooting
- [Installation Guide](docs/install_wizard.md) - Detailed setup instructions
- [Pipeline Reference](docs/run_pipeline.md) - Command-line usage and options
- [Component Scripts](docs/component_scripts.md) - Individual tool documentation
- [Maintenance](docs/janitor.md) - System health and updates
- [Windows Compatibility](docs/windows-compatibility.md) - Roadmap for Windows support

<details>
<summary><strong>Project Structure</strong></summary>

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

</details>

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
- Model checkpoints (Depth Anything V3, SAM3): 1.0 GB

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

<details>
<summary><strong>Tool Limitations by Shot Type</strong></summary>

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

</details>

## License

See individual component licenses. This pipeline integrates multiple open-source projects with varying licenses.
