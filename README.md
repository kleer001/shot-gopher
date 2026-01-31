![ShotGopher Banner](https://i.imgur.com/VP9rmor.png)

![License](https://img.shields.io/github/license/kleer001/shot-gopher)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20|%20macOS%20|%20Windows-lightgrey)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia)
![Tests](https://img.shields.io/github/actions/workflow/status/kleer001/shot-gopher/test.yml?label=tests)

An automated VFX pipeline built on ComfyUI for production-ready outputs from raw footage. Combines modern ML models with traditional computer vision to generate depth maps, segmentation masks, clean plates, camera solves, and 3D reconstructions with minimal manual intervention.

## Overview

This pipeline automates first-pass VFX prep work. Ingest a movie file, get production-ready outputs following industry conventions (PNG sequences, etc.). Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

**Target workflow:** Run the pipeline overnight, come back to usable first-pass outputs ready for VFX compositing and matchmove.

<details>
<summary><strong>Capabilities</strong></summary>

- **Frame extraction** - Convert video files to PNG frame sequences
- **Depth estimation** - Monocular depth maps with temporal consistency (Video Depth Anything)
- **Segmentation/Rotoscoping** - Text-prompted video segmentation for dynamic object masking (SAM3)
- **Matte refinement** - Alpha matte generation for human subjects (VideoMaMa)
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
- **Human motion capture** - World-grounded skeleton tracking and clothed mesh reconstruction (GVHMR preferred, WHAM fallback)
- **Batch processing** - Automated multi-stage pipeline orchestration
- **Web interface** - Browser-based GUI for drag-and-drop operation

</details>

<details>
<summary><strong>Tools & Dependencies</strong></summary>

### Core Pipeline
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based workflow engine for ML inference
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) - Temporally consistent depth estimation
- [Segment Anything Model 2/3](https://github.com/facebookresearch/segment-anything-2) - Text-prompted video segmentation
- [VideoMaMa](https://github.com/hywang66/VideoMaMa) - Video matting for human alpha mattes
- [ProPainter](https://github.com/sczhou/ProPainter) - Video inpainting for clean plates
- [COLMAP](https://colmap.github.io/) - Structure-from-Motion and Multi-View Stereo
- [FFmpeg](https://ffmpeg.org/) - Video/image processing

### Optional Components
- [GS-IR](https://github.com/lzhnb/GS-IR) - Gaussian Splatting for PBR material decomposition
- [GVHMR](https://github.com/zju3dv/GVHMR) - World-grounded human motion tracking (SIGGRAPH Asia 2024, preferred)
- [WHAM](https://github.com/yohanshin/WHAM) - World-grounded human motion tracking (fallback)
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

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.sh | bash
```

**Prerequisites:** NVIDIA GPU with driver, Conda or Miniconda

**Run:** `python scripts/run_pipeline.py video.mp4 -s ingest,interactive,depth,roto,mama,cleanplate,colmap,camera`

---

### Windows

One-liner bootstrap in **PowerShell** (not Command Prompt):

```powershell
irm https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.ps1 | iex
```

> **Getting "irm is not recognized"?** You're in Command Prompt. Open **PowerShell** instead:
> Press `Win+X` → select "Windows PowerShell" or "Terminal", then run the command above.

**Prerequisites:** Windows 10 2004+ or Windows 11, NVIDIA GPU with driver

**Run:** `python scripts/run_pipeline.py video.mp4 -s ingest,interactive,depth,roto,mama,cleanplate,colmap,camera`

---

### macOS

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.sh | bash
```

**Prerequisites:** macOS 11+, Apple Silicon recommended (Intel Macs slower)

**Run:** `python scripts/run_pipeline.py video.mp4 -s ingest,interactive,depth,roto,mama,cleanplate,colmap,camera`

---

### Manual Installation

For step-by-step installation without the wizard, see the [Manual Installation Guide](docs/manual-install.md).

## Running Your First Project

After installation, you're ready to process your first video.

**Quick start examples:**

```bash
# User-friendly TUI (recommended for new users)
./shot-gopher                # Linux/macOS
src\shot-gopher.bat          # Windows

# All stages
python scripts/run_pipeline.py video.mp4 -s ingest,interactive,depth,roto,mama,cleanplate,colmap,mocap,gsir,camera

# 8GB VRAM (skip high-memory stages)
python scripts/run_pipeline.py video.mp4 -s ingest,interactive,depth,roto,cleanplate,colmap,camera

# Re-run stages on last project (auto-detects most recent)
python scripts/run_pipeline.py -s roto,cleanplate
```

**See [Your First Project Guide](docs/first-project.md) for complete walkthrough, examples, and troubleshooting.**

## Documentation

Complete documentation available in [docs/](docs/):
- [Your First Project](docs/first-project.md) - Complete walkthrough for running your first pipeline
- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [CLI Reference](docs/reference/cli.md) - Command-line usage and options
- [Pipeline Stages](docs/reference/stages.md) - Individual stage documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Windows Guide](docs/platforms/windows.md) - Windows support and troubleshooting

<details>
<summary><strong>Project Structure</strong></summary>

Output follows VFX production conventions:
```
../vfx_projects/Shot_Name/
├── source/frames/      # Input frames (frame_0001.png, ...)
├── depth/              # Depth maps
├── roto/               # Segmentation masks
│   ├── mask/           # Combined mask
│   ├── person_00/      # First person instance
│   └── person_01/      # Second person instance
├── matte/              # Refined alpha mattes (person_00/, person_01/, etc.)
├── cleanplate/         # Inpainted backgrounds
├── camera/             # Camera data (Alembic, JSON, point clouds, meshes)
└── colmap/             # COLMAP reconstruction data
```

**Note on frame numbering:** Frame sequences start at 0001 rather than the VFX industry standard of 1001. ComfyUI's SaveImage node output constraints make custom start frame numbering infeasible.

</details>

## System Requirements

**Platform:** Linux, macOS, Windows
**Python:** 3.10 or newer

**Requirements:**
- Git, FFmpeg
- NVIDIA GPU with CUDA support
- Conda or Miniconda

**Note:** macOS supports Conda installation (CPU-only, no GPU acceleration). See [Windows Guide](docs/platforms/windows.md) for Windows-specific details.

## Installation Requirements

### Download Sizes (Approximate)

**Core Pipeline:**
- ComfyUI: 2.0 GB
- PyTorch (with CUDA): 6.0 GB
- Custom nodes (VideoHelperSuite, SAM3, ProPainter, etc.): 5.3 GB
- Model checkpoints (Video Depth Anything, SAM3): 1.0 GB

**Core Total: ~14 GB**

**Optional Components:**
- GVHMR (motion capture, preferred): 4.0 GB
- WHAM (motion capture, fallback): 3.0 GB
- COLMAP (camera tracking): 0.5 GB
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
- Video Depth Anything: ~7 GB (Small model)
- SAM3 (segmentation): ~4 GB
- ProPainter (clean plates): ~6 GB
- VideoMaMa (matte refinement): 12+ GB
- COLMAP: CPU-based (minimal GPU usage)
- GS-IR (material decomposition): 12+ GB
- GVHMR/WHAM (motion capture): 12+ GB

**Minimum Recommendation: 12 GB VRAM** (covers core pipeline including VideoMaMa)
**Comfortable Recommendation: 12 GB VRAM** (supports all features including motion capture and material decomposition)
**Optimal: 24 GB VRAM** (allows higher batch sizes and parallel processing)

Note: NVIDIA GPU with CUDA support required for all ML models.

<details>
<summary><strong>Tool Limitations by Shot Type</strong></summary>

Different components perform best under specific conditions:

| Shot Type | Depth (VDA) | Roto (SAM3) | Clean Plate | Camera (COLMAP) | Material (GS-IR) | MoCap (GVHMR) |
|-----------|-------------|-------------|-------------|-----------------|------------------|-------------------|
| **Static camera** | ✅ | ✅ | ✅ | 🚫 | 🚫 | ⚠️ |
| **Moving camera** | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **Handheld/shaky** | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **Fast motion** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **Low texture** | ✅ | ✅ | ✅ | 🚫 | ⚠️ | ✅ |
| **Full body person** | ✅ | ✅ | ✅ | ✅ | N/A | ✅ |
| **Partial body/occluded** | ✅ | ⚠️ | ⚠️ | ✅ | N/A | ⚠️ |
| **Multiple people** | ✅ | ⚠️ | ⚠️ | ✅ | N/A | ✅ |
| **In-focus background** | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| **Shallow DOF/bokeh** | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ |
| **High contrast lighting** | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| **150+ frames** | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ |

**Legend:**
- ✅ Works well
- ⚠️ Limited/challenging
- 🚫 Not suitable/fails
- N/A Not applicable

</details>

## License

See [LICENSE](LICENSE) for details. This pipeline integrates multiple open-source projects with varying licenses - see [License Audit](docs/LICENSE_AUDIT_REPORT.md) for component details.
