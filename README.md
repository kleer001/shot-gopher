![ShotGopher Banner](https://i.imgur.com/VP9rmor.png)

![License](https://img.shields.io/github/license/kleer001/shot-gopher)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20|%20macOS%20|%20Windows-lightgrey)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia)
![Tests](https://img.shields.io/github/actions/workflow/status/kleer001/shot-gopher/test.yml?label=tests)

Automated VFX Ingest pipeline. Start with footage, get first pass depth maps, roto, clean plates, camera solves, human matchmove, and 3D reconstructions. 

<details>
<summary><strong>Capabilities</strong></summary>

- **Frame extraction** - Convert video files to PNG frame sequences
- **Depth estimation** - Monocular depth maps with temporal consistency (Video Depth Anything)
- **Rotoscoping** - Text-prompted roto for dynamic object masking (SAM3)
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
- **Human motion capture** - World-grounded skeleton tracking and mesh reconstruction (GVHMR)
- **Batch processing** - Automated multi-stage pipeline orchestration
- **Web interface** - Browser-based GUI for drag-and-drop operation

</details>

<details>
<summary><strong>Tools & Dependencies</strong></summary>

### Core Pipeline
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based workflow engine for ML inference
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) - Temporally consistent depth estimation
- [Segment Anything Model 2/3](https://github.com/facebookresearch/segment-anything-2) - Text-prompted rotoscoping
- [VideoMaMa](https://github.com/hywang66/VideoMaMa) - Video matting for human alpha mattes
- [ProPainter](https://github.com/sczhou/ProPainter) - Video inpainting for clean plates
- [COLMAP](https://colmap.github.io/) - Structure-from-Motion and Multi-View Stereo
- [FFmpeg](https://ffmpeg.org/) - Video/image processing

### Optional Components
- [Blender 4.2 LTS](https://www.blender.org/) - Mesh-to-Alembic export for camera/geometry data
- [GS-IR](https://github.com/lzhnb/GS-IR) - Gaussian Splatting for PBR material decomposition
- [GVHMR](https://github.com/zju3dv/GVHMR) - World-grounded human motion tracking (SIGGRAPH Asia 2024)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) - Parametric body model for motion capture

### ComfyUI Custom Nodes
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) - Frame I/O handling
- [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3) - Depth estimation node

### Python Dependencies
- PyTorch - Deep learning framework
- NumPy, OpenCV, Pillow - Image processing
- trimesh, smplx - 3D geometry (motion capture only)

</details>

---

### ⚠️🚧 Tool Limitations by Shot Type 🚧⚠️

> **🔦 READ THIS BEFORE RUNNING THE PIPELINE 🔦**
>
> Different tools have different requirements. Not all footage works with all stages!

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

🛠️ **Legend:**
- ✅ Works well
- ⚠️ Limited/challenging — may require parameter tuning or produce imperfect results
- 🚫 Not suitable/fails — tool requires different input (e.g., COLMAP needs camera motion)
- N/A Not applicable

🦺 **Key Gotchas:**
- **COLMAP** and **GS-IR** require camera movement — static tripod shots will fail
- **GVHMR** needs full or mostly-visible human bodies
- **Long sequences (150+ frames)** may hit VRAM limits on 12GB cards

## Getting Started

### Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.sh | bash
```

This script will:
- Clone the repository
- Set up a Conda environment
- Install all required dependencies
- Download ML models

**Prerequisites:** Git, Conda/Miniconda, NVIDIA GPU with driver (CUDA)

---

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.ps1 | iex
```

Run PowerShell as Administrator for best results.

**Prerequisites:** Git, Conda/Miniconda, NVIDIA GPU with driver (CUDA)

---

See [Manual Installation](docs/manual-install.md) for step-by-step setup or troubleshooting.



## Documentation

[First Project](docs/first-project.md) · [CLI Reference](docs/reference/cli.md) · [Pipeline Stages](docs/reference/stages.md) · [Troubleshooting](docs/troubleshooting.md)

<details>
<summary><strong>Project Structure</strong></summary>

Output follows VFX production conventions:
```
../vfx_projects/Shot_Name/
├── source/frames/      # Input frames (frame_0001.png, ...)
├── depth/              # Depth maps
├── roto/               # Roto masks
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

## Requirements

**Platform:** Linux, macOS, Windows | **Python:** 3.10+ | **GPU:** NVIDIA with CUDA (12GB+ VRAM recommended)

**Disk:** ~15 GB core, ~35 GB full install | **Dependencies:** Git, FFmpeg, Conda

<details>
<summary><strong>VRAM & Download Details</strong></summary>

| Component | VRAM | Download |
|-----------|------|----------|
| PyTorch + CUDA | — | 6.0 GB |
| ComfyUI + nodes | — | 2.5 GB |
| Video Depth Anything | ~7 GB | 1.5 GB |
| SAM2/3 (roto) | ~4 GB | 0.9 GB |
| ProPainter (clean plates) | ~6 GB | 0.5 GB |
| Blender 4.2 LTS | — | 0.5 GB |
| COLMAP (camera) | CPU | 0.2 GB |
| VideoMaMa (matte) | 12+ GB | 10.0 GB |
| GS-IR (materials) | 12+ GB | 2.0 GB |
| GVHMR (mocap) | 12+ GB | 4.0 GB |

**Motion capture** requires [SMPL-X registration](https://smpl-x.is.tue.mpg.de/) (free academic license, 24-48h approval).

</details>

## License

See [LICENSE](LICENSE) for details. This pipeline integrates multiple open-source projects with varying licenses - see [License Audit](docs/LICENSE_AUDIT_REPORT.md) for component details.
