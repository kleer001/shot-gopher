# VFX Pipeline

An automated VFX pipeline built on ComfyUI for production-ready outputs from raw footage. Combines modern ML models with traditional computer vision to generate depth maps, segmentation masks, clean plates, camera solves, and 3D reconstructions with minimal manual intervention.

## Overview

This pipeline automates first-pass VFX prep work that traditionally requires manual labor. Ingest a movie file, get production-ready outputs following industry conventions (1001 frame numbering, PNG sequences, etc.). Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

**Target workflow:** Run the pipeline overnight, come back to usable first-pass outputs ready for VFX compositing and matchmove.

## Capabilities

- **Frame extraction** - Convert video files to industry-standard PNG sequences (1001+ numbering)
- **Depth estimation** - Monocular depth maps with temporal consistency (Depth Anything V3)
- **Segmentation/Rotoscoping** - Text-prompted video segmentation for dynamic object masking (SAM3)
- **Matte refinement** - Alpha matte generation for human subjects (MatAnyone)
- **Clean plate generation** - Automated inpainting to remove objects from footage (ProPainter)
- **Camera tracking** - Structure-from-Motion camera solves with bundle adjustment (COLMAP)
- **3D reconstruction** - Dense point clouds and mesh generation from multi-view footage
- **Material decomposition** - PBR material extraction via GS-IR
  - Albedo maps (diffuse color)
  - Roughness maps (surface specularity)
  - Metallic maps (metallic vs dielectric)
  - Normal maps (surface orientation)
  - Environment lighting (HDR environment map)
- **Camera export** - Export to Alembic/JSON for Nuke, Maya, Houdini, Blender
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

**Quick install** (recommended):
```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap.sh | bash
```

Or manually:
```bash
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python scripts/install_wizard.py
```

**Basic usage:**
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

## Project Structure

Output follows VFX production conventions:
```
../vfx_projects/Shot_Name/
├── source/frames/      # Input frames (frame_1001.png, ...)
├── depth/              # Depth maps
├── roto/               # Segmentation masks
├── matte/              # Refined alpha mattes
├── cleanplate/         # Inpainted backgrounds
├── camera/             # Camera data (Alembic, JSON, point clouds, meshes)
└── colmap/             # COLMAP reconstruction data
```

## License

See individual component licenses. This pipeline integrates multiple open-source projects with varying licenses.
