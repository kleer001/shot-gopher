# VFX Pipeline Documentation

Complete documentation for the comfyui_ingest VFX pipeline toolset.

## Quick Links

- **[Installation Wizard](install_wizard.md)** - Interactive setup and component installation
- **[Pipeline Orchestrator](run_pipeline.md)** - Automated end-to-end VFX processing
- **[Janitor Tool](janitor.md)** - Maintenance, updates, and diagnostics

## Overview

This VFX pipeline provides a complete workflow for processing footage through:
- Frame extraction
- Depth analysis (ComfyUI + Depth-Anything-V3)
- Segmentation/rotoscoping (ComfyUI + SAM2)
- Camera tracking (COLMAP or Depth-Anything-V3)
- Human motion capture (WHAM + TAVA + ECON)
- Material decomposition (GS-IR)
- Camera export (Alembic format)

## Getting Started

### 1. Installation

Use the one-liner bootstrap script:

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap.sh | bash
```

Or manually:

```bash
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python3 scripts/install_wizard.py
```

See **[Installation Wizard Documentation](install_wizard.md)** for details.

### 2. Process Footage

Once installed, process footage with a single command:

```bash
python scripts/run_pipeline.py /path/to/footage.mp4 -n "MyShot"
```

See **[Pipeline Documentation](run_pipeline.md)** for all options.

### 3. Maintenance

Keep your installation healthy:

```bash
python scripts/janitor.py -a  # Run all maintenance checks
```

See **[Janitor Documentation](janitor.md)** for details.

## Tool Reference

### Primary Tools

| Tool | Purpose | Documentation |
|------|---------|---------------|
| `install_wizard.py` | Interactive installation and setup | [View](install_wizard.md) |
| `run_pipeline.py` | Automated VFX pipeline orchestration | [View](run_pipeline.md) |
| `janitor.py` | Maintenance and diagnostics | [View](janitor.md) |

### Component Scripts

These are called by `run_pipeline.py` but can also be used standalone:

| Script | Purpose |
|--------|---------|
| `setup_project.py` | Initialize project directory structure |
| `run_colmap.py` | COLMAP Structure-from-Motion reconstruction |
| `run_segmentation.py` | Video segmentation with dynamic scene detection |
| `run_mocap.py` | Human motion capture pipeline |
| `run_gsir.py` | GS-IR material decomposition |
| `export_camera.py` | Export camera data to Alembic |
| `texture_projection.py` | Project textures onto SMPL-X meshes |

## Architecture

```
comfyui_ingest/
├── scripts/                  # All executable tools
│   ├── install_wizard.py    # Installation wizard
│   ├── run_pipeline.py      # Pipeline orchestrator
│   ├── janitor.py           # Maintenance tool
│   └── ...                  # Component scripts
├── workflow_templates/       # ComfyUI workflow templates
│   ├── 01_analysis.json     # Depth analysis
│   ├── 02_segmentation.json # Segmentation
│   └── 03_cleanplate.json   # Clean plate generation
├── .vfx_pipeline/           # Installation directory (created by wizard)
│   ├── WHAM/                # Motion capture - pose estimation
│   ├── tava/                # Motion capture - avatar generation
│   ├── ECON/                # Motion capture - 3D reconstruction
│   ├── ComfyUI/             # ComfyUI and custom nodes
│   ├── config.json          # Generated configuration
│   └── activate.sh          # Environment activation script
└── docs/                    # This documentation
```

## Project Structure

When you process footage, the pipeline creates this structure:

```
./projects/MyShot/           # Default projects directory
├── source/
│   └── frames/              # Extracted frames (1001.png, 1002.png, ...)
├── workflows/               # ComfyUI workflow copies
├── depth/                   # Depth maps
├── roto/                    # Segmentation masks
├── cleanplate/              # Clean plates
├── colmap/                  # COLMAP reconstruction
│   ├── sparse/              # Sparse 3D model
│   ├── dense/               # Dense point cloud (optional)
│   └── meshed/              # Mesh (optional)
├── mocap/                   # Motion capture data
│   ├── wham/                # WHAM pose estimates
│   ├── tava/                # TAVA avatars
│   └── econ/                # ECON 3D meshes
├── gsir/                    # GS-IR material decomposition
└── camera/                  # Exported camera data
    ├── extrinsics.json      # Camera transforms
    ├── intrinsics.json      # Camera parameters
    └── camera.abc           # Alembic camera export
```

## Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **Disk**: 50GB+ free space
- **RAM**: 16GB+ recommended
- **Python**: 3.8+
- **Conda**: Miniconda or Anaconda

## Common Workflows

### Workflow 1: Quick Depth + Camera

Process footage for depth and camera tracking only:

```bash
python scripts/run_pipeline.py footage.mp4 -s depth,camera
```

### Workflow 2: Full Segmentation Pipeline

Extract, segment, and track:

```bash
python scripts/run_pipeline.py footage.mp4 -s ingest,depth,roto,colmap,camera
```

### Workflow 3: Motion Capture

Full pipeline with human motion capture:

```bash
python scripts/run_pipeline.py footage.mp4 -s all
```

### Workflow 4: Resume After Interruption

Skip already-processed stages:

```bash
python scripts/run_pipeline.py footage.mp4 -s all -e
```

## Troubleshooting

### Installation Issues

Run validation to diagnose problems:

```bash
python scripts/install_wizard.py -v
```

Check health with janitor:

```bash
python scripts/janitor.py -H
```

### ComfyUI Not Running

ComfyUI workflows require the ComfyUI server to be running:

```bash
cd .vfx_pipeline/ComfyUI
python main.py --listen
```

Then run the pipeline in another terminal.

### Missing Checkpoints

Re-download missing model checkpoints:

```bash
python scripts/janitor.py -r
```

### Out of Date Components

Update all git repositories:

```bash
python scripts/janitor.py -u
```

## Environment Activation

After installation, activate the environment:

```bash
source .vfx_pipeline/activate.sh
```

This sets up:
- Conda environment (`vfx-pipeline`)
- Python paths for WHAM, TAVA, ECON
- Environment variables for checkpoint paths

## Contributing

See main repository README for contribution guidelines.

## License

See main repository for license information.

## Support

- **Issues**: https://github.com/kleer001/comfyui_ingest/issues
- **Documentation**: This docs folder
- **Testing Guide**: See `TESTING.md` in repository root
