# Pipeline Usage

Run the VFX pipeline from a single command.

**Quick links**: [Stages](stages.md) | [Troubleshooting](../troubleshooting.md) | [Installation](../installation.md)

---

## Overview

`run_pipeline.py` processes footage through multiple stages:

| Stage | Purpose | VRAM |
|-------|---------|------|
| [ingest](stages.md#ingest) | Extract frames | CPU |
| [depth](stages.md#depth) | Depth maps | 7 GB |
| [roto](stages.md#roto) | Segmentation masks | 4 GB |
| [matanyone](stages.md#matanyone) | Matte refinement | 9 GB |
| [cleanplate](stages.md#cleanplate) | Object removal | 6 GB |
| [colmap](stages.md#colmap) | Camera tracking | 2-4 GB |
| [mocap](stages.md#mocap) | Motion capture | 12 GB |
| [gsir](stages.md#gsir) | PBR materials | 8 GB |
| [camera](stages.md#camera) | Export camera | CPU |

---

## Quick Start

```bash
# Full pipeline
python scripts/run_pipeline.py footage.mp4 -n "MyShot"

# Specific stages
python scripts/run_pipeline.py footage.mp4 -s depth,roto,cleanplate

# List stages
python scripts/run_pipeline.py footage.mp4 -l
```

---

## Command Line Reference

### Core Options

| Short | Long | Description |
|-------|------|-------------|
| `-n` | `--name` | Project name (default: filename) |
| `-p` | `--projects-dir` | Output directory (default: `../vfx_projects`) |
| `-s` | `--stages` | Stages to run, comma-separated or `all` |
| `-f` | `--fps` | Override frame rate (default: auto-detect) |
| `-e` | `--skip-existing` | Skip stages with existing output |
| `-l` | `--list-stages` | List available stages and exit |
| `-c` | `--comfyui-url` | ComfyUI URL (default: `http://127.0.0.1:8188`) |

### Segmentation Options

| Long | Description |
|------|-------------|
| `--prompt` | Segmentation targets (default: `person`). Comma-separated: `person,bag,ball` |
| `--separate-instances` | Split multi-person masks into `person_0/`, `person_1/`, etc. |

### COLMAP Options

| Short | Long | Description |
|-------|------|-------------|
| `-q` | `--colmap-quality` | `low`, `medium` (default), `high`, or `slow` |
| `-d` | `--colmap-dense` | Run dense reconstruction |
| `-m` | `--colmap-mesh` | Generate mesh (requires `-d`) |
| `-M` | `--colmap-no-masks` | Don't use roto masks for tracking |

### GS-IR Options

| Short | Long | Description |
|-------|------|-------------|
| `-i` | `--gsir-iterations` | Training iterations (default: 35000) |
| `-g` | `--gsir-path` | GS-IR installation path |

### Automation Options

| Long | Description |
|------|-------------|
| `--no-auto-comfyui` | Don't auto-start ComfyUI |
| `--auto-movie` | Generate preview MP4s for each stage |
| `--no-overwrite` | Keep existing output files |

---

## Examples

### Matchmove Only

```bash
python scripts/run_pipeline.py footage.mp4 -s depth,camera
```

### Object Removal

```bash
python scripts/run_pipeline.py footage.mp4 -s roto,cleanplate
```

### Multi-Object Segmentation

```bash
python scripts/run_pipeline.py footage.mp4 -s roto --prompt "person,bag,ball"
```

### Multi-Person Separation

```bash
python scripts/run_pipeline.py footage.mp4 -s roto --prompt "person" --separate-instances
```

### High-Quality COLMAP

```bash
python scripts/run_pipeline.py footage.mp4 -s colmap -q high -d -m
```

### Resume After Interruption

```bash
python scripts/run_pipeline.py footage.mp4 -s all -e
```

### Generate Preview Movies

```bash
python scripts/run_pipeline.py footage.mp4 -s depth,roto,cleanplate --auto-movie
```

---

## Project Structure

Pipeline creates this directory structure:

```
../vfx_projects/MyShot/
├── source/frames/       # Extracted frames
├── workflows/           # ComfyUI workflow copies
├── depth/               # Depth maps
├── roto/                # Segmentation masks
│   ├── person/
│   └── combined/        # Consolidated for cleanplate
├── matte/               # MatAnyone refined mattes
├── cleanplate/          # Clean plates
├── colmap/
│   ├── sparse/0/        # Sparse reconstruction
│   ├── dense/           # Dense point cloud (optional)
│   └── meshed/          # Mesh (optional)
├── mocap/
│   ├── wham/            # Pose estimates
│   └── econ/            # 3D reconstructions
├── gsir/
│   ├── model/           # Checkpoints
│   └── materials/       # Albedo, roughness, metallic
├── camera/
│   ├── extrinsics.json
│   ├── intrinsics.json
│   └── camera.abc       # Alembic export
└── preview/             # Preview movies (if --auto-movie)
```

---

## ComfyUI

Pipeline auto-starts ComfyUI for depth, roto, matanyone, and cleanplate stages.

**Manual control:**
```bash
# Disable auto-start
python scripts/run_pipeline.py footage.mp4 -s depth --no-auto-comfyui

# Custom server
python scripts/run_pipeline.py footage.mp4 -c http://192.168.1.100:8188
```

**Workflow customization:** Edit `MyShot/workflows/*.json` for per-shot adjustments.

---

## Advanced

### Individual Scripts

```bash
python scripts/run_colmap.py MyShot -q high
python scripts/run_mocap.py MyShot
python scripts/export_camera.py MyShot --fps 24
```

### Batch Processing

```bash
for video in footage/*.mp4; do
    name=$(basename "$video" .mp4)
    python scripts/run_pipeline.py "$video" -n "$name" -s all -e
done
```

### DCC Import

| Application | Camera | Point Cloud |
|-------------|--------|-------------|
| Maya | `.abc` | — |
| Houdini | `.abc` | `.ply` |
| Blender | `.abc` or `.fbx` | `.ply` |
| Nuke | Convert from `.abc` | — |

---

## Related Documentation

- [Stages](stages.md) — Detailed stage documentation
- [Troubleshooting](../troubleshooting.md) — Common issues and performance tips
- [Installation](../installation.md) — Setup guide
- [Scripts](scripts.md) — Individual script docs
