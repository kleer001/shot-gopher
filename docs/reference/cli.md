# Pipeline Usage

> **Before you run:** Shot-Gopher uses destructive workflows. See [The Gopher's Rules](../RulesAndGotchas.md).

Run the VFX pipeline from a single command.

**Quick links**: [Stages](stages.md) | [Troubleshooting](../troubleshooting.md) | [Installation](../installation.md)

---

## Overview

`run_pipeline.py` processes footage through multiple stages:

| Stage | Purpose | VRAM |
|-------|---------|------|
| [ingest](stages.md#ingest) | Extract frames | CPU |
| [interactive](stages.md#interactive) | Interactive roto | 4 GB |
| [depth](stages.md#depth) | Depth maps | 7 GB |
| [roto](stages.md#roto) | Roto masks | 4 GB |
| [mama](stages.md#mama) | Matte refinement | 12 GB |
| [cleanplate](stages.md#cleanplate) | Clean plate (static camera) | ~2 GB |
| [matchmove_camera](stages.md#matchmove_camera) | Camera tracking | 2-4 GB |
| [mocap](stages.md#mocap) | Motion capture | 12 GB |
| [gsir](stages.md#gsir) | PBR materials | 8 GB |

---

## Quick Start

```bash
# User-friendly TUI (recommended for new users)
./shot-gopher

# Full pipeline
python scripts/run_pipeline.py footage.mp4 -n "MyShot"

# Re-run last project (auto-detects most recent)
python scripts/run_pipeline.py -s roto,cleanplate

# Specific stages
python scripts/run_pipeline.py footage.mp4 -s depth,roto,cleanplate

# List stages
python scripts/run_pipeline.py footage.mp4 --list-stages
```

---

## Command Line Reference

### Core Options

| Short | Long | Description |
|-------|------|-------------|
| `-n` | `--name` | Project name (default: filename) |
| `-p` | `--projects-dir` | Output directory (default: `../vfx_projects`) |
| `-s` | `--stages` | Stages to run, comma-separated |
| `-f` | `--fps` | Override frame rate (default: auto-detect) |
| `-e` | `--skip-existing` | Skip stages with existing output |
| | `--list-stages` | List available stages and exit |
| `-c` | `--comfyui-url` | ComfyUI URL (default: `http://127.0.0.1:8188`) |

### Roto Options

| Long | Description |
|------|-------------|
| `--prompt` | Roto targets (default: `person`). Comma-separated: `person,bag,ball` |
| `--separate-instances` | Split multi-person masks into `person_0/`, `person_1/`, etc. (default: on) |
| `--no-separate-instances` | Combine all instances into single mask |
| `--start-frame` | Frame to start roto from (enables bidirectional propagation). Use when subject isn't visible on first frame |

### Matchmove Camera Options

| Short | Long | Description |
|-------|------|-------------|
| `-q` | `--matchmove-camera-quality` | `low`, `medium` (default), `high`, or `slow` |
| `-d` | `--matchmove-camera-dense` | Run dense reconstruction |
| `-m` | `--matchmove-camera-mesh` | Generate mesh (requires `-d`) |
| `-M` | `--matchmove-camera-no-masks` | Don't use roto masks for tracking |
| | `--matchmove-camera-max-size` | Max image dimension (downscales larger, use 1000-2000 for speed) |

### GS-IR Options

| Short | Long | Description |
|-------|------|-------------|
| `-i` | `--gsir-iterations` | Training iterations (default: 35000) |
| `-g` | `--gsir-path` | GS-IR installation path |

### Mocap Options

| Long | Description |
|------|-------------|
| `--mocap-person` | Roto person to isolate (e.g., `person_00`). Composites source frames with roto matte for single-person tracking |
| `--mocap-start-frame` | Start frame for mocap (1-indexed). Use when person enters late |
| `--mocap-end-frame` | End frame for mocap (1-indexed). Use when person exits early |
| `--mocap-gender` | Body model gender: `neutral`, `male`, `female` (default: `neutral`) |
| `--mocap-export` | Auto-export formats: `abc`, `usd`, `obj`, `none` (default: `abc,usd`) |

### Automation Options

| Long | Description |
|------|-------------|
| `--no-auto-comfyui` | Don't auto-start ComfyUI |
| `--auto-movie` | Generate preview MP4s for each stage |
| `--no-overwrite` | Keep existing output files |
| `--gpu-profile` | Log GPU VRAM usage to `project/gpu_profile.log` |

---

## Examples

### Interactive Roto (Recommended for Multiple People)

```bash
python scripts/run_pipeline.py footage.mp4 -s interactive
```

Opens ComfyUI in browser for manual point/box selection of objects.

**⚠️ Use interactive roto when:**
- Shot has multiple people (automatic roto may miss people or track inconsistently)
- Objects can't be reliably identified by text prompts
- You need precise control over what gets segmented

**Automatic roto works well for:** Single person or single easily-described object.

### Matchmove Only

```bash
python scripts/run_pipeline.py footage.mp4 -s matchmove_camera
```

### Object Removal

```bash
python scripts/run_pipeline.py footage.mp4 -s roto,cleanplate
```

### Multi-Object Roto

```bash
python scripts/run_pipeline.py footage.mp4 -s roto --prompt "person,bag,ball"
```

### Multi-Person Separation

```bash
# Automatic: multiple instances are separated by default
python scripts/run_pipeline.py footage.mp4 -s roto --prompt "person"
# Creates: roto/mask/, roto/person_00/, roto/person_01/, etc.
```

**⚠️ Automatic multi-person roto is unreliable.** Common issues include:
- **Dropped roto:** People vanish from masks when occluded or near frame edges
- **Identity swapping:** Masks switch between people after they cross paths
- **Stuttering/flickering:** Erratic mask boundaries frame-to-frame
- **Merged masks:** Nearby people combined into one mask
- **Inconsistent detection:** Different person count across frames

For production work with multiple people, use [interactive roto](#interactive-roto-recommended-for-multiple-people) instead.

### Multi-Person Mocap

When a shot has multiple people, use roto isolation to track each person separately:

```bash
# Track first person (uses roto/person_00 matte)
python scripts/run_mocap.py ./projects/MyShot --mocap-person person_00

# Track second person with custom frame range (enters at frame 34, exits at 101)
python scripts/run_mocap.py ./projects/MyShot --mocap-person person_01 \
    --start-frame 34 --end-frame 101
```

Creates separate output folders: `mocap/person_00/`, `mocap/person_01/`

### High-Quality Matchmove Camera

```bash
python scripts/run_pipeline.py footage.mp4 -s matchmove_camera -q high -d -m
```

### Resume After Interruption

```bash
python scripts/run_pipeline.py footage.mp4 -s depth,roto,cleanplate -e
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
├── roto/                # Roto masks
│   ├── mask/            # Combined mask (all prompts)
│   ├── person_00/       # First person instance
│   ├── person_01/       # Second person instance
│   └── combined/        # Consolidated for cleanplate
├── matte/               # VideoMaMa refined mattes
│   ├── person_00/
│   └── person_01/
├── cleanplate/          # Clean plates
├── mmcam/
│   ├── sparse/0/        # Sparse reconstruction
│   ├── dense/           # Dense point cloud (optional)
│   └── meshed/          # Mesh (optional)
├── mocap/
│   └── person/          # Default person output (or person_00/, person_01/)
│       ├── motion.pkl   # GVHMR pose estimates
│       ├── mesh_sequence/  # SMPL-X mesh sequence
│       └── export/      # Exported formats
│           ├── tpose.obj   # T-pose reference mesh
│           ├── motion.abc  # Alembic animation
│           └── motion.usd  # USD animation
├── gsir/
│   ├── model/           # Checkpoints
│   └── materials/       # Albedo, roughness, metallic
├── camera/              # From matchmove_camera stage
│   ├── extrinsics.json  # 4x4 matrices per frame
│   ├── intrinsics.json  # fx, fy, cx, cy
│   ├── camera.abc       # Alembic export
│   ├── camera.chan      # Nuke export
│   └── camera.jsx       # After Effects export
└── preview/             # Preview movies (if --auto-movie)
```

---

## ComfyUI

Pipeline auto-starts ComfyUI for depth and roto stages.

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
python scripts/run_matchmove_camera.py MyShot -q high
python scripts/run_mocap.py MyShot
python scripts/export_camera.py MyShot --fps 24
```

### Batch Processing

```bash
for video in footage/*.mp4; do
    name=$(basename "$video" .mp4)
    python scripts/run_pipeline.py "$video" -n "$name" -s depth,roto,cleanplate -e
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
