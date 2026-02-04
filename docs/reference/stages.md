# Pipeline Stages

> **Before you run:** Shot-Gopher uses destructive workflows. See [The Gopher's Rules](../RulesAndGotchas.md).

Detailed documentation for each processing stage.

**Quick links**: [CLI Reference](cli.md) | [Troubleshooting](../troubleshooting.md) | [Installation](../installation.md)

---

## Stage Overview

| Stage | Purpose | VRAM | Input → Output |
|-------|---------|------|----------------|
| [ingest](#ingest) | Extract frames | CPU | Video → PNGs |
| [interactive](#interactive) | Interactive roto | 4 GB | Browser-based point/box selection |
| [depth](#depth) | Depth maps | 7 GB | Frames → Depth |
| [roto](#roto) | Roto masks | 4 GB | Frames → Masks |
| [mama](#mama) | Matte refinement | 12 GB | Roto masks → Alpha mattes |
| [cleanplate](#cleanplate) | Object removal | 6 GB | Frames + masks → Clean plates |
| [colmap](#colmap) | Camera tracking | 2-4 GB | Frames → 3D reconstruction |
| [mocap](#mocap) | Motion capture | 12 GB | Frames + camera → Pose + mesh |
| [gsir](#gsir) | PBR materials | 8 GB | COLMAP → Albedo, roughness, metallic |

---

## ingest

Extracts frames from video using ffmpeg.

| | |
|---|---|
| **VRAM** | CPU only |
| **Input** | Video file (mp4, mov, avi, mkv, webm, mxf) |
| **Output** | `source/frames/frame_0001.png`, `frame_0002.png`, ... |
| **Workflow** | None (ffmpeg) |

**Options:**
- `--fps` — Override frame rate (default: auto-detect)

```bash
python scripts/run_pipeline.py footage.mp4 -s ingest -f 24
```

**Notes:**
- Frame numbering starts at 0001 (ComfyUI requirement)
- Zero-padded to 4 digits (supports up to 9999 frames)

---

## interactive

Browser-based interactive roto using SAM3 with point/box prompts.

| | |
|---|---|
| **VRAM** | ~4 GB |
| **Input** | `source/frames/*.png` |
| **Output** | `roto/<label>/*.png` |
| **Workflow** | `05_interactive_segmentation.json` |

**📖 Tutorials:**
- [**Interactive Roto Guide**](interactive-segmentation.md) — Full walkthrough with tips for multi-object selection
- [ComfyUI-SAM3 GitHub](https://github.com/1038lab/ComfyUI-SAM3) — Node documentation and examples

**Requirements:**
- ComfyUI server running
- SAM3 custom node

```bash
python scripts/run_pipeline.py footage.mp4 -s interactive
```

**How it works:**
1. Opens ComfyUI in browser with interactive roto workflow
2. You manually select objects using point/box prompts
3. Name each selection (e.g., "car", "building")
4. Press Enter in terminal when done to continue pipeline

**Use cases:**
- Objects that text prompts can't identify reliably
- Precise control over roto boundaries
- One-off objects that don't need automation

---

## depth

Generates depth maps using Depth-Anything-V3.

| | |
|---|---|
| **VRAM** | ~7 GB |
| **Input** | `source/frames/*.png` |
| **Output** | `depth/*.png` |
| **Workflow** | `01_analysis.json` |

**Requirements:**
- ComfyUI server running
- DepthAnythingV3 custom node

```bash
python scripts/run_pipeline.py footage.mp4 -s depth
```

**Notes:**
- Generates temporally consistent depth maps for compositing
- For camera tracking, use the [colmap](#colmap) stage instead

---

## roto

Creates roto masks using SAM3 (Segment Anything Model 3).

| | |
|---|---|
| **VRAM** | ~4 GB |
| **Input** | `source/frames/*.png` |
| **Output** | `roto/<prompt>/*.png` (per-prompt subdirectories) |
| **Workflow** | `02_segmentation.json` |

**📖 Tutorials:**
- [ComfyUI-SAM3 GitHub](https://github.com/1038lab/ComfyUI-SAM3) — Node documentation and workflow examples
- [1038lab/sam3 Model](https://huggingface.co/1038lab/sam3) — Model weights and usage info

**Requirements:**
- ComfyUI server running
- SAM3 custom node (~3.2 GB model, auto-downloads from public repo)

**⚠️ When to use automatic vs interactive roto:**
- **Single person/object:** Automatic roto works well
- **Multiple people:** Use [interactive](#interactive) roto instead
- **Non-standard objects:** Use [interactive](#interactive) for objects that text prompts can't reliably identify (specific props, partial views, unusual angles)

**Common issues with automatic multi-person roto:**
- **Dropped roto:** People disappear from masks mid-shot when occluded or at frame edges
- **Identity swapping:** Person A's mask suddenly contains Person B after they cross paths
- **Stuttering/flickering:** Mask boundaries jump erratically frame-to-frame
- **Merged masks:** Two people combined into one mask when standing close together
- **Phantom detection:** Background objects incorrectly identified as people
- **Inconsistent instance count:** Different number of people detected across frames

These issues compound in downstream stages (mama, cleanplate) and are difficult to fix in post.

### Basic Usage

```bash
python scripts/run_pipeline.py footage.mp4 -s roto --prompt "person"
```

### Multi-Object Roto

Segment multiple objects in one run:

```bash
python scripts/run_pipeline.py footage.mp4 -s roto --prompt "person,bag,ball"
```

Creates separate directories:
```
roto/
├── person/
├── bag/
└── ball/
```

### Instance Separation

When multiple instances of the same object are detected, masks are automatically separated into individual directories:

```bash
python scripts/run_pipeline.py footage.mp4 -s roto --prompt "person"
```

Creates:
```
roto/
├── mask/         # Combined mask (all instances)
├── person_00/    # First person
├── person_01/    # Second person
└── person_02/    # Third person
```

Use `--separate-instances` to explicitly enable this behavior (enabled by default when multiple instances detected).

**Use cases:** Object removal, selective grading, COLMAP masking, per-person mocap

---

## mama

Refines roto masks into production-quality alpha mattes using VideoMaMa diffusion-based matting.

| | |
|---|---|
| **VRAM** | ~12 GB (24 GB recommended) |
| **Input** | `roto/<prompt>_00/*.png`, `roto/<prompt>_01/*.png`, etc. (numbered instance directories) |
| **Output** | `matte/<prompt>_00/*.png`, `matte/<prompt>_01/*.png`, etc. |
| **Workflow** | None (VideoMaMa conda environment) |

**📖 Tutorials:**
- [VideoMaMa GitHub](https://github.com/hywang66/VideoMaMa) — Official repository with paper and examples
- [Video Matting Explained (YouTube)](https://www.youtube.com/watch?v=PJgPrRRq9Cs) — Background on video matting techniques

**Requirements:**
- VideoMaMa installed via `python scripts/video_mama_install.py` (~12 GB disk space)
- Separate conda environment (created automatically)

```bash
python scripts/run_pipeline.py footage.mp4 -s roto,mama
```

**What it does:**
- Refines coarse SAM3 masks into soft alpha mattes with fine edge detail
- Uses Stable Video Diffusion for temporal consistency
- Processes videos in chunks (auto-sized based on VRAM)
- Combines multiple mattes into `roto/combined/` for cleanplate

**Processing:**
- Auto-detects GPU VRAM to set optimal chunk size (14 frames for 24GB)
- Clears VRAM between chunks to prevent OOM errors
- Auto-reduces chunk size on OOM and retries

**Notes:**
- Processes numbered roto directories only (e.g., `person_00/`, `bag_01/`)
- Skips unnumbered directories (`person/`, `combined/`, `mask/`)

---

## cleanplate

Removes masked objects from footage. Two methods available:

| Method | Best For | VRAM |
|--------|----------|------|
| **ProPainter** (default) | Moving cameras, complex scenes | ~6 GB (peak ~18 GB) |
| **Temporal Median** | Static cameras, fast turnaround | ~2 GB |

| | |
|---|---|
| **Input** | `source/frames/*.png`, `roto/*/*.png`, optionally `matte/*.png` |
| **Output** | `cleanplate/*.png` |
| **Workflow** | `03_cleanplate.json` (ProPainter only) |

**Requirements:**
- ComfyUI server running (ProPainter only)
- Roto masks from roto stage

```bash
# ProPainter (default) - handles moving cameras
python scripts/run_pipeline.py footage.mp4 -s roto,cleanplate

# Temporal median - fast, static camera only
python scripts/run_pipeline.py footage.mp4 -s roto,cleanplate --cleanplate-median
```

**Mask handling:**
1. Collects all masks from `roto/` subdirectories
2. Uses VideoMaMa refined mattes if available (from `matte/` directory)
3. Combines into single mask for inpainting

### ProPainter Method

AI-powered video inpainting that handles camera motion and complex occlusions.

**Quality tuning (environment variables):**

| Variable | Default | Description |
|----------|---------|-------------|
| `PROPAINTER_INTERNAL_SCALE` | `0.5` | Internal processing resolution (0.1-1.0). Higher = sharper but more VRAM |
| `PROPAINTER_REFINE_ITERS` | `16` | Refinement iterations (compute-bound, not VRAM) |
| `PROPAINTER_NUM_FLOWS` | `20` | Optical flow frames for temporal consistency |

**Example:**
```bash
PROPAINTER_REFINE_ITERS=20 python scripts/run_pipeline.py footage.mp4 -s cleanplate
```

### Temporal Median Method

Computes per-pixel temporal median using only unmasked (background) samples. Fast but requires:
- **Static camera** (locked-off tripod shot)
- **Moving foreground** (subject must move enough to reveal all background pixels)

```bash
python scripts/run_pipeline.py footage.mp4 -s cleanplate --cleanplate-median
```

**GPU profiling:** Use `--gpu-profile` to log VRAM usage per stage to `project/gpu_profile.log`.

---

## colmap

Structure-from-Motion camera tracking and 3D reconstruction.

| | |
|---|---|
| **VRAM** | ~2-4 GB |
| **Input** | `source/frames/*.png`, optionally `roto/*.png` |
| **Output** | `colmap/sparse/0/`, `camera/*.abc`, `.chan`, `.csv`, `.jsx`, `.clip` |
| **Workflow** | None (COLMAP binary) |

**Automatic camera export:** After COLMAP completes, camera data is automatically exported to VFX formats:

| Format | Application | Notes |
|--------|-------------|-------|
| `.abc` (Alembic) | Maya, Houdini, Blender | Requires Blender |
| `.chan` | Nuke | Text format, ZXY rotation order |
| `.clip` | Houdini | CHOP channel data |
| `.jsx` | After Effects | JavaScript import script |
| `.csv` | Any | Spreadsheet-compatible |

**📖 Tutorials:**
- [COLMAP Documentation](https://colmap.github.io/tutorial.html) — Official tutorial and parameter reference
- [COLMAP SfM Workflow (YouTube)](https://www.youtube.com/watch?v=P-EC0DzeVEU) — Visual walkthrough of the reconstruction process
- [Understanding SfM for VFX](https://www.fxguide.com/fxfeatured/the-art-of-photogrammetry-introduction-to-structure-from-motion/) — FXGuide's photogrammetry guide

**Options:**

| Flag | Description |
|------|-------------|
| `-q low` | Fast preview |
| `-q medium` | Default quality |
| `-q high` | Production quality |
| `-q slow` | Minimal camera motion |
| `-d` | Dense reconstruction |
| `-m` | Generate mesh (requires `-d`) |
| `-M` | Disable mask usage |

```bash
# High quality with dense reconstruction
python scripts/run_pipeline.py footage.mp4 -s colmap -q high -d
```

**Mask integration:** If `roto/` masks exist, COLMAP uses them to ignore moving objects. Disable with `-M` if masks cause issues.

**Troubleshooting:** See [COLMAP issues](troubleshooting.md#colmap-reconstruction-failed)

---

## mocap

Human motion capture using GVHMR.

| | |
|---|---|
| **VRAM** | ~12 GB |
| **Input** | `source/frames/*.png`, `camera/extrinsics.json` |
| **Output** | `mocap/motion.pkl`, `mocap/mesh_sequence/` |
| **Workflow** | None (GVHMR) |

**📖 Tutorials:**
- [GVHMR Project Page](https://zju3dv.github.io/gvhmr/) — Official project with paper and demo videos
- [SMPL-X Body Model](https://smpl-x.is.tue.mpg.de/) — Understanding the output body model format

**Requirements:**
- GVHMR installed ([Installation guide](../installation.md))
- Camera data from `colmap` stage

```bash
python scripts/run_pipeline.py footage.mp4 -s colmap,mocap
```

**Pipeline:**
1. **GVHMR** — Extracts world-grounded pose from video (SMPL-X compatible)
2. **Mesh Generation** — Creates animated SMPL-X mesh sequence

**Troubleshooting:** See [Mocap issues](troubleshooting.md#motion-capture-requires-camera-data)

---

## gsir

PBR material extraction using Gaussian Splatting Inverse Rendering.

| | |
|---|---|
| **VRAM** | ~8 GB |
| **Input** | `colmap/sparse/0/` |
| **Output** | `gsir/model/`, `gsir/materials/` (albedo, roughness, metallic) |
| **Workflow** | None (GS-IR training) |

**Options:**
- `-i N` — Training iterations (default: 35000)
- `-g PATH` — GS-IR installation path

```bash
python scripts/run_pipeline.py footage.mp4 -s colmap,gsir -i 50000
```

**Training time:**

| Iterations | Time | Quality |
|------------|------|---------|
| 30k | ~30 min | Preview |
| 35k | ~40 min | Default |
| 50k | ~60 min | High |

---

## Related Documentation

- [CLI Reference](cli.md) — Command reference and examples
- [Troubleshooting](../troubleshooting.md) — Common issues and solutions
- [Installation](../installation.md) — Component setup
