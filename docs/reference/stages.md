# Pipeline Stages

Detailed documentation for each processing stage.

**Quick links**: [CLI Reference](cli.md) | [Troubleshooting](../troubleshooting.md) | [Installation](../installation.md)

---

## Stage Overview

| Stage | Purpose | VRAM | Input → Output |
|-------|---------|------|----------------|
| [ingest](#ingest) | Extract frames | CPU | Video → PNGs |
| [interactive](#interactive) | Interactive segmentation | 4 GB | Browser-based point/box selection |
| [depth](#depth) | Depth maps | 7 GB | Frames → Depth + camera |
| [roto](#roto) | Segmentation | 4 GB | Frames → Masks |
| [mama](#mama) | Matte refinement | 12 GB | Roto masks → Alpha mattes |
| [cleanplate](#cleanplate) | Object removal | 6 GB | Frames + masks → Clean plates |
| [colmap](#colmap) | Camera tracking | 2-4 GB | Frames → 3D reconstruction |
| [mocap](#mocap) | Motion capture | 12 GB | Frames + camera → Pose + mesh |
| [gsir](#gsir) | PBR materials | 8 GB | COLMAP → Albedo, roughness, metallic |
| [camera](#camera) | Export camera | CPU | Camera JSON → Alembic/FBX |

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
- Frame numbering starts at 0001 (ComfyUI/WHAM requirement)
- Zero-padded to 4 digits (supports up to 9999 frames)

---

## interactive

Browser-based interactive segmentation using SAM3 with point/box prompts.

| | |
|---|---|
| **VRAM** | ~4 GB |
| **Input** | `source/frames/*.png` |
| **Output** | `roto/<label>/*.png` |
| **Workflow** | `05_interactive_segmentation.json` |

**Requirements:**
- ComfyUI server running
- SAM3 custom node

```bash
python scripts/run_pipeline.py footage.mp4 -s interactive
```

**How it works:**
1. Opens ComfyUI in browser with interactive segmentation workflow
2. You manually select objects using point/box prompts
3. Name each selection (e.g., "car", "building")
4. Press Enter in terminal when done to continue pipeline

**Use cases:**
- Objects that text prompts can't identify reliably
- Precise control over segmentation boundaries
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

Creates segmentation masks using SAM3 (Segment Anything Model 3).

| | |
|---|---|
| **VRAM** | ~4 GB |
| **Input** | `source/frames/*.png` |
| **Output** | `roto/<prompt>/*.png` (per-prompt subdirectories) |
| **Workflow** | `02_segmentation.json` |

**Requirements:**
- ComfyUI server running
- SAM3 custom node (~3.2 GB model, auto-downloads from public repo)

**⚠️ When to use automatic vs interactive segmentation:**
- **Single person/object:** Automatic roto works well
- **Multiple people:** Use [interactive](#interactive) segmentation instead
- **Non-standard objects:** Use [interactive](#interactive) for objects that text prompts can't reliably identify (specific props, partial views, unusual angles)

**Common issues with automatic multi-person roto:**
- **Dropped segmentation:** People disappear from masks mid-shot when occluded or at frame edges
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

### Multi-Object Segmentation

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

Removes masked objects using ProPainter video inpainting.

| | |
|---|---|
| **VRAM** | ~6 GB |
| **Input** | `source/frames/*.png`, `roto/*/*.png`, optionally `matte/*.png` |
| **Output** | `cleanplate/*.png` |
| **Workflow** | `03_cleanplate.json` |

**Requirements:**
- ComfyUI server running
- Segmentation masks from roto stage

```bash
python scripts/run_pipeline.py footage.mp4 -s roto,cleanplate
```

**Mask handling:**
1. Collects all masks from `roto/` subdirectories
2. Uses VideoMaMa refined mattes if available (from `matte/` directory)
3. Combines into single mask for inpainting

---

## colmap

Structure-from-Motion camera tracking and 3D reconstruction.

| | |
|---|---|
| **VRAM** | ~2-4 GB |
| **Input** | `source/frames/*.png`, optionally `roto/*.png` |
| **Output** | `colmap/sparse/0/`, optionally `colmap/dense/`, `colmap/meshed/` |
| **Workflow** | None (COLMAP binary) |

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

Human motion capture using WHAM + ECON.

| | |
|---|---|
| **VRAM** | ~12 GB |
| **Input** | `source/frames/*.png`, `camera/extrinsics.json` |
| **Output** | `mocap/wham/`, `mocap/econ/`, `mocap/mesh_sequence/` |
| **Workflow** | None (WHAM/ECON binaries) |

**Requirements:**
- WHAM and ECON installed ([Installation guide](../installation.md))
- Camera data from `colmap` or `depth` stage

```bash
python scripts/run_pipeline.py footage.mp4 -s colmap,mocap
```

**Pipeline:**
1. **WHAM** — Extracts world-grounded pose from video
2. **ECON** — Reconstructs clothed 3D human (SMPL-X compatible)
3. **Texture** — Projects video frames onto mesh

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

## camera

Exports camera data to DCC-compatible formats.

| | |
|---|---|
| **VRAM** | CPU only |
| **Input** | `camera/extrinsics.json`, `camera/intrinsics.json` (from COLMAP) |
| **Output** | `camera/camera.abc`, `.chan`, `.csv`, `.jsx`, `.clip` |
| **Workflow** | None (Python export) |

**Important:** Camera data is generated by the **colmap** stage, not depth. Run COLMAP first:

```bash
python scripts/run_pipeline.py footage.mp4 -s colmap,camera
```

**Export formats:**

| Format | Application | Notes |
|--------|-------------|-------|
| `.abc` (Alembic) | Maya, Houdini, Blender | Requires `alembic` package |
| `.chan` | Nuke | Text format, ZXY rotation order |
| `.clip` | Houdini | CHOP channel data |
| `.jsx` | After Effects | JavaScript import script |
| `.csv` | Any | Spreadsheet-compatible |

**Options:**
- `--rotation-order` — Euler rotation order: `xyz` (Maya), `zxy` (Nuke default), `zyx`

**DCC import:**

| Application | Method |
|-------------|--------|
| Maya | File → Import → Alembic |
| Houdini | Alembic SOP or File CHOP (`.clip`) |
| Blender | File → Import → Alembic |
| Nuke | Camera → File → Import chan file |
| After Effects | File → Scripts → Run Script File (`.jsx`) |

---

## Related Documentation

- [CLI Reference](cli.md) — Command reference and examples
- [Troubleshooting](../troubleshooting.md) — Common issues and solutions
- [Installation](../installation.md) — Component setup
