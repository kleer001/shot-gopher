# Pipeline Stages

Detailed documentation for each processing stage.

**Quick links**: [Pipeline Usage](run_pipeline.md) | [Troubleshooting](troubleshooting.md) | [Installation](install_wizard.md)

---

## Stage Overview

| Stage | Purpose | VRAM | Input → Output |
|-------|---------|------|----------------|
| [ingest](#ingest) | Extract frames | CPU | Video → PNGs |
| [interactive](#interactive) | Interactive segmentation | 4 GB | Browser-based point/box selection |
| [depth](#depth) | Depth maps | 7 GB | Frames → Depth + camera |
| [roto](#roto) | Segmentation | 4 GB | Frames → Masks |
| [matanyone](#matanyone) | Matte refinement | 9 GB | Person masks → Alpha mattes |
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
| **Output** | `depth/*.png`, `camera/extrinsics.json`, `camera/intrinsics.json` |
| **Workflow** | `01_analysis.json` |

**Requirements:**
- ComfyUI server running
- DepthAnythingV3 custom node

```bash
python scripts/run_pipeline.py footage.mp4 -s depth
```

**Notes:**
- Also generates camera data from depth (useful when COLMAP fails)
- Camera data enables mocap stage without running COLMAP

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
- **Multiple people:** Use [interactive](#interactive) segmentation instead—automatic detection may miss people or produce inconsistent tracking across frames
- **Non-standard objects:** Use [interactive](#interactive) for objects that text prompts can't reliably identify (specific props, partial views, unusual angles)

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

## matanyone

Refines person masks into production-quality alpha mattes.

| | |
|---|---|
| **VRAM** | ~9 GB |
| **Input** | `roto/person_00/*.png`, `roto/person_01/*.png`, etc. |
| **Output** | `matte/person_00/*.png`, `matte/person_01/*.png`, etc. |
| **Workflow** | `04_matanyone.json` |

**Requirements:**
- ComfyUI server running
- ComfyUI-MatAnyone custom node
- MatAnyone checkpoint (~450 MB)

```bash
python scripts/run_pipeline.py footage.mp4 -s roto,matanyone
```

**What it does:**
- Refines rough SAM3 masks into clean edges (hair, clothing detail)
- Applies temporal consistency across frames
- Combines multiple mattes into `roto/combined/` for cleanplate

**Limitations:**
- **People only** — trained on humans, won't improve car/object masks
- Skipped automatically if no person masks exist

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
2. Substitutes MatAnyone mattes for person masks (if available)
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
- WHAM and ECON installed ([Installation guide](install_wizard.md))
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
| **Input** | `camera/extrinsics.json`, `camera/intrinsics.json` |
| **Output** | `camera/camera.abc`, optionally `camera/camera.fbx` |
| **Workflow** | None (Python export) |

```bash
python scripts/run_pipeline.py footage.mp4 -s depth,camera
```

**DCC import:**

| Application | Method |
|-------------|--------|
| Maya | File → Import → Alembic |
| Houdini | Alembic SOP |
| Blender | File → Import → Alembic |
| Nuke | ReadGeo node |

---

## Related Documentation

- [Pipeline Usage](run_pipeline.md) — Command reference and examples
- [Troubleshooting](troubleshooting.md) — Common issues and solutions
- [Installation](install_wizard.md) — Component setup
