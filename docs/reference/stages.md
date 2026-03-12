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
| [cleanplate](#cleanplate) | Clean plate (static camera) | ~2 GB | Frames + masks → Clean plates |
| [matchmove_camera](#matchmove_camera) | Camera tracking (VGGSfM) | 2-4 GB | Frames → 3D reconstruction |
| [mocap](#mocap) | Motion capture (SLAHMR) | 12 GB | Frames → Pose + mesh + camera |
| [hands](#hands) | Hand pose estimation (WiLoR) | ~4 GB | Mocap + video → Hand poses |
| [foot_contact](#foot_contact) | Foot contact cleanup (UnderPressure) | ~2 GB | Mocap → Foot-planted motion |
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
- For camera tracking, use the [matchmove_camera](#matchmove_camera) stage instead

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

**Use cases:** Object removal, selective grading, matchmove_camera masking, per-person mocap

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

Generates clean plates by computing per-pixel temporal median using only unmasked (background) samples.

| | |
|---|---|
| **VRAM** | ~2 GB |
| **Input** | `source/frames/*.png`, `roto/*/*.png`, optionally `matte/*.png` |
| **Output** | `cleanplate/*.png` |
| **Workflow** | None (NumPy temporal median) |

**Requirements:**
- Roto masks from roto stage
- **Static camera** (locked-off tripod shot)
- **Moving foreground** (subject must move enough to reveal all background pixels)

```bash
python scripts/run_pipeline.py footage.mp4 -s roto,cleanplate
```

**Mask handling:**
1. Collects all masks from `roto/` subdirectories
2. Uses VideoMaMa refined mattes if available (from `matte/` directory)
3. Combines into single mask for median computation

**GPU profiling:** Use `--gpu-profile` to log VRAM usage per stage to `project/gpu_profile.log`.

---

## matchmove_camera

Structure-from-Motion camera tracking using VGGSfM.

| | |
|---|---|
| **VRAM** | ~2-4 GB |
| **Input** | `source/frames/*.png`, optionally `roto/*.png` |
| **Output** | `mmcam/sparse/0/`, `camera/*.abc`, `.chan`, `.csv`, `.jsx`, `.clip` |
| **Workflow** | None (VGGSfM) |

**Automatic camera export:** After reconstruction completes, camera data is automatically exported to VFX formats:

| Format | Application | Notes |
|--------|-------------|-------|
| `.abc` (Alembic) | Maya, Houdini, Blender | Requires Blender |
| `.chan` | Nuke | Text format, ZXY rotation order |
| `.clip` | Houdini | CHOP channel data |
| `.jsx` | After Effects | JavaScript import script |
| `.csv` | Any | Spreadsheet-compatible |

**📖 Tutorials:**
- [VGGSfM GitHub](https://github.com/facebookresearch/vggsfm) — VGGSfM v2 repository
- [Understanding SfM for VFX](https://www.fxguide.com/fxfeatured/the-art-of-photogrammetry-introduction-to-structure-from-motion/) — FXGuide's photogrammetry guide

VGGSfM is a learned SfM pipeline that handles handheld tracking, narrow-baseline sequences, and large dynamic foreground well. It uses PINHOLE camera model (no distortion estimation).

**VGGSfM details:**
- Non-commercial license (CC BY-NC 4.0)
- Requires separate conda environment and ~200MB model weights (auto-downloaded)
- Long sequences are auto-subsampled (every Nth frame) to fit in VRAM; missing cameras are interpolated

```bash
# Default
python scripts/run_pipeline.py footage.mp4 -s matchmove_camera

# With downscaling for faster processing
python scripts/run_pipeline.py footage.mp4 -s matchmove_camera --mmcam-max-size 1500
```

### Options

| Flag | Description |
|------|-------------|
| `--mmcam-max-size N` | Downscale images to max dimension N |
| `-M` | Disable mask usage |

**Mask integration:** If `roto/` or `matte/` masks exist, VGGSfM uses them to ignore moving objects. Disable with `-M` if masks cause issues.

**COLMAP:** COLMAP is still available for the dense reconstruction stage (`-s dense` — point clouds, mesh, depth/normal maps) but is no longer used for camera tracking.

---

## mocap

Human motion capture using SLAHMR (joint camera + body optimization).

| | |
|---|---|
| **VRAM** | ~12 GB |
| **Input** | `source/frames/*.png` (or video) |
| **Output** | `mocap/<person>/motion.pkl`, `mocap/<person>/export/`, `mocap_camera/` |
| **Workflow** | None (SLAHMR → SMPLX conversion) |

**📖 Tutorials:**
- [SLAHMR Project Page](https://slahmr.github.io/) — Official project (CVPR 2023)
- [SMPL-X Body Model](https://smpl-x.is.tue.mpg.de/) — Understanding the output body model format

**Requirements:**
- SLAHMR installed via install wizard (`slahmr` conda environment)
- Runs for 1-3 hours depending on sequence length

```bash
# Full pipeline: camera tracking + mocap + alignment
python scripts/run_pipeline.py footage.mp4 -s matchmove_camera,mocap

# Mocap only (SLAHMR estimates its own camera)
python scripts/run_pipeline.py footage.mp4 -s ingest,mocap
```

### Engine Selection

| Engine | Flag | Status |
|--------|------|--------|
| **SLAHMR** (default) | `--mocap-engine slahmr` | Joint camera + body optimization, best accuracy |
| **GVHMR** | `--mocap-engine gvhmr` | **Deprecated** — will be removed in a future version |

SLAHMR jointly optimizes camera trajectory and body pose, producing both motion data and camera extrinsics. It uses SMPL-H internally; the pipeline automatically converts to SMPLX for consistent output.

### Camera Data

SLAHMR produces its own camera estimate alongside the body motion:

| Source | Location | When Available |
|--------|----------|----------------|
| VGGSfM | `camera/` | When matchmove_camera stage has run |
| SLAHMR estimate | `mocap_camera/` | Always (auto-exported by mocap stage) |

When both VGGSfM and SLAHMR cameras exist, the pipeline automatically aligns the mocap body into VGGSfM's world space, producing a combined output in `mocap_and_mmcam/` with scene files containing both the animated body mesh and camera.

**Pipeline:**
1. **SLAHMR** — Joint camera + body optimization (SMPL-H)
2. **SMPL-H → SMPLX** — Body model conversion with betas fitting
3. **Camera Export** — SLAHMR camera to Alembic/USD
4. **Mesh Export** — SMPLX animated mesh to Alembic/USD
5. **Alignment** (if mmcam exists) — Body mesh aligned to VGGSfM world space → `mocap_and_mmcam/`
6. **Scene Geometry** — Ground plane at foot height + sparse pointcloud (scaled from VGGSfM to body-mesh metric scale)

### Multi-Person Tracking

For shots with multiple people, use roto isolation to track each person separately:

```bash
# Track person using their roto matte
python scripts/run_mocap.py ./projects/MyShot --mocap-person person_00

# Track person with custom frame range (enters late, exits early)
python scripts/run_mocap.py ./projects/MyShot --mocap-person person_01 \
    --start-frame 34 --end-frame 101
```

The `--mocap-person` flag composites source frames with the corresponding roto matte, isolating a single person on a black background for clean tracking.

### Output Structure

```
mocap/
├── person/                 # Default output (single person)
│   ├── motion.pkl          # SMPLX pose data
│   ├── hand_poses.npz      # Hand poses (after hands stage)
│   ├── foot_contacts.npz   # Foot contacts (after foot_contact stage)
│   ├── slahmr/             # Raw SLAHMR output
│   │   ├── slahmr_stitched.npz  # Stitched SMPL-H chunks
│   │   └── run/            # Optimizer checkpoints
│   └── export/             # Production-ready formats
│       ├── tpose.obj       # T-pose bind reference
│       ├── body_motion.abc # Alembic animation
│       └── body_motion.usd # USD animation
├── person_00/              # First person (when using --mocap-person)
└── person_01/              # Second person
mocap_camera/               # SLAHMR camera estimate
├── extrinsics.json         # Camera-to-world 4x4 matrices
├── intrinsics.json         # Focal length, principal point
├── camera.abc              # Alembic camera
└── camera.usd              # USD camera
mocap_and_mmcam/            # Aligned body + camera (when mmcam exists)
├── body_motion.abc         # Body in VGGSfM world space
├── body_motion.usd
├── scene.abc               # Animated mesh + camera
├── scene.usd
├── combined.abc            # Everything: mesh + camera + ground plane + pointcloud
├── camera.abc              # Output camera (SLAHMR focal + VGGSfM trajectory)
├── camera.usd
├── ground_plane.ply        # Grid mesh at foot-contact height
├── ground_plane.abc
├── ground_plane.json       # Ground height + normal metadata
├── sparse_pointcloud.ply   # VGGSfM reconstruction (scaled to body)
├── sparse_pointcloud.abc
├── extrinsics.json
├── intrinsics.json
└── tpose.obj
```

### Options

| Flag | Description |
|------|-------------|
| `--mocap-engine` | Engine: `slahmr` (default) or `gvhmr` (deprecated) |
| `--mocap-person` | Roto person to isolate (e.g., `person_00`) |
| `--mocap-start-frame` | First frame to process (1-indexed) |
| `--mocap-end-frame` | Last frame to process (1-indexed) |
| `--mocap-gender` | Body model: `neutral`, `male`, `female` |
| `--mocap-no-export` | Skip mesh export |
| `--mocap-fps` | Override export FPS (default: project FPS) |

### Export Formats

| Format | Contents | Use Case |
|--------|----------|----------|
| `tpose.obj` | T-pose mesh with UVs | Bind pose for rigging, cloth sim reference |
| `body_motion.abc` | Animated mesh sequence | Maya, Houdini, Blender, Nuke |
| `body_motion.usd` | Animated mesh sequence | Houdini/Solaris, USD pipelines |
| `scene.abc` | Mesh + camera combined | Single-file import in DCC apps |
| `scene.usd` | Mesh + camera combined | USD pipelines |
| `combined.abc` | Mesh + camera + ground plane + sparse pointcloud | Full scene context in one file |
| `ground_plane.abc` | Grid mesh at foot-contact Y height | Reference plane for compositing |
| `sparse_pointcloud.abc` | VGGSfM 3D reconstruction (scaled to body) | Scene context, layout reference |

### How to Import

| Application | Method |
|-------------|--------|
| Maya | File → Import → Alembic (`combined.abc` for full scene, or individual files) |
| Houdini | Alembic SOP → `combined.abc` or `scene.abc` |
| Blender | File → Import → Alembic |
| Nuke | ReadGeo → `body_motion.abc` or `scene.abc` |
| Solaris | Reference LOP → `body_motion.usd` or `scene.usd` |

**Rigging workflow:**
1. Import `tpose.obj` as bind pose
2. Build rig/skeleton on T-pose
3. Import `body_motion.abc` as animated geometry
4. Transfer animation or use as reference

**Troubleshooting:** See [Mocap issues](troubleshooting.md#motion-capture-requires-camera-data)

---

## hands

Hand pose estimation using WiLoR-mini.

| | |
|---|---|
| **VRAM** | ~4 GB |
| **Input** | `mocap/<person>/` (requires completed mocap stage) |
| **Output** | `mocap/<person>/hand_poses.npz`, re-exported `body_motion.abc`/`.usd` with articulated fingers |
| **Workflow** | None (WiLoR-mini) |

Estimates per-frame 3D hand poses from video using WiLoR-mini, producing SMPLX-compatible axis-angle rotations (15 joints x 3 per hand). After estimation, the mocap body mesh is re-exported with articulated fingers.

**Requirements:**
- Completed mocap stage
- `gvhmr` conda environment (WiLoR-mini is installed there)

```bash
# Full pipeline: mocap then hands
python scripts/run_pipeline.py footage.mp4 -s mocap,hands

# Hands only (mocap already complete)
python scripts/run_pipeline.py footage.mp4 -s hands --mocap-person person
```

**Processing:**
1. Runs WiLoR-mini per-frame on the source video
2. Selects the largest-bbox detection per hand per frame
3. Fills detection gaps via linear interpolation
4. Applies Savitzky-Golay temporal smoothing
5. Saves to `hand_poses.npz` and re-exports body mesh

**Output:**
- `hand_poses.npz` — Left/right hand poses (N x 45 axis-angle) with detection masks
- Re-exported `body_motion.abc`/`.usd` with articulated fingers

---

## foot_contact

Foot contact detection and footskate cleanup using UnderPressure.

| | |
|---|---|
| **VRAM** | ~2 GB |
| **Input** | `mocap/<person>/motion.pkl` (requires completed mocap stage) |
| **Output** | `mocap/<person>/foot_contacts.npz`, re-exported `body_motion.abc`/`.usd` |
| **Workflow** | None (UnderPressure) |

Detects per-frame foot-ground contact and applies IK foot planting to reduce footskate artifacts in the mocap output.

**Requirements:**
- Completed mocap stage
- `gvhmr` conda environment

```bash
# Full pipeline: mocap then foot cleanup
python scripts/run_pipeline.py footage.mp4 -s mocap,foot_contact

# Foot contact only (mocap already complete)
python scripts/run_pipeline.py footage.mp4 -s foot_contact --mocap-person person
```

**Output:**
- `foot_contacts.npz` — Per-frame contact labels
- Re-exported `body_motion.abc`/`.usd` with foot-planted motion

---

## gsir

PBR material extraction using Gaussian Splatting Inverse Rendering.

| | |
|---|---|
| **VRAM** | ~8 GB |
| **Input** | `mmcam/sparse/0/` |
| **Output** | `gsir/model/`, `gsir/materials/` (albedo, roughness, metallic) |
| **Workflow** | None (GS-IR training) |

**Options:**
- `-i N` — Training iterations (default: 35000)
- `-g PATH` — GS-IR installation path

```bash
python scripts/run_pipeline.py footage.mp4 -s matchmove_camera,gsir -i 50000
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
