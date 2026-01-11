# VFX Pipeline

A FOSS automated VFX pipeline built on ComfyUI for first-pass rotoscoping, depth extraction, 3D reconstruction, and clean plate generation.

## Goal

**Hands-off batch processing.** Ingest a movie file, get production-ready outputs (depth maps, segmentation masks, clean plates, camera solves, 3D point clouds) with minimal manual intervention. Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

## Why This Exists

Traditional VFX prep work (roto, depth, clean plates) is tedious. Modern ML models (SAM3, Depth Anything V3, ProPainter) can automate 80% of it. This pipeline stitches them together into a single workflow that:

- Takes raw footage in
- Outputs VFX-ready passes out
- Follows real production folder conventions (frame numbering starts at 1001, etc.)

## Quick Start

```bash
# Single command - processes everything
python scripts/run_pipeline.py /path/to/footage.mp4 --name "My_Shot"

# Or run specific stages
python scripts/run_pipeline.py footage.mp4 --stages depth,camera

# Run with COLMAP for accurate camera reconstruction + mesh
python scripts/run_pipeline.py footage.mp4 --stages ingest,colmap,camera --colmap-dense --colmap-mesh
```

Requires:
- ComfyUI running: `cd /path/to/ComfyUI && python main.py --listen`
- COLMAP (for colmap stage): `sudo apt install colmap` or `conda install -c conda-forge colmap`
- GS-IR (for gsir stage): https://github.com/lzhnb/GS-IR

## Architecture

```
movie.mp4 → run_pipeline.py → project folder → VFX passes
                   │
                   ├── Extract frames (ffmpeg)
                   ├── Setup project structure
                   ├── Populate workflow templates with project paths
                   ├── Queue ComfyUI workflows (depth, roto, cleanplate)
                   ├── COLMAP reconstruction (camera, point cloud, mesh)
                   └── Export camera to Alembic/JSON
```

**Core components:**
- **ComfyUI** - Node-based workflow engine (not for image generation here—just ML inference)
- **SAM3** - Segment Anything Model 3 for text-prompted rotoscoping
- **Depth Anything V3** - Monocular depth estimation with temporal consistency
- **ProPainter** - Video inpainting for clean plate generation
- **VideoHelperSuite** - Frame I/O handling
- **COLMAP** - Structure-from-Motion for accurate camera reconstruction and 3D geometry

**Project structure** (per-shot):
```
projects/My_Shot_Name/
├── source/frames/    # Input frames (frame_1001.png, ...)
├── depth/            # Depth maps
├── roto/             # Segmentation masks
├── cleanplate/       # Inpainted plates
├── camera/           # Camera/geometry data
│   ├── extrinsics.json    # Per-frame camera matrices
│   ├── intrinsics.json    # Camera calibration
│   ├── camera.abc         # Alembic camera (for Houdini/Nuke/Maya)
│   ├── pointcloud.ply     # Dense point cloud (if --colmap-dense)
│   ├── mesh.ply           # Scene mesh (if --colmap-mesh)
│   ├── materials/         # PBR material maps (if gsir stage run)
│   └── normals/           # Normal maps (if gsir stage run)
├── colmap/           # COLMAP working directory
│   ├── sparse/       # Sparse reconstruction
│   └── dense/        # Dense reconstruction (if --colmap-dense)
└── workflows/        # Project-specific workflows (with absolute paths)
```

## Current State

### Working
- `scripts/run_pipeline.py` - **Main entry point** - automated pipeline runner
- `scripts/setup_project.py` - Project setup with workflow templating
- `scripts/export_camera.py` - Camera data → Alembic/JSON export (supports both DA3 and COLMAP)
- `scripts/run_colmap.py` - COLMAP SfM/MVS reconstruction wrapper
- `scripts/run_gsir.py` - GS-IR material decomposition wrapper
- `workflow_templates/01_analysis.json` - Depth Anything V3 + camera estimation

### Needs Testing
- `02_segmentation.json` - SAM3 video segmentation (text prompt → masks)
- `03_cleanplate.json` - ProPainter inpainting

### Known Issues
- SAM3 text prompts like "person" don't capture carried items (bags, purses) - need multi-prompt approach
- Frame numbering in ComfyUI SaveImage doesn't support custom start numbers (1001+) - outputs need post-rename or custom node
- Large frame counts (150+) can stall SAM3 propagation - need batching strategy

### Not Yet Implemented
- Automated mask combination (when using multi-prompt segmentation)
- Depth-to-normals conversion for relighting
- Batch processing wrapper (multiple shots in parallel)

## Design Decisions

1. **Frames, not video files** - All processing uses PNG sequences. More flexible, easier to debug, standard in VFX.

2. **1001 frame numbering** - Industry convention. Leaves room for handles/pre-roll.

3. **Templated workflows per project** - Each project gets its own copy of workflow JSONs with absolute paths populated. Avoids symlink management and makes projects self-contained.

4. **Text prompts over manual selection** - For first-pass automation. Manual point-clicking is available but defeats the hands-off goal.

5. **Separate passes, not monolithic workflow** - Depth, roto, and clean plate as individual workflows. Easier to re-run one stage without redoing everything.

6. **COLMAP for accurate camera, DA3 for fast depth** - Two camera sources serve different needs:
   - **Depth Anything V3**: Fast monocular depth maps for compositing. Camera estimates are approximate.
   - **COLMAP**: Accurate Structure-from-Motion camera solves via bundle adjustment. Produces 3D geometry.

## COLMAP Integration

COLMAP provides geometric 3D reconstruction from multiple views, producing:
- **Sparse reconstruction**: Feature matching + bundle adjustment → accurate camera poses
- **Dense reconstruction**: Multi-view stereo → dense 3D point cloud
- **Mesh**: Poisson surface reconstruction from point cloud

### Usage

```bash
# Basic COLMAP reconstruction (sparse only, fast)
python scripts/run_pipeline.py footage.mp4 --stages ingest,colmap,camera

# With dense point cloud
python scripts/run_pipeline.py footage.mp4 --stages ingest,colmap,camera --colmap-dense

# With mesh generation (requires --colmap-dense)
python scripts/run_pipeline.py footage.mp4 --stages ingest,colmap,camera --colmap-dense --colmap-mesh

# Quality presets: low (fast), medium (default), high (accurate)
python scripts/run_pipeline.py footage.mp4 --stages colmap --colmap-quality high
```

### Standalone COLMAP

```bash
# Run COLMAP directly on an existing project
python scripts/run_colmap.py /path/to/projects/My_Shot --dense --mesh

# Check if COLMAP is installed
python scripts/run_colmap.py --check
```

### COLMAP vs DA3

| Aspect | Depth Anything V3 | COLMAP |
|--------|-------------------|--------|
| Camera accuracy | Approximate (monocular) | Geometric (bundle adjustment) |
| Scale | Relative only | Consistent across scene |
| Speed | Fast (GPU inference) | Slower (CPU-bound matching) |
| 3D output | Depth maps only | Point cloud + mesh |
| Requirements | ComfyUI + model | COLMAP binary |
| Best for | Compositing depth | Matchmove, 3D reconstruction |

**Recommendation**: Use COLMAP for camera solves if you need accurate matchmove. Use DA3 depth maps for compositing tasks (holdouts, depth-based effects).

## GS-IR Integration (Material Decomposition)

GS-IR (Gaussian Splatting for Inverse Rendering) extracts PBR material properties from multi-view images:
- **Albedo maps** — Diffuse color without lighting
- **Roughness maps** — Surface roughness for specular
- **Metallic maps** — Metallic vs dielectric
- **Normal maps** — Surface orientation
- **Environment lighting** — Estimated HDR environment

### Prerequisites

1. Install GS-IR from https://github.com/lzhnb/GS-IR
2. Run COLMAP first (GS-IR needs camera poses)
3. CUDA GPU with 12GB+ VRAM

### Usage

```bash
# Full pipeline: ingest → COLMAP → GS-IR → camera export
python scripts/run_pipeline.py footage.mp4 --stages ingest,colmap,gsir,camera

# With custom iterations (longer = better quality)
python scripts/run_pipeline.py footage.mp4 --stages colmap,gsir --gsir-iterations 50000

# Standalone on existing project
python scripts/run_gsir.py /path/to/projects/My_Shot

# Check if GS-IR is installed
python scripts/run_gsir.py --check
```

### Output Structure

```
camera/
├── materials/           # PBR material maps per frame
│   ├── 00000_brdf.png   # Combined: albedo | roughness | metallic
│   ├── 00000_albedo.png # Extracted albedo
│   └── ...
├── normals/             # Normal maps per frame
├── depth_gsir/          # Depth maps from GS-IR
├── environment.png      # Estimated environment lighting
└── gsir_metadata.json   # Export metadata
```

### Pipeline Flow

```
Frames → COLMAP (cameras) → GS-IR (materials) → VFX-ready outputs
              ↓                    ↓
         camera.abc          albedo/roughness/normal maps
```

**Note**: GS-IR training takes 1-3 hours depending on scene complexity and iteration count. The default 35,000 iterations balances quality and time.

## For the Agent

When continuing development:

- **Test with real footage** - The workflows were built from node documentation, not extensive testing. Expect connection/type errors.
- **Check actual node signatures** - ComfyUI node inputs/outputs change between versions. Use `grep -A 20 "INPUT_TYPES" ...` to verify.
- **VRAM is the constraint** - 12GB minimum, 24GB comfortable. Batch sizes and model choices are driven by this.
- **User's system** - Ubuntu, NVIDIA GPU, conda environment at `/media/menser/fauna/META_VFX/VFX/`

The goal is a pipeline that a VFX artist can run overnight and come back to usable first-pass outputs.
