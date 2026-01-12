# Pipeline Orchestrator Documentation

Automated end-to-end VFX processing pipeline.

## Overview

`run_pipeline.py` orchestrates the entire VFX pipeline from a single command. It processes footage through multiple stages:

1. **Ingest** - Extract frames from video
2. **Depth** - Depth map generation (ComfyUI + Depth-Anything-V3)
3. **Roto** - Segmentation masks (ComfyUI + SAM2)
4. **Cleanplate** - Clean plate generation (ComfyUI)
5. **COLMAP** - Camera tracking and 3D reconstruction
6. **Mocap** - Human motion capture (WHAM + TAVA + ECON)
7. **GS-IR** - Material decomposition (Gaussian splatting)
8. **Camera** - Export camera to Alembic format

## Quick Start

### Process Full Pipeline

```bash
python scripts/run_pipeline.py /path/to/footage.mp4 -n "MyShot"
```

### Process Specific Stages

```bash
python scripts/run_pipeline.py footage.mp4 -s depth,camera
```

### List Available Stages

```bash
python scripts/run_pipeline.py -l
```

## Command Line Options

### Short Options

```bash
-n NAME            # Project name
-p DIR             # Projects directory
-s STAGES          # Stages to run (comma-separated)
-c URL             # ComfyUI API URL
-f FPS             # Override frame rate
-e                 # Skip existing output
-l                 # List available stages

# COLMAP options
-q QUALITY         # Quality (low/medium/high)
-d                 # Dense reconstruction
-m                 # Generate mesh
-M                 # Disable masks

# GS-IR options
-i ITERATIONS      # Training iterations
-g PATH            # GS-IR installation path
```

### Long Options

```bash
--name NAME                  # Project name
--projects-dir DIR           # Projects directory (default: ./projects)
--stages STAGES              # Stages to run (comma-separated or "all")
--comfyui-url URL           # ComfyUI API URL (default: http://127.0.0.1:8188)
--fps FPS                    # Override frame rate (default: auto-detect)
--skip-existing             # Skip stages with existing output
--list-stages               # List available stages and exit

# COLMAP options
--colmap-quality QUALITY    # Quality preset: low, medium, high
--colmap-dense             # Run dense reconstruction (slower)
--colmap-mesh              # Generate mesh from dense reconstruction
--colmap-no-masks          # Disable automatic use of segmentation masks

# GS-IR options
--gsir-iterations N        # Total training iterations (default: 35000)
--gsir-path PATH           # Path to GS-IR installation (default: auto-detect)
```

## Usage Examples

### Example 1: Quick Depth + Camera

Extract frames, generate depth maps, and export camera:

```bash
python scripts/run_pipeline.py footage.mp4 -n "Shot01" -s depth,camera
```

This is the fastest workflow for basic VFX work.

### Example 2: Full Segmentation Pipeline

Complete pipeline with segmentation for clean plates:

```bash
python scripts/run_pipeline.py footage.mp4 -n "Shot02" -s ingest,depth,roto,cleanplate,colmap,camera
```

### Example 3: Motion Capture

Process footage with human motion capture:

```bash
python scripts/run_pipeline.py footage.mp4 -n "Actor01" -s all
```

Requires:
- WHAM, TAVA, ECON installed
- Camera data (from COLMAP or Depth-Anything-V3)

### Example 4: High-Quality COLMAP

Run COLMAP with dense reconstruction and mesh generation:

```bash
python scripts/run_pipeline.py footage.mp4 -s colmap,camera -q high -d -m
```

Parameters:
- `-q high` - High quality settings (slower, more accurate)
- `-d` - Dense reconstruction (point cloud)
- `-m` - Mesh generation (requires dense)

### Example 5: Resume After Interruption

Skip already-processed stages:

```bash
python scripts/run_pipeline.py footage.mp4 -s all -e
```

The `-e` flag checks for existing output before running each stage.

### Example 6: Custom FPS

Override automatic frame rate detection:

```bash
python scripts/run_pipeline.py footage.mp4 -f 30
```

Useful for:
- Image sequences without metadata
- Forcing specific frame rates

### Example 7: Custom Project Location

Specify output directory:

```bash
python scripts/run_pipeline.py footage.mp4 -p /mnt/storage/vfx_shots
```

### Example 8: Material Decomposition

Run GS-IR for PBR material extraction:

```bash
python scripts/run_pipeline.py footage.mp4 -s colmap,gsir -i 50000
```

Higher iterations (`-i`) produce better quality but take longer.

## Stages

### ingest - Frame Extraction

Extracts frames from video file using ffmpeg.

**Input**: Movie file (mp4, mov, avi, mkv, webm, mxf)
**Output**: `source/frames/frame_1001.png, frame_1002.png, ...`

**Options**:
- `--fps` - Override frame rate (default: auto-detect from video metadata)

**Example**:
```bash
python scripts/run_pipeline.py footage.mp4 -s ingest -f 24
```

**Frame Numbering**:
- Starts at 1001 (standard VFX convention)
- Zero-padded 4 digits (1001, 1002, ..., 9999)

### depth - Depth Analysis

Generates depth maps using Depth-Anything-V3.

**Input**: `source/frames/*.png`
**Output**: `depth/*.png`, `camera/extrinsics.json`, `camera/intrinsics.json`

**Requirements**:
- ComfyUI server running
- DepthAnythingV3 custom node installed

**Workflow**: `01_analysis.json`

**Example**:
```bash
# Start ComfyUI first
cd .vfx_pipeline/ComfyUI
python main.py --listen &

# Run depth stage
python scripts/run_pipeline.py footage.mp4 -s depth
```

### roto - Segmentation

Creates segmentation masks using SAM2.

**Input**: `source/frames/*.png`
**Output**: `roto/*.png` (binary masks)

**Requirements**:
- ComfyUI server running
- SAM2 custom node installed

**Workflow**: `02_segmentation.json`

**Use Cases**:
- Object removal (clean plates)
- Selective color grading
- COLMAP masking (improves camera tracking)

### cleanplate - Clean Plate Generation

Generates clean plates by removing objects from segmented areas.

**Input**: `source/frames/*.png`, `roto/*.png`
**Output**: `cleanplate/*.png`

**Requirements**:
- ComfyUI server running
- Segmentation masks from `roto` stage

**Workflow**: `03_cleanplate.json`

### colmap - Camera Tracking

COLMAP Structure-from-Motion reconstruction.

**Input**: `source/frames/*.png`, optional: `roto/*.png` (masks)
**Output**:
- `colmap/sparse/0/` - Sparse 3D model
- `colmap/dense/` - Dense point cloud (if `--colmap-dense`)
- `colmap/meshed/` - Mesh (if `--colmap-mesh`)

**Options**:
- `-q low|medium|high` - Quality preset (default: medium)
- `-d` - Run dense reconstruction
- `-m` - Generate mesh (requires `-d`)
- `-M` - Disable automatic mask usage

**Quality Presets**:

| Preset | Feature Extraction | Matching | Speed | Accuracy |
|--------|-------------------|----------|-------|----------|
| Low | Fast | Vocab tree | Fast | Lower |
| Medium | Normal | Vocab tree | Medium | Good |
| High | Detailed | Exhaustive | Slow | Best |

**Example**:
```bash
# High quality with dense reconstruction
python scripts/run_pipeline.py footage.mp4 -s colmap -q high -d
```

**Mask Integration**:
If `roto/*.png` exists, COLMAP automatically uses masks to ignore moving objects and improve camera tracking.

### mocap - Motion Capture

Human motion capture using WHAM + TAVA + ECON.

**Input**:
- `source/frames/*.png`
- `camera/extrinsics.json` (from COLMAP or depth stage)

**Output**:
- `mocap/wham/` - WHAM pose estimates
- `mocap/tava/` - TAVA avatar mesh sequences
- `mocap/econ/` - ECON 3D reconstructions

**Requirements**:
- WHAM, TAVA, ECON installed
- Camera data (run `colmap` or `depth` stage first)

**Example**:
```bash
python scripts/run_pipeline.py actor_footage.mp4 -s colmap,mocap
```

**Pipeline**:
1. WHAM extracts pose from video
2. TAVA generates animatable avatar
3. ECON reconstructs detailed 3D clothed human
4. Texture projection (optional, use `--skip-texture` to disable)

### gsir - Material Decomposition

GS-IR (Gaussian Splatting Inverse Rendering) for PBR material extraction.

**Input**: `colmap/sparse/0/` (COLMAP sparse model)
**Output**:
- `gsir/model/chkpnt{N}.pth` - Trained model checkpoints
- `gsir/materials/` - Extracted PBR textures (albedo, roughness, metallic)

**Options**:
- `-i ITERATIONS` - Total training iterations (default: 35000)
- `-g PATH` - Path to GS-IR installation

**Example**:
```bash
python scripts/run_pipeline.py footage.mp4 -s colmap,gsir -i 50000
```

**Training Time**:
- 30k iterations: ~30 minutes (quick preview)
- 35k iterations: ~40 minutes (default quality)
- 50k iterations: ~60 minutes (high quality)

### camera - Camera Export

Export camera data to Alembic format for use in DCCs (Maya, Houdini, Blender).

**Input**: `camera/extrinsics.json`, `camera/intrinsics.json`
**Output**:
- `camera/camera.abc` - Alembic camera export
- `camera/camera.fbx` - FBX camera export (if available)

**Example**:
```bash
python scripts/run_pipeline.py footage.mp4 -s depth,camera
```

**Format**:
- Alembic (.abc) - Industry standard, works in all major DCCs
- FBX (.fbx) - Optional, if fbx SDK available

## Project Structure

Pipeline creates this directory structure:

```
./projects/MyShot/               # Project directory
├── source/
│   └── frames/                  # Extracted frames
│       ├── frame_1001.png
│       ├── frame_1002.png
│       └── ...
├── workflows/                   # ComfyUI workflow copies
│   ├── 01_analysis.json
│   ├── 02_segmentation.json
│   └── 03_cleanplate.json
├── depth/                       # Depth maps
│   ├── depth_1001.png
│   ├── depth_1002.png
│   └── ...
├── roto/                        # Segmentation masks
│   ├── mask_1001.png
│   ├── mask_1002.png
│   └── ...
├── cleanplate/                  # Clean plates
│   ├── clean_1001.png
│   ├── clean_1002.png
│   └── ...
├── colmap/                      # COLMAP reconstruction
│   ├── sparse/
│   │   └── 0/                   # Sparse 3D model
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   ├── dense/                   # Dense point cloud (optional)
│   │   ├── fused.ply
│   │   └── stereo/
│   └── meshed/                  # Mesh (optional)
│       └── meshed-poisson.ply
├── mocap/                       # Motion capture
│   ├── wham/
│   │   └── poses.pkl
│   ├── tava/
│   │   └── mesh_sequence.pkl
│   └── econ/
│       └── textured_meshes/
├── gsir/                        # GS-IR materials
│   ├── model/
│   │   ├── chkpnt30000.pth
│   │   └── chkpnt35000.pth
│   └── materials/
│       ├── albedo.png
│       ├── roughness.png
│       └── metallic.png
└── camera/                      # Camera export
    ├── extrinsics.json          # Camera transforms per frame
    ├── intrinsics.json          # Camera parameters
    ├── camera.abc               # Alembic export
    └── camera.fbx               # FBX export (optional)
```

## ComfyUI Integration

### Starting ComfyUI

Pipeline requires ComfyUI server for depth, roto, and cleanplate stages:

```bash
# Terminal 1: Start ComfyUI
cd .vfx_pipeline/ComfyUI
python main.py --listen

# Terminal 2: Run pipeline
python scripts/run_pipeline.py footage.mp4 -s depth,roto
```

### Custom ComfyUI URL

If ComfyUI runs on different host/port:

```bash
python scripts/run_pipeline.py footage.mp4 -c http://192.168.1.100:8188
```

### Workflow Files

Pipeline copies workflow templates to project directory:

```
workflow_templates/01_analysis.json → MyShot/workflows/01_analysis.json
```

Edit workflows in project directory to customize per-shot.

### API Format

Workflows must be in API format (not UI format). Save workflows using:

```
ComfyUI → Save API Format
```

## Performance Tips

### Parallel Processing

Some stages can run in parallel:

```bash
# Terminal 1: Depth analysis
python scripts/run_pipeline.py footage.mp4 -s depth

# Terminal 2: Segmentation (runs simultaneously)
python scripts/run_pipeline.py footage.mp4 -s roto
```

### Skip Existing Output

Use `-e` to avoid re-processing:

```bash
python scripts/run_pipeline.py footage.mp4 -s all -e
```

Checks for:
- Frame files in `source/frames/`
- Depth maps in `depth/`
- Masks in `roto/`
- COLMAP model in `colmap/sparse/0/`
- Camera file in `camera/camera.abc`

### Quality vs Speed

**Fast preview** (low quality):
```bash
python scripts/run_pipeline.py footage.mp4 -s depth,colmap,camera -q low
```

**Production quality** (slow):
```bash
python scripts/run_pipeline.py footage.mp4 -s all -q high -d -m
```

### Disk Space Management

Pipeline generates large amounts of data:

| Stage | Approximate Size (per 1000 frames) |
|-------|-----------------------------------|
| Frames | 2-5 GB (PNG) |
| Depth | 1-2 GB |
| Roto | 500 MB (binary masks) |
| Cleanplate | 2-5 GB |
| COLMAP sparse | 50-200 MB |
| COLMAP dense | 5-20 GB |
| Mocap | 500 MB - 2 GB |
| GS-IR | 2-5 GB |

**Total for full pipeline**: 15-40 GB per shot

Clean up intermediate data after export:

```bash
# Keep only final outputs
rm -rf MyShot/depth MyShot/roto
```

## Troubleshooting

### "ComfyUI not running"

Start ComfyUI server:

```bash
cd .vfx_pipeline/ComfyUI
python main.py --listen
```

Verify server is accessible:

```bash
curl http://127.0.0.1:8188/system_stats
```

### "Workflow not found"

Pipeline copies workflows from `workflow_templates/` to project directory.

If missing:
1. Check `workflow_templates/` in repository root
2. Manually copy workflows to `MyShot/workflows/`

### "COLMAP reconstruction failed"

Common causes:

1. **Insufficient features**:
   - Use `-q high` for better feature detection
   - Ensure footage has trackable features (not pure white/black)

2. **Moving objects**:
   - Use segmentation masks: run `roto` stage first
   - Pipeline automatically uses masks if available

3. **Motion blur**:
   - No easy fix - requires sharp frames
   - Try lower resolution input

### "Motion capture requires camera data"

Mocap stage needs camera transforms:

```bash
# Run COLMAP or depth stage first
python scripts/run_pipeline.py footage.mp4 -s colmap,mocap
```

Or:

```bash
# Use Depth-Anything-V3 camera (faster, less accurate)
python scripts/run_pipeline.py footage.mp4 -s depth,mocap
```

### "Out of memory" (GPU)

Reduce batch size or resolution:

1. Edit ComfyUI workflows to use smaller batch sizes
2. Process fewer frames at once
3. Use CPU mode (much slower)

### "Stage failed" (general)

Check logs in project directory:

```bash
ls -lh MyShot/*.log
cat MyShot/colmap.log  # Example
```

## Advanced Usage

### Custom Stages

Run individual component scripts:

```bash
# COLMAP only
python scripts/run_colmap.py MyShot -q high

# Motion capture only
python scripts/run_mocap.py MyShot

# Camera export only
python scripts/export_camera.py MyShot --fps 24
```

### Batch Processing

Process multiple shots:

```bash
for video in footage/*.mp4; do
    name=$(basename "$video" .mp4)
    python scripts/run_pipeline.py "$video" -n "$name" -s all -e
done
```

### Integration with Other Tools

Pipeline outputs are compatible with:

- **Maya**: Import `.abc` camera
- **Houdini**: Import `.abc` camera, load `.ply` point clouds
- **Blender**: Import `.abc` or `.fbx` camera, load `.ply`
- **Nuke**: Import frame sequences, camera data (convert from ABC)
- **After Effects**: Import frame sequences

## Related Tools

- **[Installation Wizard](install_wizard.md)** - Set up components before running pipeline
- **[Janitor](janitor.md)** - Maintain and update installed components

## See Also

- Main documentation: [README.md](README.md)
- Testing guide: `TESTING.md` (repository root)
- Component scripts: `scripts/run_*.py`
