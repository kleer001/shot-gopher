# VFX Pipeline

A FOSS automated VFX pipeline built on ComfyUI for first-pass rotoscoping, depth extraction, 3D reconstruction, and clean plate generation.

## Goal

**Hands-off batch processing.** Ingest a movie file, get production-ready outputs (depth maps, segmentation masks, clean plates, camera solves, 3D point clouds) with minimal manual intervention. Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

## Why This Exists

Traditional VFX prep work (roto, depth, clean plates) is tedious. Modern ML models (SAM2, Depth Anything V3, ProPainter) can automate 80% of it. This pipeline stitches them together into a single workflow that:

- Takes raw footage in
- Outputs VFX-ready passes out
- Follows real production folder conventions (frame numbering starts at 1001, etc.)

## Installation

### Quick Install (Recommended)

**One-liner bootstrap script** - clones repository and runs wizard automatically:

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap.sh | bash
```

Or manually:

```bash
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python scripts/install_wizard.py
```

### What Gets Installed

The wizard will:
- ✓ Check system requirements (Python, GPU, Git)
- ✓ Create dedicated conda environment automatically
- ✓ Install core dependencies (PyTorch, NumPy, OpenCV)
- ✓ Download model checkpoints automatically (WHAM, TAVA, ECON)
- ✓ Clone git repositories (WHAM, TAVA, ECON, ComfyUI)
- ✓ Generate configuration files (`config.json`, `activate.sh`)
- ✓ Run validation tests

**Installation options:**
1. **Core pipeline only** - COLMAP, segmentation, depth (no motion capture)
2. **Core + ComfyUI** - Add workflows and custom nodes
3. **Full stack** - Everything including motion capture
4. **Custom selection** - Pick exactly what you need

**Quick commands:**
```bash
# Check what's installed
python scripts/install_wizard.py -c

# Validate installation
python scripts/install_wizard.py -v

# Resume interrupted installation
python scripts/install_wizard.py -r
```

See **[docs/install_wizard.md](docs/install_wizard.md)** for complete documentation or [manual installation](#manual-installation) below.

## Quick Start

After installation, process footage with a single command:

```bash
# Basic pipeline - depth + camera
python scripts/run_pipeline.py /path/to/footage.mp4 -n "My_Shot"

# With segmentation and clean plates
python scripts/run_pipeline.py footage.mp4 -s ingest,roto,cleanplate,colmap,camera

# Full pipeline with motion capture
python scripts/run_pipeline.py footage.mp4 -s all

# Skip already-processed stages
python scripts/run_pipeline.py footage.mp4 -s all -e
```

**Single-letter shortcuts** available for all options (e.g., `-n` for `--name`, `-s` for `--stages`, `-e` for `--skip-existing`).

**Prerequisites:**
- ComfyUI running: `cd .vfx_pipeline/ComfyUI && python main.py --listen`
- Dependencies installed via wizard (or manually)

See **[docs/run_pipeline.md](docs/run_pipeline.md)** for complete documentation.

## Maintenance

Keep your installation healthy with the janitor tool:

```bash
# Quick health check
python scripts/janitor.py -H

# Update all components
python scripts/janitor.py -u -y

# Clean temporary files
python scripts/janitor.py -c

# Full maintenance (health + update + clean + report)
python scripts/janitor.py -a
```

See **[docs/janitor.md](docs/janitor.md)** for complete documentation.

## Documentation

Complete documentation is available in the **[docs/](docs/)** folder:

- **[docs/README.md](docs/README.md)** - Overview and getting started
- **[docs/install_wizard.md](docs/install_wizard.md)** - Installation wizard reference
- **[docs/run_pipeline.md](docs/run_pipeline.md)** - Pipeline orchestrator reference
- **[docs/janitor.md](docs/janitor.md)** - Maintenance tool reference
- **[docs/component_scripts.md](docs/component_scripts.md)** - Individual component scripts

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
- **SAM2** - Segment Anything Model 2 for text-prompted rotoscoping
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
- `scripts/run_pipeline.py` - **Main entry point** - automated pipeline runner with full stage support
- `scripts/setup_project.py` - Project setup with workflow templating
- `scripts/export_camera.py` - Camera data → Alembic/JSON export (supports both DA3 and COLMAP)
- `scripts/run_colmap.py` - COLMAP SfM/MVS reconstruction wrapper with mask support
- `scripts/run_gsir.py` - GS-IR material decomposition wrapper
- `scripts/run_segmentation.py` - Standalone SAM2 segmentation runner with multi-prompt support
- `workflow_templates/01_analysis.json` - Depth Anything V3 + camera estimation
- `workflow_templates/02_segmentation.json` - SAM2 video segmentation (text prompt → masks)
- `workflow_templates/03_cleanplate.json` - ProPainter inpainting

### Known Issues
- SAM2 text prompts like "person" don't capture carried items (bags, purses) - use `run_segmentation.py --prompts` for multi-prompt
- Frame numbering in ComfyUI SaveImage doesn't support custom start numbers (1001+) - outputs need post-rename or custom node
- Large frame counts (150+) can stall SAM2 propagation - need batching strategy

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

## Dynamic Scene Segmentation

For footage with moving subjects (people, vehicles, etc.), the pipeline supports automatic segmentation to separate dynamic elements from static backgrounds. This enables:
- **Static-only camera solving**: COLMAP reconstructs only the background by masking out moving objects
- **Clean plate generation**: Inpainted backgrounds for compositing
- **Object isolation**: Separate dynamic elements for individual processing

### How It Works

```
Frames → SAM2 Segmentation → Masks → COLMAP (masked) → Static reconstruction
                  ↓
              Clean plates (ProPainter inpainting)
```

The segmentation masks tell COLMAP to ignore features in dynamic regions during camera solving, producing accurate camera tracks even with moving subjects in frame.

### Basic Usage

```bash
# Full pipeline with segmentation (roto + colmap)
python scripts/run_pipeline.py footage.mp4 --stages ingest,roto,colmap,camera

# Run all stages including clean plates
python scripts/run_pipeline.py footage.mp4 --stages ingest,roto,cleanplate,colmap,camera
```

**Default behavior**: If masks exist in `roto/`, COLMAP automatically uses them. To disable:
```bash
python scripts/run_pipeline.py footage.mp4 --stages colmap --colmap-no-masks
```

### Standalone Segmentation

For custom prompts or multi-object scenes:

```bash
# Single prompt (default: "person")
python scripts/run_segmentation.py /path/to/projects/My_Shot --prompt "person"

# Multiple prompts for complex scenes (person + carried objects)
python scripts/run_segmentation.py /path/to/projects/My_Shot --prompts "person,bag,backpack"

# Different object types
python scripts/run_segmentation.py /path/to/projects/My_Shot --prompt "car"
```

### Workflow Stages

1. **Segmentation (roto)**: SAM2 video segmentation
   - Text-prompted object detection
   - Temporal propagation across frames
   - Output: Binary masks in `roto/`

2. **Clean Plate (cleanplate)**: ProPainter inpainting
   - Uses masks from roto stage
   - Fills dynamic regions with static background
   - Output: Clean plates in `cleanplate/`

3. **COLMAP (with masks)**: Camera solving
   - Automatically detects masks in `roto/`
   - Excludes masked regions from feature extraction
   - Produces accurate static-scene reconstruction

### Multi-Prompt Segmentation

For scenes where a single prompt doesn't capture all dynamic elements:

```bash
# Segment person + objects they're carrying
python scripts/run_segmentation.py project/ --prompts "person,bag,phone,cup"

# Multiple actors or objects
python scripts/run_segmentation.py project/ --prompts "person,car,bicycle"
```

**Note**: Multiple prompts run sequentially and combine masks. For complex scenes with 150+ frames, consider batching or using frame ranges to avoid SAM2 propagation stalls.

### Pipeline Integration

```bash
# Typical dynamic scene workflow:
# 1. Extract frames
python scripts/run_pipeline.py footage.mp4 --stages ingest

# 2. Segment dynamic objects (custom prompt if needed)
python scripts/run_segmentation.py projects/footage --prompt "person"

# 3. Run COLMAP with masks + generate clean plates
python scripts/run_pipeline.py footage.mp4 --stages cleanplate,colmap,camera
```

### Output Structure

```
projects/My_Shot/
├── roto/                    # Segmentation masks (dynamic regions)
│   ├── mask_00001.png
│   ├── mask_00002.png
│   └── ...
├── cleanplate/              # Inpainted backgrounds (static only)
│   ├── clean_00001.png
│   ├── clean_00002.png
│   └── ...
├── colmap/
│   └── sparse/0/            # Static-only reconstruction
└── camera/
    ├── extrinsics.json      # Camera from static features only
    └── camera.abc
```

### Known Limitations

- **Text prompt specificity**: SAM2's "person" prompt doesn't capture carried items (bags, purses). Use multi-prompt for complete coverage.
- **Large frame counts**: 150+ frames can stall SAM2 propagation. Batch processing or frame range support planned.
- **Mask combination**: Multiple prompt runs currently write to same output directory. Manual mask merging may be needed for complex scenes.

## Human Motion Capture (Experimental)

**Status:** Experimental - requires additional dependencies (WHAM, TAVA, ECON)

Reconstruct people from video with temporally consistent geometry and textures ready for VFX matchmove.

### What It Produces

```
mocap/
├── wham/
│   └── motion.pkl              # World-grounded skeleton animation
├── econ/
│   ├── mesh_0001.obj           # Clothed geometry keyframes
│   ├── mesh_0025.obj
│   └── ...
├── tava/
│   └── mesh_sequence.pkl       # Consistent topology sequence
├── obj_sequence/               # Exported mesh per frame
│   ├── frame_0001.obj          # SMPL-X topology (10,475 verts)
│   ├── frame_0002.obj          # Same vertex IDs across frames
│   └── ...
└── texture.png                 # Canonical UV texture (1024x1024)
```

**Key features:**
- **Consistent topology**: Same vertex count/connectivity across all frames (SMPL-X: 10,475 vertices)
- **Standard UV layout**: SMPL-X UV coordinates - no texture swimming
- **World-space alignment**: Matches COLMAP camera/scene coordinates
- **Clothed geometry**: Captures actual clothing, not template body

### Pipeline Stages

```
Frames → WHAM (motion) → ECON (geometry) → TAVA (tracking) → Textured mesh
           ↓                  ↓                  ↓
    World-space         Clothed body      Consistent topology
    skeleton            keyframes         + UV texture
```

1. **WHAM**: World-grounded motion tracking
   - Estimates SMPL-X skeleton animation
   - Output in world coordinates (aligns with COLMAP)
   - Includes foot contact detection

2. **ECON**: Clothed body reconstruction
   - Captures actual clothing geometry (not template)
   - Runs on keyframes (every 25 frames by default)
   - Produces high-quality meshes

3. **TAVA**: Temporal tracking
   - Registers ECON geometry to SMPL-X topology
   - Tracks consistent vertex IDs through time
   - Interpolates between keyframes

4. **Texture projection**: Multi-view texturing
   - Projects camera images to canonical UV space
   - Weighted by viewing angle + visibility
   - Temporal consistency filtering

### Usage

**Prerequisites:**
```bash
# Check dependencies
python scripts/run_mocap.py --check

# Install WHAM
git clone https://github.com/yohanshin/WHAM.git
cd WHAM && pip install -e .
# Download WHAM checkpoints from project page

# Install TAVA
git clone https://github.com/facebookresearch/tava.git
cd tava && pip install -e .

# Install ECON
git clone https://github.com/YuliangXiu/ECON.git
cd ECON && pip install -r requirements.txt
# Download SMPL-X models + ECON checkpoints

# Core dependencies
pip install numpy torch smplx trimesh opencv-python pillow
```

**Basic usage:**
```bash
# Full pipeline (requires COLMAP camera data first)
python scripts/run_pipeline.py footage.mp4 \
  --stages ingest,roto,colmap,mocap,camera

# Standalone mocap
python scripts/run_mocap.py /path/to/projects/My_Shot

# Skip texture (faster, mesh only)
python scripts/run_mocap.py /path/to/projects/My_Shot --skip-texture

# Custom keyframe interval
python scripts/run_mocap.py /path/to/projects/My_Shot --keyframe-interval 30
```

**Testing individual stages:**
```bash
# Test motion tracking only
python scripts/run_mocap.py project/ --test-stage motion

# Test ECON reconstruction
python scripts/run_mocap.py project/ --test-stage econ

# Test TAVA tracking
python scripts/run_mocap.py project/ --test-stage tava

# Test texture projection
python scripts/run_mocap.py project/ --test-stage texture
```

### Integration with VFX Tools

**Import to Maya/Houdini/Blender:**
```python
# Load OBJ sequence
obj_dir = "mocap/obj_sequence/"
frames = sorted(glob("frame_*.obj"))

# All frames have:
# - Same vertex count (10,475)
# - Same connectivity (no topology changes)
# - SMPL-X UV coordinates

# Load texture
texture = "mocap/texture.png"
# Apply to mesh with SMPL-X UV layout
```

**Nuke compositing:**
```python
# Load mesh sequence + camera
ReadGeo {
  file "mocap/obj_sequence/frame_####.obj"
}
Camera {
  file "camera/camera.abc"
}

# Both in world space - already aligned!
```

### Output Format

**Mesh sequence:**
- Format: OBJ per frame (Alembic export planned)
- Topology: SMPL-X (10,475 vertices, standard connectivity)
- UVs: SMPL-X layout (consistent across frames)
- Scale: Meters (world-space aligned with COLMAP)

**Texture:**
- Resolution: 1024x1024 (configurable)
- Format: PNG, RGB
- UV layout: SMPL-X canonical space
- Source: Aggregated from all camera views

### Known Limitations

- **Experimental**: Requires installing multiple research codebases
- **GPU intensive**: Needs 12GB+ VRAM for ECON/WHAM
- **Slow**: Full pipeline can take hours (TAVA training is bottleneck)
- **Single person**: Multi-person support requires per-person segmentation
- **Clothing detail**: Quality depends on ECON keyframe density
- **Occlusions**: Heavy occlusions may cause tracking drift

### Validation & Testing

**Check motion quality:**
```bash
# Visualize WHAM skeleton
python -c "
import pickle
with open('mocap/wham/motion.pkl', 'rb') as f:
    data = pickle.load(f)
print('Frames:', len(data['poses']))
print('Root trajectory:', data['trans'])
"
```

**Check mesh consistency:**
```bash
# Verify vertex count is consistent
for obj in mocap/obj_sequence/*.obj; do
    echo "$obj: $(grep -c '^v ' $obj) vertices"
done
# Should all print same number (10,475 for SMPL-X)
```

**Check texture coverage:**
```bash
# View texture
display mocap/texture.png  # ImageMagick
# Check for unfilled regions (black areas)
```

### Best Practices

1. **Run COLMAP first** - Mocap needs world-space camera data
2. **Use segmentation masks** - Helps WHAM focus on person
3. **Adjust keyframe interval** - More keyframes = better detail, slower processing
4. **Test stages independently** - Use `--test-stage` to debug failures
5. **Validate topology** - Check vertex counts match before importing to DCC
6. **Check scale** - SMPL-X is in meters, verify it matches your scene

### Troubleshooting

**"WHAM not available":**
```bash
python scripts/run_mocap.py --check
# Follow installation instructions printed
```

**"Motion file not found":**
```bash
# Run motion stage first
python scripts/run_mocap.py project/ --test-stage motion
```

**"Camera data required":**
```bash
# Run COLMAP before mocap
python scripts/run_pipeline.py footage.mp4 --stages colmap
```

**Mesh quality issues:**
- Increase `--keyframe-interval` (e.g., 15 instead of 25)
- Check person is clearly visible in frames
- Verify segmentation masks are accurate

## Manual Installation

If you prefer to install components manually instead of using the wizard:

### Core Dependencies

```bash
# Python packages
pip install numpy torch opencv-python pillow

# System packages (Ubuntu)
sudo apt install ffmpeg colmap git
```

### ComfyUI Setup

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI && pip install -r requirements.txt

# Install custom nodes
cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/your-repo/ComfyUI-Depth-Anything-V3.git
git clone https://github.com/your-repo/ComfyUI-SAM2.git

# Start server
cd .. && python main.py --listen
```

### Motion Capture Dependencies (Optional)

```bash
# Core packages
pip install smplx trimesh

# WHAM (world-grounded motion)
git clone https://github.com/yohanshin/WHAM.git
cd WHAM && pip install -e .
# Download checkpoints from project page

# TAVA (consistent topology)
git clone https://github.com/facebookresearch/tava.git
cd tava && pip install -e .

# ECON (clothed reconstruction)
git clone https://github.com/YuliangXiu/ECON.git
cd ECON && pip install -r requirements.txt

# SMPL-X models
# 1. Register at https://smpl-x.is.tue.mpg.de/
# 2. Download models
# 3. Place in ~/.smplx/
```

### Verify Installation

```bash
# Check core pipeline
python scripts/run_pipeline.py --help

# Check motion capture
python scripts/run_mocap.py --check

# Check all components
python scripts/install_wizard.py --check-only
```

## For the Agent

When continuing development:

- **Test with real footage** - The workflows were built from node documentation, not extensive testing. Expect connection/type errors.
- **Check actual node signatures** - ComfyUI node inputs/outputs change between versions. Use `grep -A 20 "INPUT_TYPES" ...` to verify.
- **VRAM is the constraint** - 12GB minimum, 24GB comfortable. Batch sizes and model choices are driven by this.
- **User's system** - Ubuntu, NVIDIA GPU, conda environment at `/media/menser/fauna/META_VFX/VFX/`

The goal is a pipeline that a VFX artist can run overnight and come back to usable first-pass outputs.
