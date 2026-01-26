# Component Scripts Reference

Detailed reference for individual component scripts that can be run standalone.

## Overview

While `run_pipeline.py` orchestrates the full pipeline, each component can also be run independently for fine-grained control or debugging.

## Quick Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `setup_project.py` | Initialize project structure | Project path | Directory tree |
| `run_colmap.py` | COLMAP reconstruction | Frames | 3D model |
| `run_segmentation.py` | Dynamic scene segmentation | Frames | Masks |
| `run_mocap.py` | Human motion capture | Frames + camera | Mesh sequences |
| `run_gsir.py` | Material decomposition | COLMAP model | PBR materials |
| `export_camera.py` | Camera export | Camera JSON | Alembic/FBX |
| `texture_projection.py` | Texture SMPL-X meshes | Meshes + frames | Textured meshes |
| `smplx_from_motion.py` | Generate SMPL-X from WHAM | motion.pkl | Mesh sequence |
| `mesh_deform.py` | Deform ECON with SMPL-X | SMPL-X + ECON meshes | Animated ECON |

## setup_project.py

Initialize VFX project directory structure.

### Usage

```bash
python scripts/setup_project.py <project_dir> [--workflows-dir DIR]
```

### Arguments

- `project_dir` - Path to project directory (will be created if doesn't exist)
- `--workflows-dir` - Path to workflow templates (default: `workflow_templates/`)

### Example

```bash
python scripts/setup_project.py ./projects/MyShot
```

### Output Structure

```
MyShot/
├── source/
│   └── frames/          # Place input frames here
├── workflows/           # ComfyUI workflows (copied from templates)
├── depth/               # Depth maps output
├── roto/                # Segmentation masks output
├── cleanplate/          # Clean plates output
├── colmap/              # COLMAP reconstruction output
├── mocap/               # Motion capture output
├── gsir/                # GS-IR materials output
└── camera/              # Camera export output
```

### Use Cases

- Set up project structure before manual processing
- Create multiple project directories at once
- Use custom workflow templates

---

## run_colmap.py

COLMAP Structure-from-Motion reconstruction with automatic configuration.

### Usage

```bash
python scripts/run_colmap.py <project_dir> [options]
```

### Arguments

**Positional**:
- `project_dir` - Project directory containing `source/frames/`

**Quality**:
- `--quality`, `-q` - Quality preset: `low`, `medium`, `high` (default: `medium`)

**Features**:
- `--dense`, `-d` - Run dense reconstruction (point cloud)
- `--mesh`, `-m` - Generate mesh (requires `--dense`)
- `--no-masks` - Disable automatic mask usage

**Advanced**:
- `--matcher` - Matching method: `sequential`, `exhaustive`, `vocab_tree` (default: auto)
- `--vocab-tree-path` - Path to vocabulary tree file (for vocab_tree matcher)

### Examples

**Basic reconstruction**:
```bash
python scripts/run_colmap.py ./projects/Shot01
```

**High quality with dense + mesh**:
```bash
python scripts/run_colmap.py ./projects/Shot01 -q high -d -m
```

**Without segmentation masks**:
```bash
python scripts/run_colmap.py ./projects/Shot01 --no-masks
```

### Input

- `source/frames/*.png` - Input frames (required)
- `roto/*.png` - Segmentation masks (optional, used if available unless `--no-masks`)

### Output

- `colmap/sparse/0/` - Sparse 3D reconstruction
  - `cameras.bin` - Camera parameters
  - `images.bin` - Camera poses
  - `points3D.bin` - 3D points
- `colmap/dense/` - Dense point cloud (if `--dense`)
  - `fused.ply` - Fused point cloud
- `colmap/meshed/` - Mesh (if `--mesh`)
  - `meshed-poisson.ply` - Poisson surface reconstruction

### Quality Presets

| Preset | Feature Extraction | Matching | Use Case |
|--------|-------------------|----------|----------|
| `low` | Fast, fewer features | Sequential | Quick previews, simple scenes |
| `medium` | Balanced | Vocab tree | Production work (default) |
| `high` | Detailed, more features | Exhaustive | Complex scenes, highest accuracy |

### Tips

- Use masks to ignore moving objects (improves tracking)
- Dense reconstruction takes 10-100x longer than sparse
- Mesh generation requires dense reconstruction
- Vocabulary tree matching is fastest for long sequences

---

## run_segmentation.py

Dynamic scene segmentation with automatic shot boundary detection.

### Usage

```bash
python scripts/run_segmentation.py <input_dir> <output_dir> [options]
```

### Arguments

**Positional**:
- `input_dir` - Directory with input frames
- `output_dir` - Directory for output masks

**Segmentation**:
- `--model` - Segmentation model: `sam3`, `yolov8`, `detectron2` (default: `sam3`)
- `--classes` - Object classes to segment (comma-separated)
- `--confidence` - Detection confidence threshold (default: 0.5)

**Scene Detection**:
- `--scene-threshold` - Scene change threshold (default: 0.3)
- `--min-scene-length` - Minimum frames per scene (default: 10)

### Examples

**Basic segmentation**:
```bash
python scripts/run_segmentation.py source/frames roto
```

**Custom classes**:
```bash
python scripts/run_segmentation.py source/frames roto --classes "person,car,bicycle"
```

**Adjust sensitivity**:
```bash
python scripts/run_segmentation.py source/frames roto --scene-threshold 0.5
```

### Input

- Input frames (any image format)

### Output

- Binary masks (white = foreground, black = background)
- One mask per input frame
- `_scene_info.json` - Scene boundary metadata

### Scene Detection

Automatically detects scene changes based on:
- Color histogram differences
- Feature matching quality
- Motion discontinuities

Scene boundaries prevent mask propagation across cuts.

### Tips

- Lower `--scene-threshold` for more sensitive detection
- Higher `--confidence` reduces false positives
- SAM3 is most accurate but slowest
- YOLOv8 is fastest but less accurate

---

## run_mocap.py

Human motion capture pipeline using WHAM + ECON.

### Usage

```bash
python scripts/run_mocap.py <project_dir> [options]
```

### Arguments

**Positional**:
- `project_dir` - Project directory with frames and camera data

**Options**:
- `--skip-texture` - Skip texture projection (faster, kept for compatibility)
- `--check` - Validate installation without processing

### Examples

**Full motion capture**:
```bash
python scripts/run_mocap.py ./projects/Actor01
```

**Skip texture projection** (faster):
```bash
python scripts/run_mocap.py ./projects/Actor01 --skip-texture
```

**Check installation**:
```bash
python scripts/run_mocap.py ./projects/Actor01 --check
```

### Input

- `source/frames/*.png` - Input frames
- `camera/extrinsics.json` - Camera transforms (required)
- `camera/intrinsics.json` - Camera parameters (required)

### Output

- `mocap/wham/` - WHAM pose estimates
  - `poses.pkl` - Per-frame poses
  - `tracking.json` - Tracking metadata
- `mocap/econ/` - ECON 3D reconstructions
  - `mesh_*.obj` - Clothed mesh per keyframe
- `mocap/mesh_sequence/` - Exported mesh sequence
  - `mesh_*.obj` - OBJ mesh files

### Pipeline

1. **WHAM** extracts 3D pose from video (world-grounded motion)
2. **ECON** reconstructs detailed clothed human with SMPL-X compatibility
3. **Texture Projection** (optional) projects video texture onto meshes

### Tips

- Requires good camera tracking (from COLMAP or Depth-Anything-V3)
- Actor should be visible in majority of frames
- Better results with frontal views
- `--skip-texture` saves significant time
- Higher `--keyframe-interval` processes fewer frames (faster, less detail)

---

## run_gsir.py

GS-IR (Gaussian Splatting Inverse Rendering) for material decomposition.

### Usage

```bash
python scripts/run_gsir.py <project_dir> [options]
```

### Arguments

**Positional**:
- `project_dir` - Project directory with COLMAP reconstruction

**Training**:
- `--iterations-stage1` - Stage 1 iterations (default: 30000)
- `--iterations-stage2` - Total iterations (default: 35000)

**Paths**:
- `--gsir-path` - Path to GS-IR installation (default: auto-detect)

**Options**:
- `--skip-training` - Skip training if checkpoint exists
- `--check` - Check if GS-IR is available and exit

### Examples

**Basic material decomposition**:
```bash
python scripts/run_gsir.py ./projects/Shot01
```

**High quality** (more iterations):
```bash
python scripts/run_gsir.py ./projects/Shot01 --iterations-stage2 50000
```

**Skip training if checkpoint exists**:
```bash
python scripts/run_gsir.py ./projects/Shot01 --skip-training
```

### Input

- `colmap/sparse/0/` - COLMAP sparse reconstruction (required)
- `source/frames/*.png` - Input frames

### Output

- `gsir/model/chkpnt{N}.pth` - Model checkpoints (every 5k iterations)
- `gsir/materials/` - Extracted PBR textures
  - `albedo.exr` - Base color (RGB)
  - `roughness.exr` - Surface roughness (grayscale)
  - `metallic.exr` - Metallic (grayscale)
  - `normal.exr` - Normal map (RGB)
- `gsir/renders/` - Rendered views for validation

### Training Time

| Iterations | Time (RTX 3090) | Quality |
|-----------|-----------------|---------|
| 30k | ~30 min | Preview |
| 35k | ~40 min | Good (default) |
| 50k | ~60 min | High quality |
| 100k | ~2 hours | Best quality |

### Tips

- Requires good COLMAP reconstruction
- More iterations = better quality but longer training
- Save interval determines disk usage (more checkpoints = more space)
- Resume capability useful for long training runs
- Extracted materials can be used in Blender, Houdini, Unreal

---

## export_camera.py

Export camera data to industry-standard VFX formats.

### Usage

```bash
python scripts/export_camera.py <project_dir> [options]
```

### Arguments

**Positional**:
- `project_dir` - Project directory with camera data

**Export Options**:
- `--format` - Export format: `chan`, `csv`, `clip`, `cmd`, `json`, `abc`, `jsx`, `all` (default: `all`)
- `--fps` - Frame rate (default: 24.0)
- `--start-frame` - Starting frame number (default: 1)
- `--width` - Image width in pixels (default: 1920)
- `--height` - Image height in pixels (default: 1080)

### Examples

**Export all formats** (default):
```bash
python scripts/export_camera.py ./projects/Shot01
```

**Export specific format**:
```bash
# Nuke .chan format
python scripts/export_camera.py ./projects/Shot01 --format chan

# After Effects JSX script
python scripts/export_camera.py ./projects/Shot01 --format jsx

# Alembic (most compatible)
python scripts/export_camera.py ./projects/Shot01 --format abc

# Houdini .cmd script
python scripts/export_camera.py ./projects/Shot01 --format cmd
```

**Custom frame rate**:
```bash
python scripts/export_camera.py ./projects/Shot01 --fps 30
```

### Input

- `camera/extrinsics.json` - Camera transforms per frame (required)
- `camera/intrinsics.json` - Camera parameters (required)

### Output

When using `--format all` (default), exports to all available formats:
- `camera/camera.chan` - Nuke .chan format
- `camera/camera.csv` - CSV spreadsheet format
- `camera/camera.clip` - Houdini CHOP clip format
- `camera/camera.cmd` + `camera/camera.py` - Houdini Python scripts
- `camera/camera.camera.json` - JSON format with detailed transforms
- `camera/camera.abc` - Alembic format (requires conda install)
- `camera/camera.jsx` - After Effects JavaScript script

### Format Compatibility

| Format | Nuke | After Effects | Maya | Houdini | Blender | Unreal | Unity |
|--------|------|---------------|------|---------|---------|--------|-------|
| .chan | ✓ | - | - | - | - | - | - |
| .jsx | - | ✓ | - | - | - | - | - |
| .abc | ✓ | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| .clip | - | - | - | ✓ | - | - | - |
| .cmd/.py | - | - | - | ✓ | - | - | - |
| .csv | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| .json | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Notes:
- Alembic (.abc) is the most widely supported format for 3D DCCs
- Nuke .chan is the simplest format for Nuke imports
- After Effects .jsx provides native scripted import
- CSV and JSON are universal and can be imported via custom scripts

### Tips

- Use `--format all` to export all formats at once (default behavior)
- Alembic (.abc) requires: `conda install -c conda-forge alembic`
- For Nuke, use .chan for simplest import
- For After Effects, run the .jsx script via File → Scripts → Run Script File
- For Houdini, use .cmd via File → Run Script, or .clip via File CHOP
- CSV and JSON formats work universally with custom import scripts
- Frame rate should match your edit timeline

---

## texture_projection.py

Project video textures onto SMPL-X mesh sequences.

### Usage

```bash
python scripts/texture_projection.py <mesh_dir> <frames_dir> <camera_dir> <output_dir> [options]
```

### Arguments

**Positional**:
- `mesh_dir` - Directory with mesh sequence (`.obj` or `.ply` files)
- `frames_dir` - Directory with video frames
- `camera_dir` - Directory with camera data (`extrinsics.json`, `intrinsics.json`)
- `output_dir` - Output directory for textured meshes

**Options**:
- `--resolution` - Texture resolution (default: 2048)
- `--format` - Output format: `obj`, `ply`, `fbx` (default: `obj`)
- `--unwrap-method` - UV unwrapping method: `smart`, `conformal`, `angle` (default: `smart`)

### Examples

**Basic texture projection**:
```bash
python scripts/texture_projection.py \
    mocap/econ/meshes \
    source/frames \
    camera \
    mocap/econ/textured_meshes
```

**High resolution textures**:
```bash
python scripts/texture_projection.py \
    mocap/econ/meshes \
    source/frames \
    camera \
    mocap/econ/textured_meshes \
    --resolution 4096
```

**Different output format**:
```bash
python scripts/texture_projection.py \
    mocap/econ/meshes \
    source/frames \
    camera \
    mocap/econ/textured_meshes \
    --format fbx
```

### Input

- Mesh sequence (one mesh per frame or keyframe)
- Video frames (matching mesh frames)
- Camera data (extrinsics and intrinsics)

### Output

- Textured meshes (one per input mesh)
- Texture maps (.png or .jpg)
- Material files (if applicable to format)

### UV Unwrapping Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `smart` | Automatic seam placement | Best for most cases (default) |
| `conformal` | Preserves angles | Characters, organic shapes |
| `angle` | Angle-based unwrapping | Hard surface objects |

### Tips

- Higher resolution = better quality but larger files
- Smart unwrapping usually works best
- Requires accurate camera tracking
- Best results with frontal views of subject

---

## smplx_from_motion.py

Generate animated SMPL-X mesh sequences from WHAM motion data.

### Usage

```bash
python scripts/smplx_from_motion.py <project_dir> [options]
```

### Arguments

**Positional**:
- `project_dir` - Project directory

**Required**:
- `--motion` - Path to WHAM motion.pkl file
- `--output` - Output directory for mesh sequence

**Optional**:
- `--rest-pose` - Save rest pose (frame 0) to this path
- `--model-path` - Path to SMPL-X models (auto-detected if not specified)
- `--gender` - Body model gender: `neutral`, `male`, `female` (default: `neutral`)
- `--device` - Torch device: `cuda`, `cpu` (default: `cuda`)

### Examples

**Basic mesh generation**:
```bash
python scripts/smplx_from_motion.py ./projects/Actor01 \
    --motion mocap/wham/motion.pkl \
    --output mocap/smplx_animated/
```

**With rest pose export**:
```bash
python scripts/smplx_from_motion.py ./projects/Actor01 \
    --motion mocap/wham/motion.pkl \
    --output mocap/smplx_animated/ \
    --rest-pose mocap/smplx_rest.obj
```

**Specific gender model**:
```bash
python scripts/smplx_from_motion.py ./projects/Actor01 \
    --motion mocap/wham/motion.pkl \
    --output mocap/smplx_animated/ \
    --gender female
```

### Input

- `mocap/wham/motion.pkl` - WHAM motion data containing:
  - `poses`: [N, 72] SMPL pose parameters
  - `trans`: [N, 3] root translation
  - `betas`: [10] or [N, 10] shape parameters

### Output

- `mocap/smplx_animated/frame_0000.obj` ... `frame_NNNN.obj` - Per-frame SMPL-X meshes
- Optional: Rest pose mesh at specified path

### Prerequisites

- SMPL-X models downloaded from https://smpl-x.is.tue.mpg.de/
- Models placed in `.vfx_pipeline/smplx_models/` or specified via `--model-path`
- PyTorch installed with CUDA support (for GPU acceleration)

### Tips

- Use `--device cpu` if CUDA is unavailable (slower)
- Rest pose is required for mesh_deform.py workflow
- SMPL-X meshes include UV coordinates for texturing

---

## mesh_deform.py

Transfer animation from SMPL-X to ECON clothed meshes using UV-based correspondence.

### Usage

```bash
python scripts/mesh_deform.py <project_dir> [options]
```

### Arguments

**Positional**:
- `project_dir` - Project directory

**Required**:
- `--smplx-rest` - SMPL-X rest pose mesh (.obj)
- `--econ-rest` - ECON rest pose mesh (.obj)
- `--smplx-sequence` - Directory with animated SMPL-X meshes
- `--output` - Output directory for deformed meshes

**Optional**:
- `--smoothing-map` - UV-space smoothing weight image (grayscale PNG)
- `--offset-mode` - Offset transformation mode: `smooth`, `rigid`, `normal` (default: `smooth`)
- `--cache` - Path to cache correspondence data (.npz)
- `--create-smoothing-template` - Create a default smoothing map template

### Examples

**Basic deformation**:
```bash
python scripts/mesh_deform.py ./projects/Actor01 \
    --smplx-rest mocap/smplx_rest.obj \
    --econ-rest mocap/econ/mesh_0001.obj \
    --smplx-sequence mocap/smplx_animated/ \
    --output mocap/econ_animated/
```

**With correspondence caching** (faster iteration):
```bash
python scripts/mesh_deform.py ./projects/Actor01 \
    --smplx-rest mocap/smplx_rest.obj \
    --econ-rest mocap/econ/mesh_0001.obj \
    --smplx-sequence mocap/smplx_animated/ \
    --output mocap/econ_animated/ \
    --cache mocap/correspondence.npz
```

**Create smoothing template**:
```bash
python scripts/mesh_deform.py ./projects/Actor01 \
    --create-smoothing-template mocap/smoothing_weights.png
```

**With custom smoothing map**:
```bash
python scripts/mesh_deform.py ./projects/Actor01 \
    --smplx-rest mocap/smplx_rest.obj \
    --econ-rest mocap/econ/mesh_0001.obj \
    --smplx-sequence mocap/smplx_animated/ \
    --output mocap/econ_animated/ \
    --smoothing-map mocap/smoothing_weights.png
```

### Input

- SMPL-X rest pose mesh with UV coordinates
- ECON rest pose mesh with matching UV coordinates
- Animated SMPL-X mesh sequence (from smplx_from_motion.py)

### Output

- Deformed ECON meshes (one per input SMPL-X frame)
- Preserves ECON mesh topology and UV coordinates

### Offset Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `smooth` | Linear interpolation (fastest) | Tight clothing, default |
| `normal` | Offset along surface normal | Medium clothing |
| `rigid` | Full local frame transformation | Loose clothing, most accurate |

### Smoothing Map

The smoothing map controls per-region deformation behavior:
- **White (255)**: Maximum smoothing/damping (soft fabric)
- **Black (0)**: Rigid offset (stiff clothing moves with body)
- **Gray (128)**: Moderate smoothing (default)

Paint the map in Photoshop/GIMP using the SMPL-X UV layout to control areas like:
- Armpits, thighs (more smoothing to avoid artifacts)
- Belt, shoes (rigid, moves with body)
- Shirt, pants (moderate)

### Why UV-Based (Not Distance-Based)

Distance-based mesh binding creates artifacts where geometry folds (armpits, inner thighs) because vertices that are far apart on the body surface become close in 3D space when the mesh folds.

UV-based correspondence avoids this because:
- UV coordinates are pose-invariant
- Vertices far apart on the surface stay far apart in UV space
- Folding in 3D doesn't affect UV-space relationships

### Tips

- Use `--cache` to speed up iteration (correspondence is expensive to compute)
- Start with `smooth` mode, try `rigid` if clothing looks too damped
- Low UV coverage (<90%) indicates UV mismatch between meshes
- ECON and SMPL-X must have compatible UV layouts

---

## Common Patterns

### Running Components in Sequence

Manual pipeline execution:

```bash
# 1. Set up project
python scripts/setup_project.py ./projects/MyShot

# 2. Place frames in source/frames/

# 3. Run COLMAP
python scripts/run_colmap.py ./projects/MyShot -q high

# 4. Export camera
python scripts/export_camera.py ./projects/MyShot

# 5. Run mocap (if applicable)
python scripts/run_mocap.py ./projects/MyShot

# 6. Run GS-IR (if needed)
python scripts/run_gsir.py ./projects/MyShot
```

### Batch Processing

Process multiple projects:

```bash
for project in ./projects/*; do
    echo "Processing $project"
    python scripts/run_colmap.py "$project" -q medium
    python scripts/export_camera.py "$project"
done
```

### Debugging Individual Stages

Test one component:

```bash
# Test COLMAP only
python scripts/run_colmap.py ./projects/Test -q low

# Check output
ls ./projects/Test/colmap/sparse/0/
```

## See Also

- **[CLI Reference](cli.md)** - Run all components automatically
- **[Installation](../installation.md)** - Install dependencies
- **[Maintenance](maintenance.md)** - Maintain installations
- **Main Documentation**: [README.md](../README.md)
