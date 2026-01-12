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
- `--model` - Segmentation model: `sam2`, `yolov8`, `detectron2` (default: `sam2`)
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
- SAM2 is most accurate but slowest
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
- `--skip-texture` - Skip texture projection (faster)
- `--keyframe-interval` - ECON keyframe interval (default: 25)
- `--test-stage` - Test specific stage: `motion`, `econ`, `texture`
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

**Custom keyframe interval**:
```bash
python scripts/run_mocap.py ./projects/Actor01 --keyframe-interval 50
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
- `--batch-size` - Batch size (default: 4)
- `--learning-rate` - Learning rate (default: 0.001)

**Paths**:
- `--gsir-path` - Path to GS-IR installation (default: auto-detect)
- `--output-dir` - Output directory (default: `project_dir/gsir`)

**Debug**:
- `--resume` - Resume from checkpoint
- `--save-interval` - Save checkpoint every N iterations (default: 5000)

### Examples

**Basic material decomposition**:
```bash
python scripts/run_gsir.py ./projects/Shot01
```

**High quality** (more iterations):
```bash
python scripts/run_gsir.py ./projects/Shot01 --iterations-stage2 50000
```

**Resume interrupted training**:
```bash
python scripts/run_gsir.py ./projects/Shot01 --resume
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

Export camera data to industry-standard formats (Alembic, FBX).

### Usage

```bash
python scripts/export_camera.py <project_dir> [options]
```

### Arguments

**Positional**:
- `project_dir` - Project directory with camera data

**Export Options**:
- `--format` - Export format: `abc`, `fbx`, `both` (default: `both`)
- `--fps` - Frame rate (default: 24.0)
- `--scale` - Scene scale factor (default: 1.0)

**Camera Parameters**:
- `--focal-length` - Override focal length (mm)
- `--sensor-width` - Override sensor width (mm)

### Examples

**Export to Alembic and FBX**:
```bash
python scripts/export_camera.py ./projects/Shot01
```

**Alembic only** (most compatible):
```bash
python scripts/export_camera.py ./projects/Shot01 --format abc
```

**Custom frame rate**:
```bash
python scripts/export_camera.py ./projects/Shot01 --fps 30
```

**Scale adjustment** (for different scene units):
```bash
python scripts/export_camera.py ./projects/Shot01 --scale 0.01
```

### Input

- `camera/extrinsics.json` - Camera transforms per frame (required)
- `camera/intrinsics.json` - Camera parameters (required)

### Output

- `camera/camera.abc` - Alembic camera export
- `camera/camera.fbx` - FBX camera export (if FBX SDK available)

### Format Compatibility

| Format | Maya | Houdini | Blender | Unreal | Unity |
|--------|------|---------|---------|--------|-------|
| Alembic (.abc) | ✓ | ✓ | ✓ | ✓ | ✓ |
| FBX (.fbx) | ✓ | ✓ | ✓ | ✓ | ✓ |

Both formats are widely supported. Alembic is generally more reliable.

### Tips

- Alembic is the safer choice (more compatible)
- FBX requires FBX SDK (may not be available)
- Scale factor depends on your DCC's scene units
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

- **[Pipeline Orchestrator](run_pipeline.md)** - Run all components automatically
- **[Installation Wizard](install_wizard.md)** - Install dependencies
- **[Janitor](janitor.md)** - Maintain installations
- **Main Documentation**: [README.md](README.md)
