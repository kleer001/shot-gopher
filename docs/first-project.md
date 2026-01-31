# Running Your First Project

This guide walks you through running your first VFX pipeline project after installation.

## Prerequisites

Before running the pipeline, ensure you've completed installation:

```bash
./scripts/bootstrap_conda.sh
# or
python scripts/install_wizard.py
```

Verify your installation is working:
```bash
python scripts/run_pipeline.py --help
```

## Basic Usage

Process a video with selected stages:

```bash
python scripts/run_pipeline.py /path/to/video.mp4 -s ingest,depth,roto,cleanplate,colmap,camera
```

This processes the video through the specified stages in order.

**Output location:** `../vfx_projects/MyProject/` (relative to repository root)

### Specific Stages

Run only depth estimation:

```bash
python scripts/run_pipeline.py /path/to/video.mp4 -s depth --name DepthTest
```

### Common Workflows

**Quick preview** (depth and segmentation):
```bash
python scripts/run_pipeline.py ~/Videos/test_shot.mp4 \
  -s ingest,depth,roto \
  --name TestShot
```

**Full VFX prep**:
```bash
python scripts/run_pipeline.py ~/Videos/shot001.mp4 \
  -s ingest,depth,roto,matte,cleanplate,colmap,camera \
  --name Shot001
```

**Motion capture** (requires GVHMR + SMPL-X models):
```bash
python scripts/run_pipeline.py ~/Videos/person_walking.mp4 \
  -s ingest,colmap,mocap \
  --name WalkingTest
```

### With Custom Options

Custom segmentation prompt:
```bash
python scripts/run_pipeline.py video.mp4 \
  -s roto \
  --prompt "person, car, building" \
  --name CustomRoto
```

High-quality COLMAP reconstruction:
```bash
python scripts/run_pipeline.py video.mp4 \
  -s colmap \
  --colmap-quality high
```

### Available Stages

| Stage | Description | Output |
|-------|-------------|--------|
| `ingest` | Extract frames from video | PNG sequence in `source/frames/` |
| `depth` | Monocular depth estimation | Depth maps in `depth/` |
| `roto` | Text-prompted segmentation | Binary masks in `roto/` |
| `mama` | Alpha matte refinement (VideoMaMa) | Alpha mattes in `matte/` |
| `cleanplate` | Video inpainting | Clean backgrounds in `cleanplate/` |
| `colmap` | Structure-from-Motion camera solve | Point cloud, camera poses in `colmap/` |
| `camera` | Export camera to multiple formats | Alembic, JSON, meshes in `camera/` |
| `mocap` | Human motion capture (experimental) | SMPL-X parameters, meshes in `mocap/` |

### Environment Variables

Configure output locations:

```bash
# Custom projects directory
export VFX_PROJECTS_DIR=/path/to/projects
python scripts/run_pipeline.py video.mp4 -s depth
```

**See [CLI Reference](reference/cli.md) for complete command-line options.**

---

## Understanding Pipeline Output

### Frame Numbering

Frames start at `0001` rather than the VFX industry standard `1001`:
- `frame_0001.png` (first frame)
- `frame_0002.png` (second frame)
- etc.

This is due to ComfyUI constraints.

### File Formats

- **Frames/Images:** PNG (8-bit or 16-bit depending on stage)
- **Depth maps:** 16-bit PNG (normalized to 0-65535 range)
- **Masks:** 8-bit PNG (binary: 0 or 255)
- **Camera data:** Alembic (`.abc`), JSON, PLY point clouds
- **Material decomposition:** OpenEXR (`.exr`) with multiple channels

### Viewing Output

Use standard VFX tools to view results:

**Nuke:**
```tcl
# Read frame sequence
Read {
  file /path/to/MyProject/depth/frame_%04d.png
  first 1
  last 100
}
```

**Blender:**
- Import Alembic camera: File → Import → Alembic (.abc)
- Import PLY point cloud: File → Import → Stanford (.ply)

**DJV View** (free frame viewer):
```bash
djv_view ../vfx_projects/MyProject/depth/
```

---

## Troubleshooting First Run

### Common Issues

**Missing dependencies:**
```bash
# Reinstall environment
conda activate vfx-pipeline
pip install -r requirements.txt
```

**CUDA not detected:**
```bash
# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**ComfyUI not found:**
```bash
# Verify ComfyUI installation
ls .vfx_pipeline/ComfyUI/

# Reinstall if missing
python scripts/install_wizard.py
```

### General Issues

**Out of VRAM:**
- Process fewer frames at once
- Use smaller models (if available)
- Close other GPU-intensive applications
- See component-specific memory requirements in main README

**Slow processing:**
- First run downloads models (can take time)
- Verify GPU is being used (check `nvidia-smi`)
- Check PyTorch CUDA installation

**Bad results:**
- Check input video quality (resolution, lighting, motion blur)
- Review tool limitations by shot type (see main README)
- Try different stages or parameters
- Some tools work better with specific shot types

---

## Next Steps

After running your first project:

1. **Review output** in `../vfx_projects/ProjectName/`
2. **Import to VFX tools** (Nuke, Blender, Houdini, etc.)
3. **Refine results** manually downstream as needed
4. **Experiment** with different stages and options
5. **Read component docs** for advanced usage: [Scripts Reference](reference/scripts.md)

### Recommended Learning Path

1. Start with simple stages: `ingest,depth`
2. Add segmentation: `ingest,depth,roto`
3. Try camera tracking: `ingest,colmap,camera`
4. Experiment with full pipeline: `all`
5. Customize per shot type (see tool limitations table)

### Getting Help

- **Documentation:** Check [docs/](../) for detailed guides
- **Issues:** https://github.com/kleer001/shot-gopher/issues
- **Discussions:** https://github.com/kleer001/shot-gopher/discussions

---

## Example: Complete Walkthrough

Here's a complete example from video to VFX-ready output:

### Step 1: Prepare Video

```bash
# Copy or symlink your video
cp ~/Videos/my_shot.mp4 ~/Desktop/
```

### Step 2: Run Pipeline

```bash
# Full pipeline
python scripts/run_pipeline.py \
  ~/Desktop/my_shot.mp4 \
  --name MyShot \
  -s ingest,depth,roto,mama,cleanplate,colmap,camera
```

### Step 3: Check Output

```bash
# List generated assets
ls -lh ../vfx_projects/MyShot/

# View frames
djv_view ../vfx_projects/MyShot/depth/

# Check camera data
ls ../vfx_projects/MyShot/camera/*.abc
```

### Step 4: Import to Nuke/Blender

**Nuke:**
```tcl
Read {
  file ../vfx_projects/MyShot/depth/frame_%04d.png
}
ReadGeo {
  file ../vfx_projects/MyShot/camera/camera.abc
}
```

**Blender:**
- File → Import → Alembic
- Select `../vfx_projects/MyShot/camera/camera.abc`
- File → Import → Stanford (.ply) for point cloud

### Step 5: Iterate

```bash
# Re-run specific stage with different settings
python scripts/run_pipeline.py \
  ~/Desktop/my_shot.mp4 \
  --name MyShot \
  -s roto \
  --prompt "person only"  # Different prompt
```

---

**Version:** 1.1
**Last Updated:** 2026-01-30
