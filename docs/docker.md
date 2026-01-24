# VFX Ingest Platform - Docker Guide

This guide covers using the VFX Ingest Platform with Docker, which eliminates the need for complex conda environment setup and COLMAP compilation.

## Prerequisites

### Required
- **Docker Desktop** (Linux/macOS/Windows)
  - Linux: https://docs.docker.com/engine/install/
  - macOS: https://docs.docker.com/desktop/mac/install/
  - Windows: https://docs.docker.com/desktop/windows/install/
- **NVIDIA GPU** with CUDA support
- **NVIDIA Docker runtime** (for GPU passthrough)
  - Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Disk Space
- **Docker image**: ~10-15GB (CUDA, Python, ComfyUI, dependencies)
- **Models**: ~15-20GB (downloaded separately to host)
- **Projects**: Variable (your output data)

## Quick Start

### 1. Download Models

Models are stored on your host machine (not in the Docker image) to avoid re-downloading.

```bash
# Run the model download script
./scripts/download_models.sh
```

This downloads:
- SAM3 (Segment Anything Model 3)
- Video Depth Anything
- WHAM (4D Human Motion Capture)
- MatAnyone (Matte Refinement)

**Optional:** SMPL-X models (required for mocap stage only):
- Can be downloaded automatically with the install wizard
- Register at https://smpl-x.is.tue.mpg.de/
- Create `SMPL.login.dat` with credentials (email on line 1, password on line 2)
- Run: `python3 scripts/install_wizard.py --component mocap`
- The wizard handles authentication and download automatically

Verify models:
```bash
python3 scripts/verify_models.py
```

### 2. Build Docker Image

```bash
# Build the image (one-time, 10-15 minutes)
docker-compose build
```

Or use the wrapper script which builds automatically:
```bash
bash scripts/run_docker.sh --help
```

### 3. Run Pipeline

Using wrapper script (recommended):
```bash
bash scripts/run_docker.sh --help
bash scripts/run_docker.sh video.mp4 --name MyProject --stages depth,roto
```

Or directly with docker-compose:
```bash
docker-compose run --rm vfx-ingest --help
docker-compose run --rm vfx-ingest /workspace/video.mp4 --name MyProject
```

## Directory Structure

### Volume Mounts

The Docker container uses two volume mounts:

1. **Models** (`/models`) - Read-only
   - Host: `~/.vfx_pipeline/models`
   - Container: `/models`
   - Contains ML models (SAM3, Depth, WHAM, etc.)

2. **Projects** (`/workspace`) - Read-write
   - Host: `~/VFX-Projects`
   - Container: `/workspace`
   - Contains your projects and output

### Environment Variables

Configure paths via environment variables:

```bash
# Custom model directory
export VFX_MODELS_DIR=/path/to/models
docker-compose run --rm vfx-ingest --help

# Custom projects directory
export VFX_PROJECTS_DIR=/path/to/projects
docker-compose run --rm vfx-ingest --help
```

Or edit `docker-compose.yml`:
```yaml
volumes:
  - /custom/path/models:/models:ro
  - /custom/path/projects:/workspace/projects
```

## Usage Examples

### Basic Pipeline

```bash
# Full pipeline (all stages)
bash scripts/run_docker.sh \
  ~/Videos/shot001.mp4 \
  --name Shot001 \
  --stages all

# Specific stages
bash scripts/run_docker.sh \
  ~/Videos/shot001.mp4 \
  --name Shot001 \
  --stages ingest,depth,roto
```

### With Options

```bash
# COLMAP camera tracking
bash scripts/run_docker.sh \
  ~/Videos/shot001.mp4 \
  --name Shot001 \
  --stages ingest,colmap \
  --colmap-quality high

# Custom segmentation prompt
bash scripts/run_docker.sh \
  ~/Videos/shot001.mp4 \
  --name Shot001 \
  --stages roto \
  --prompt "person, car"
```

### ComfyUI Web Interface

Access ComfyUI while the pipeline runs:

```bash
# Start container with ComfyUI exposed
docker-compose up

# Open browser to http://localhost:8188
```

## Troubleshooting

### GPU Not Detected

**Symptom:** "CUDA not available" or GPU not used

**Solution:**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# If fails, install NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Models Not Found

**Symptom:** "Model not found" or "WARNING: Model not found: /models/sam3"

**Solution:**
```bash
# Verify models on host
ls -lh ~/.vfx_pipeline/models/

# Run download script
./scripts/download_models.sh

# Verify
python3 scripts/verify_models.py
```

### Volume Mount Issues

**Symptom:** "Permission denied" or files not visible

**Solution:**
```bash
# Check volume mounts
docker-compose config

# Verify directory exists
mkdir -p ~/.vfx_pipeline/models
mkdir -p ~/VFX-Projects

# Check permissions
ls -ld ~/.vfx_pipeline/models
ls -ld ~/VFX-Projects
```

### Out of Memory

**Symptom:** Container killed or "Out of memory"

**Solution:**
```bash
# Increase Docker memory limit in Docker Desktop settings
# Recommended: 16GB+ for full pipeline

# Or process fewer frames
bash scripts/run_docker.sh \
  video.mp4 \
  --name Test \
  --stages depth \
  --skip-existing  # Skip already-processed stages
```

### Build Failures

**Symptom:** Docker build fails

**Solutions:**

1. **Network issues:**
   ```bash
   # Retry build
   docker-compose build --no-cache
   ```

2. **Disk space:**
   ```bash
   # Check available space
   df -h

   # Clean old images
   docker system prune -a
   ```

3. **Custom node dependencies:**
   ```bash
   # If a custom node fails to install, check logs:
   docker build -t vfx-ingest:latest . 2>&1 | tee build.log
   tail -100 build.log
   ```

## Advanced Usage

### Running Tests

```bash
# Integration tests (requires Docker)
./tests/integration/test_docker_build.sh

# Download test video
./tests/fixtures/download_football.sh

# Copy to projects directory (tests/ is not in container)
cp tests/fixtures/football_short.mp4 ~/VFX-Projects/

# Test with Football CIF
bash scripts/run_docker.sh \
  /workspace/projects/football_short.mp4 \
  --name FootballTest \
  --stages ingest,depth
```

### Accessing Container Shell

```bash
# Interactive shell in container
docker-compose run --rm vfx-ingest bash

# Inside container:
cd /app
python3 scripts/verify_models.py
ls /models
ls /workspace
```

### Custom Entrypoint

```bash
# Run specific Python script
docker-compose run --rm vfx-ingest python3 /app/scripts/export_camera.py /workspace/MyProject

# Run with different command
docker-compose run --rm --entrypoint bash vfx-ingest
```

### Multi-GPU

```bash
# Use specific GPU
docker-compose run --rm -e CUDA_VISIBLE_DEVICES=0 vfx-ingest --help

# Use multiple GPUs (modify docker-compose.yml)
# In docker-compose.yml:
#   environment:
#     - NVIDIA_VISIBLE_DEVICES=0,1
```

## Performance

### Expected Build Time
- First build: 10-15 minutes
- Rebuilds (code changes): 1-2 minutes (cached layers)

### Expected Runtime
Container overhead is minimal (~5% compared to local):

| Stage | 30 frames | 260 frames |
|-------|-----------|------------|
| Ingest | 10s | 1m |
| Depth | 1m | 8m |
| Roto | 1.5m | 12m |
| COLMAP | 2m | 10m |

(Times approximate, varies by GPU/CPU)

## Migration from Local

Already using local conda installation? Both work simultaneously:

```bash
# Local conda (as before)
conda activate vfx-pipeline
python scripts/run_pipeline.py video.mp4 --name Local

# Docker (new)
bash scripts/run_docker.sh video.mp4 --name Docker
```

Projects are compatible - you can process with Docker and view with local tools.

## What's Inside the Container

### System Packages
- Ubuntu 22.04
- CUDA 12.1 + cuDNN 8
- COLMAP (structure from motion)
- FFmpeg (video processing)
- Python 3.10

### Python Packages
- PyTorch (CUDA enabled)
- NumPy, SciPy, Trimesh, Pillow
- FastAPI, Uvicorn (web interface)
- All `requirements.txt` dependencies

### ComfyUI
- Main ComfyUI installation
- 5 custom nodes:
  - VideoHelperSuite
  - Video-Depth-Anything
  - SAM3
  - ProPainter
  - MatAnyone

### Your Code
- All `scripts/` Python modules
- All `workflow_templates/` ComfyUI workflows
- Web interface (`web/`)

## FAQ

**Q: Do I need to rebuild when code changes?**
A: Yes, rebuild after modifying scripts: `docker-compose build`

**Q: Can I use the web GUI in Docker?**
A: Yes! Run `docker-compose up` and visit http://localhost:8188

**Q: Where are my projects saved?**
A: On host at `~/VFX-Projects` (configurable via `VFX_PROJECTS_DIR`)

**Q: Can I use my own COLMAP installation?**
A: No need - COLMAP is installed in the container

**Q: How do I update?**
A: `git pull && docker-compose build`

**Q: Do models stay downloaded?**
A: Yes - models are on host (`~/.vfx_pipeline/models`), not in container

## Support

- Issues: https://github.com/kleer001/shot-gopher/issues
- Discussions: https://github.com/kleer001/shot-gopher/discussions

## See Also

- [Main README](../README.md) - General platform documentation
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Installation Guide](installation.md) - Detailed setup instructions
