# VFX Ingest Platform - Developer Quick Start

Get the Docker-based VFX pipeline running in under 10 minutes.

## Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully Supported | Native Docker + NVIDIA runtime |
| **Windows** | ✅ Supported | Via WSL2 + Docker Desktop + NVIDIA CUDA on WSL |
| **macOS** | ❌ Not Supported | Docker Desktop doesn't support GPU passthrough |

**Why macOS doesn't work:** This pipeline requires NVIDIA CUDA for GPU acceleration. Docker on macOS cannot access NVIDIA GPUs, making it incompatible with the containerized workflow.

**macOS users:** Use the local conda installation instead (see main README.md).

---

## Prerequisites

### All Platforms
- **NVIDIA GPU** with CUDA support (GTX 1060 or better recommended)
- **16GB+ RAM** (32GB recommended)
- **50GB+ free disk space** (Docker image + models + projects)

### Linux
- Docker Engine 20.10+
- NVIDIA Docker runtime (Container Toolkit)

### Windows
- Windows 10/11 with WSL2 enabled
- Docker Desktop for Windows
- NVIDIA CUDA on WSL2

---

## Quick Start (Linux)

### 1. Install Prerequisites

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Clone Repository

```bash
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
```

### 3. Download Models

```bash
# Create models directory
mkdir -p ~/.vfx_pipeline/models

# Download models (15-20GB, takes 10-20 minutes)
./scripts/download_models.sh

# Verify downloads
python3 scripts/verify_models.py
```

**Manual step required:** SMPL-X models need registration:
1. Visit https://smpl-x.is.tue.mpg.de/
2. Register and download SMPL-X models
3. Extract to `~/.vfx_pipeline/models/smplx/`

### 4. Build Docker Image

```bash
# First build takes 10-15 minutes
docker-compose build

# Verify build
docker images | grep vfx-ingest
```

### 5. Download Test Video

```bash
# Download Football CIF test video (academic standard, ~2MB)
./tests/fixtures/download_football.sh

# Copy to projects directory
mkdir -p ~/VFX-Projects
cp tests/fixtures/football_short.mp4 ~/VFX-Projects/
```

### 6. Run Test Pipeline

```bash
# Test depth analysis stage (fastest, ~1 minute)
./scripts/run_docker.sh \
  --input /workspace/projects/football_short.mp4 \
  --name FootballTest \
  --stages depth

# Check output
ls -lh ~/VFX-Projects/FootballTest/depth/
```

**Success looks like:**
```
~/VFX-Projects/FootballTest/
├── source/frames/         # 30 PNG frames
├── depth/                 # 30 depth maps
├── workflows/             # ComfyUI workflow JSON
└── project.json           # Project metadata
```

### 7. Run Full Pipeline (Optional)

```bash
# All stages (takes 5-10 minutes for 30 frames)
./scripts/run_docker.sh \
  --input /workspace/projects/football_short.mp4 \
  --name FootballFull \
  --stages all
```

---

## Quick Start (Windows + WSL2)

### 1. Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install
wsl --set-default-version 2

# Install Ubuntu
wsl --install -d Ubuntu-22.04

# Restart computer
```

### 2. Install NVIDIA CUDA on WSL2

Follow NVIDIA's official guide:
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

```bash
# Inside WSL2 Ubuntu terminal, verify CUDA
nvidia-smi
```

### 3. Install Docker Desktop

1. Download Docker Desktop for Windows: https://www.docker.com/products/docker-desktop
2. Install with WSL2 backend enabled
3. In Docker Desktop settings:
   - Enable "Use the WSL 2 based engine"
   - Under "Resources > WSL Integration", enable your Ubuntu distro

### 4. Verify GPU Access

```bash
# Inside WSL2 Ubuntu terminal
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 5. Follow Linux Steps

Once GPU access works in WSL2, follow the Linux quick start steps above (steps 2-7).

**Windows-specific notes:**
- All commands run inside WSL2 Ubuntu terminal
- Projects saved to `~/VFX-Projects` in WSL2 filesystem (accessible from Windows at `\\wsl$\Ubuntu-22.04\home\<username>\VFX-Projects`)
- For best performance, keep all files in WSL2 filesystem (not `/mnt/c/`)

---

## Common Issues

### "CUDA not available" or GPU not detected

**Linux:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

**Windows:**
```bash
# In WSL2, verify CUDA
nvidia-smi

# Verify Docker Desktop is using WSL2 backend
docker info | grep "Operating System"
# Should show: Operating System: Docker Desktop
```

### "Models not found"

```bash
# Verify models downloaded
ls -lh ~/.vfx_pipeline/models/

# Expected directories:
# - sam3/
# - videodepthanything/
# - wham/
# - matanyone/
# - smplx/ (manual download required)

# Re-run download script
./scripts/download_models.sh

# Check with verification script
python3 scripts/verify_models.py
```

### "Permission denied" mounting volumes

**Linux:**
```bash
# Ensure directories exist with correct permissions
mkdir -p ~/.vfx_pipeline/models
mkdir -p ~/VFX-Projects
chmod 755 ~/.vfx_pipeline/models
chmod 755 ~/VFX-Projects
```

**Windows/WSL2:**
```bash
# Use WSL2 home directory, not /mnt/c/
# ✓ Good: ~/VFX-Projects
# ✗ Bad:  /mnt/c/Users/YourName/VFX-Projects
```

### Docker build fails

```bash
# Check disk space
df -h

# Clean old Docker images
docker system prune -a

# Retry build with no cache
docker-compose build --no-cache
```

### Out of memory during pipeline

```bash
# Check Docker memory limit
docker info | grep Memory

# Increase in Docker Desktop: Settings > Resources > Memory (set to 16GB+)

# Or process fewer frames
ffmpeg -i input.mp4 -vframes 10 short.mp4
```

---

## Verify Installation

Run all tests:

```bash
# Python test suite (7 tests)
python3 tests/test_phase_1_complete.py

# Docker build tests (requires Docker, 7 tests)
./tests/integration/test_docker_build.sh

# All tests should pass ✓
```

---

## Next Steps

**Option 1: Process your own video**
```bash
# Copy your video to projects directory
cp /path/to/your/video.mp4 ~/VFX-Projects/

# Run pipeline
./scripts/run_docker.sh \
  --input /workspace/projects/your_video.mp4 \
  --name YourProject \
  --stages depth,roto,colmap
```

**Option 2: Use ComfyUI web interface**
```bash
# Start container with web interface
docker-compose up

# Open browser to http://localhost:8188
# Load workflows from project's workflows/ directory
```

**Option 3: Explore other stages**
```bash
# List all available stages
./scripts/run_docker.sh --list-stages

# Run specific stages
./scripts/run_docker.sh \
  --input /workspace/projects/video.mp4 \
  --name Test \
  --stages roto \
  --prompt "person, ball, car"  # Custom segmentation
```

---

## Platform-Specific Performance

**Linux (Native):**
- Best performance
- Minimal overhead (~2-5%)
- Direct GPU access

**Windows (WSL2):**
- Good performance
- Slight overhead (~5-10%)
- GPU access via WSL2 passthrough
- Keep files in WSL2 filesystem for best speed

**macOS:**
- Not supported (no GPU access)
- Use local conda installation instead

---

## Getting Help

- **Documentation:** [docs/README-DOCKER.md](docs/README-DOCKER.md)
- **Issues:** https://github.com/kleer001/comfyui_ingest/issues
- **Discussions:** https://github.com/kleer001/comfyui_ingest/discussions

## See Also

- [Main README](README.md) - Full platform documentation
- [Docker Guide](docs/README-DOCKER.md) - Comprehensive Docker usage
- [ATLAS](docs/ATLAS.md) - Development roadmap
- [Roadmap 1](docs/ROADMAP-1-DOCKER.md) - Docker migration technical details
