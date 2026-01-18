# VFX Ingest Platform - Quick Start

Get the Docker-based VFX pipeline running in **under 10 minutes** with the automated installer.

## One-Line Install

**Linux / WSL2:**
```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.sh | bash
```

**Windows (PowerShell as Administrator):**
```powershell
irm https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.ps1 | iex
```

**Or manual clone:**
```bash
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
python3 scripts/install_wizard_docker.py
```

That's it! The bootstrap script will:
1. ✓ Clone the repository (or update if already exists)
2. ✓ Launch the installation wizard
3. ✓ Detect your platform (Linux/Windows/macOS)
4. ✓ Check prerequisites (Docker, NVIDIA driver, etc.)
5. ✓ Guide you through any missing installations
6. ✓ Download ML models (~15-20GB)
7. ✓ Build Docker image (~10-15 minutes)
8. ✓ Run test pipeline to verify everything works

**Windows note:** The PowerShell script sets up WSL2 and Ubuntu, then runs the Linux bootstrap inside WSL.

---

## Platform Support

| Platform | Status | Installation Method |
|----------|--------|---------------------|
| **Linux** | ✅ Fully Supported | Automated wizard |
| **Windows** | ✅ Supported via WSL2 | Automated wizard |
| **macOS** | ❌ Not Supported | Use local conda (see below) |

**Why macOS won't work:** Docker on macOS cannot access NVIDIA GPUs. macOS users must use the local conda installation:

```bash
# macOS only - use conda installation
python3 scripts/install_wizard.py
```

---

## What You Need

### Hardware
- **NVIDIA GPU** with CUDA support (GTX 1060 or better)
- **16GB+ RAM** (32GB recommended)
- **50GB+ free disk space**

### Software (wizard will check)
- Docker Engine (or Docker Desktop for Windows)
- NVIDIA drivers
- NVIDIA Container Toolkit (Linux) or NVIDIA CUDA on WSL (Windows)

---

## Platform-Specific Notes

### Linux
The wizard will guide you through installing:
- Docker via official install script
- NVIDIA Container Toolkit
- All prerequisites automatically detected

### Windows (WSL2)
Before running the wizard:

1. **Enable WSL2** (PowerShell as Administrator):
   ```powershell
   wsl --install
   wsl --set-default-version 2
   wsl --install -d Ubuntu-22.04
   ```

2. **Install NVIDIA CUDA on WSL2**:
   - Follow: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
   - Verify with `nvidia-smi` in WSL2

3. **Install Docker Desktop**:
   - Download: https://www.docker.com/products/docker-desktop
   - Enable WSL2 backend in settings
   - Enable integration with your Ubuntu distro

4. **Run wizard in WSL2 Ubuntu terminal**:
   ```bash
   python3 scripts/install_wizard_docker.py
   ```

---

## Usage After Installation

### Process Your Own Video

```bash
# Copy video to projects directory
cp /path/to/your/video.mp4 ~/VFX-Projects/

# Run pipeline
./scripts/run_docker.sh \
  --input /workspace/projects/video.mp4 \
  --name MyProject \
  --stages all
```

### Available Stages

```bash
# List all stages
./scripts/run_docker.sh --list-stages

# Common workflows:
./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name Test --stages depth,roto
./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name Test --stages colmap,camera
./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name Test --stages all
```

### Use ComfyUI Web Interface

```bash
# Start container with web interface
docker-compose up

# Open browser to http://localhost:8188
```

### Output Location

All projects saved to `~/VFX-Projects/` on your host machine:

```
~/VFX-Projects/
└── MyProject/
    ├── source/frames/      # Extracted frames
    ├── depth/              # Depth maps
    ├── roto/               # Segmentation masks
    ├── cleanplate/         # Clean plates
    ├── colmap/             # Camera tracking
    ├── mocap/              # Motion capture
    ├── camera/             # Camera exports (.abc, .chan, .clip)
    └── preview/            # Preview videos
```

---

## Wizard Options

```bash
# Check prerequisites without installing
python3 scripts/install_wizard_docker.py --check-only

# Skip test pipeline (faster)
python3 scripts/install_wizard_docker.py --skip-test

# Custom models directory
python3 scripts/install_wizard_docker.py --models-dir /path/to/models

# Show help
python3 scripts/install_wizard_docker.py --help
```

---

## Troubleshooting

### Wizard Says Prerequisites Missing

The wizard checks for:
1. **NVIDIA driver** - Install from: https://www.nvidia.com/Download/index.aspx
2. **Docker** - Follow wizard's instructions
3. **NVIDIA Container Toolkit** - Follow wizard's instructions

Re-run wizard after installing missing components.

### Docker Build Fails

```bash
# Check disk space
df -h

# Clean old images
docker system prune -a

# Retry build
docker-compose build --no-cache
```

### GPU Not Detected

**Linux:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

**Windows/WSL2:**
```bash
# In WSL2, verify CUDA
nvidia-smi

# Should show Windows GPU driver version
```

### Models Download Fails

```bash
# Retry download manually
./scripts/download_models.sh

# Verify
python3 scripts/verify_models.py
```

### Out of Memory

```bash
# Increase Docker memory limit
# Linux: Edit daemon.json
# Windows: Docker Desktop > Settings > Resources > Memory (16GB+)

# Or process shorter videos
ffmpeg -i input.mp4 -t 5 short.mp4  # First 5 seconds
```

---

## Manual Installation (Not Recommended)

If you prefer manual installation or the wizard doesn't work, see [docs/README-DOCKER.md](docs/README-DOCKER.md) for step-by-step instructions.

---

## Verify Installation

After wizard completes:

```bash
# Run test suite
python3 tests/test_phase_1_complete.py

# All 7 tests should pass ✓
```

---

## Performance Expectations

**First-time setup:** ~20-30 minutes
- Model downloads: 10-20 minutes
- Docker build: 10-15 minutes
- Test pipeline: 1 minute

**Processing 30 frames:**
- Depth: ~1 minute
- Roto: ~1.5 minutes
- COLMAP: ~2 minutes
- Full pipeline: ~5-10 minutes

**Processing 260 frames (10 seconds @ 24fps):**
- Depth: ~8 minutes
- Roto: ~12 minutes
- COLMAP: ~10 minutes
- Full pipeline: ~30-40 minutes

*(Times vary by GPU/CPU)*

---

## Next Steps

**Explore the pipeline:**
- [Main README](README.md) - Full platform documentation
- [Docker Guide](docs/README-DOCKER.md) - Comprehensive Docker usage
- [ATLAS](docs/ATLAS.md) - Development roadmap

**Try different stages:**
```bash
# Depth only
./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name Test --stages depth

# Segmentation with custom prompt
./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name Test --stages roto --prompt "person, car"

# Camera tracking
./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name Test --stages colmap,camera
```

**Access ComfyUI workflows:**
- Start: `docker-compose up`
- Open: http://localhost:8188
- Load workflows from `~/VFX-Projects/YourProject/workflows/`

---

## Getting Help

- **Documentation:** [docs/README-DOCKER.md](docs/README-DOCKER.md)
- **Issues:** https://github.com/kleer001/comfyui_ingest/issues
- **Discussions:** https://github.com/kleer001/comfyui_ingest/discussions

---

**That's it!** You should now have a fully functional Docker-based VFX pipeline. The wizard handles all the complexity - just run it and follow the prompts!
