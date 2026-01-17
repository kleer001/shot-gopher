# 📋 Roadmap 1: Docker Migration

**Goal:** Replicate current CLI functionality using Docker containers

**Status:** 🟡 In Progress

**Dependencies:** None (starting point)

---

## Overview

This roadmap transitions the VFX Ingest Platform from conda-based local installation to Docker containers while maintaining 100% feature parity. Users still interact via CLI, but all execution happens inside containers.

### Key Principles
- **No Feature Changes** - Exact same functionality as current installation
- **Volume Mounts** - Models and projects persist on host filesystem
- **Container-Aware Code** - Scripts detect and adapt to containerized environment
- **Backwards Compatible** - Local installation still works during transition

---

## Phase 1A: Container Foundation ⚪

**Goal:** Build base Docker image with all system dependencies

### Deliverables
- `Dockerfile` - Multi-stage container image
- `docker-compose.yml` - Service orchestration and volume configuration
- `.dockerignore` - Build optimization
- `docker/` directory with supporting files

### Tasks

#### Task 1A.1: Create Dockerfile Base Layer
**File:** `Dockerfile`

```dockerfile
# Stage 1: Base image with system dependencies
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Install system packages
RUN apt-get update && apt-get install -y \
    colmap \
    ffmpeg \
    git \
    python3.10 \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive
```

**Validation:**
```bash
docker build --target base -t vfx-ingest:base .
docker run --rm vfx-ingest:base colmap --help
docker run --rm vfx-ingest:base ffmpeg -version
docker run --rm vfx-ingest:base python3 --version
```

**Success Criteria:**
- [ ] Image builds without errors
- [ ] COLMAP installed and executable
- [ ] FFmpeg installed and functional
- [ ] Python 3.10 available

---

#### Task 1A.2: Python Dependencies Layer
**File:** `Dockerfile` (continued)

```dockerfile
# Stage 2: Python dependencies
FROM base AS python-deps

# Copy requirements
COPY requirements.txt /tmp/

# Install Python packages
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Validation:**
```bash
docker build --target python-deps -t vfx-ingest:python .
docker run --rm --gpus all vfx-ingest:python python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
docker run --rm vfx-ingest:python python3 -c "import numpy, scipy, trimesh, PIL; print('Core deps OK')"
```

**Success Criteria:**
- [ ] All requirements.txt packages installed
- [ ] PyTorch detects CUDA (with --gpus flag)
- [ ] No dependency conflicts

---

#### Task 1A.3: ComfyUI Installation Layer
**File:** `Dockerfile` (continued)

```dockerfile
# Stage 3: ComfyUI and custom nodes
FROM python-deps AS comfyui

# Create .vfx_pipeline directory structure
RUN mkdir -p /app/.vfx_pipeline

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/.vfx_pipeline/ComfyUI

# Clone custom nodes
WORKDIR /app/.vfx_pipeline/ComfyUI/custom_nodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git && \
    git clone https://github.com/PozzettiAndrea/ComfyUI-SAM3.git && \
    git clone https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git && \
    git clone https://github.com/FuouM/ComfyUI-MatAnyone.git

# Install custom node dependencies
RUN for dir in */; do \
        if [ -f "$dir/requirements.txt" ]; then \
            pip3 install --no-cache-dir -r "$dir/requirements.txt"; \
        fi; \
    done

WORKDIR /app
```

**Validation:**
```bash
docker build --target comfyui -t vfx-ingest:comfyui .
docker run --rm vfx-ingest:comfyui ls /app/.vfx_pipeline/ComfyUI/custom_nodes/
docker run --rm vfx-ingest:comfyui python3 -c "import sys; sys.path.insert(0, '/app/.vfx_pipeline/ComfyUI'); import folder_paths; print('ComfyUI imports OK')"
```

**Success Criteria:**
- [ ] ComfyUI cloned successfully
- [ ] All 5 custom nodes present
- [ ] Custom node dependencies installed
- [ ] ComfyUI Python modules importable

---

#### Task 1A.4: Pipeline Scripts Layer
**File:** `Dockerfile` (continued)

```dockerfile
# Stage 4: Pipeline scripts
FROM comfyui AS pipeline

# Copy pipeline scripts
COPY scripts/ /app/scripts/
COPY workflow_templates/ /app/workflow_templates/

# Set Python path
ENV PYTHONPATH=/app/scripts:$PYTHONPATH

# Copy entrypoint
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Mark as container environment
ENV CONTAINER=true \
    VFX_INSTALL_DIR=/app/.vfx_pipeline \
    VFX_MODELS_DIR=/models \
    VFX_PROJECTS_DIR=/workspace/projects \
    COMFYUI_OUTPUT_DIR=/workspace

# Expose ports
EXPOSE 8188

# Volumes
VOLUME ["/models", "/workspace"]

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
```

**Validation:**
```bash
docker build -t vfx-ingest:latest .
docker run --rm vfx-ingest:latest --help
docker run --rm vfx-ingest:latest ls /app/scripts/
```

**Success Criteria:**
- [ ] Image builds completely
- [ ] Scripts copied into image
- [ ] Entrypoint executable
- [ ] `--help` displays usage info

---

#### Task 1A.5: Create Entrypoint Script
**File:** `docker/entrypoint.sh`

```bash
#!/bin/bash
set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== VFX Ingest Platform (Container) ===${NC}"

# Validate mounted volumes
if [ ! -d "/models" ]; then
    echo -e "${RED}ERROR: /models volume not mounted${NC}"
    echo "Docker run must include: -v /path/to/models:/models"
    exit 1
fi

if [ ! -d "/workspace" ]; then
    echo -e "${RED}ERROR: /workspace volume not mounted${NC}"
    echo "Docker run must include: -v /path/to/workspace:/workspace"
    exit 1
fi

# Check for required models
echo -e "${YELLOW}Checking models...${NC}"
REQUIRED_MODELS=(
    "/models/sam3"
    "/models/videodepthanything"
    "/models/wham"
)

MISSING_MODELS=0
for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -d "$model" ]; then
        echo -e "${YELLOW}  WARNING: Model not found: $model${NC}"
        MISSING_MODELS=1
    else
        echo -e "${GREEN}  ✓ Found: $model${NC}"
    fi
done

if [ $MISSING_MODELS -eq 1 ]; then
    echo -e "${YELLOW}Some models are missing. Pipeline may fail on certain stages.${NC}"
    echo "Run model download script on host: ./scripts/download_models.sh"
fi

# Start ComfyUI in background if requested
if [ "$START_COMFYUI" = "true" ]; then
    echo -e "${GREEN}Starting ComfyUI...${NC}"
    cd /app/.vfx_pipeline/ComfyUI
    python3 main.py --listen 0.0.0.0 --port 8188 \
        --output-directory /workspace > /tmp/comfyui.log 2>&1 &
    COMFYUI_PID=$!

    # Wait for ComfyUI to be ready
    echo "Waiting for ComfyUI to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
            echo -e "${GREEN}✓ ComfyUI ready on port 8188${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo -e "${RED}ERROR: ComfyUI failed to start${NC}"
            cat /tmp/comfyui.log
            exit 1
        fi
    done
fi

# Execute the main command
echo -e "${GREEN}Running pipeline...${NC}"
cd /app
exec python3 /app/scripts/run_pipeline.py "$@"
```

**Validation:**
```bash
chmod +x docker/entrypoint.sh
docker run --rm \
  -v ~/.vfx_pipeline/models:/models \
  -v $(pwd)/test:/workspace \
  vfx-ingest:latest --help
```

**Success Criteria:**
- [ ] Entrypoint validates volumes
- [ ] Entrypoint checks for models
- [ ] Entrypoint can start ComfyUI
- [ ] Entrypoint executes pipeline scripts

---

#### Task 1A.6: Create docker-compose.yml
**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  vfx-ingest:
    build: .
    image: vfx-ingest:latest
    container_name: vfx-ingest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - START_COMFYUI=true
      - VFX_MODELS_DIR=/models
      - VFX_PROJECTS_DIR=/workspace/projects
    volumes:
      # Model storage (read-only for safety)
      - ${HOME}/.vfx_pipeline/models:/models:ro
      # Project workspace (read-write)
      - ${HOME}/VFX-Projects:/workspace/projects
      # Optional: mount specific project
      # - ./my_project:/workspace/project
    ports:
      - "8188:8188"  # ComfyUI web interface
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Validation:**
```bash
docker-compose build
docker-compose run --rm vfx-ingest --help
```

**Success Criteria:**
- [ ] Compose file valid syntax
- [ ] GPU passthrough configured
- [ ] Volumes mounted correctly
- [ ] Ports exposed

---

#### Task 1A.7: Create .dockerignore
**File:** `.dockerignore`

```
# Git
.git/
.gitignore
.github/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/

# Testing
.pytest_cache/
tests/
test_*/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Docs (don't need in image)
docs/
*.md
README.md

# Large files that should be mounted
.vfx_pipeline/models/
.vfx_pipeline/ComfyUI/output/

# Project files (mounted at runtime)
../vfx_projects/

# OS
.DS_Store
Thumbs.db

# CI/CD
.github/
.gitlab-ci.yml
```

**Success Criteria:**
- [ ] Image build time reduced
- [ ] Image size minimized
- [ ] No sensitive files in image

---

### Phase 1A Exit Criteria

- [ ] Complete Dockerfile builds successfully
- [ ] All system dependencies installed and functional
- [ ] ComfyUI and custom nodes installed
- [ ] Pipeline scripts embedded in image
- [ ] Entrypoint script validates and starts services
- [ ] docker-compose.yml configuration valid
- [ ] Image size reasonable (<15GB)

---

## Phase 1B: Code Modifications ⚪

**Goal:** Make Python scripts container-aware

### Deliverables
- Modified `scripts/env_config.py` with container detection
- Modified `scripts/comfyui_manager.py` with container paths
- Modified `scripts/run_pipeline.py` with container compatibility
- Backward compatibility with local installation maintained

### Tasks

#### Task 1B.1: Add Container Detection to env_config.py
**File:** `scripts/env_config.py`

**Changes:**
```python
def is_in_container() -> bool:
    """
    Detect if running inside a container.

    Returns:
        bool: True if in container, False otherwise
    """
    # Check for Docker
    if os.path.exists("/.dockerenv"):
        return True

    # Check environment variable
    if os.environ.get("CONTAINER") == "true":
        return True

    # Check for Kubernetes
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True

    # Check cgroup (Linux containers)
    try:
        with open("/proc/1/cgroup", "rt") as f:
            return "docker" in f.read() or "kubepods" in f.read()
    except Exception:
        pass

    return False


def check_conda_env_or_warn(env_name: str = None) -> bool:
    """Check environment and print a warning if wrong, but continue execution."""

    # Skip conda checks inside containers
    if is_in_container():
        print("Running in container environment - skipping conda checks")
        return True

    # ... existing conda check logic ...


# Path configuration - allow environment variable overrides
INSTALL_DIR = Path(os.environ.get(
    "VFX_INSTALL_DIR",
    _REPO_ROOT / ".vfx_pipeline"
))

DEFAULT_PROJECTS_DIR = Path(os.environ.get(
    "VFX_PROJECTS_DIR",
    _REPO_ROOT.parent / "vfx_projects"
))
```

**Validation:**
```bash
# Test local (should check conda)
python scripts/env_config.py

# Test container (should skip conda)
docker run --rm vfx-ingest:latest python /app/scripts/env_config.py
```

**Success Criteria:**
- [ ] `is_in_container()` correctly detects container
- [ ] Conda checks skipped in container
- [ ] Conda checks still work in local mode
- [ ] Environment variable overrides respected

---

#### Task 1B.2: Update comfyui_manager.py for Container Paths
**File:** `scripts/comfyui_manager.py`

**Changes:**
```python
def start_comfyui(
    gpu: int = None,
    listen: str = "127.0.0.1",
    port: int = 8188,
    lowvram: bool = False
) -> bool:
    """Start ComfyUI server."""

    # ... existing code ...

    # Container-aware output directory
    if os.environ.get("CONTAINER") == "true":
        output_base = Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/workspace"))
        listen = "0.0.0.0"  # Must listen on all interfaces in container
    else:
        output_base = COMFYUI_DIR.parent.parent.parent

    cmd = [
        sys.executable,
        "main.py",
        "--listen", listen,
        "--port", str(port),
        "--output-directory", str(output_base),
    ]

    # ... rest of existing code ...
```

**Validation:**
```bash
# Test that ComfyUI starts with correct output directory
docker run --rm \
  -e START_COMFYUI=true \
  -v ~/.vfx_pipeline/models:/models \
  -v $(pwd)/test:/workspace \
  vfx-ingest:latest \
  python /app/scripts/comfyui_manager.py
```

**Success Criteria:**
- [ ] ComfyUI starts in container with correct output path
- [ ] ComfyUI listens on 0.0.0.0 in container
- [ ] Local installation still works with 127.0.0.1

---

#### Task 1B.3: Update run_pipeline.py Path Handling
**File:** `scripts/run_pipeline.py`

**Changes:**
```python
def main():
    parser = argparse.ArgumentParser(...)
    # ... existing argument parsing ...

    # Resolve project directory - container-aware
    if args.name:
        projects_dir = Path(os.environ.get(
            "VFX_PROJECTS_DIR",
            env_config.DEFAULT_PROJECTS_DIR
        ))
        project_dir = projects_dir / args.name
    elif args.project:
        project_dir = Path(args.project)
        # In container, ensure path is under /workspace
        if env_config.is_in_container() and not str(project_dir).startswith("/workspace"):
            print(f"ERROR: In container, project must be under /workspace")
            print(f"  Got: {project_dir}")
            print(f"  Mount your project directory to /workspace")
            return 1
    else:
        project_dir = Path.cwd()

    # ... rest of existing code ...
```

**Validation:**
```bash
# Test path validation in container
docker run --rm \
  -v $(pwd)/test_project:/workspace/test_project \
  vfx-ingest:latest \
  --project /workspace/test_project \
  --stages ingest
```

**Success Criteria:**
- [ ] Path resolution works in container
- [ ] Error messages helpful for container users
- [ ] Local path handling unchanged

---

#### Task 1B.4: Update All Path References
**Files:** All scripts in `scripts/`

**Pattern to find:**
```bash
grep -r "INSTALL_DIR" scripts/
grep -r "DEFAULT_PROJECTS_DIR" scripts/
grep -r "COMFYUI_DIR" scripts/
```

**Changes:**
Replace hardcoded path assumptions with environment-aware lookups:
```python
# Before
models_dir = INSTALL_DIR / "models"

# After
models_dir = Path(os.environ.get("VFX_MODELS_DIR", INSTALL_DIR / "models"))
```

**Success Criteria:**
- [ ] All scripts use environment variables for paths
- [ ] Backward compatibility maintained
- [ ] No hardcoded assumptions about filesystem structure

---

### Phase 1B Exit Criteria

- [ ] All scripts run successfully in container
- [ ] All scripts still run successfully in local mode
- [ ] No conda warnings in container logs
- [ ] Paths resolve correctly in both modes
- [ ] Unit tests pass in both environments

---

## Phase 1C: Model Management ⚪

**Goal:** External model download script that populates host directory

### Deliverables
- `scripts/download_models.sh` - Host-side model downloader
- `scripts/verify_models.py` - Model validation script
- Documentation for manual model downloads (SMPL-X)

### Tasks

#### Task 1C.1: Create Model Download Script
**File:** `scripts/download_models.sh`

```bash
#!/bin/bash
set -e

# Model download script (runs on HOST, not in container)
MODEL_DIR="${HOME}/.vfx_pipeline/models"
mkdir -p "$MODEL_DIR"

echo "=== VFX Ingest Platform - Model Downloader ==="
echo "Models will be downloaded to: $MODEL_DIR"
echo ""

# Function to download with progress
download_model() {
    local name=$1
    local url=$2
    local dest=$3

    echo "Downloading $name..."
    mkdir -p "$(dirname "$dest")"

    if command -v wget &> /dev/null; then
        wget -O "$dest" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$dest" "$url"
    else
        echo "ERROR: Neither wget nor curl found. Install one and try again."
        exit 1
    fi
}

# SAM3 (via HuggingFace)
echo "[1/4] SAM3..."
if [ ! -d "$MODEL_DIR/sam3" ]; then
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download('1038lab/sam3', local_dir='$MODEL_DIR/sam3')"
    echo "✓ SAM3 downloaded"
else
    echo "✓ SAM3 already exists"
fi

# Video Depth Anything
echo "[2/4] Video Depth Anything..."
if [ ! -d "$MODEL_DIR/videodepthanything" ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Video-Depth-Anything-Small', local_dir='$MODEL_DIR/videodepthanything')"
    echo "✓ Video Depth Anything downloaded"
else
    echo "✓ Video Depth Anything already exists"
fi

# WHAM (via Google Drive - requires gdown)
echo "[3/4] WHAM..."
if [ ! -d "$MODEL_DIR/wham" ]; then
    pip install gdown
    mkdir -p "$MODEL_DIR/wham"
    gdown "1234567890abcdef" -O "$MODEL_DIR/wham/checkpoint.pth"
    echo "✓ WHAM downloaded"
else
    echo "✓ WHAM already exists"
fi

# MatAnyone
echo "[4/4] MatAnyone..."
if [ ! -d "$MODEL_DIR/matanyone" ]; then
    mkdir -p "$MODEL_DIR/matanyone"
    download_model "MatAnyone" \
        "https://github.com/FuouM/ComfyUI-MatAnyone/releases/download/v1.0/matanyone.pth" \
        "$MODEL_DIR/matanyone/matanyone.pth"
    echo "✓ MatAnyone downloaded"
else
    echo "✓ MatAnyone already exists"
fi

echo ""
echo "=== Public Models Downloaded ==="
echo ""
echo "⚠ MANUAL DOWNLOAD REQUIRED:"
echo "SMPL-X models require registration:"
echo "  1. Register at: https://smpl-x.is.tue.mpg.de/"
echo "  2. Download models to: $MODEL_DIR/smplx/"
echo "  3. Run: python scripts/verify_models.py"
echo ""
echo "Once SMPL-X is downloaded, you're ready to run the pipeline!"
```

**Success Criteria:**
- [ ] Script downloads public models successfully
- [ ] Script is idempotent (safe to re-run)
- [ ] Clear instructions for manual downloads
- [ ] Models placed in correct directory structure

---

#### Task 1C.2: Create Model Verification Script
**File:** `scripts/verify_models.py`

```python
"""Verify all required models are present and valid."""

import os
from pathlib import Path
import sys

MODEL_DIR = Path(os.environ.get("VFX_MODELS_DIR", Path.home() / ".vfx_pipeline/models"))

REQUIRED_MODELS = {
    "sam3": {
        "path": MODEL_DIR / "sam3",
        "files": ["model.safetensors", "config.json"],
        "size_mb": 3200,
    },
    "videodepthanything": {
        "path": MODEL_DIR / "videodepthanything",
        "files": ["model.safetensors"],
        "size_mb": 120,
    },
    "wham": {
        "path": MODEL_DIR / "wham",
        "files": ["checkpoint.pth"],
        "size_mb": 1200,
    },
    "matanyone": {
        "path": MODEL_DIR / "matanyone",
        "files": ["matanyone.pth"],
        "size_mb": 141,
    },
    "smplx": {
        "path": MODEL_DIR / "smplx",
        "files": ["SMPLX_NEUTRAL.pkl", "SMPLX_MALE.pkl", "SMPLX_FEMALE.pkl"],
        "size_mb": 830,
    },
}

def check_model(name, config):
    """Check if a model is present and valid."""
    path = config["path"]

    if not path.exists():
        return False, f"Directory not found: {path}"

    missing_files = []
    for file in config["files"]:
        if not (path / file).exists():
            missing_files.append(file)

    if missing_files:
        return False, f"Missing files: {', '.join(missing_files)}"

    return True, "OK"

def main():
    print(f"Checking models in: {MODEL_DIR}\n")

    all_ok = True
    for name, config in REQUIRED_MODELS.items():
        ok, msg = check_model(name, config)
        status = "✓" if ok else "✗"
        print(f"{status} {name:20s} {msg}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("✓ All models present and valid!")
        return 0
    else:
        print("✗ Some models are missing or invalid")
        print("\nRun: ./scripts/download_models.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Success Criteria:**
- [ ] Script detects missing models
- [ ] Script validates file presence
- [ ] Clear output for troubleshooting

---

### Phase 1C Exit Criteria

- [ ] Model download script works on all platforms
- [ ] All public models download successfully
- [ ] Verification script accurately detects issues
- [ ] Documentation clear for manual downloads
- [ ] Models persist between container restarts

---

## Phase 1D: Integration Testing ⚪

**Goal:** End-to-end validation of containerized pipeline

### Test Cases

#### Test 1D.1: Ingest Stage
```bash
docker-compose run --rm vfx-ingest \
  --input /workspace/test_video.mp4 \
  --name TestShot \
  --stages ingest
```

**Expected:**
- Frames extracted to `/workspace/projects/TestShot/source/frames/`
- FFmpeg runs without errors
- Frame count matches video

---

#### Test 1D.2: Depth Estimation
```bash
docker-compose run --rm vfx-ingest \
  --name TestShot \
  --stages depth
```

**Expected:**
- ComfyUI starts successfully
- Depth maps generated in `/workspace/projects/TestShot/depth/`
- GPU utilized correctly

---

#### Test 1D.3: COLMAP Camera Tracking
```bash
docker-compose run --rm vfx-ingest \
  --name TestShot \
  --stages colmap \
  --colmap-quality medium
```

**Expected:**
- COLMAP runs without installation errors
- Sparse reconstruction succeeds
- Camera data exported to JSON and Alembic

---

#### Test 1D.4: Full Pipeline
```bash
docker-compose run --rm vfx-ingest \
  --input /workspace/test_video.mp4 \
  --name FullTest \
  --stages ingest,depth,roto,colmap
```

**Expected:**
- All stages complete successfully
- Output files present in project directory
- Performance comparable to local installation

---

#### Test 1D.5: Volume Mount Validation
```bash
# Create test file on host
echo "test" > ~/VFX-Projects/mount_test.txt

# Check visibility in container
docker-compose run --rm vfx-ingest ls /workspace/projects/

# Modify from container
docker-compose run --rm vfx-ingest \
  bash -c "echo 'from container' > /workspace/projects/container_test.txt"

# Verify on host
cat ~/VFX-Projects/container_test.txt
```

**Expected:**
- Files visible bidirectionally
- Permissions correct
- No data loss

---

### Phase 1D Exit Criteria

- [ ] All pipeline stages work in container
- [ ] Output identical to local installation
- [ ] Performance within 10% of local
- [ ] No file permission issues
- [ ] GPU utilization correct
- [ ] Error messages helpful

---

## Phase 1E: Documentation & User Tools ⚪

**Goal:** User-facing documentation and helper scripts

### Deliverables
- `README-DOCKER.md` - Docker usage guide
- `scripts/run_docker.sh` - Simplified wrapper script
- `scripts/setup_docker.sh` - First-time Docker setup
- Migration guide from local to Docker

### Tasks

#### Task 1E.1: Create Wrapper Script
**File:** `scripts/run_docker.sh`

```bash
#!/bin/bash
# Simplified wrapper for Docker execution

set -e

# Defaults
MODELS_DIR="${HOME}/.vfx_pipeline/models"
PROJECTS_DIR="${HOME}/VFX-Projects"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running"
    echo "Start Docker Desktop and try again"
    exit 1
fi

# Check models exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models not found at $MODELS_DIR"
    echo "Run: ./scripts/download_models.sh"
    exit 1
fi

# Check image exists
if ! docker image inspect vfx-ingest:latest > /dev/null 2>&1; then
    echo "Building Docker image (first time only)..."
    docker-compose build
fi

# Run container
docker-compose run --rm vfx-ingest "$@"
```

**Success Criteria:**
- [ ] Checks preconditions before running
- [ ] Clear error messages
- [ ] Passes arguments through correctly

---

#### Task 1E.2: Create Docker Setup Guide
**File:** `docs/README-DOCKER.md`

Content: Installation guide, troubleshooting, common issues

**Success Criteria:**
- [ ] Covers all platforms (Linux, macOS, Windows)
- [ ] Troubleshooting section for common errors
- [ ] Examples for all use cases

---

### Phase 1E Exit Criteria

- [ ] Documentation complete and accurate
- [ ] Wrapper scripts tested on all platforms
- [ ] Migration path documented
- [ ] Troubleshooting guide comprehensive

---

## Roadmap 1 Success Criteria

**Ready to move to Roadmap 2 when:**

- [ ] All Phase 1A-1E tasks complete
- [ ] Docker image builds reliably
- [ ] All pipeline stages functional in container
- [ ] Models persist correctly via volume mounts
- [ ] Performance acceptable (within 10% of local)
- [ ] Documentation complete
- [ ] At least 2 users have successfully deployed
- [ ] COLMAP works without any host installation

**Known Limitations (to address in Roadmap 2):**
- Still requires command-line usage
- No progress visualization
- No multi-project management UI
- Manual ComfyUI monitoring if needed

---

**Next:** [Roadmap 2: Web Interface](ROADMAP-2-WEB.md)
