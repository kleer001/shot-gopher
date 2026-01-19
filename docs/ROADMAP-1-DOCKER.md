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

### SOLID/DRY Application in Infrastructure

While SOLID principles primarily apply to object-oriented code, we adapt them for Docker/infrastructure:

**Single Responsibility Principle (SRP):**
- Each Dockerfile stage has one purpose (base deps → Python deps → ComfyUI → pipeline)
- Entrypoint script handles validation, startup, and execution as separate functions
- Separate scripts for different concerns: `download_models.sh`, `verify_models.py`, `run_docker.sh`

**Open/Closed Principle (OCP):**
- Environment variables allow extension without modifying code
- Scripts work in both local and container modes through environment detection
- Volume mounts enable model/project customization without image rebuilds

**DRY (Don't Repeat Yourself):**
- Path configuration centralized in `env_config.py` with environment variable overrides
- Model validation logic in single `verify_models.py` (used by both entrypoint and users)
- Reuse existing Python scripts instead of duplicating logic in shell scripts
- Docker multi-stage build reuses layers efficiently

**Interface Segregation:**
- CLI interface unchanged (users don't need to learn Docker internals)
- Clear volume mount contracts: `/models` (read-only), `/workspace` (read-write)
- Environment variables as documented interfaces: `VFX_MODELS_DIR`, `VFX_PROJECTS_DIR`

**Dependency Inversion:**
- Scripts depend on abstractions (environment variables) not concrete paths
- `is_in_container()` abstracts detection logic from usage
- ComfyUI manager uses configurable paths, not hardcoded locations

**Separation of Concerns:**
- **Infrastructure Layer** (Docker): Dependencies, networking, volumes
- **Configuration Layer** (Environment): Paths, settings, feature flags
- **Logic Layer** (Python scripts): Unchanged business logic from local installation

**Code Review Checklist:**
- [ ] No hardcoded paths in Python scripts (use environment variables)
- [ ] No duplicated validation logic across scripts
- [ ] Each Docker stage builds on previous (no redundant installs)
- [ ] Entrypoint functions are focused and testable
- [ ] Local and container modes share maximum code

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

## Phase 1D: Comprehensive Testing ⚪

**Goal:** End-to-end validation of containerized pipeline with test fixtures

### Overview

Comprehensive testing strategy with three test tiers:
1. **Unit Tests** - Synthetic fixtures (fast, no GPU)
2. **Integration Tests** - Standard test video (full pipeline validation)
3. **Performance Tests** - Benchmark against local installation

---

### Test Data Preparation

#### Task 1D.0: Setup Test Fixtures

**Standard Test Video:**
- **Source:** Football CIF sequence (academic standard)
- **URL:** https://media.xiph.org/video/derf/
- **Format:** YUV 4:2:0, 352×288, ~260 frames
- **Size:** ~1-2MB
- **Content:** Two football players (good for mocap/segmentation)
- **Why:** Industry standard, reproducible, includes people

**Download and prepare:**
```bash
# Create test fixtures directory
mkdir -p tests/fixtures

# Download Football CIF (YUV format)
wget https://media.xiph.org/video/derf/y4m/football_cif.y4m -O tests/fixtures/football_cif.y4m

# Convert to MP4 for pipeline testing
ffmpeg -i tests/fixtures/football_cif.y4m \
  -c:v libx264 -preset slow -crf 18 \
  tests/fixtures/football_test.mp4

# Extract first 30 frames for quick tests
ffmpeg -i tests/fixtures/football_test.mp4 -vframes 30 \
  tests/fixtures/football_short.mp4
```

**Synthetic Test Fixtures (for unit tests):**
```python
# tests/fixtures/generate_synthetic.py
"""Generate synthetic test data for unit tests."""

import numpy as np
from PIL import Image
from pathlib import Path

def generate_checkerboard(width=640, height=480, square_size=40):
    """Generate checkerboard pattern (good for COLMAP feature detection)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = 255
    return img

def generate_gradient(width=640, height=480):
    """Generate gradient image (for depth map testing)."""
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    img = np.tile(gradient, (height, 1))
    return np.stack([img, img, img], axis=-1)

def generate_test_masks():
    """Generate test segmentation masks."""
    # Simple shapes for roto validation
    pass

if __name__ == "__main__":
    fixtures_dir = Path("tests/fixtures/synthetic")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Generate checkerboard frames (simulates static camera)
    for i in range(10):
        img = generate_checkerboard()
        Image.fromarray(img).save(fixtures_dir / f"frame_{i:04d}.png")

    # Generate test depth map
    depth = generate_gradient()
    Image.fromarray(depth).save(fixtures_dir / "test_depth.png")

    print("✓ Synthetic fixtures generated")
```

**Success Criteria:**
- [ ] Football CIF video downloaded and converted
- [ ] Short test video (30 frames) created
- [ ] Synthetic fixtures generated
- [ ] Test fixtures documented in `tests/README.md`

---

### Unit Tests (Synthetic Data)

#### Test 1D.1: Container Environment Detection
**File:** `tests/test_env_detection.py`

```python
import os
import pytest
from scripts import env_config

def test_container_detection_dockerenv():
    """Test /.dockerenv file detection."""
    # Mock /.dockerenv existence
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        assert env_config.is_in_container() == True

def test_container_detection_env_var():
    """Test CONTAINER env var detection."""
    with patch.dict(os.environ, {'CONTAINER': 'true'}):
        assert env_config.is_in_container() == True

def test_conda_check_skipped_in_container():
    """Test conda checks are skipped in containers."""
    with patch.dict(os.environ, {'CONTAINER': 'true'}):
        result = env_config.check_conda_env_or_warn()
        assert result == True  # Should skip and return True
```

**Success Criteria:**
- [ ] Container detection works with multiple methods
- [ ] Conda checks properly skipped in containers
- [ ] Environment variable overrides respected

---

#### Test 1D.2: Path Resolution
**File:** `tests/test_path_handling.py`

```python
def test_container_path_validation():
    """Test paths are validated in container mode."""
    with patch.dict(os.environ, {'CONTAINER': 'true'}):
        # Should reject paths outside /workspace
        with pytest.raises(SystemExit):
            validate_project_path('/home/user/project')

        # Should accept /workspace paths
        assert validate_project_path('/workspace/project') is not None

def test_model_dir_override():
    """Test VFX_MODELS_DIR override."""
    with patch.dict(os.environ, {'VFX_MODELS_DIR': '/custom/models'}):
        assert get_models_dir() == Path('/custom/models')
```

**Success Criteria:**
- [ ] Path validation works correctly
- [ ] Environment overrides respected
- [ ] Error messages helpful

---

#### Test 1D.3: Frame Extraction (Synthetic)
**File:** `tests/test_frame_extraction.py`

```python
def test_frame_extraction_synthetic(tmp_path):
    """Test frame extraction with synthetic checkerboard video."""
    # Use synthetic test video (10 frames)
    input_video = Path("tests/fixtures/synthetic/checkerboard.mp4")
    output_dir = tmp_path / "frames"

    extract_frames(input_video, output_dir)

    frames = list(output_dir.glob("*.png"))
    assert len(frames) == 10
    assert all(f.stat().st_size > 0 for f in frames)
```

**Success Criteria:**
- [ ] Frame extraction works with synthetic data
- [ ] Frame count accurate
- [ ] No GPU required

---

#### Test 1D.4: COLMAP Input Validation (Synthetic)
**File:** `tests/test_colmap_validation.py`

```python
def test_colmap_checkerboard_features():
    """Test COLMAP can detect features in checkerboard."""
    image_dir = Path("tests/fixtures/synthetic")

    # Run feature extraction only (fast)
    result = extract_colmap_features(image_dir)

    assert result['num_images'] == 10
    assert result['num_features'] > 100  # Checkerboard has many corners
```

**Success Criteria:**
- [ ] COLMAP can process synthetic images
- [ ] Feature detection works on known patterns
- [ ] Fast execution (<10 seconds)

---

### Integration Tests (Real Video)

#### Test 1D.5: Full Ingest Stage
**File:** `tests/integration/test_ingest.py`

```bash
#!/bin/bash
# Run in container with Football test video

docker-compose run --rm vfx-ingest \
  /workspace/fixtures/football_short.mp4 \
  --name FootballTest \
  --stages ingest
```

**Expected Output:**
```
✓ Frames extracted: 30
✓ Output directory: /workspace/projects/FootballTest/source/frames/
✓ Frame dimensions: 352×288
✓ All frames valid PNG format
```

**Validation:**
```python
def test_ingest_stage():
    project_dir = Path("/workspace/projects/FootballTest")
    frames_dir = project_dir / "source/frames"

    frames = list(frames_dir.glob("*.png"))
    assert len(frames) == 30

    # Check first frame is readable
    img = Image.open(frames[0])
    assert img.size == (352, 288)
```

**Success Criteria:**
- [ ] All 30 frames extracted
- [ ] Correct frame dimensions
- [ ] FFmpeg runs without errors
- [ ] Execution time <30 seconds

---

#### Test 1D.6: Depth Estimation Stage
```bash
docker-compose run --rm vfx-ingest \
  --name FootballTest \
  --stages depth
```

**Expected:**
- ComfyUI starts successfully
- 30 depth maps generated
- GPU memory used efficiently
- Depth maps have correct dimensions

**Validation:**
```python
def test_depth_stage():
    depth_dir = Path("/workspace/projects/FootballTest/depth")
    depth_maps = list(depth_dir.glob("*.png"))

    assert len(depth_maps) == 30

    # Check depth map is grayscale and valid range
    depth_img = np.array(Image.open(depth_maps[0]))
    assert depth_img.ndim == 2 or depth_img.shape[2] == 1
    assert 0 <= depth_img.min() <= depth_img.max() <= 255
```

**Success Criteria:**
- [ ] All depth maps generated
- [ ] GPU utilized (check nvidia-smi)
- [ ] Execution time reasonable (~2-3 min for 30 frames)
- [ ] Memory usage acceptable

---

#### Test 1D.7: COLMAP Camera Tracking
```bash
docker-compose run --rm vfx-ingest \
  --name FootballTest \
  --stages colmap \
  --colmap-quality medium
```

**Expected:**
- COLMAP runs without installation errors
- Camera poses estimated for most frames
- Sparse point cloud generated
- Camera data exported (JSON, Alembic)

**Validation:**
```python
def test_colmap_stage():
    colmap_dir = Path("/workspace/projects/FootballTest/colmap")

    # Check outputs exist
    assert (colmap_dir / "sparse/0/cameras.bin").exists()
    assert (colmap_dir / "sparse/0/images.bin").exists()
    assert (colmap_dir / "camera_data.json").exists()
    assert (colmap_dir / "camera_track.abc").exists()

    # Validate JSON structure
    with open(colmap_dir / "camera_data.json") as f:
        data = json.load(f)
        assert len(data['frames']) > 20  # Should track most frames
        assert all('quat' in frame for frame in data['frames'].values())
```

**Success Criteria:**
- [ ] COLMAP runs without host installation
- [ ] Camera poses estimated for >70% of frames
- [ ] JSON export valid and complete
- [ ] Alembic file readable by Houdini/Maya
- [ ] Execution time <5 minutes

---

#### Test 1D.8: Segmentation Stage
```bash
docker-compose run --rm vfx-ingest \
  --name FootballTest \
  --stages roto
```

**Expected:**
- SAM3 detects people (2 football players)
- Masks generated for all frames
- Alpha mattes exported

**Validation:**
```python
def test_segmentation_stage():
    roto_dir = Path("/workspace/projects/FootballTest/roto")
    masks = list(roto_dir.glob("*.png"))

    assert len(masks) == 30

    # Check mask has alpha channel
    mask = np.array(Image.open(masks[0]))
    assert mask.ndim == 3 and mask.shape[2] == 4  # RGBA
```

**Success Criteria:**
- [ ] Masks generated for all frames
- [ ] People detected in frames
- [ ] Alpha channel present
- [ ] Execution time reasonable

---

#### Test 1D.9: Full Pipeline (All Stages)
```bash
docker-compose run --rm vfx-ingest \
  /workspace/fixtures/football_short.mp4 \
  --name FootballFull \
  --stages ingest,depth,roto,colmap,mocap
```

**Expected:**
- All stages complete successfully
- Complete project structure created
- All output files present

**Validation:**
```python
def test_full_pipeline():
    project = Path("/workspace/projects/FootballFull")

    # Check all output directories exist
    assert (project / "source/frames").exists()
    assert (project / "depth").exists()
    assert (project / "roto").exists()
    assert (project / "colmap").exists()
    assert (project / "mocap").exists()

    # Count outputs
    assert len(list((project / "source/frames").glob("*.png"))) == 30
    assert len(list((project / "depth").glob("*.png"))) == 30
    assert (project / "colmap/camera_data.json").exists()
```

**Success Criteria:**
- [ ] End-to-end pipeline completes
- [ ] All stages produce output
- [ ] No errors or warnings
- [ ] Total execution time <15 minutes
- [ ] Output files valid and complete

---

### Performance Tests

#### Test 1D.10: Performance Comparison
**Goal:** Validate container performance is within 10% of local installation

```python
# tests/performance/test_benchmarks.py

import time
from pathlib import Path

def benchmark_stage(stage_name, command, iterations=3):
    """Run a stage multiple times and measure performance."""
    times = []
    for i in range(iterations):
        start = time.time()
        run_command(command)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }

def test_performance_parity():
    """Compare container vs local performance."""

    # Benchmark in container
    container_times = benchmark_stage(
        "depth",
        "docker-compose run --rm vfx-ingest --name PerfTest --stages depth"
    )

    # Benchmark local (if available)
    # local_times = benchmark_stage("depth", "python scripts/run_pipeline.py ...")

    # Container should be within 10% of local
    # assert container_times['mean'] < local_times['mean'] * 1.10
```

**Metrics to Collect:**
- Execution time per stage
- GPU utilization (nvidia-smi)
- Memory usage (docker stats)
- Disk I/O
- CPU usage

**Success Criteria:**
- [ ] Container within 10% of local performance
- [ ] GPU fully utilized in compute stages
- [ ] No memory leaks over multiple runs
- [ ] Consistent performance across runs (low std dev)

---

### Volume Mount Tests

#### Test 1D.11: Volume Persistence
```bash
# Test 1: Write from container, read from host
docker-compose run --rm vfx-ingest \
  bash -c "echo 'test data' > /workspace/test_persistence.txt"

cat ~/VFX-Projects/test_persistence.txt
# Should output: test data

# Test 2: Write from host, read from container
echo "host data" > ~/VFX-Projects/host_test.txt

docker-compose run --rm vfx-ingest \
  cat /workspace/host_test.txt
# Should output: host data

# Test 3: Model volume is read-only
docker-compose run --rm vfx-ingest \
  bash -c "touch /models/should_fail.txt"
# Should fail with "read-only file system"
```

**Success Criteria:**
- [ ] Bidirectional file visibility
- [ ] Permissions preserved
- [ ] Model volume read-only (safety)
- [ ] Large files (>1GB) handled correctly

---

### Regression Tests

#### Test 1D.12: Backwards Compatibility
**Goal:** Ensure local installation still works

```bash
# Test local execution (without Docker)
python scripts/run_pipeline.py \
  tests/fixtures/football_short.mp4 \
  --name LocalTest \
  --stages ingest,depth

# Validate it still works
pytest tests/test_local_installation.py
```

**Success Criteria:**
- [ ] Local installation unchanged
- [ ] Can switch between local and Docker modes
- [ ] Scripts detect environment correctly

---

### Automated Test Suite

#### Task 1D.13: CI/CD Test Runner
**File:** `tests/run_docker_tests.sh`

```bash
#!/bin/bash
set -e

echo "=== VFX Ingest Platform - Docker Test Suite ==="

# 1. Setup test fixtures
echo "[1/5] Setting up test fixtures..."
python tests/fixtures/generate_synthetic.py
./tests/fixtures/download_football.sh

# 2. Build Docker image
echo "[2/5] Building Docker image..."
docker-compose build

# 3. Run unit tests (fast, no GPU)
echo "[3/5] Running unit tests..."
pytest tests/ -m "not integration" --verbose

# 4. Run integration tests (requires GPU)
echo "[4/5] Running integration tests..."
./tests/integration/test_all_stages.sh

# 5. Generate test report
echo "[5/5] Generating test report..."
pytest tests/ --html=tests/report.html --self-contained-html

echo "✓ All tests passed!"
```

**Success Criteria:**
- [ ] All tests automated
- [ ] Can run in CI/CD pipeline
- [ ] Test report generated
- [ ] <15 minute total execution

---

### Test Documentation

#### Task 1D.14: Testing Guide
**File:** `tests/README.md`

Document:
- How to run tests locally
- How to add new tests
- Test fixture management
- CI/CD integration
- Troubleshooting test failures

---

### Phase 1D Exit Criteria

- [ ] All unit tests pass (synthetic data)
- [ ] All integration tests pass (Football CIF video)
- [ ] Performance within 10% of local installation
- [ ] Volume mounts validated
- [ ] No file permission issues
- [ ] GPU utilization correct
- [ ] Error messages helpful
- [ ] Test fixtures documented
- [ ] Automated test suite functional
- [ ] Backwards compatibility verified

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
