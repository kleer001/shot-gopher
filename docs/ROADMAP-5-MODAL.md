# Roadmap 5: Modal Cloud Deployment

**Goal:** Deploy VFX Ingest Platform to Modal for serverless GPU processing

**Status:** Planning

**Dependencies:** Roadmap 1 (Docker concepts), Roadmap 3 (Web UI)

---

## Overview

This roadmap deploys the VFX pipeline to Modal's serverless GPU infrastructure. Modal uses a Python-first approach where infrastructure is defined in code, offering excellent developer experience and 1-second cold starts.

### Why Modal?

| Factor | Modal Advantage |
|--------|-----------------|
| **Free Tier** | **$30/month free credits** - best in class |
| **Cold Starts** | ~1 second (exceptional) |
| **Developer Experience** | Python-first, no Docker required |
| **Billing** | Per-second, scale to zero |
| **Custom Domains** | Native web endpoint support |
| **Secret Management** | Built-in secrets store |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                             │
│                   https://vfx--web.modal.run                     │
│                   (or custom: vfx.yourdomain.com)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MODAL CLOUD                              │
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │  Web App     │────────▶│  GPU Worker  │                      │
│  │  (FastAPI)   │  Queue  │  (A10G/A100) │                      │
│  │  Always-on   │         │  Scale 0→N   │                      │
│  └──────────────┘         └──────────────┘                      │
│         │                        │                               │
│         ▼                        ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                Modal Volume (Persistent)                     ││
│  │   /models (15GB)  │  /projects (variable)                   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Modal vs RunPod Comparison

| Feature | Modal | RunPod |
|---------|-------|--------|
| **Approach** | Python-native | Docker-native |
| **Free Tier** | $30/mo | $5-10 one-time |
| **Cold Start** | ~1s | ~30-60s |
| **Docker Support** | Yes (but not primary) | Primary |
| **Learning Curve** | Lower (Python devs) | Lower (Docker devs) |
| **Custom Domain** | Native | Via tunnel |
| **GPU Pricing** | A10G ~$1.10/hr | RTX 4090 ~$0.39/hr |

---

## Phase 5A: Modal Setup & Environment

**Goal:** Set up Modal account and understand the platform

### Tasks

#### Task 5A.1: Create Modal Account

1. Sign up at [modal.com](https://modal.com/)
2. Verify email
3. Install Modal CLI:

```bash
pip install modal
modal setup  # Opens browser for authentication
```

**Validation:**
```bash
modal token show
# Should display your token info
```

**Success Criteria:**
- [ ] Account created
- [ ] CLI authenticated
- [ ] $30 free credits visible

---

#### Task 5A.2: Understand Modal Concepts

Modal has unique concepts different from traditional cloud:

| Concept | Description |
|---------|-------------|
| **App** | Container for related functions and resources |
| **Function** | Serverless function that runs on Modal infrastructure |
| **Image** | Container image built from Python code |
| **Volume** | Persistent storage across function invocations |
| **Secret** | Secure credential storage |
| **Web Endpoint** | HTTP endpoint for functions |
| **Stub** | (Legacy) Now called App |

**Key Insight:** Modal builds containers from your Python code - no Dockerfile needed (though you can use one).

---

#### Task 5A.3: Create Modal Volume for Models

```python
# scripts/modal_setup.py
"""One-time setup for Modal volumes and secrets."""

import modal

app = modal.App("vfx-setup")

# Create persistent volume for models
models_volume = modal.Volume.from_name("vfx-models", create_if_missing=True)

# Create persistent volume for projects
projects_volume = modal.Volume.from_name("vfx-projects", create_if_missing=True)

@app.function(volumes={"/models": models_volume})
def check_models():
    """Check what models exist in the volume."""
    import os
    from pathlib import Path

    models_dir = Path("/models")
    if not models_dir.exists():
        return {"status": "empty", "models": []}

    models = list(models_dir.iterdir())
    return {
        "status": "populated" if models else "empty",
        "models": [m.name for m in models]
    }

if __name__ == "__main__":
    with app.run():
        result = check_models.remote()
        print(result)
```

```bash
# Run setup
modal run scripts/modal_setup.py
```

**Success Criteria:**
- [ ] Volumes created
- [ ] Can list volume contents

---

#### Task 5A.4: Upload Models to Modal Volume

```python
# scripts/modal_upload_models.py
"""Upload models to Modal volume."""

import modal
from pathlib import Path

app = modal.App("vfx-model-upload")
models_volume = modal.Volume.from_name("vfx-models")

# Image with huggingface_hub for downloading
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "gdown", "requests")
)

@app.function(
    image=download_image,
    volumes={"/models": models_volume},
    timeout=3600,  # 1 hour for large downloads
)
def download_models():
    """Download all required models to the volume."""
    from huggingface_hub import snapshot_download
    from pathlib import Path
    import subprocess

    models_dir = Path("/models")
    models_dir.mkdir(exist_ok=True)

    results = {}

    # SAM3
    sam3_dir = models_dir / "sam3"
    if not sam3_dir.exists():
        print("Downloading SAM3...")
        snapshot_download("1038lab/sam3", local_dir=str(sam3_dir))
        results["sam3"] = "downloaded"
    else:
        results["sam3"] = "exists"

    # Video Depth Anything
    depth_dir = models_dir / "videodepthanything"
    if not depth_dir.exists():
        print("Downloading Video Depth Anything...")
        snapshot_download(
            "depth-anything/Video-Depth-Anything-Small",
            local_dir=str(depth_dir)
        )
        results["videodepthanything"] = "downloaded"
    else:
        results["videodepthanything"] = "exists"

    # Commit changes to volume
    models_volume.commit()

    return results


@app.local_entrypoint()
def main():
    result = download_models.remote()
    print(f"Model download results: {result}")
```

```bash
# Run model download (runs on Modal's infrastructure)
modal run scripts/modal_upload_models.py
```

**Alternative: Upload local models**
```python
@app.function(volumes={"/models": models_volume})
def upload_local_models():
    """Upload models from local machine."""
    import shutil
    from pathlib import Path

    # This runs locally, uploads to volume
    local_models = Path.home() / ".vfx_pipeline" / "models"

    for model_dir in local_models.iterdir():
        if model_dir.is_dir():
            dest = Path("/models") / model_dir.name
            if not dest.exists():
                shutil.copytree(model_dir, dest)
                print(f"Uploaded: {model_dir.name}")

    models_volume.commit()
```

**Success Criteria:**
- [ ] Models downloaded/uploaded to volume
- [ ] Volume persists across function calls
- [ ] ~15-20GB of models stored

---

### Phase 5A Exit Criteria

- [ ] Modal account active with credits
- [ ] CLI configured and authenticated
- [ ] Volumes created for models and projects
- [ ] Models uploaded to volume

---

## Phase 5B: Modal Image Definition

**Goal:** Define the Modal container image with all dependencies

### Tasks

#### Task 5B.1: Create Base Image Definition

Modal builds images from Python code - no Dockerfile needed.

**File:** `modal_app/image.py`

```python
"""Modal image definition for VFX Ingest Pipeline."""

import modal

def create_vfx_image() -> modal.Image:
    """
    Create the Modal image with all VFX pipeline dependencies.

    This replaces the Dockerfile with Python code.
    """
    image = (
        # Start with CUDA base
        modal.Image.from_registry(
            "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
            add_python="3.10"
        )
        # System dependencies
        .apt_install(
            "git",
            "ffmpeg",
            "colmap",
            "wget",
            "curl",
            "libgl1-mesa-glx",
            "libglib2.0-0",
        )
        # Python dependencies
        .pip_install(
            "torch",
            "torchvision",
            "torchaudio",
            index_url="https://download.pytorch.org/whl/cu121"
        )
        .pip_install(
            "numpy",
            "scipy",
            "trimesh",
            "pillow",
            "fastapi",
            "uvicorn",
            "websockets",
            "jinja2",
            "httpx",
            "aiofiles",
        )
        # Clone ComfyUI and custom nodes
        .run_commands(
            "mkdir -p /app/.vfx_pipeline",
            "git clone https://github.com/comfyanonymous/ComfyUI.git /app/.vfx_pipeline/ComfyUI",
        )
        .run_commands(
            "cd /app/.vfx_pipeline/ComfyUI/custom_nodes && "
            "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && "
            "git clone https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git && "
            "git clone https://github.com/PozzettiAndrea/ComfyUI-SAM3.git && "
            "git clone https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git && "
            "git clone https://github.com/FuouM/ComfyUI-MatAnyone.git"
        )
        # Install custom node requirements
        .run_commands(
            "cd /app/.vfx_pipeline/ComfyUI/custom_nodes && "
            "for dir in */; do "
            "  if [ -f \"$dir/requirements.txt\" ]; then "
            "    pip install -r \"$dir/requirements.txt\"; "
            "  fi; "
            "done"
        )
        # Copy local pipeline code
        .copy_local_dir("scripts", "/app/scripts")
        .copy_local_dir("workflow_templates", "/app/workflow_templates")
        .copy_local_dir("web", "/app/web")
        .copy_local_file("start_web.py", "/app/start_web.py")
        # Set environment
        .env({
            "CONTAINER": "true",
            "MODAL": "true",
            "VFX_INSTALL_DIR": "/app/.vfx_pipeline",
            "VFX_MODELS_DIR": "/models",
            "VFX_PROJECTS_DIR": "/projects",
            "PYTHONPATH": "/app/scripts:/app",
        })
    )

    return image


# Pre-build the image (cached across deployments)
vfx_image = create_vfx_image()
```

**Success Criteria:**
- [ ] Image definition compiles
- [ ] All dependencies specified
- [ ] Local code mounted correctly

---

#### Task 5B.2: Test Image Build

```python
# modal_app/test_image.py
"""Test that the image builds and works correctly."""

import modal
from image import vfx_image

app = modal.App("vfx-image-test")

@app.function(image=vfx_image, gpu="any")
def test_dependencies():
    """Test that all dependencies are available."""
    import subprocess
    results = {}

    # Test PyTorch + CUDA
    import torch
    results["pytorch"] = torch.__version__
    results["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)

    # Test FFmpeg
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    results["ffmpeg"] = "ok" if result.returncode == 0 else "failed"

    # Test COLMAP
    result = subprocess.run(["colmap", "--help"], capture_output=True, text=True)
    results["colmap"] = "ok" if result.returncode == 0 else "failed"

    # Test ComfyUI import
    import sys
    sys.path.insert(0, "/app/.vfx_pipeline/ComfyUI")
    try:
        import folder_paths
        results["comfyui"] = "ok"
    except ImportError as e:
        results["comfyui"] = f"failed: {e}"

    # Test pipeline scripts
    sys.path.insert(0, "/app/scripts")
    try:
        import env_config
        results["env_config"] = "ok"
    except ImportError as e:
        results["env_config"] = f"failed: {e}"

    return results

@app.local_entrypoint()
def main():
    result = test_dependencies.remote()
    print("Dependency test results:")
    for key, value in result.items():
        status = "✓" if value in ["ok", True] or "ok" in str(value) else "✗"
        print(f"  {status} {key}: {value}")
```

```bash
modal run modal_app/test_image.py
```

**Expected Output:**
```
Dependency test results:
  ✓ pytorch: 2.1.0
  ✓ cuda_available: True
  ✓ gpu_name: NVIDIA A10G
  ✓ ffmpeg: ok
  ✓ colmap: ok
  ✓ comfyui: ok
  ✓ env_config: ok
```

**Success Criteria:**
- [ ] Image builds successfully
- [ ] PyTorch detects GPU
- [ ] All system tools available
- [ ] Pipeline code accessible

---

### Phase 5B Exit Criteria

- [ ] Image definition complete
- [ ] Image builds without errors
- [ ] All dependencies verified
- [ ] GPU access confirmed

---

## Phase 5C: GPU Processing Functions

**Goal:** Create Modal functions for each pipeline stage

### Tasks

#### Task 5C.1: Create Pipeline App Structure

**File:** `modal_app/app.py`

```python
"""Main Modal application for VFX Ingest Pipeline."""

import modal
from pathlib import Path

from image import vfx_image

# Create the Modal app
app = modal.App("vfx-ingest")

# Persistent volumes
models_volume = modal.Volume.from_name("vfx-models")
projects_volume = modal.Volume.from_name("vfx-projects")

# Common function configuration
gpu_config = dict(
    image=vfx_image,
    volumes={
        "/models": models_volume,
        "/projects": projects_volume,
    },
    timeout=3600,  # 1 hour max
)
```

---

#### Task 5C.2: Create Ingest Stage Function

```python
# modal_app/stages/ingest.py
"""Frame extraction stage."""

import modal
from ..app import app, gpu_config, projects_volume

@app.function(**gpu_config, gpu=None)  # No GPU needed for ingest
def run_ingest(video_bytes: bytes, project_name: str) -> dict:
    """
    Extract frames from video.

    Args:
        video_bytes: Raw video file bytes
        project_name: Name for the project

    Returns:
        dict with frame count and paths
    """
    from pathlib import Path
    import subprocess
    import tempfile

    project_dir = Path(f"/projects/{project_name}")
    frames_dir = project_dir / "source" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Write video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    # Extract frames with FFmpeg
    output_pattern = str(frames_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", "fps=24",
        "-q:v", "2",
        output_pattern
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {"status": "failed", "error": result.stderr}

    # Count extracted frames
    frames = list(frames_dir.glob("*.png"))

    # Commit volume changes
    projects_volume.commit()

    return {
        "status": "completed",
        "project_name": project_name,
        "frame_count": len(frames),
        "frames_dir": str(frames_dir),
    }
```

---

#### Task 5C.3: Create Depth Estimation Function

```python
# modal_app/stages/depth.py
"""Depth estimation stage using Video Depth Anything."""

import modal
from ..app import app, gpu_config, models_volume, projects_volume

@app.function(
    **gpu_config,
    gpu="A10G",  # 24GB VRAM, good balance of cost/performance
    memory=32768,  # 32GB RAM
)
def run_depth(project_name: str) -> dict:
    """
    Generate depth maps for all frames.

    Args:
        project_name: Project to process

    Returns:
        dict with depth map count and paths
    """
    from pathlib import Path
    import subprocess
    import sys

    project_dir = Path(f"/projects/{project_name}")
    frames_dir = project_dir / "source" / "frames"
    depth_dir = project_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Verify frames exist
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return {"status": "failed", "error": "No frames found"}

    # Link models to ComfyUI expected location
    comfyui_models = Path("/app/.vfx_pipeline/ComfyUI/models")
    comfyui_models.mkdir(parents=True, exist_ok=True)

    depth_model_src = Path("/models/videodepthanything")
    depth_model_dst = comfyui_models / "depth_anything"
    if depth_model_src.exists() and not depth_model_dst.exists():
        depth_model_dst.symlink_to(depth_model_src)

    # Run depth estimation via pipeline script
    sys.path.insert(0, "/app/scripts")
    from run_depth import process_frames

    result = process_frames(
        input_dir=str(frames_dir),
        output_dir=str(depth_dir),
        model_path=str(depth_model_src),
    )

    # Commit volume changes
    projects_volume.commit()

    depth_maps = list(depth_dir.glob("*.png"))

    return {
        "status": "completed",
        "project_name": project_name,
        "depth_count": len(depth_maps),
        "depth_dir": str(depth_dir),
    }
```

---

#### Task 5C.4: Create Segmentation Function

```python
# modal_app/stages/roto.py
"""Segmentation/rotoscoping stage using SAM3."""

import modal
from ..app import app, gpu_config, models_volume, projects_volume

@app.function(
    **gpu_config,
    gpu="A10G",
    memory=32768,
)
def run_roto(project_name: str, prompts: list[str] = None) -> dict:
    """
    Generate segmentation masks.

    Args:
        project_name: Project to process
        prompts: Optional text prompts for segmentation targets

    Returns:
        dict with mask count and paths
    """
    from pathlib import Path
    import sys

    project_dir = Path(f"/projects/{project_name}")
    frames_dir = project_dir / "source" / "frames"
    roto_dir = project_dir / "roto"
    roto_dir.mkdir(parents=True, exist_ok=True)

    # Default prompts if none provided
    if prompts is None:
        prompts = ["person", "human"]

    # Link SAM3 model
    sam3_src = Path("/models/sam3")
    comfyui_models = Path("/app/.vfx_pipeline/ComfyUI/models/sam3")
    if sam3_src.exists() and not comfyui_models.exists():
        comfyui_models.parent.mkdir(parents=True, exist_ok=True)
        comfyui_models.symlink_to(sam3_src)

    # Run segmentation
    sys.path.insert(0, "/app/scripts")
    from run_roto import process_segmentation

    result = process_segmentation(
        input_dir=str(frames_dir),
        output_dir=str(roto_dir),
        prompts=prompts,
    )

    projects_volume.commit()

    masks = list(roto_dir.glob("*.png"))

    return {
        "status": "completed",
        "project_name": project_name,
        "mask_count": len(masks),
        "roto_dir": str(roto_dir),
    }
```

---

#### Task 5C.5: Create COLMAP Function

```python
# modal_app/stages/colmap.py
"""Camera tracking stage using COLMAP."""

import modal
from ..app import app, gpu_config, projects_volume

@app.function(
    **gpu_config,
    gpu="A10G",
    memory=65536,  # 64GB RAM - COLMAP can be memory hungry
    timeout=7200,  # 2 hours - COLMAP can take a while
)
def run_colmap(project_name: str, quality: str = "medium") -> dict:
    """
    Run COLMAP camera tracking.

    Args:
        project_name: Project to process
        quality: 'low', 'medium', or 'high'

    Returns:
        dict with camera data
    """
    from pathlib import Path
    import subprocess
    import json

    project_dir = Path(f"/projects/{project_name}")
    frames_dir = project_dir / "source" / "frames"
    colmap_dir = project_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Quality presets
    quality_presets = {
        "low": {"SiftExtraction.max_num_features": 2048},
        "medium": {"SiftExtraction.max_num_features": 8192},
        "high": {"SiftExtraction.max_num_features": 16384},
    }

    # Run COLMAP pipeline
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    # Feature extraction
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
    ]
    subprocess.run(cmd, check=True)

    # Feature matching
    cmd = [
        "colmap", "sequential_matcher",
        "--database_path", str(database_path),
    ]
    subprocess.run(cmd, check=True)

    # Sparse reconstruction
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ]
    subprocess.run(cmd, check=True)

    # Export camera data to JSON
    camera_data = export_camera_json(sparse_dir / "0", colmap_dir / "camera_data.json")

    projects_volume.commit()

    return {
        "status": "completed",
        "project_name": project_name,
        "camera_data_path": str(colmap_dir / "camera_data.json"),
        "num_cameras": len(camera_data.get("frames", {})),
    }


def export_camera_json(sparse_path: Path, output_path: Path) -> dict:
    """Export COLMAP sparse reconstruction to JSON."""
    # Implementation details...
    pass
```

---

#### Task 5C.6: Create Pipeline Orchestrator

```python
# modal_app/pipeline.py
"""Pipeline orchestration - runs stages in sequence."""

import modal
from .app import app
from .stages.ingest import run_ingest
from .stages.depth import run_depth
from .stages.roto import run_roto
from .stages.colmap import run_colmap

@app.function()
def run_pipeline(
    video_bytes: bytes,
    project_name: str,
    stages: list[str] = None,
) -> dict:
    """
    Run the complete VFX pipeline.

    Args:
        video_bytes: Raw video file
        project_name: Project name
        stages: List of stages to run (default: all)

    Returns:
        dict with results from each stage
    """
    if stages is None:
        stages = ["ingest", "depth", "roto", "colmap"]

    results = {"project_name": project_name}

    # Stage execution map
    stage_functions = {
        "ingest": lambda: run_ingest.remote(video_bytes, project_name),
        "depth": lambda: run_depth.remote(project_name),
        "roto": lambda: run_roto.remote(project_name),
        "colmap": lambda: run_colmap.remote(project_name),
    }

    # Execute stages in order
    for stage in stages:
        if stage not in stage_functions:
            results[stage] = {"status": "skipped", "error": f"Unknown stage: {stage}"}
            continue

        print(f"Running stage: {stage}")
        try:
            result = stage_functions[stage]()
            results[stage] = result
            if result.get("status") != "completed":
                print(f"Stage {stage} failed, stopping pipeline")
                break
        except Exception as e:
            results[stage] = {"status": "failed", "error": str(e)}
            break

    return results


@app.local_entrypoint()
def main(video_path: str, project_name: str, stages: str = "ingest,depth"):
    """CLI entrypoint for testing."""
    from pathlib import Path

    video_bytes = Path(video_path).read_bytes()
    stage_list = stages.split(",")

    result = run_pipeline.remote(video_bytes, project_name, stage_list)
    print(f"Pipeline result: {result}")
```

```bash
# Test pipeline
modal run modal_app/pipeline.py --video-path test.mp4 --project-name TestProject --stages ingest,depth
```

**Success Criteria:**
- [ ] All stage functions defined
- [ ] Pipeline orchestrator works
- [ ] Stages can run independently
- [ ] Volume persistence verified

---

### Phase 5C Exit Criteria

- [ ] All pipeline stages implemented
- [ ] GPU allocation correct per stage
- [ ] Volume mounts working
- [ ] Pipeline orchestrator functional
- [ ] End-to-end test passes

---

## Phase 5D: Web UI Deployment

**Goal:** Deploy FastAPI web UI as a Modal web endpoint

### Tasks

#### Task 5D.1: Create Web App Function

```python
# modal_app/web.py
"""FastAPI Web UI deployment."""

import modal
from .app import app, vfx_image, models_volume, projects_volume

@app.function(
    image=vfx_image,
    volumes={
        "/models": models_volume,
        "/projects": projects_volume,
    },
    allow_concurrent_inputs=100,
    container_idle_timeout=300,  # Keep warm for 5 minutes
)
@modal.asgi_app()
def web_app():
    """
    Serve the FastAPI web application.

    This creates a persistent web endpoint at:
    https://vfx-ingest--web-app.modal.run
    """
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/scripts")

    # Import the FastAPI app
    from web.server import create_app

    # Create app with Modal-specific config
    fastapi_app = create_app(
        models_dir="/models",
        projects_dir="/projects",
        modal_mode=True,
    )

    return fastapi_app
```

---

#### Task 5D.2: Modify FastAPI App for Modal

**File:** `web/server.py` (modifications)

```python
"""FastAPI server with Modal support."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

def create_app(
    models_dir: str = None,
    projects_dir: str = None,
    modal_mode: bool = False,
) -> FastAPI:
    """Create FastAPI application."""

    app = FastAPI(
        title="VFX Ingest Platform",
        description="Automated VFX pipeline",
        version="1.0.0",
    )

    # Configure paths
    if modal_mode:
        app.state.models_dir = Path(models_dir or "/models")
        app.state.projects_dir = Path(projects_dir or "/projects")
        app.state.use_modal_functions = True
    else:
        from scripts.env_config import INSTALL_DIR, DEFAULT_PROJECTS_DIR
        app.state.models_dir = INSTALL_DIR / "models"
        app.state.projects_dir = DEFAULT_PROJECTS_DIR
        app.state.use_modal_functions = False

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Include routers
    from .api import router as api_router
    app.include_router(api_router, prefix="/api")

    # Include web routes
    from .routes import router as web_router
    app.include_router(web_router)

    return app


# For local development
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

---

#### Task 5D.3: Add Modal Job Submission to API

**File:** `web/api.py` (additions)

```python
"""API routes with Modal integration."""

from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import modal

router = APIRouter()

class ProcessRequest(BaseModel):
    project_name: str
    stages: list[str] = ["ingest", "depth"]

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: dict = None

# Track running jobs
jobs = {}

@router.post("/process")
async def process_video(
    file: UploadFile = File(...),
    project_name: str = "NewProject",
    stages: str = "ingest,depth",
):
    """Submit a video for processing."""
    from fastapi import Request

    video_bytes = await file.read()
    stage_list = stages.split(",")

    # Check if running in Modal
    if hasattr(router, "app") and getattr(router.app.state, "use_modal_functions", False):
        # Submit to Modal
        from modal_app.pipeline import run_pipeline

        # Spawn async job
        call = run_pipeline.spawn(video_bytes, project_name, stage_list)
        job_id = call.object_id

        jobs[job_id] = {"status": "running", "call": call}

        return {"job_id": job_id, "status": "submitted"}
    else:
        # Run locally
        # ... local processing logic
        pass


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    if job_id not in jobs:
        return {"error": "Job not found"}

    job = jobs[job_id]
    call = job.get("call")

    if call:
        try:
            result = call.get(timeout=0)  # Non-blocking check
            jobs[job_id] = {"status": "completed", "result": result}
        except modal.exception.OutputExpiredError:
            pass  # Still running
        except Exception as e:
            jobs[job_id] = {"status": "failed", "error": str(e)}

    return jobs[job_id]
```

---

#### Task 5D.4: Deploy Web App

```bash
# Deploy the web app
modal deploy modal_app/web.py

# Output:
# ✓ Created web endpoint: https://vfx-ingest--web-app.modal.run
```

**Validation:**
```bash
# Test the endpoint
curl https://vfx-ingest--web-app.modal.run/health
# {"status": "ok"}

# Open in browser
open https://vfx-ingest--web-app.modal.run
```

**Success Criteria:**
- [ ] Web app deployed
- [ ] Endpoint accessible
- [ ] Static files served
- [ ] Job submission works

---

#### Task 5D.5: Configure Custom Domain

Modal supports custom domains natively.

```python
# modal_app/web.py (update)

@app.function(
    # ... existing config ...
    custom_domains=["vfx.yourdomain.com"],  # Add custom domain
)
@modal.asgi_app()
def web_app():
    # ... existing code ...
```

**DNS Configuration:**
1. Go to Modal dashboard > Deployments > web_app
2. Click "Custom Domain"
3. Add your domain: `vfx.yourdomain.com`
4. Modal provides a CNAME record to add to your DNS
5. Add CNAME: `vfx.yourdomain.com` -> `modal.run` (or provided value)

**Validation:**
```bash
# After DNS propagation (5-30 minutes)
curl https://vfx.yourdomain.com/health
```

**Success Criteria:**
- [ ] Custom domain configured
- [ ] DNS records added
- [ ] HTTPS working automatically

---

### Phase 5D Exit Criteria

- [ ] Web app deployed to Modal
- [ ] Public URL accessible
- [ ] Custom domain configured (optional)
- [ ] Job submission working
- [ ] Static assets served correctly

---

## Phase 5E: Production Optimization

**Goal:** Optimize for production use

### Tasks

#### Task 5E.1: Implement Model Preloading

Reduce cold starts by keeping models in memory.

```python
# modal_app/app.py (update)

# Use cls() pattern for persistent state
@app.cls(
    image=vfx_image,
    gpu="A10G",
    volumes={"/models": models_volume, "/projects": projects_volume},
    container_idle_timeout=600,  # Keep warm for 10 minutes
)
class DepthProcessor:
    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import torch
        from pathlib import Path

        self.device = torch.device("cuda")
        self.model = load_depth_model(Path("/models/videodepthanything"))
        self.model.to(self.device)
        self.model.eval()
        print("Depth model loaded and ready")

    @modal.method()
    def process(self, project_name: str) -> dict:
        """Process with pre-loaded model."""
        # Use self.model directly - already loaded
        pass

    @modal.exit()
    def cleanup(self):
        """Cleanup when container shuts down."""
        del self.model
        torch.cuda.empty_cache()
```

---

#### Task 5E.2: Add Secrets Management

Store API keys and credentials securely.

```bash
# Add secrets via CLI
modal secret create vfx-secrets \
    COMFYUI_API_KEY=xxx \
    S3_ACCESS_KEY=xxx \
    S3_SECRET_KEY=xxx
```

```python
# Use secrets in functions
@app.function(
    secrets=[modal.Secret.from_name("vfx-secrets")],
)
def function_with_secrets():
    import os
    api_key = os.environ["COMFYUI_API_KEY"]
```

---

#### Task 5E.3: Add Monitoring and Logging

```python
# modal_app/monitoring.py
"""Monitoring and logging utilities."""

import modal
from datetime import datetime

@app.function()
def log_job_metrics(
    job_id: str,
    stage: str,
    duration_seconds: float,
    gpu_memory_mb: float,
):
    """Log job metrics for monitoring."""
    print(f"[METRIC] job={job_id} stage={stage} duration={duration_seconds:.2f}s gpu_mem={gpu_memory_mb:.0f}MB")

    # Optionally send to external monitoring
    # send_to_datadog(...)
    # send_to_prometheus(...)
```

---

#### Task 5E.4: Cost Tracking

```python
# modal_app/costs.py
"""Cost estimation utilities."""

# Modal pricing (approximate, check modal.com/pricing for current)
MODAL_PRICING = {
    "A10G": 1.10,      # $/hour
    "A100-40GB": 3.40,  # $/hour
    "A100-80GB": 4.50,  # $/hour
    "T4": 0.59,         # $/hour
}

def estimate_cost(gpu_type: str, duration_seconds: float) -> float:
    """Estimate cost for a job."""
    hourly_rate = MODAL_PRICING.get(gpu_type, 1.10)
    return (duration_seconds / 3600) * hourly_rate


@app.function()
def get_monthly_usage():
    """Get usage stats (requires Modal API)."""
    # Modal provides usage stats in dashboard
    # Could also implement via their API
    pass
```

---

### Phase 5E Exit Criteria

- [ ] Model preloading implemented
- [ ] Cold starts minimized
- [ ] Secrets properly managed
- [ ] Monitoring in place
- [ ] Cost tracking understood

---

## Phase 5F: Testing & Documentation

**Goal:** Comprehensive testing and documentation

### Tasks

#### Task 5F.1: Integration Tests

```python
# tests/test_modal_integration.py
"""Integration tests for Modal deployment."""

import pytest
import modal

@pytest.fixture
def modal_app():
    from modal_app.app import app
    return app

def test_ingest_stage(modal_app):
    """Test frame extraction on Modal."""
    from modal_app.stages.ingest import run_ingest

    # Small test video (base64 or bytes)
    test_video = create_test_video(frames=10)

    with modal_app.run():
        result = run_ingest.remote(test_video, "TestProject")

    assert result["status"] == "completed"
    assert result["frame_count"] == 10

def test_full_pipeline(modal_app):
    """Test complete pipeline."""
    from modal_app.pipeline import run_pipeline

    test_video = create_test_video(frames=30)

    with modal_app.run():
        result = run_pipeline.remote(
            test_video,
            "FullTest",
            ["ingest", "depth"]
        )

    assert result["ingest"]["status"] == "completed"
    assert result["depth"]["status"] == "completed"
```

---

#### Task 5F.2: Create Deployment Documentation

**File:** `docs/MODAL-DEPLOYMENT.md`

Document:
- Account setup
- Local development workflow
- Deployment commands
- Custom domain setup
- Cost management
- Troubleshooting

---

### Phase 5F Exit Criteria

- [ ] Integration tests passing
- [ ] Documentation complete
- [ ] Deployment process documented

---

## Cost Summary

### Modal Pricing (as of 2025)

| Resource | Price |
|----------|-------|
| **A10G GPU** | ~$1.10/hour |
| **A100 40GB** | ~$3.40/hour |
| **CPU** | ~$0.03/hour |
| **Memory** | ~$0.01/GB/hour |
| **Storage** | ~$0.50/GB/month |

### Estimated Monthly Costs

| Usage | Pattern | Estimated Cost |
|-------|---------|----------------|
| **Light** | 10 jobs/month, 5 min each | $5-10 |
| **Medium** | 50 jobs/month, 10 min each | $20-50 |
| **Heavy** | 200 jobs/month, 15 min each | $80-150 |

### Free Tier

- **$30/month in credits** - covers significant testing/light use
- No credit card required initially
- Credits refresh monthly

---

## Roadmap 5 Success Criteria

**Ready for production when:**

- [ ] All pipeline stages working on Modal
- [ ] Web UI deployed with custom domain
- [ ] Cold starts under 5 seconds
- [ ] Cost tracking in place
- [ ] Documentation complete
- [ ] At least 5 successful end-to-end tests
- [ ] Monitoring and logging functional

**Comparison with RunPod:**

| Factor | Modal | RunPod |
|--------|-------|--------|
| Setup complexity | Lower | Medium |
| Free tier | Better ($30/mo) | $5-10 one-time |
| Cold starts | Faster (~1s) | Slower (~30s) |
| GPU pricing | Higher (~$1.10/hr) | Lower (~$0.39/hr) |
| Docker reuse | Requires adaptation | Works as-is |

**Recommendation:**
- Choose **Modal** if: Python-first, need fast cold starts, prefer DX
- Choose **RunPod** if: Already have Docker setup, need cheapest GPU, batch processing

---

**Previous:** [Roadmap 4: RunPod Deployment](ROADMAP-4-RUNPOD.md)
