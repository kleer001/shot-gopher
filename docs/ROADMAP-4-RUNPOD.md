# Roadmap 4: RunPod Cloud Deployment

**Goal:** Deploy VFX Ingest Platform to RunPod for cloud-based GPU processing

**Status:** Planning

**Dependencies:** Roadmap 1 (Docker), Roadmap 3 (Web UI)

---

## Prerequisites

### Option A: Use Existing Local Installation (Recommended)

If you've already run the install wizard locally, you have models downloaded to `~/.vfx_pipeline/models/`. This roadmap can upload those directly to RunPod instead of re-downloading.

```bash
# Check if models exist locally
ls -la ~/.vfx_pipeline/models/
# Should show: sam3/, videodepthanything/, wham/, matanyone/, smplx/ (optional)
```

### Option B: Fresh Cloud Installation

If you don't have local models, this roadmap will download them directly to RunPod's network volumes (slower, but works).

### Shared Configuration

Model metadata (URLs, checksums, sizes) is defined in `scripts/install_wizard/downloader.py:CheckpointDownloader.CHECKPOINTS`. Both local and cloud installations use the same source of truth:

| Model | Source | Size | Config Key |
|-------|--------|------|------------|
| SAM3 | HuggingFace `1038lab/sam3` | ~3.2GB | `sam3` |
| Video Depth Anything | HuggingFace `depth-anything/Video-Depth-Anything-Small` | ~120MB | `video_depth_anything` |
| WHAM | Google Drive | ~1.2GB | `wham` |
| MatAnyone | GitHub Release | ~141MB | `matanyone` |
| SMPL-X | Requires registration | ~830MB | `smplx` |

---

## Overview

This roadmap deploys the existing Docker-based VFX pipeline to RunPod's cloud GPU infrastructure. Users access the web UI via a public URL while heavy GPU processing happens on RunPod's servers.

### Why RunPod?

| Factor | RunPod Advantage |
|--------|------------------|
| **Compatibility** | Your existing Docker image works as-is |
| **Pricing** | RTX 4090: $0.39/hr, A100: $1.89/hr, H100: $2.99/hr |
| **Free Credits** | $5-10 for new users (~15-25 hours on RTX 4090) |
| **Zero Egress** | No data transfer fees |
| **Storage** | Persistent network volumes at $0.07/GB/month |
| **Serverless** | Scale to zero when not in use |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                             │
│                    https://your-app.runpod.net                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RUNPOD POD / SERVERLESS                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   FastAPI Web   │───▶│   ComfyUI       │───▶│    NVIDIA    │ │
│  │   UI (Port 5000)│    │   (Port 8188)   │    │    GPU       │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Network Volume (/workspace)                     ││
│  │   /models (read-only)  │  /projects (read-write)            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Options

| Option | Best For | Cost Model |
|--------|----------|------------|
| **GPU Pod** | Development, continuous use | Hourly ($0.39-2.99/hr) |
| **Serverless** | Production, variable load | Per-second (scale to zero) |
| **Pod + Spot** | Batch processing | Discounted hourly |

---

## Phase 4A: RunPod Account & Infrastructure Setup

**Goal:** Set up RunPod account, network volumes, and understand the platform

### Tasks

#### Task 4A.1: Create RunPod Account

1. Sign up at [runpod.io](https://www.runpod.io/)
2. Add payment method (credit card or crypto)
3. Claim free credits ($5-10 for new users)
4. Enable two-factor authentication

**Validation:**
- [ ] Account created and verified
- [ ] Free credits visible in dashboard
- [ ] Billing method configured

---

#### Task 4A.2: Create Network Volume for Models

Network volumes persist data independently of pods and can be shared across multiple pods.

**Via Web Console:**
1. Navigate to Storage > Network Volumes
2. Click "Create Network Volume"
3. Configure:
   - Name: `vfx-models`
   - Region: Select closest to your users (e.g., `US-East`, `EU-West`)
   - Size: 50GB (expandable later)
   - Type: Standard ($0.07/GB/month)

**Via CLI:**
```bash
pip install runpod
runpod config  # Enter API key

runpod volume create \
  --name vfx-models \
  --size 50 \
  --region US-EAST-1
```

**Validation:**
```bash
runpod volume list
# Should show: vfx-models, 50GB, US-EAST-1
```

**Success Criteria:**
- [ ] Network volume created
- [ ] Region selected appropriately
- [ ] Volume ID noted for later use

---

#### Task 4A.3: Create Network Volume for Projects

```bash
runpod volume create \
  --name vfx-projects \
  --size 100 \
  --region US-EAST-1  # Same region as models
```

**Success Criteria:**
- [ ] Projects volume created
- [ ] Same region as models volume

---

#### Task 4A.4: Upload Models to Network Volume

Models must be uploaded once; they persist across pod restarts.

**Option A: Upload Existing Local Models (Recommended)**

If you've already run `python scripts/install_wizard.py --docker`, your models are at `~/.vfx_pipeline/models/`. Upload them directly:

```bash
# Install runpodctl
brew install runpod/runpodctl/runpodctl  # macOS
# or: curl -fsSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -o runpodctl && chmod +x runpodctl

# Upload local models (fastest method)
runpodctl send ~/.vfx_pipeline/models --volume vfx-models

# This uploads the exact same models the install wizard downloaded,
# using the structure defined in scripts/install_wizard/docker.py:DockerCheckpointDownloader.DOCKER_DEST_DIRS
```

**Option B: Download Fresh to RunPod**

If you don't have local models, use a temporary pod to download them. This reuses the same download logic from the install wizard:

```bash
# Start a cheap CPU pod with volume attached
runpod pod create \
  --name model-uploader \
  --image ubuntu:22.04 \
  --volume vfx-models:/models \
  --gpu-count 0

# SSH into pod
runpod pod ssh model-uploader
```

Inside the pod, download using the same sources as `scripts/install_wizard/downloader.py`:

```bash
apt update && apt install -y wget python3 python3-pip git
pip3 install huggingface_hub gdown

# SAM3 (from CheckpointDownloader.CHECKPOINTS['sam3'])
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('1038lab/sam3', local_dir='/models/sam3')"

# Video Depth Anything (from CheckpointDownloader.CHECKPOINTS['video_depth_anything'])
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Video-Depth-Anything-Small', local_dir='/models/videodepthanything')"

# WHAM (from CheckpointDownloader.CHECKPOINTS['wham'])
pip3 install gdown
gdown "https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB" -O /models/wham/wham_vit_w_3dpw.pth.tar
mkdir -p /models/wham && mv wham_vit_w_3dpw.pth.tar /models/wham/

# MatAnyone (from CheckpointDownloader.CHECKPOINTS['matanyone'])
mkdir -p /models/matanyone
wget -O /models/matanyone/matanyone.pth "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"

# SMPL-X requires manual registration - see scripts/install_wizard/downloader.py for instructions
# Register at: https://smpl-x.is.tue.mpg.de/register.php

exit
runpod pod terminate model-uploader
```

**Validation:**
```bash
# Start test pod to verify
runpod pod create --name verify-models --image ubuntu:22.04 --volume vfx-models:/models --gpu-count 0
runpod pod ssh verify-models
ls -la /models/
# Should show: sam3/, videodepthanything/, wham/, matanyone/, smplx/ (if registered)
```

**Success Criteria:**
- [ ] All required models uploaded
- [ ] Models accessible from volume mount
- [ ] Directory structure matches `DockerCheckpointDownloader.DOCKER_DEST_DIRS`
- [ ] Total size ~15-20GB

---

### Phase 4A Exit Criteria

- [ ] RunPod account active with credits
- [ ] Network volume for models created and populated
- [ ] Network volume for projects created
- [ ] Volumes in same region
- [ ] Model upload verified

---

## Phase 4B: Docker Image Adaptation

**Goal:** Modify Docker image for RunPod compatibility

### Tasks

#### Task 4B.1: Extend Existing Dockerfile for RunPod

Rather than duplicating the existing `Dockerfile`, create a thin extension that adds RunPod-specific requirements (SSH, environment variables).

**Approach:** Use a multi-stage build that inherits from the existing image.

**File:** `Dockerfile.runpod`

```dockerfile
# RunPod extension of existing VFX Ingest image
# Inherits all dependencies from the main Dockerfile

# Build the base image first (reuses existing Dockerfile)
FROM vfx-ingest:latest AS base

# Add RunPod-specific requirements
USER root

# SSH server for RunPod web terminal
RUN apt-get update && apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    echo 'root:runpod' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# RunPod-specific entrypoint (extends docker/entrypoint.sh)
COPY docker/runpod_entrypoint.sh /app/runpod_entrypoint.sh
RUN chmod +x /app/runpod_entrypoint.sh

# Override environment for RunPod volume paths
# These match the volume mount points configured in RunPod
ENV RUNPOD=true \
    VFX_MODELS_DIR=/runpod-volume/models \
    VFX_PROJECTS_DIR=/runpod-volume/projects \
    COMFYUI_OUTPUT_DIR=/runpod-volume/projects

# Expose additional port for SSH
EXPOSE 22 5000 8188

ENTRYPOINT ["/app/runpod_entrypoint.sh"]
```

**Alternative: Build from scratch (if base image not available)**

If you can't build the base image locally first, use the full Dockerfile but note that most of the code is **identical to the existing `Dockerfile`**. The only additions are:

1. SSH server installation (3 lines)
2. `RUNPOD=true` environment variable
3. Different volume paths (`/runpod-volume/` instead of `/workspace/`)
4. RunPod-specific entrypoint

**Build sequence:**
```bash
# Option 1: Build base first, then extend
docker build -t vfx-ingest:latest .
docker build -f Dockerfile.runpod -t vfx-ingest:runpod .

# Option 2: Build directly (if Dockerfile.runpod is self-contained)
docker build -f Dockerfile.runpod -t vfx-ingest:runpod .
```

**Success Criteria:**
- [ ] Dockerfile builds successfully
- [ ] Reuses existing Dockerfile logic (DRY)
- [ ] SSH server configured for RunPod terminal
- [ ] All ports exposed correctly

---

#### Task 4B.2: Create RunPod Entrypoint Script

**File:** `docker/runpod_entrypoint.sh`

```bash
#!/bin/bash
set -e

echo "=== VFX Ingest Platform (RunPod) ==="

# Start SSH server for RunPod web terminal
service ssh start

# Validate volume mounts
if [ ! -d "/runpod-volume" ]; then
    echo "ERROR: RunPod volume not mounted at /runpod-volume"
    echo "Configure your pod with a network volume mounted to /runpod-volume"
    exit 1
fi

# Create symlinks to models if using separate volumes
if [ -d "/runpod-volume/models" ]; then
    echo "Models volume detected"
    ln -sf /runpod-volume/models /models
else
    echo "WARNING: No models found at /runpod-volume/models"
    echo "Upload models using runpodctl or create a setup pod"
fi

# Ensure projects directory exists
mkdir -p /runpod-volume/projects

# Link models to ComfyUI expected locations
COMFYUI_MODELS="/app/.vfx_pipeline/ComfyUI/models"
mkdir -p "$COMFYUI_MODELS"

# Symlink each model type to ComfyUI's expected structure
if [ -d "/runpod-volume/models/sam3" ]; then
    ln -sf /runpod-volume/models/sam3 "$COMFYUI_MODELS/sam3"
fi
if [ -d "/runpod-volume/models/videodepthanything" ]; then
    ln -sf /runpod-volume/models/videodepthanything "$COMFYUI_MODELS/depth_anything"
fi

# Start ComfyUI in background
echo "Starting ComfyUI on port 8188..."
cd /app/.vfx_pipeline/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 \
    --output-directory /runpod-volume/projects > /tmp/comfyui.log 2>&1 &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI..."
for i in {1..60}; do
    if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI ready!"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "WARNING: ComfyUI not responding after 120s"
        cat /tmp/comfyui.log
    fi
done

# Start FastAPI Web UI
echo "Starting Web UI on port 5000..."
cd /app
python3 start_web.py --host 0.0.0.0 --port 5000

# Keep container running
tail -f /dev/null
```

**Success Criteria:**
- [ ] SSH starts for RunPod terminal access
- [ ] Volume validation works
- [ ] ComfyUI starts and waits correctly
- [ ] Web UI accessible

---

#### Task 4B.3: Build and Push to Docker Hub

RunPod pulls images from Docker Hub or other registries.

```bash
# Build image
docker build -f Dockerfile.runpod -t yourusername/vfx-ingest:runpod .

# Test locally first
docker run --gpus all -p 5000:5000 -p 8188:8188 \
    -v ~/.vfx_pipeline/models:/runpod-volume/models \
    -v ~/VFX-Projects:/runpod-volume/projects \
    yourusername/vfx-ingest:runpod

# Push to Docker Hub
docker login
docker push yourusername/vfx-ingest:runpod
```

**Validation:**
- Open http://localhost:5000 - should see Web UI
- Open http://localhost:8188 - should see ComfyUI

**Success Criteria:**
- [ ] Image builds successfully
- [ ] Local test passes
- [ ] Image pushed to Docker Hub
- [ ] Image size reasonable (<15GB)

---

### Phase 4B Exit Criteria

- [ ] RunPod-specific Dockerfile created
- [ ] Entrypoint script handles RunPod environment
- [ ] Image tested locally with GPU
- [ ] Image pushed to registry

---

## Phase 4C: GPU Pod Deployment

**Goal:** Deploy and test on RunPod GPU Pod (hourly billing)

### Tasks

#### Task 4C.1: Create GPU Pod via Web Console

1. Navigate to Pods > Deploy
2. Configure:
   - **Template:** Custom
   - **Container Image:** `yourusername/vfx-ingest:runpod`
   - **GPU:** RTX 4090 (24GB VRAM, $0.39/hr) or A100 for heavy workloads
   - **Volume:** Mount `vfx-models` to `/runpod-volume`
   - **Expose Ports:** HTTP: 5000, 8188
   - **Volume Disk:** 50GB (for container storage)

3. Click Deploy

**Via CLI:**
```bash
runpod pod create \
    --name vfx-ingest \
    --image yourusername/vfx-ingest:runpod \
    --gpu-type "NVIDIA RTX 4090" \
    --gpu-count 1 \
    --volume vfx-models:/runpod-volume \
    --ports "5000/http,8188/http" \
    --disk 50
```

**Success Criteria:**
- [ ] Pod starts without errors
- [ ] GPU detected and accessible
- [ ] Volumes mounted correctly

---

#### Task 4C.2: Access and Test Web UI

Once pod is running, RunPod provides public URLs.

```bash
# Get pod info
runpod pod list

# Output includes:
# Pod ID: abc123xyz
# Status: RUNNING
# Proxy URLs:
#   - https://abc123xyz-5000.proxy.runpod.net (Web UI)
#   - https://abc123xyz-8188.proxy.runpod.net (ComfyUI)
```

**Test Web UI:**
1. Open `https://abc123xyz-5000.proxy.runpod.net`
2. Should see VFX Ingest Platform dashboard
3. Create a test project
4. Upload a short video clip
5. Run ingest + depth stages

**Validation:**
- [ ] Web UI loads correctly
- [ ] Project creation works
- [ ] Video upload succeeds
- [ ] Pipeline stages execute
- [ ] Output files accessible

---

#### Task 4C.3: Test ComfyUI Direct Access

```bash
# Access ComfyUI directly
open https://abc123xyz-8188.proxy.runpod.net

# Should see ComfyUI interface
# Verify custom nodes are loaded
```

**Success Criteria:**
- [ ] ComfyUI accessible via proxy URL
- [ ] All custom nodes visible
- [ ] Workflows load correctly

---

#### Task 4C.4: Test SSH Terminal Access

RunPod provides web-based SSH terminal.

1. In RunPod console, click "Connect" on your pod
2. Select "Web Terminal"
3. Verify you can access the container:

```bash
# Check GPU
nvidia-smi

# Check models
ls -la /runpod-volume/models/

# Check ComfyUI logs
tail -f /tmp/comfyui.log

# Run pipeline manually
cd /app
python3 scripts/run_pipeline.py --help
```

**Success Criteria:**
- [ ] SSH terminal accessible
- [ ] GPU visible via nvidia-smi
- [ ] Can run commands inside container

---

#### Task 4C.5: Performance Benchmarking

Compare RunPod performance to local installation.

```bash
# Inside RunPod pod
cd /app

# Time depth estimation on test video
time python3 scripts/run_pipeline.py \
    /runpod-volume/projects/test/input.mp4 \
    --name BenchmarkTest \
    --stages depth

# Record: GPU utilization, execution time, memory usage
watch -n 1 nvidia-smi
```

**Expected Performance (30 frames, 1080p):**

| Stage | RTX 4090 | A100 40GB | Local 3090 |
|-------|----------|-----------|------------|
| Depth | ~45s | ~30s | ~60s |
| Roto (SAM3) | ~90s | ~60s | ~120s |
| COLMAP | ~180s | ~150s | ~200s |

**Success Criteria:**
- [ ] Performance within expected ranges
- [ ] GPU utilization >80% during compute
- [ ] No OOM errors

---

### Phase 4C Exit Criteria

- [ ] GPU pod deployed successfully
- [ ] Web UI accessible via public URL
- [ ] Pipeline executes correctly
- [ ] Performance acceptable
- [ ] Cost tracking understood

---

## Phase 4D: Custom Domain Setup

**Goal:** Configure custom domain for professional access

### Tasks

#### Task 4D.1: Option A - Cloudflare Tunnel (Recommended)

Cloudflare Tunnel provides free, secure tunneling with custom domains.

**Prerequisites:**
- Cloudflare account (free)
- Domain managed by Cloudflare DNS

**Setup:**

1. Install cloudflared in your container (add to Dockerfile):
```dockerfile
# Add to Dockerfile.runpod
RUN wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
    && dpkg -i cloudflared-linux-amd64.deb
```

2. Create tunnel via Cloudflare dashboard:
   - Go to Zero Trust > Access > Tunnels
   - Create tunnel named `vfx-ingest`
   - Note the tunnel token

3. Add tunnel startup to entrypoint:
```bash
# In runpod_entrypoint.sh, add before web UI startup:

# Start Cloudflare Tunnel (if token provided)
if [ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]; then
    echo "Starting Cloudflare Tunnel..."
    cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" &
fi
```

4. Configure tunnel routes in Cloudflare:
   - `vfx.yourdomain.com` -> `http://localhost:5000`
   - `comfyui.yourdomain.com` -> `http://localhost:8188` (optional)

5. Set environment variable in RunPod:
   - Pod Settings > Environment Variables
   - Add: `CLOUDFLARE_TUNNEL_TOKEN=your_token_here`

**Validation:**
```bash
# Test custom domain
curl https://vfx.yourdomain.com/health
# Should return: {"status": "ok"}
```

**Success Criteria:**
- [ ] Tunnel established
- [ ] Custom domain resolves
- [ ] HTTPS works automatically
- [ ] No RunPod proxy URL needed

---

#### Task 4D.2: Option B - nginx Reverse Proxy

Alternative using a small always-on proxy server.

```nginx
# nginx.conf on a small VPS ($5/mo)
server {
    listen 443 ssl;
    server_name vfx.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass https://abc123xyz-5000.proxy.runpod.net;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

**Trade-offs:**
- Pro: More control, can add authentication
- Con: Additional server cost, RunPod URL changes on restart

---

#### Task 4D.3: Option C - RunPod Endpoint (Serverless)

For serverless deployment, RunPod provides stable endpoint URLs.

```bash
# Create serverless endpoint
runpod endpoint create \
    --name vfx-ingest-api \
    --image yourusername/vfx-ingest:runpod \
    --gpu-type "NVIDIA RTX 4090"

# Returns stable endpoint:
# https://api.runpod.ai/v2/vfx-ingest-api/run
```

Then use Cloudflare Workers or similar to proxy with custom domain.

**Success Criteria:**
- [ ] Custom domain configured
- [ ] HTTPS working
- [ ] Domain persists across pod restarts

---

### Phase 4D Exit Criteria

- [ ] Custom domain option selected and configured
- [ ] Domain resolves to running instance
- [ ] HTTPS enabled
- [ ] Documentation updated with domain setup

---

## Phase 4E: Serverless Deployment (Production)

**Goal:** Convert to serverless for cost-effective production use

### Overview

Serverless scales to zero when not in use - you only pay during active processing.

### Tasks

#### Task 4E.1: Create Serverless Handler

RunPod Serverless requires a handler function.

**File:** `runpod_handler.py`

```python
"""RunPod Serverless Handler for VFX Ingest Pipeline."""

import runpod
import subprocess
import json
import os
from pathlib import Path

def handler(event):
    """
    Process incoming requests.

    Input event format:
    {
        "input": {
            "action": "process_video",
            "video_url": "https://...",  # or base64 encoded
            "project_name": "MyProject",
            "stages": ["ingest", "depth", "roto"],
            "options": {}
        }
    }

    Returns:
    {
        "output": {
            "status": "completed",
            "project_path": "/workspace/projects/MyProject",
            "outputs": {...}
        }
    }
    """
    try:
        job_input = event.get("input", {})
        action = job_input.get("action", "process_video")

        if action == "health_check":
            return {"status": "healthy", "gpu": check_gpu()}

        elif action == "process_video":
            return process_video(job_input)

        elif action == "get_status":
            return get_project_status(job_input.get("project_name"))

        else:
            return {"error": f"Unknown action: {action}"}

    except Exception as e:
        return {"error": str(e)}


def check_gpu():
    """Check GPU availability."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def process_video(job_input):
    """Run the VFX pipeline on input video."""
    video_url = job_input.get("video_url")
    project_name = job_input.get("project_name", "ServerlessProject")
    stages = job_input.get("stages", ["ingest", "depth"])

    # Download video if URL provided
    if video_url:
        video_path = download_video(video_url, project_name)
    else:
        return {"error": "No video_url provided"}

    # Build pipeline command
    cmd = [
        "python3", "/app/scripts/run_pipeline.py",
        str(video_path),
        "--name", project_name,
        "--stages", ",".join(stages)
    ]

    # Run pipeline
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        return {
            "status": "failed",
            "error": result.stderr,
            "stdout": result.stdout
        }

    # Collect outputs
    project_path = Path(f"/runpod-volume/projects/{project_name}")
    outputs = collect_outputs(project_path)

    return {
        "status": "completed",
        "project_name": project_name,
        "outputs": outputs
    }


def download_video(url, project_name):
    """Download video from URL."""
    import urllib.request

    project_dir = Path(f"/runpod-volume/projects/{project_name}")
    project_dir.mkdir(parents=True, exist_ok=True)

    video_path = project_dir / "input.mp4"
    urllib.request.urlretrieve(url, video_path)

    return video_path


def collect_outputs(project_path):
    """Collect output file information."""
    outputs = {}

    for subdir in ["source/frames", "depth", "roto", "colmap"]:
        dir_path = project_path / subdir
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            outputs[subdir] = {
                "count": len(files),
                "total_size_mb": sum(f.stat().st_size for f in files) / (1024*1024)
            }

    return outputs


# Start serverless worker
runpod.serverless.start({"handler": handler})
```

**Success Criteria:**
- [ ] Handler processes video requests
- [ ] Proper error handling
- [ ] Output collection works

---

#### Task 4E.2: Create Serverless Dockerfile

**File:** `Dockerfile.runpod-serverless`

```dockerfile
# Serverless variant - optimized for cold starts
FROM yourusername/vfx-ingest:runpod

# Install runpod SDK
RUN pip3 install runpod

# Copy serverless handler
COPY runpod_handler.py /app/runpod_handler.py

# Serverless uses different entrypoint
CMD ["python3", "/app/runpod_handler.py"]
```

```bash
# Build and push
docker build -f Dockerfile.runpod-serverless -t yourusername/vfx-ingest:serverless .
docker push yourusername/vfx-ingest:serverless
```

---

#### Task 4E.3: Deploy Serverless Endpoint

**Via Web Console:**
1. Navigate to Serverless > Endpoints
2. Click "New Endpoint"
3. Configure:
   - Name: `vfx-ingest`
   - Docker Image: `yourusername/vfx-ingest:serverless`
   - GPU: RTX 4090 (or as needed)
   - Workers:
     - Min: 0 (scale to zero)
     - Max: 3 (parallel processing)
   - Idle Timeout: 5 seconds
   - Volume: Mount `vfx-models` to `/runpod-volume`

**Via CLI:**
```bash
runpod endpoint create \
    --name vfx-ingest \
    --image yourusername/vfx-ingest:serverless \
    --gpu-type "NVIDIA RTX 4090" \
    --min-workers 0 \
    --max-workers 3 \
    --idle-timeout 5 \
    --volume vfx-models:/runpod-volume
```

**Output:**
```
Endpoint created:
  ID: vfx-ingest-abc123
  URL: https://api.runpod.ai/v2/vfx-ingest-abc123
```

---

#### Task 4E.4: Test Serverless Endpoint

```bash
# Set API key
export RUNPOD_API_KEY="your_api_key"

# Health check
curl -X POST https://api.runpod.ai/v2/vfx-ingest-abc123/run \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": {"action": "health_check"}}'

# Process video
curl -X POST https://api.runpod.ai/v2/vfx-ingest-abc123/run \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "action": "process_video",
            "video_url": "https://example.com/test.mp4",
            "project_name": "ServerlessTest",
            "stages": ["ingest", "depth"]
        }
    }'

# Response:
# {"id": "job-123", "status": "IN_QUEUE"}

# Check job status
curl https://api.runpod.ai/v2/vfx-ingest-abc123/status/job-123 \
    -H "Authorization: Bearer $RUNPOD_API_KEY"
```

**Success Criteria:**
- [ ] Endpoint responds to requests
- [ ] Jobs queue and process correctly
- [ ] Scale to zero works
- [ ] Cold start time acceptable (<60s)

---

#### Task 4E.5: Web UI Integration with Serverless

Modify web UI to use serverless backend instead of local processing.

**File:** `web/services/runpod_client.py`

```python
"""RunPod Serverless API client."""

import os
import httpx
from typing import Optional

class RunPodClient:
    def __init__(self):
        self.api_key = os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"

    async def submit_job(self, video_url: str, project_name: str, stages: list) -> str:
        """Submit a processing job, return job ID."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/run",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "input": {
                        "action": "process_video",
                        "video_url": video_url,
                        "project_name": project_name,
                        "stages": stages
                    }
                }
            )
            return response.json()["id"]

    async def get_job_status(self, job_id: str) -> dict:
        """Get job status and results."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/status/{job_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.json()
```

**Success Criteria:**
- [ ] Web UI can submit serverless jobs
- [ ] Job status polling works
- [ ] Results displayed correctly

---

### Phase 4E Exit Criteria

- [ ] Serverless endpoint deployed
- [ ] Scale to zero verified
- [ ] Cold start time acceptable
- [ ] API integration complete
- [ ] Cost tracking understood

---

## Phase 4F: Cost Optimization

**Goal:** Optimize for cost-effective operation

### Tasks

#### Task 4F.1: Implement Job Queuing

Batch multiple requests to reduce cold starts.

#### Task 4F.2: Use Spot Instances for Batch Work

RunPod offers discounted "Spot" GPU instances.

```bash
runpod pod create \
    --name batch-processor \
    --image yourusername/vfx-ingest:runpod \
    --gpu-type "NVIDIA RTX 4090" \
    --spot  # 50-70% cheaper, but can be interrupted
```

#### Task 4F.3: Implement Model Caching

Pre-load models to reduce cold start time.

#### Task 4F.4: Cost Monitoring Dashboard

Track spending via RunPod API:

```python
# Get usage stats
runpod usage --last 30d
```

---

### Cost Estimates

| Usage Pattern | Pod (24/7) | Pod (8hr/day) | Serverless |
|---------------|------------|---------------|------------|
| **RTX 4090** | $280/mo | $93/mo | ~$0.39/hr active |
| **A100 40GB** | $1,360/mo | $453/mo | ~$1.89/hr active |
| **Storage (50GB)** | $3.50/mo | $3.50/mo | $3.50/mo |

**Break-even Analysis:**
- If using <240 hours/month: Serverless is cheaper
- If using >240 hours/month: Pod is cheaper

---

## Roadmap 4 Success Criteria

**Ready for production when:**

- [ ] Docker image deployed to RunPod
- [ ] GPU pod or serverless endpoint running
- [ ] Custom domain configured
- [ ] Web UI accessible via custom URL
- [ ] Pipeline executes correctly in cloud
- [ ] Cost tracking in place
- [ ] Documentation complete
- [ ] At least 3 successful end-to-end tests

**Estimated Costs:**
- Development/testing: Free credits cover ~15-25 hours
- Light production: $20-50/month (serverless + storage)
- Heavy production: $100-300/month (dedicated pod)

---

**Next:** [Roadmap 5: Modal Deployment](ROADMAP-5-MODAL.md) (alternative cloud platform)
