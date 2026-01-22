# VFX Ingest Platform - Docker Image
# Multi-stage build for optimized layer caching

# Stage 1: Get COLMAP from official pre-built image
# Using official image which is Ubuntu 24.04 based with all dependencies resolved
FROM colmap/colmap:latest AS colmap-source

# Stage 2: Base image with system dependencies
# Using Ubuntu 24.04 to match COLMAP image (avoids glibc mismatch)
# CUDA 12.4 has PyTorch wheels available
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages and COLMAP runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    xvfb \
    ninja-build \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglew2.2 \
    libgomp1 \
    libboost-filesystem1.83.0 \
    libboost-program-options1.83.0 \
    libboost-graph1.83.0 \
    libgoogle-glog0v2t64 \
    libceres3t64 \
    libmetis5 \
    libopenimageio2.5t64 \
    libsqlite3-0 \
    libqt6core6t64 \
    libqt6widgets6t64 \
    libqt6openglwidgets6 \
    libcurl4t64 \
    libmkl-intel-lp64 \
    libmkl-intel-thread \
    libmkl-core \
    && rm -rf /var/lib/apt/lists/*

# Copy COLMAP from official image
COPY --from=colmap-source /usr/local/bin/colmap /usr/local/bin/colmap
COPY --from=colmap-source /usr/local/lib/ /usr/local/lib/
COPY --from=colmap-source /usr/local/share/ /usr/local/share/

# Ensure COLMAP is executable and update library cache
RUN chmod +x /usr/local/bin/colmap && ldconfig

# Ensure /usr/local/bin is in PATH (for shutil.which to find colmap)
ENV PATH="/usr/local/bin:$PATH"

# Create application directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Stage 3: Python dependencies
FROM base AS python-deps

# Copy requirements
COPY requirements.txt /tmp/

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install smplx (required for mocap)
RUN pip3 install --no-cache-dir --break-system-packages smplx

# Install kornia (required for GS-IR)
RUN pip3 install --no-cache-dir --break-system-packages kornia

# Stage 4: GS-IR (Gaussian Splatting Inverse Rendering)
FROM python-deps AS gsir

# CUDA architecture for CUDA extension builds (GPU not visible during docker build)
# Override: export CUDA_ARCH=$(./scripts/detect_cuda_arch.sh) && docker compose build
# Common values: 7.5 (RTX 20xx/T4), 8.6 (RTX 30xx), 8.9 (RTX 40xx)
ARG CUDA_ARCH="7.5 8.6 8.9"

WORKDIR /app/.vfx_pipeline

# Clone GS-IR with submodules
RUN git clone --recursive https://github.com/lzhnb/GS-IR.git GS-IR

# Fix missing <cstdint> includes in GS-IR source (upstream bug)
# Required for uint32_t type definition in CUDA compilation
RUN cd GS-IR/gs-ir && \
    sed -i '1i#include <cstdint>' src/utils.h && \
    sed -i '1i#include <cstdint>' src/pbr_utils.cuh

# Install nvdiffrast (required for GS-IR rendering)
# --no-build-isolation required so it can find PyTorch during build
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" \
    pip3 install --no-build-isolation --break-system-packages git+https://github.com/NVlabs/nvdiffrast.git

# Build and install GS-IR submodules (CUDA extensions)
WORKDIR /app/.vfx_pipeline/GS-IR
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" \
    pip3 install --no-build-isolation --break-system-packages ./submodules/diff-gaussian-rasterization && \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" \
    pip3 install --no-build-isolation --break-system-packages ./submodules/simple-knn

# Install gs-ir module (has CUDA extensions, needs TORCH_CUDA_ARCH_LIST)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd gs-ir && TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" pip3 install --no-build-isolation --break-system-packages -e .

WORKDIR /app

# Stage 5: ComfyUI and custom nodes
FROM gsir AS comfyui

# Create .vfx_pipeline directory structure
RUN mkdir -p /app/.vfx_pipeline

# Clone ComfyUI and install its requirements
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/.vfx_pipeline/ComfyUI && \
    pip3 install --no-cache-dir --break-system-packages -r /app/.vfx_pipeline/ComfyUI/requirements.txt

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
            pip3 install --no-cache-dir --break-system-packages -r "$dir/requirements.txt"; \
        fi; \
    done

# Install SAM3 GPU-accelerated NMS (speeds up video tracking 5-10x)
# Only attempt if nvcc (CUDA compiler) is available
# Note: SAM3's install.py uses an outdated comfy_env API (passes config= argument)
# The new comfy_env API auto-discovers config from cwd, so we call it directly
RUN cd ComfyUI-SAM3 && \
    if command -v nvcc >/dev/null 2>&1; then \
        echo "CUDA toolkit found, installing comfy-env and SAM3 GPU NMS..." && \
        pip3 install --no-cache-dir --break-system-packages comfy-env && \
        python3 -c "from comfy_env import install; install()" || \
        echo "WARNING: SAM3 GPU NMS installation failed. Will use CPU fallback at runtime."; \
    else \
        echo "Skipping SAM3 GPU NMS (nvcc not available - will use CPU fallback at runtime)"; \
    fi

WORKDIR /app

# Stage 6: Pipeline scripts
FROM comfyui AS pipeline

# Copy pipeline scripts
COPY scripts/ /app/scripts/
COPY workflow_templates/ /app/workflow_templates/

# Copy web application (if exists)
COPY web/ /app/web/

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
    COMFYUI_OUTPUT_DIR=/workspace \
    GSIR_PATH=/app/.vfx_pipeline/GS-IR \
    QT_QPA_PLATFORM=offscreen

# Expose ports
EXPOSE 8188

# Volumes
VOLUME ["/models", "/workspace"]

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
