# VFX Ingest Platform - Docker Image
# Multi-stage build for optimized layer caching

# Stage 1: Get COLMAP from official pre-built image (with CUDA + FreeImage support)
# Using 20231029.4 tag which is compatible with Ubuntu 22.04 (before 24.04 release)
# This image is built properly, so FreeImage_Initialise() works
FROM colmap/colmap:20231029.4 AS colmap-source

# Stage 2: Base image with system dependencies
# Using devel image for nvcc (CUDA compiler) needed by SAM3 GPU NMS
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages and COLMAP runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3.10 \
    python3-pip \
    wget \
    curl \
    xvfb \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglew2.2 \
    libgomp1 \
    libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 \
    libboost-graph1.74.0 \
    libgoogle-glog0v5 \
    libceres2 \
    libmetis5 \
    libfreeimage3 \
    libsqlite3-0 \
    libflann1.9 \
    libqt5core5a \
    libqt5widgets5 \
    && rm -rf /var/lib/apt/lists/*

# Copy COLMAP from official image
# The official image has proper FreeImage initialization built in
COPY --from=colmap-source /usr/local/bin/colmap /usr/local/bin/colmap
# Copy COLMAP's shared libraries (official image uses dynamic linking)
COPY --from=colmap-source /usr/local/lib/libcolmap* /usr/local/lib/
# Update library cache
RUN ldconfig || true

# Create application directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Stage 3: Python dependencies
FROM base AS python-deps

# Copy requirements
COPY requirements.txt /tmp/

# Install Python packages
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install smplx (required for mocap)
RUN pip3 install --no-cache-dir smplx

# Stage 4: ComfyUI and custom nodes
FROM python-deps AS comfyui

# Create .vfx_pipeline directory structure
RUN mkdir -p /app/.vfx_pipeline

# Clone ComfyUI and install its requirements
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/.vfx_pipeline/ComfyUI && \
    pip3 install --no-cache-dir -r /app/.vfx_pipeline/ComfyUI/requirements.txt

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

# Install SAM3 GPU-accelerated NMS (speeds up video tracking 5-10x)
# Only attempt if nvcc (CUDA compiler) is available
# UV_SYSTEM_PYTHON=1 tells uv to install into system Python (no venv in Docker)
RUN cd ComfyUI-SAM3 && \
    if command -v nvcc >/dev/null 2>&1; then \
        echo "CUDA toolkit found, installing SAM3 GPU NMS..." && \
        UV_SYSTEM_PYTHON=1 python3 install.py; \
    else \
        echo "Skipping SAM3 GPU NMS (nvcc not available - will use CPU fallback at runtime)"; \
    fi

WORKDIR /app

# Stage 5: Pipeline scripts
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
    QT_QPA_PLATFORM=offscreen

# Expose ports
EXPOSE 8188

# Volumes
VOLUME ["/models", "/workspace"]

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
