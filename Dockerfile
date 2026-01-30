# VFX Ingest Platform - Docker Image
# Multi-stage build for optimized layer caching

# Build argument to skip COLMAP (speeds up rebuilds for testing)
# Usage: docker build --build-arg SKIP_COLMAP=true .
ARG SKIP_COLMAP=false

# Stage 1: Build COLMAP from source with CUDA support
# The official colmap/colmap:latest image does NOT have GPU support
# We must build from source with CUDA enabled
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04 AS colmap-builder

ARG SKIP_COLMAP
ENV DEBIAN_FRONTEND=noninteractive

# Install COLMAP build dependencies (skip if SKIP_COLMAP=true)
RUN if [ "$SKIP_COLMAP" = "true" ]; then \
        echo "Skipping COLMAP build (SKIP_COLMAP=true)" && \
        mkdir -p /colmap-install/bin /colmap-install/lib; \
    else \
        apt-get update && apt-get install -y \
            git \
            cmake \
            ninja-build \
            build-essential \
            libboost-program-options-dev \
            libboost-graph-dev \
            libboost-system-dev \
            libeigen3-dev \
            libcgal-dev \
            libceres-dev \
            libgoogle-glog-dev \
            libgflags-dev \
            libglew-dev \
            libsqlite3-dev \
            libfreeimage-dev \
            libflann-dev \
            libopenimageio-dev \
            libsuitesparse-dev \
            libmetis-dev \
            libmkl-full-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Clone and build COLMAP with CUDA support (skip if SKIP_COLMAP=true)
ARG COLMAP_VERSION=3.11.1
ARG CUDA_ARCHITECTURES="75;86;89"

RUN if [ "$SKIP_COLMAP" = "true" ]; then \
        echo "COLMAP build skipped"; \
    else \
        git clone --branch ${COLMAP_VERSION} --depth 1 https://github.com/colmap/colmap.git /colmap-src && \
        mkdir -p /colmap-src/build && cd /colmap-src/build && \
        cmake .. -GNinja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/colmap-install \
            -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
            -DGUI_ENABLED=OFF \
            -DBLA_VENDOR=Intel10_64lp && \
        ninja install && \
        rm -rf /colmap-src; \
    fi

# Stage 2: Base image with system dependencies
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04 AS base

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
    gosu \
    libgl1 \
    libglx-mesa0 \
    libglu1-mesa \
    libglew2.2 \
    libgomp1 \
    libboost-filesystem1.83.0 \
    libboost-program-options1.83.0 \
    libboost-graph1.83.0 \
    libgoogle-glog0v6t64 \
    libceres4 \
    libmetis5 \
    libfreeimage3 \
    libsqlite3-0 \
    libcurl4t64 \
    libmkl-intel-lp64 \
    libmkl-intel-thread \
    libmkl-core \
    libomp5 \
    libflann1.9 \
    libopenimageio2.4t64 \
    libcholmod5 \
    && rm -rf /var/lib/apt/lists/*

# Copy COLMAP from builder stage (GPU-enabled build)
COPY --from=colmap-builder /colmap-install/ /usr/local/

# Update library cache
RUN ldconfig

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

# Install GS-IR Python dependencies (not in environment.yml but required)
# Install GS-IR Python dependencies (extracted from render.py, train.py, relight.py, baking.py)
RUN pip3 install --no-cache-dir --break-system-packages \
    plyfile \
    tqdm \
    "imageio[ffmpeg]" \
    tensorboard \
    lpips \
    opencv-python-headless \
    Pillow

# Fix missing includes in GS-IR submodules (upstream bugs, GCC 13+ compatibility)
# - cstdint: uint32_t/uint64_t/uintptr_t type definitions
# - cfloat: FLT_MAX constant
RUN cd GS-IR/gs-ir && \
    sed -i '1i#include <cstdint>' src/utils.h && \
    sed -i '1i#include <cstdint>' src/pbr_utils.cuh && \
    cd ../submodules/diff-gaussian-rasterization/cuda_rasterizer && \
    sed -i '1i#include <cstdint>' rasterizer_impl.h && \
    cd ../../simple-knn && \
    sed -i '1i#include <cfloat>' simple_knn.cu

# Fix PyTorch 2.6+ weights_only=True default (GS-IR uses pickle in checkpoints)
# Patch all torch.load calls to explicitly set weights_only=False
RUN cd GS-IR && \
    for f in $(find . -name "*.py" -exec grep -l "torch.load" {} \;); do \
        sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' "$f"; \
    done

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
    git clone https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git

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

# Stage 6: VideoMaMa (diffusion-based video matting)
FROM comfyui AS videomama

# Install Miniconda for VideoMaMa's isolated environment
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda init bash && \
    ${CONDA_DIR}/bin/conda clean -afy

ENV PATH="${CONDA_DIR}/bin:$PATH"

# Create VideoMaMa conda environment with Python 3.10
RUN conda create -n videomama python=3.10 -y && \
    conda clean -afy

# Install PyTorch with CUDA in videomama environment
RUN conda run -n videomama pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install VideoMaMa dependencies in conda environment
RUN conda run -n videomama pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    huggingface_hub \
    opencv-python \
    einops \
    omegaconf \
    safetensors

# Clone VideoMaMa repository
WORKDIR /app/.vfx_pipeline/tools
RUN git clone https://github.com/cvlab-kaist/VideoMaMa.git

# Install VideoMaMa requirements if they exist
RUN if [ -f "VideoMaMa/requirements.txt" ]; then \
        conda run -n videomama pip install --no-cache-dir -r VideoMaMa/requirements.txt; \
    fi

WORKDIR /app

# Stage 7: Pipeline scripts
FROM videomama AS pipeline

# Build-time UID/GID for pre-setting ownership (most Linux users have UID 1000)
ARG VFX_UID=1000
ARG VFX_GID=1000

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

# Set ownership of application directories to UID 1000 at build time
# This eliminates runtime chown when HOST_UID=1000 (common case)
RUN chown -R ${VFX_UID}:${VFX_GID} /app/.vfx_pipeline

# Mark as container environment
# HOST_UID/HOST_GID: Set at runtime to match host user for correct file ownership
ENV CONTAINER=true \
    VFX_INSTALL_DIR=/app/.vfx_pipeline \
    VFX_MODELS_DIR=/models \
    VFX_PROJECTS_DIR=/workspace/projects \
    COMFYUI_OUTPUT_DIR=/workspace \
    GSIR_PATH=/app/.vfx_pipeline/GS-IR \
    QT_QPA_PLATFORM=offscreen \
    HOST_UID=0 \
    HOST_GID=0 \
    VIDEOMAMA_TOOLS_DIR=/app/.vfx_pipeline/tools/VideoMaMa \
    VIDEOMAMA_MODELS_DIR=/models/videomama \
    CONDA_DIR=/opt/conda \
    PATH="/opt/conda/bin:$PATH"

# Expose ports
EXPOSE 8188

# Volumes
VOLUME ["/models", "/workspace"]

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
