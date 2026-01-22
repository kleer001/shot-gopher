# VFX Ingest Platform - Docker Image
# Multi-stage build for optimized layer caching

# Stage 1: Build COLMAP from source
# The official colmap/colmap:latest image now uses Ubuntu 24.04 (PR #3363, June 2025)
# which is incompatible with our Ubuntu 22.04 base due to glibc version mismatch.
# Building from source ensures compatibility with Ubuntu 22.04 + CUDA 12.1.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS colmap-builder

ENV DEBIAN_FRONTEND=noninteractive

# CUDA architectures to build for (RTX 30xx = 8.6, RTX 40xx = 8.9, etc.)
# Can be overridden at build time: docker build --build-arg COLMAP_CUDA_ARCHS="86"
ARG COLMAP_CUDA_ARCHS="75;80;86;89"

# Install GCC 10 (required for Ubuntu 22.04 + CUDA compilation)
# and all COLMAP build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    gcc-10 \
    g++-10 \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libmkl-full-dev \
    libglew-dev \
    libcgal-dev \
    libceres-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 10 as the compiler (required for Ubuntu 22.04 + CUDA builds)
ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
ENV CUDAHOSTCXX=/usr/bin/g++-10

# Clone and build COLMAP 3.13.0 (latest stable release)
RUN git clone --branch 3.13.0 --depth 1 https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${COLMAP_CUDA_ARCHS}" \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DBLA_VENDOR=Intel10_64lp \
        -DTESTS_ENABLED=OFF && \
    ninja && \
    ninja install

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
    ninja-build \
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
    libopenimageio2.2 \
    libsqlite3-0 \
    libqt5core5a \
    libqt5widgets5 \
    libcurl4 \
    libmkl-intel-lp64 \
    libmkl-intel-thread \
    libmkl-core \
    && rm -rf /var/lib/apt/lists/*

# Copy COLMAP from the build stage
COPY --from=colmap-builder /colmap-install/bin/colmap /usr/local/bin/colmap
COPY --from=colmap-builder /colmap-install/lib/ /usr/local/lib/
COPY --from=colmap-builder /colmap-install/share/ /usr/local/share/

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
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install smplx (required for mocap)
RUN pip3 install --no-cache-dir smplx

# Install kornia (required for GS-IR)
RUN pip3 install --no-cache-dir kornia

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
    pip3 install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# Build and install GS-IR submodules (CUDA extensions)
WORKDIR /app/.vfx_pipeline/GS-IR
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" \
    pip3 install --no-build-isolation ./submodules/diff-gaussian-rasterization && \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" \
    pip3 install --no-build-isolation ./submodules/simple-knn

# Install gs-ir module (has CUDA extensions, needs TORCH_CUDA_ARCH_LIST)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd gs-ir && TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" pip3 install --no-build-isolation -e .

WORKDIR /app

# Stage 5: ComfyUI and custom nodes
FROM gsir AS comfyui

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
# Note: SAM3's install.py uses an outdated comfy_env API (passes config= argument)
# The new comfy_env API auto-discovers config from cwd, so we call it directly
RUN cd ComfyUI-SAM3 && \
    if command -v nvcc >/dev/null 2>&1; then \
        echo "CUDA toolkit found, installing comfy-env and SAM3 GPU NMS..." && \
        pip3 install --no-cache-dir comfy-env && \
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
