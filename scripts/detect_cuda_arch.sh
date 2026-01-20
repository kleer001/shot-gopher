#!/bin/bash
# Detect CUDA compute capability for the current GPU
# Usage: ./detect_cuda_arch.sh
#        docker compose build --build-arg CUDA_ARCH="$(./scripts/detect_cuda_arch.sh)"

if ! command -v nvidia-smi &> /dev/null; then
  echo "7.5 8.6 8.9" # fallback default (RTX 20xx/30xx/40xx)
  exit 0
fi

# Get compute capability (e.g., "8.6")
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')

if [ -z "$ARCH" ]; then
  echo "7.5 8.6 8.9" # fallback default (RTX 20xx/30xx/40xx)
else
  echo "$ARCH"
fi
