#!/bin/bash
# Detect CUDA compute capability for the current GPU
# Usage: ./detect_cuda_arch.sh
#        docker compose build --build-arg CUDA_ARCH="$(./scripts/detect_cuda_arch.sh)"

if ! command -v nvidia-smi &> /dev/null; then
  echo "7.5;8.6;8.9" # fallback default
  exit 0
fi

# Get compute capability (e.g., "8.6")
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')

if [ -z "$ARCH" ]; then
  echo "7.5;8.6;8.9" # fallback default
else
  echo "$ARCH"
fi
