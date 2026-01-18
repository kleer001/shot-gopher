#!/bin/bash
# VFX Ingest Platform - Docker Wrapper Script
# Simplified wrapper for running the pipeline in Docker

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default paths
MODELS_DIR="${VFX_MODELS_DIR:-${HOME}/.vfx_pipeline/models}"
PROJECTS_DIR="${VFX_PROJECTS_DIR:-${HOME}/VFX-Projects}"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    echo "Start Docker Desktop and try again"
    exit 1
fi

# Check models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${YELLOW}WARNING: Models directory not found at $MODELS_DIR${NC}"
    echo ""
    echo "The pipeline requires ML models to function."
    echo "Download them first:"
    echo "  ./scripts/download_models.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create projects directory if it doesn't exist
if [ ! -d "$PROJECTS_DIR" ]; then
    echo -e "${YELLOW}Creating projects directory: $PROJECTS_DIR${NC}"
    mkdir -p "$PROJECTS_DIR"
fi

# Check if image exists
if ! docker image inspect vfx-ingest:latest > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker image not found. Building...${NC}"
    echo "(This is a one-time operation and may take 10-15 minutes)"
    echo ""
    docker-compose build
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
    echo ""
fi

# Run container
echo -e "${GREEN}Running VFX Ingest Platform in Docker...${NC}"
echo "Models: $MODELS_DIR (read-only)"
echo "Projects: $PROJECTS_DIR"
echo ""

docker-compose run --rm vfx-ingest "$@"
