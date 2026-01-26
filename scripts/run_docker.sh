#!/bin/bash
# VFX Ingest Platform - Docker Wrapper Script
# Simplified wrapper for running the pipeline in Docker

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get repo root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Default paths relative to repo (not home directory)
# Models: <repo>/.vfx_pipeline/models/
# Projects: <repo>/../vfx_projects/ (sibling to repo)
MODELS_DIR="${VFX_MODELS_DIR:-${REPO_ROOT}/.vfx_pipeline/models}"
PROJECTS_DIR="${VFX_PROJECTS_DIR:-$(dirname "$REPO_ROOT")/vfx_projects}"

# Detect docker compose command (plugin vs standalone)
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo -e "${RED}ERROR: Neither 'docker compose' nor 'docker-compose' found${NC}"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    echo "Start Docker Desktop and try again"
    exit 1
fi

# Check if .env exists (required for Docker build configuration)
if [ ! -f "${REPO_ROOT}/.env" ]; then
    echo -e "${YELLOW}No .env file found - running GPU detection...${NC}"
    echo ""
    bash "${REPO_ROOT}/scripts/setup_build.sh"
    echo ""
fi

# Check models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${YELLOW}WARNING: Models directory not found at $MODELS_DIR${NC}"
    echo ""
    echo "The pipeline requires ML models to function."
    echo "Download them first:"
    echo "  ./scripts/download_models.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r < /dev/tty
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

# Export paths for docker-compose volume mounts
export VFX_MODELS_DIR="$MODELS_DIR"
export VFX_PROJECTS_DIR="$PROJECTS_DIR"

# Export host user UID/GID for correct file ownership in container
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
echo -e "${GREEN}File ownership: UID=${HOST_UID} GID=${HOST_GID}${NC}"

# Check if image exists
if ! docker image inspect vfx-ingest:latest > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker image not found. Building...${NC}"
    echo "(This is a one-time operation and may take 10-15 minutes)"
    echo ""
    $DOCKER_COMPOSE build
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
    echo ""
fi

# Parse arguments to find input file and mount it
INPUT_FILE=""
DOCKER_ARGS=()
FOUND_INPUT=false

for arg in "$@"; do
    if [[ "$arg" != -* ]] && [ "$FOUND_INPUT" = false ]; then
        INPUT_FILE="$arg"
        FOUND_INPUT=true
    else
        DOCKER_ARGS+=("$arg")
    fi
done

# Prepare volume mount for input file
INPUT_VOLUME_ARGS=""
CONTAINER_INPUT_PATH=""

if [ -n "$INPUT_FILE" ]; then
    ABSOLUTE_INPUT="$(cd "$(dirname "$INPUT_FILE")" 2>/dev/null && pwd)/$(basename "$INPUT_FILE")"

    if [ ! -f "$ABSOLUTE_INPUT" ]; then
        echo -e "${RED}ERROR: Input file not found: $INPUT_FILE${NC}"
        echo "Resolved path: $ABSOLUTE_INPUT"
        exit 1
    fi

    INPUT_DIR="$(dirname "$ABSOLUTE_INPUT")"
    INPUT_FILENAME="$(basename "$ABSOLUTE_INPUT")"
    CONTAINER_INPUT_PATH="/input/$INPUT_FILENAME"
    INPUT_VOLUME_ARGS="-v ${INPUT_DIR}:/input:ro"

    echo -e "${GREEN}Running VFX Ingest Platform in Docker...${NC}"
    echo "Input: $ABSOLUTE_INPUT -> $CONTAINER_INPUT_PATH"
    echo "Models: $MODELS_DIR (read-only)"
    echo "Projects: $PROJECTS_DIR"
    echo ""
else
    echo -e "${GREEN}Running VFX Ingest Platform in Docker...${NC}"
    echo "Models: $MODELS_DIR (read-only)"
    echo "Projects: $PROJECTS_DIR"
    echo ""
fi

# Run container with input volume mount if needed
if [ -n "$INPUT_VOLUME_ARGS" ]; then
    $DOCKER_COMPOSE run --rm $INPUT_VOLUME_ARGS vfx-ingest "$CONTAINER_INPUT_PATH" "${DOCKER_ARGS[@]}"
else
    $DOCKER_COMPOSE run --rm vfx-ingest "$@"
fi
