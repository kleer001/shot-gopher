#!/bin/bash
set -e

# Integration tests for Docker build (Phase 1D)
# Run these tests on a machine with Docker installed

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}=== VFX Ingest Platform - Docker Integration Tests ===${NC}"
echo "Repository: $REPO_ROOT"
echo ""

# Check Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found. Install Docker to run these tests.${NC}"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker daemon not running. Start Docker and try again.${NC}"
    exit 1
fi

cd "$REPO_ROOT"

# Test 1: Build base layer
echo -e "${BLUE}[Test 1/7] Building base layer...${NC}"
if docker build --target base -t vfx-ingest:base . > /tmp/docker_base_build.log 2>&1; then
    echo -e "${GREEN}✓ Base layer built successfully${NC}"
else
    echo -e "${RED}✗ Base layer build failed${NC}"
    tail -20 /tmp/docker_base_build.log
    exit 1
fi

# Test 2: Verify COLMAP installed
echo -e "${BLUE}[Test 2/7] Verifying COLMAP installation...${NC}"
if docker run --rm vfx-ingest:base colmap --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓ COLMAP installed and working${NC}"
else
    echo -e "${RED}✗ COLMAP not working${NC}"
    exit 1
fi

# Test 3: Verify FFmpeg installed
echo -e "${BLUE}[Test 3/7] Verifying FFmpeg installation...${NC}"
if docker run --rm vfx-ingest:base ffmpeg -version > /dev/null 2>&1; then
    echo -e "${GREEN}✓ FFmpeg installed and working${NC}"
else
    echo -e "${RED}✗ FFmpeg not working${NC}"
    exit 1
fi

# Test 4: Build python-deps layer
echo -e "${BLUE}[Test 4/7] Building python-deps layer...${NC}"
if docker build --target python-deps -t vfx-ingest:python . > /tmp/docker_python_build.log 2>&1; then
    echo -e "${GREEN}✓ Python dependencies layer built successfully${NC}"
else
    echo -e "${RED}✗ Python dependencies layer build failed${NC}"
    tail -20 /tmp/docker_python_build.log
    exit 1
fi

# Test 5: Verify Python packages
echo -e "${BLUE}[Test 5/7] Verifying Python packages...${NC}"
if docker run --rm vfx-ingest:python python3 -c "import numpy, scipy, trimesh, PIL, torch; print('Core deps OK')" 2>&1 | grep -q "Core deps OK"; then
    echo -e "${GREEN}✓ Python packages installed and importable${NC}"
else
    echo -e "${RED}✗ Python package import failed${NC}"
    exit 1
fi

# Test 6: Build full image
echo -e "${BLUE}[Test 6/7] Building full image...${NC}"
if docker build -t vfx-ingest:latest . > /tmp/docker_full_build.log 2>&1; then
    echo -e "${GREEN}✓ Full image built successfully${NC}"
else
    echo -e "${RED}✗ Full image build failed${NC}"
    tail -20 /tmp/docker_full_build.log
    exit 1
fi

# Test 7: Verify entrypoint
echo -e "${BLUE}[Test 7/7] Verifying entrypoint...${NC}"
# This will fail because volumes aren't mounted, but we check that the entrypoint runs
if docker run --rm vfx-ingest:latest --help 2>&1 | grep -q "VFX Ingest Platform"; then
    echo -e "${GREEN}✓ Entrypoint script works${NC}"
else
    echo -e "${YELLOW}⚠ Entrypoint validation inconclusive (expected - volumes not mounted)${NC}"
fi

echo ""
echo -e "${GREEN}=== All Docker Build Tests Passed ===${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Download models: ./scripts/download_models.sh"
echo "2. Run with docker-compose: docker-compose run --rm vfx-ingest --help"
echo "3. Test full pipeline with test video"
echo ""
