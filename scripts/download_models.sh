#!/bin/bash
set -e

# VFX Ingest Platform - Model Download Script
# Runs on HOST (not in container) to populate model directory

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get repo root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Model directory (default: <repo>/.vfx_pipeline/models)
MODEL_DIR="${VFX_MODELS_DIR:-${REPO_ROOT}/.vfx_pipeline/models}"

echo -e "${BLUE}=== VFX Ingest Platform - Model Downloader ===${NC}"
echo -e "Models will be downloaded to: ${GREEN}${MODEL_DIR}${NC}"
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# Check for required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}ERROR: $1 not found. Please install it first.${NC}"
        exit 1
    fi
}

# Download with progress
download_file() {
    local name=$1
    local url=$2
    local dest=$3

    echo -e "${YELLOW}Downloading $name...${NC}"
    mkdir -p "$(dirname "$dest")"

    if command -v wget &> /dev/null; then
        wget -O "$dest" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$dest" "$url"
    else
        echo -e "${RED}ERROR: Neither wget nor curl found${NC}"
        exit 1
    fi
}

# Check Python available
check_tool python3

# Install huggingface_hub if needed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip3 install --user huggingface_hub
fi

# SAM3 (via HuggingFace)
echo ""
echo -e "${BLUE}[1/4] SAM3 (Segment Anything Model 3)${NC}"
if [ -d "$MODEL_DIR/sam3" ] && [ "$(ls -A "$MODEL_DIR/sam3")" ]; then
    echo -e "${GREEN}✓ SAM3 already exists${NC}"
else
    echo -e "${YELLOW}Downloading SAM3 from HuggingFace (1038lab/sam3)...${NC}"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('1038lab/sam3', local_dir='$MODEL_DIR/sam3')"
    echo -e "${GREEN}✓ SAM3 downloaded${NC}"
fi

# Video Depth Anything
echo ""
echo -e "${BLUE}[2/4] Video Depth Anything${NC}"
if [ -d "$MODEL_DIR/videodepthanything" ] && [ "$(ls -A "$MODEL_DIR/videodepthanything")" ]; then
    echo -e "${GREEN}✓ Video Depth Anything already exists${NC}"
else
    echo -e "${YELLOW}Downloading Video Depth Anything from HuggingFace...${NC}"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Video-Depth-Anything-Small', local_dir='$MODEL_DIR/videodepthanything')"
    echo -e "${GREEN}✓ Video Depth Anything downloaded${NC}"
fi

# WHAM (4D Human Motion Capture)
echo ""
echo -e "${BLUE}[3/4] WHAM (4D Human MoCap)${NC}"
if [ -d "$MODEL_DIR/wham" ] && [ -f "$MODEL_DIR/wham/wham_vit_w_3dpw.pth.tar" ]; then
    echo -e "${GREEN}✓ WHAM already exists${NC}"
else
    echo -e "${YELLOW}Downloading WHAM from HuggingFace...${NC}"
    mkdir -p "$MODEL_DIR/wham"
    # WHAM model from official repository
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='yohanshin/WHAM', filename='wham_vit_w_3dpw.pth.tar', local_dir='$MODEL_DIR/wham')"
    echo -e "${GREEN}✓ WHAM downloaded${NC}"
fi

# MatAnyone
echo ""
echo -e "${BLUE}[4/4] MatAnyone (Matte Refinement)${NC}"
if [ -f "$MODEL_DIR/matanyone/matanyone.pth" ]; then
    echo -e "${GREEN}✓ MatAnyone already exists${NC}"
else
    echo -e "${YELLOW}Downloading MatAnyone...${NC}"
    mkdir -p "$MODEL_DIR/matanyone"
    # MatAnyone from GitHub releases
    download_file "MatAnyone" \
        "https://github.com/FuouM/ComfyUI-MatAnyone/releases/download/v1.0/matanyone.pth" \
        "$MODEL_DIR/matanyone/matanyone.pth"
    echo -e "${GREEN}✓ MatAnyone downloaded${NC}"
fi

echo ""
echo -e "${GREEN}=== Public Models Downloaded ===${NC}"
echo ""
echo -e "${YELLOW}⚠ MANUAL DOWNLOAD REQUIRED:${NC}"
echo ""
echo -e "SMPL-X models require registration at:"
echo -e "  ${BLUE}https://smpl-x.is.tue.mpg.de/${NC}"
echo ""
echo -e "After registration:"
echo -e "  1. Download SMPL-X models (NEUTRAL, MALE, FEMALE)"
echo -e "  2. Extract to: ${GREEN}${MODEL_DIR}/smplx/${NC}"
echo -e "  3. Verify: python3 scripts/verify_models.py"
echo ""
echo -e "${GREEN}Once SMPL-X is downloaded, you're ready to run the pipeline!${NC}"
echo ""
