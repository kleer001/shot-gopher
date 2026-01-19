#!/bin/bash
set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== VFX Ingest Platform (Container) ===${NC}"

# Validate mounted volumes
if [ ! -d "/models" ]; then
    echo -e "${RED}ERROR: /models volume not mounted${NC}"
    echo "Docker run must include: -v /path/to/models:/models"
    exit 1
fi

if [ ! -d "/workspace" ]; then
    echo -e "${RED}ERROR: /workspace volume not mounted${NC}"
    echo "Docker run must include: -v /path/to/workspace:/workspace"
    exit 1
fi

# Check for required models
echo -e "${YELLOW}Checking models...${NC}"
REQUIRED_MODELS=(
    "/models/sam3"
    "/models/videodepthanything"
    "/models/wham"
    "/models/matanyone"
)

MISSING_MODELS=0
for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -d "$model" ]; then
        echo -e "${YELLOW}  WARNING: Model not found: $model${NC}"
        MISSING_MODELS=1
    else
        echo -e "${GREEN}  ✓ Found: $model${NC}"
    fi
done

if [ $MISSING_MODELS -eq 1 ]; then
    echo -e "${YELLOW}Some models are missing. Pipeline may fail on certain stages.${NC}"
    echo "Run model download script on host: ./scripts/download_models.sh"
fi

# Symlink models to locations expected by custom nodes
echo -e "${YELLOW}Linking models to custom node paths...${NC}"

# MatAnyone expects checkpoint in its own directory
MATANYONE_SRC="/models/matanyone/matanyone.pth"
MATANYONE_DST="/app/.vfx_pipeline/ComfyUI/custom_nodes/ComfyUI-MatAnyone/checkpoint"
if [ -f "$MATANYONE_SRC" ]; then
    mkdir -p "$MATANYONE_DST"
    if [ ! -e "$MATANYONE_DST/matanyone.pth" ]; then
        ln -sf "$MATANYONE_SRC" "$MATANYONE_DST/matanyone.pth"
        echo -e "${GREEN}  ✓ Linked MatAnyone model${NC}"
    else
        echo -e "${GREEN}  ✓ MatAnyone model already linked${NC}"
    fi
else
    echo -e "${YELLOW}  WARNING: MatAnyone model not found at $MATANYONE_SRC${NC}"
fi

# Start ComfyUI in background if requested
if [ "$START_COMFYUI" = "true" ]; then
    echo -e "${GREEN}Starting ComfyUI...${NC}"
    cd /app/.vfx_pipeline/ComfyUI
    python3 main.py --listen 0.0.0.0 --port 8188 \
        --output-directory /workspace > /tmp/comfyui.log 2>&1 &
    COMFYUI_PID=$!

    # Wait for ComfyUI to be ready
    echo "Waiting for ComfyUI to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
            echo -e "${GREEN}✓ ComfyUI ready on port 8188${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo -e "${RED}ERROR: ComfyUI failed to start${NC}"
            cat /tmp/comfyui.log
            exit 1
        fi
    done
fi

# Execute the main command
echo -e "${GREEN}Running pipeline...${NC}"
cd /app
exec python3 /app/scripts/run_pipeline.py "$@"
