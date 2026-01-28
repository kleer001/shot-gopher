#!/bin/bash
set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== VFX Ingest Platform (Container) ===${NC}"

# Key paths (avoid hardcoding throughout script)
VFX_PIPELINE_DIR="/app/.vfx_pipeline"
COMFYUI_DIR="${VFX_PIPELINE_DIR}/ComfyUI"

# Build-time UID/GID (set in Dockerfile, files already owned by this user)
# If HOST_UID matches this, we skip the runtime chown (instant startup)
BUILD_UID=1000
BUILD_GID=1000

# Handle user switching for correct file ownership
# If HOST_UID/HOST_GID are set and non-zero, create a matching user and run as them
VFX_USER="vfxuser"
VFX_HOME="/home/${VFX_USER}"

setup_user() {
    local uid="${HOST_UID:-0}"
    local gid="${HOST_GID:-0}"

    if [ "$uid" = "0" ] || [ "$gid" = "0" ]; then
        echo -e "${YELLOW}Running as root (HOST_UID/HOST_GID not set)${NC}"
        echo -e "${YELLOW}Files will be owned by root. Set HOST_UID and HOST_GID for correct ownership.${NC}"
        return 1
    fi

    echo -e "${GREEN}Setting up user with UID:${uid} GID:${gid}${NC}"

    # Fast path: if UID matches build-time user, use it directly
    if [ "$uid" = "$BUILD_UID" ] && [ "$gid" = "$BUILD_GID" ]; then
        echo -e "${GREEN}Using build-time user vfxuser (UID matches)${NC}"
        VFX_USER="vfxuser"
        VFX_HOME="/home/vfxuser"
        chown -R "$uid:$gid" /workspace 2>/dev/null || true
        return 0
    fi

    # Slow path: need to create or find a user with matching UID
    if ! getent group "$gid" > /dev/null 2>&1; then
        groupadd -g "$gid" hostgroup 2>/dev/null || true
    fi

    local existing_user=$(getent passwd "$uid" | cut -d: -f1)
    if [ -n "$existing_user" ]; then
        VFX_USER="$existing_user"
        VFX_HOME=$(getent passwd "$uid" | cut -d: -f6)
        echo -e "${GREEN}Using existing user: ${VFX_USER}${NC}"
    else
        VFX_USER="hostuser"
        VFX_HOME="/home/hostuser"
        useradd -o -u "$uid" -g "$gid" -m -d "$VFX_HOME" -s /bin/bash "$VFX_USER" 2>/dev/null || true
        echo -e "${GREEN}Created user: ${VFX_USER}${NC}"
    fi

    chown -R "$uid:$gid" "$VFX_HOME" 2>/dev/null || true
    chown -R "$uid:$gid" /workspace 2>/dev/null || true

    return 0
}

RUN_AS_USER=false
if setup_user; then
    RUN_AS_USER=true
fi

# Check for interactive mode (just run ComfyUI, no pipeline)
INTERACTIVE_MODE=false
if [[ "$1" == "interactive" || "$1" == "comfyui" ]]; then
    INTERACTIVE_MODE=true
    echo -e "${YELLOW}Running in interactive mode (ComfyUI only)${NC}"
fi

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

# Set permissions on VFX pipeline directory for non-root user
# Do this BEFORE creating symlinks/directories so they inherit correct ownership
if [ "$RUN_AS_USER" = "true" ] && [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ]; then
    if [ "$HOST_UID" = "$BUILD_UID" ] && [ "$HOST_GID" = "$BUILD_GID" ]; then
        echo -e "${GREEN}UID/GID matches build-time user (${BUILD_UID}:${BUILD_GID}) - skipping chown${NC}"
    else
        echo -e "${YELLOW}Adjusting permissions for UID:${HOST_UID} GID:${HOST_GID} (one-time operation)...${NC}"
        chown -R "$HOST_UID:$HOST_GID" "$VFX_PIPELINE_DIR" 2>/dev/null || true
    fi
fi

# Symlink models to locations expected by custom nodes
echo -e "${YELLOW}Linking models to custom node paths...${NC}"

# MatAnyone expects checkpoint in its own directory
MATANYONE_SRC="/models/matanyone/matanyone.pth"
MATANYONE_DST="${COMFYUI_DIR}/custom_nodes/ComfyUI-MatAnyone/checkpoint"
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

# SAM3 expects model at ComfyUI root (downloads automatically if missing, but symlink avoids re-download)
SAM3_SRC="/models/sam3/sam3.pt"
SAM3_DST="${COMFYUI_DIR}/sam3.pt"
if [ -f "$SAM3_SRC" ]; then
    if [ ! -e "$SAM3_DST" ]; then
        ln -sf "$SAM3_SRC" "$SAM3_DST"
        echo -e "${GREEN}  ✓ Linked SAM3 model${NC}"
    else
        echo -e "${GREEN}  ✓ SAM3 model already linked${NC}"
    fi
fi

# SMPL-X models for mocap stage (manual download required)
SMPLX_SRC="/models/smplx"
SMPLX_DST="${VFX_PIPELINE_DIR}/smplx_models"
if [ -d "$SMPLX_SRC" ]; then
    if [ ! -e "$SMPLX_DST" ]; then
        ln -sf "$SMPLX_SRC" "$SMPLX_DST"
        echo -e "${GREEN}  ✓ Linked SMPL-X models${NC}"
    else
        echo -e "${GREEN}  ✓ SMPL-X models already linked${NC}"
    fi
fi

# Determine if ComfyUI is needed based on stages
COMFYUI_STAGES="depth|roto|matanyone|cleanplate"
NEED_COMFYUI=false

# Interactive mode always needs ComfyUI
if [ "$INTERACTIVE_MODE" = "true" ]; then
    NEED_COMFYUI=true
else
    # Parse command line for stages
    for arg in "$@"; do
        if [[ "$arg" == "-s" || "$arg" == "--stages" ]]; then
            CHECKING_STAGES=true
        elif [[ "$CHECKING_STAGES" == "true" ]]; then
            if [[ "$arg" == "all" ]] || echo "$arg" | grep -qE "$COMFYUI_STAGES"; then
                NEED_COMFYUI=true
            fi
            CHECKING_STAGES=false
        fi
    done

    # If no stages specified, assume all (needs ComfyUI)
    if [[ "$@" != *"-s"* && "$@" != *"--stages"* ]]; then
        NEED_COMFYUI=true
    fi
fi

# Start ComfyUI in background only if needed
if [ "$NEED_COMFYUI" = "true" ]; then
    echo -e "${GREEN}Starting ComfyUI...${NC}"
    cd "$COMFYUI_DIR"

    # In interactive mode, run ComfyUI in foreground (never returns)
    if [ "$INTERACTIVE_MODE" = "true" ]; then
        echo -e "${GREEN}ComfyUI starting on port 8188 (interactive mode)${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        if [ "$RUN_AS_USER" = "true" ]; then
            exec gosu "$VFX_USER" python3 main.py --listen 0.0.0.0 --port 8188 --output-directory /workspace
        else
            exec python3 main.py --listen 0.0.0.0 --port 8188 --output-directory /workspace
        fi
    fi

    # For pipeline mode, run ComfyUI in background
    if [ "$RUN_AS_USER" = "true" ]; then
        gosu "$VFX_USER" python3 main.py --listen 0.0.0.0 --port 8188 \
            --output-directory /workspace > /tmp/comfyui.log 2>&1 &
    else
        python3 main.py --listen 0.0.0.0 --port 8188 \
            --output-directory /workspace > /tmp/comfyui.log 2>&1 &
    fi
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
else
    echo -e "${YELLOW}Skipping ComfyUI (not needed for requested stages)${NC}"
fi

# Execute the main command (pipeline mode only - interactive mode already exec'd above)
if [ "$INTERACTIVE_MODE" = "true" ]; then
    echo -e "${RED}ERROR: Interactive mode should have exec'd into ComfyUI${NC}"
    echo "This indicates ComfyUI failed to start. Check the logs above."
    exit 1
fi

echo -e "${GREEN}Running pipeline...${NC}"
cd /app

if [ "$RUN_AS_USER" = "true" ]; then
    gosu "$VFX_USER" python3 /app/scripts/run_pipeline.py "$@"
else
    python3 /app/scripts/run_pipeline.py "$@"
fi
EXIT_CODE=$?

# Fix ownership of output files if HOST_UID/HOST_GID are set
if [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ]; then
    echo -e "${YELLOW}Fixing file permissions for host user...${NC}"
    chown -R "$HOST_UID:$HOST_GID" /workspace/projects 2>/dev/null || true
fi

exit $EXIT_CODE
