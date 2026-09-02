#!/bin/bash
#===============================================================================
# Switch ComfyUI output to an existing project
#
# Usage: ./set_project.sh PROJECT_NAME
#        ./set_project.sh --list
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECTS_DIR="$ROOT_DIR/projects"
COMFYUI_OUTPUT="$ROOT_DIR/comfyui/ComfyUI/output"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# List projects
if [ "$1" == "--list" ] || [ "$1" == "-l" ] || [ -z "$1" ]; then
    echo -e "${CYAN}Available projects:${NC}"
    echo ""
    
    # Get current project
    CURRENT=""
    if [ -L "$COMFYUI_OUTPUT" ]; then
        CURRENT=$(basename "$(readlink -f "$COMFYUI_OUTPUT")")
    fi
    
    for dir in "$PROJECTS_DIR"/*/; do
        name=$(basename "$dir")
        if [ "$name" == "_template" ]; then
            continue
        fi
        if [ "$name" == "$CURRENT" ]; then
            echo -e "  ${GREEN}→ $name${NC} (active)"
        else
            echo "    $name"
        fi
    done
    echo ""
    echo "Usage: $0 PROJECT_NAME"
    exit 0
fi

PROJECT_NAME="$1"
PROJECT_DIR="$PROJECTS_DIR/$PROJECT_NAME"

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Error: Project not found: $PROJECT_NAME${NC}"
    echo "Use '$0 --list' to see available projects"
    exit 1
fi

# Remove existing symlink/directory
if [ -L "$COMFYUI_OUTPUT" ]; then
    rm "$COMFYUI_OUTPUT"
elif [ -d "$COMFYUI_OUTPUT" ]; then
    echo -e "${YELLOW}Warning: output/ is a directory, backing up to output.bak/${NC}"
    mv "$COMFYUI_OUTPUT" "${COMFYUI_OUTPUT}.bak"
fi

# Create symlink
ln -s "$PROJECT_DIR" "$COMFYUI_OUTPUT"

echo -e "${GREEN}Switched to project: $PROJECT_NAME${NC}"
echo "ComfyUI/output → $PROJECT_DIR"
