#!/bin/bash
#===============================================================================
# Ingest a movie file into a new project
#
# Usage: ./ingest.sh /path/to/movie.mp4
#
# Creates:
#   - Project folder (first 20 chars of filename)
#   - Symlinks ComfyUI/output to project folder
#   - Extracts frames to source/frames/
#===============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}Error: No movie file specified${NC}"
    echo "Usage: $0 /path/to/movie.mp4"
    exit 1
fi

MOVIE_FILE="$1"

if [ ! -f "$MOVIE_FILE" ]; then
    echo -e "${RED}Error: File not found: $MOVIE_FILE${NC}"
    exit 1
fi

# Get script/root directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Extract project name: first 20 chars of filename (no extension)
FILENAME=$(basename "$MOVIE_FILE")
FILENAME_NOEXT="${FILENAME%.*}"
PROJECT_NAME="${FILENAME_NOEXT:0:20}"

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  INGESTING: $FILENAME${NC}"
echo -e "${CYAN}  PROJECT:   $PROJECT_NAME${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

PROJECT_DIR="$ROOT_DIR/projects/$PROJECT_NAME"

# Check if project exists
if [ -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}Warning: Project '$PROJECT_NAME' already exists${NC}"
    read -p "Overwrite frames? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    # Create from template
    echo -e "${GREEN}[1/4]${NC} Creating project structure..."
    cp -r "$ROOT_DIR/projects/_template" "$PROJECT_DIR"
fi

# Symlink ComfyUI output to project
echo -e "${GREEN}[2/4]${NC} Linking ComfyUI output..."
COMFYUI_OUTPUT="$ROOT_DIR/comfyui/ComfyUI/output"

# Remove existing (file, link, or directory)
if [ -L "$COMFYUI_OUTPUT" ]; then
    rm "$COMFYUI_OUTPUT"
elif [ -d "$COMFYUI_OUTPUT" ]; then
    rm -rf "$COMFYUI_OUTPUT"
fi

ln -s "$PROJECT_DIR" "$COMFYUI_OUTPUT"
echo "  → ComfyUI/output → $PROJECT_DIR"

# Extract frames
echo -e "${GREEN}[3/4]${NC} Extracting frames (starting at 1001)..."
FRAMES_DIR="$PROJECT_DIR/source/frames"
mkdir -p "$FRAMES_DIR"

# Get video info
FRAME_COUNT=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$MOVIE_FILE" 2>/dev/null || echo "unknown")
echo "  → Source: $MOVIE_FILE"
echo "  → Frames: $FRAME_COUNT"

# Extract with padding starting at 1001
ffmpeg -i "$MOVIE_FILE" -start_number 1001 -qscale:v 2 "$FRAMES_DIR/frame_%04d.png" -y 2>/dev/null

EXTRACTED=$(ls -1 "$FRAMES_DIR"/*.png 2>/dev/null | wc -l)
echo "  → Extracted: $EXTRACTED frames"

# Copy original movie
echo -e "${GREEN}[4/4]${NC} Copying source movie..."
cp "$MOVIE_FILE" "$PROJECT_DIR/source/"

# Create project info
cat > "$PROJECT_DIR/project_info.txt" << EOF
Project: $PROJECT_NAME
Source:  $FILENAME
Created: $(date)
Frames:  $EXTRACTED (1001-$((1000 + EXTRACTED)))

Folder Structure:
  source/frames/    - Extracted PNG frames (frame_1001.png, ...)
  source/           - Original movie file
  depth/            - Depth map outputs
  roto/             - Segmentation masks
  cleanplate/       - Inpainted clean plates
  camera/           - Camera/geometry data

ComfyUI output symlinked to this project.
EOF

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  DONE! Project ready: $PROJECT_NAME${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Frames: $FRAMES_DIR/frame_1001.png - frame_$((1000 + EXTRACTED)).png"
echo ""
echo "Next:"
echo "  1. Launch ComfyUI:  ./scripts/run_comfyui.sh"
echo "  2. Load workflow:   01_analysis.json or 02_segmentation.json"
echo "  3. Set path to:     source/frames"
echo ""
