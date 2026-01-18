#!/bin/bash
# VFX Pipeline Bootstrap Script - Conda Edition
# Downloads and runs the installation wizard
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/scripts/bootstrap_conda.sh | bash
#   or
#   wget -qO- https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/scripts/bootstrap_conda.sh | bash

set -e  # Exit on error

REPO_URL="https://github.com/kleer001/comfyui_ingest.git"
INSTALL_DIR="$(pwd)/comfyui_ingest"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  VFX Pipeline - Automated Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo "❌ Error: git is not installed"
    echo "   Install with: sudo apt install git"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    echo "   Install with: sudo apt install python3"
    exit 1
fi

echo "✓ Prerequisites met"
echo ""

# Clone or update repository
if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR already exists."
    read -p "Update existing installation? (y/N): " -n 1 -r < /dev/tty
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating repository..."
        cd "$INSTALL_DIR"
        git pull
    else
        echo "Using existing installation at $INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
else
    echo "Cloning repository to $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching Installation Wizard"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the wizard
python3 scripts/install_wizard.py "$@"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate vfx-pipeline"
echo "  2. Or use: source $INSTALL_DIR/.vfx_pipeline/activate.sh"
echo "  3. Read: $INSTALL_DIR/README.md"
echo ""
