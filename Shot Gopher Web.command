#!/bin/bash
# Shot Gopher Web GUI Launcher for macOS
# Double-click this file to launch the web interface

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo " ===================================="
echo "  Shot Gopher - Web GUI Launcher"
echo " ===================================="
echo ""

# Initialize conda for this shell session
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
elif [ -f "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda not found"
    echo ""
    echo "Please install Miniconda or Anaconda."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

# Activate the vfx-pipeline environment
echo "Activating vfx-pipeline environment..."
conda activate vfx-pipeline
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate vfx-pipeline environment"
    echo ""
    echo "Please run the install wizard first:"
    echo "  python scripts/install_wizard.py"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

# Launch the web GUI
echo "Starting Shot Gopher Web GUI..."
echo ""
python scripts/launch_web_gui.py

# Keep terminal open if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred. Press Enter to exit..."
    read
fi
