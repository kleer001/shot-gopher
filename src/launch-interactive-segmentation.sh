#!/bin/bash
# =============================================================================
# Launch Interactive Segmentation - Auto-activating wrapper
# =============================================================================
# Activates the vfx-pipeline conda environment and launches interactive
# segmentation workflow in ComfyUI
#
# Usage:
#   ./src/launch-interactive-segmentation.sh <project_dir>
#
# Example:
#   ./src/launch-interactive-segmentation.sh /path/to/projects/My_Shot
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Save arguments before sourcing (source inherits $@ if not given explicit args)
_saved_args=("$@")
source "$REPO_ROOT/scripts/activate_env.sh" || exit 1

python "$REPO_ROOT/scripts/launch_interactive_segmentation.py" "${_saved_args[@]}"
