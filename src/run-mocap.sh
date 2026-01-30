#!/bin/bash
# =============================================================================
# Run Mocap - Auto-activating wrapper
# =============================================================================
# Activates the vfx-pipeline conda environment and runs run_mocap.py
#
# Usage:
#   ./src/run-mocap.sh <project_dir> [options]
#
# Example:
#   ./src/run-mocap.sh /path/to/projects/My_Shot
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$REPO_ROOT/scripts/activate_env.sh" || exit 1

python "$REPO_ROOT/scripts/run_mocap.py" "$@"
