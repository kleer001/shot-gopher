#!/bin/bash
# =============================================================================
# Run COLMAP - Auto-activating wrapper
# =============================================================================
# Activates the vfx-pipeline conda environment and runs run_colmap.py
#
# Usage:
#   ./src/run-colmap.sh <project_dir> [options]
#
# Example:
#   ./src/run-colmap.sh /path/to/projects/My_Shot --dense --mesh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Save arguments before sourcing (source inherits $@ if not given explicit args)
_saved_args=("$@")
source "$REPO_ROOT/scripts/activate_env.sh" || exit 1

python "$REPO_ROOT/scripts/run_colmap.py" "${_saved_args[@]}"
