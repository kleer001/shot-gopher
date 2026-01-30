#!/bin/bash
# =============================================================================
# Run Pipeline - Auto-activating wrapper
# =============================================================================
# Activates the vfx-pipeline conda environment and runs run_pipeline.py
#
# Usage:
#   ./src/run-pipeline.sh <input_movie> [options]
#
# Example:
#   ./src/run-pipeline.sh /path/to/footage.mp4 --name "My_Shot" --stages all
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Save arguments before sourcing (source inherits $@ if not given explicit args)
_saved_args=("$@")
source "$REPO_ROOT/scripts/activate_env.sh" || exit 1

python "$REPO_ROOT/scripts/run_pipeline.py" "${_saved_args[@]}"
