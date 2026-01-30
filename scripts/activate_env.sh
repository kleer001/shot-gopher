#!/bin/bash
# =============================================================================
# VFX Pipeline Conda Environment Activation Script
# =============================================================================
#
# This script ensures the vfx-pipeline conda environment is active.
# It can be sourced by other scripts or run directly.
#
# Usage:
#   source scripts/activate_env.sh        # Activate in current shell
#   source scripts/activate_env.sh --check  # Just check, don't activate
#
# In other scripts:
#   source "$(dirname "$0")/activate_env.sh" || exit 1
#
# =============================================================================

# Configuration - single source of truth
VFX_ENV_NAME="vfx-pipeline"

# Colors for output (disable if not interactive)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

_vfx_log_info() {
    echo -e "${GREEN}[vfx-env]${NC} $1"
}

_vfx_log_warn() {
    echo -e "${YELLOW}[vfx-env]${NC} $1" >&2
}

_vfx_log_error() {
    echo -e "${RED}[vfx-env]${NC} $1" >&2
}

_vfx_show_activation_warning() {
    local current_env="$1"
    echo "" >&2
    echo -e "${RED}⚠️  ╔══════════════════════════════════════════════════════════════╗  ⚠️${NC}" >&2
    echo -e "${RED}⚠️  ║                                                              ║  ⚠️${NC}" >&2
    echo -e "${RED}⚠️  ║              WRONG CONDA ENVIRONMENT ACTIVE                  ║  ⚠️${NC}" >&2
    echo -e "${RED}⚠️  ║                                                              ║  ⚠️${NC}" >&2
    echo -e "${RED}⚠️  ╚══════════════════════════════════════════════════════════════╝  ⚠️${NC}" >&2
    echo "" >&2
    if [[ -n "$current_env" ]]; then
        echo -e "    Currently active: ${YELLOW}'$current_env'${NC}" >&2
        echo -e "    Required:         ${GREEN}'$VFX_ENV_NAME'${NC}" >&2
    else
        echo "    No conda environment is currently active." >&2
        echo -e "    Required: ${GREEN}'$VFX_ENV_NAME'${NC}" >&2
    fi
    echo "" >&2
    echo "    ┌────────────────────────────────────────────────────────┐" >&2
    echo "    │  To fix this, run:                                     │" >&2
    echo "    │                                                        │" >&2
    echo -e "    │      ${GREEN}conda activate $VFX_ENV_NAME${NC}                        │" >&2
    echo "    │                                                        │" >&2
    echo "    └────────────────────────────────────────────────────────┘" >&2
    echo "" >&2
}

# -----------------------------------------------------------------------------
# Check if we're in the right environment
# -----------------------------------------------------------------------------

_vfx_check_env() {
    if [[ "$CONDA_DEFAULT_ENV" == "$VFX_ENV_NAME" ]]; then
        return 0
    fi
    return 1
}

# -----------------------------------------------------------------------------
# Find and initialize conda if needed
# -----------------------------------------------------------------------------

_vfx_init_conda() {
    # Check if conda command is available
    if command -v conda &> /dev/null; then
        return 0
    fi

    # Try common conda locations
    local conda_paths=(
        "${HOME}/miniconda3/etc/profile.d/conda.sh"
        "${HOME}/anaconda3/etc/profile.d/conda.sh"
        "${HOME}/miniforge3/etc/profile.d/conda.sh"
        "${HOME}/mambaforge/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
    )

    for conda_path in "${conda_paths[@]}"; do
        if [[ -f "$conda_path" ]]; then
            # shellcheck source=/dev/null
            source "$conda_path"
            if command -v conda &> /dev/null; then
                _vfx_log_info "Initialized conda from $conda_path"
                return 0
            fi
        fi
    done

    # Try to find conda.sh using CONDA_EXE if set
    if [[ -n "$CONDA_EXE" ]]; then
        local conda_sh="${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
        if [[ -f "$conda_sh" ]]; then
            # shellcheck source=/dev/null
            source "$conda_sh"
            if command -v conda &> /dev/null; then
                return 0
            fi
        fi
    fi

    return 1
}

# -----------------------------------------------------------------------------
# Check if the environment exists
# -----------------------------------------------------------------------------

_vfx_env_exists() {
    conda env list 2>/dev/null | grep -q "^${VFX_ENV_NAME} "
}

# -----------------------------------------------------------------------------
# Main activation logic
# -----------------------------------------------------------------------------

_vfx_activate() {
    local check_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check|-c)
                check_only=true
                shift
                ;;
            --quiet|-q)
                # Suppress most output
                _vfx_log_info() { :; }
                shift
                ;;
            --help|-h)
                echo "Usage: source activate_env.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --check, -c    Check if environment is active (don't activate)"
                echo "  --quiet, -q    Suppress informational output"
                echo "  --help, -h     Show this help message"
                echo ""
                echo "Environment: $VFX_ENV_NAME"
                return 0
                ;;
            *)
                _vfx_log_error "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # Already in the right environment?
    if _vfx_check_env; then
        _vfx_log_info "Environment '$VFX_ENV_NAME' is already active"
        return 0
    fi

    # Just checking?
    if $check_only; then
        _vfx_show_activation_warning "$CONDA_DEFAULT_ENV"
        return 1
    fi

    # Make sure conda is available
    if ! _vfx_init_conda; then
        _vfx_log_error "Could not find conda. Please install conda/miniconda first."
        _vfx_log_error "Visit: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi

    # Check if environment exists
    if ! _vfx_env_exists; then
        _vfx_log_error "Environment '$VFX_ENV_NAME' does not exist."
        _vfx_log_error "Please run the installation wizard first:"
        _vfx_log_error "  python scripts/install_wizard.py"
        return 1
    fi

    # Activate the environment
    _vfx_log_info "Activating '$VFX_ENV_NAME' environment..."

    if conda activate "$VFX_ENV_NAME" 2>/dev/null; then
        _vfx_log_info "Environment activated successfully"
        return 0
    else
        _vfx_log_error "Failed to activate environment '$VFX_ENV_NAME'"
        _vfx_show_activation_warning "$CONDA_DEFAULT_ENV"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Run activation when sourced (unless sourced with --check)
# -----------------------------------------------------------------------------

# Only run if being sourced, not executed
if [[ "${BASH_SOURCE[0]}" != "${0}" ]] || [[ -n "$ZSH_EVAL_CONTEXT" ]]; then
    _vfx_activate "$@"
else
    # Being executed directly - show instructions
    echo "This script must be sourced, not executed."
    echo ""
    echo "Usage:"
    echo "  source $0"
    echo ""
    echo "Or add to your shell startup:"
    echo "  echo 'source $(realpath "$0")' >> ~/.bashrc"
    exit 1
fi
