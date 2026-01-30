#!/bin/bash
# VFX Pipeline Bootstrap Script - Conda Edition
# Downloads and runs the installation wizard
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.sh | bash
#   or
#   wget -qO- https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.sh | bash

set -e

REPO_URL="https://github.com/kleer001/shot-gopher.git"
INSTALL_DIR="$(pwd)/shot-gopher"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'

print_banner() {
    local text="$1"
    local color="${2:-$CYAN}"
    echo ""
    echo -e "${color}============================================================${NC}"
    echo -e "${color}  $text${NC}"
    echo -e "${color}============================================================${NC}"
    echo ""
}

detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64)  echo "x86_64" ;;
        aarch64) echo "aarch64" ;;
        arm64)   echo "arm64" ;;
        *)       echo "unknown" ;;
    esac
}

find_conda() {
    # Check if conda command is available
    if command -v conda &> /dev/null; then
        command -v conda
        return 0
    fi

    # Check common installation locations
    local locations=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/.conda/bin/conda"
        "/opt/miniconda3/bin/conda"
        "/opt/anaconda3/bin/conda"
        "/usr/local/miniconda3/bin/conda"
        "/usr/local/anaconda3/bin/conda"
    )

    for loc in "${locations[@]}"; do
        if [ -x "$loc" ]; then
            echo "$loc"
            return 0
        fi
    done

    return 1
}

get_miniconda_url() {
    local os=$(detect_os)
    local arch=$(detect_arch)

    case "$os" in
        linux)
            case "$arch" in
                x86_64)  echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
                aarch64) echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
                *)       return 1 ;;
            esac
            ;;
        macos)
            case "$arch" in
                x86_64)  echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
                arm64)   echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
                *)       return 1 ;;
            esac
            ;;
        *)
            return 1
            ;;
    esac
}

install_miniconda() {
    local install_path="$HOME/miniconda3"
    local url=$(get_miniconda_url)

    if [ -z "$url" ]; then
        echo -e "${RED}✗ Could not determine Miniconda download URL for your system${NC}"
        echo -e "  OS: $(detect_os), Arch: $(detect_arch)"
        echo -e "  Please install manually from: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi

    echo -e "${YELLOW}Downloading Miniconda...${NC}"
    local installer="/tmp/miniconda_installer.sh"

    if command -v curl &> /dev/null; then
        curl -fsSL "$url" -o "$installer"
    elif command -v wget &> /dev/null; then
        wget -q "$url" -O "$installer"
    else
        echo -e "${RED}✗ Neither curl nor wget found${NC}"
        return 1
    fi

    echo -e "${YELLOW}Installing Miniconda to $install_path...${NC}"
    echo -e "${GRAY}  This may take a few minutes...${NC}"

    bash "$installer" -b -p "$install_path"
    rm -f "$installer"

    # Add to PATH for current session
    export PATH="$install_path/bin:$PATH"

    echo -e "${GREEN}✓ Miniconda installed successfully${NC}"
    return 0
}

initialize_conda_shell() {
    local conda_path="$1"
    local shell_name=$(basename "$SHELL")

    echo -e "${YELLOW}Initializing conda for $shell_name...${NC}"

    case "$shell_name" in
        bash)
            "$conda_path" init bash 2>/dev/null || true
            echo -e "${GREEN}✓ Conda initialized for bash${NC}"
            echo -e "${YELLOW}NOTE: Run 'source ~/.bashrc' or restart your terminal${NC}"
            ;;
        zsh)
            "$conda_path" init zsh 2>/dev/null || true
            echo -e "${GREEN}✓ Conda initialized for zsh${NC}"
            echo -e "${YELLOW}NOTE: Run 'source ~/.zshrc' or restart your terminal${NC}"
            ;;
        *)
            echo -e "${YELLOW}! Unknown shell: $shell_name${NC}"
            echo -e "  Run manually: $conda_path init $shell_name"
            ;;
    esac
}

print_banner "VFX Pipeline - Automated Installer"

# Check for git
echo "Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ Git is not installed${NC}"
    echo ""
    os=$(detect_os)
    case "$os" in
        linux)
            echo -e "${YELLOW}Install with:${NC}"
            echo -e "${CYAN}  Ubuntu/Debian: sudo apt install git${NC}"
            echo -e "${CYAN}  Fedora/RHEL:   sudo dnf install git${NC}"
            echo -e "${CYAN}  Arch:          sudo pacman -S git${NC}"
            ;;
        macos)
            echo -e "${YELLOW}Install with:${NC}"
            echo -e "${CYAN}  xcode-select --install${NC}"
            echo -e "${CYAN}  or: brew install git${NC}"
            ;;
    esac
    echo ""
    exit 1
fi
echo -e "${GREEN}✓ Git found${NC}"

# Check for conda
conda_path=$(find_conda 2>/dev/null || echo "")

if [ -z "$conda_path" ]; then
    echo -e "${YELLOW}✗ Conda/Miniconda not found${NC}"
    echo ""
    echo -e "${NC}Miniconda is required for the VFX Pipeline.${NC}"
    echo -e "${GRAY}It provides isolated Python environments with GPU support.${NC}"
    echo ""

    read -p "Would you like to install Miniconda automatically? (Y/n): " -n 1 -r < /dev/tty
    echo

    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo -e "${YELLOW}Manual installation required:${NC}"
        echo -e "${CYAN}  1. Download from: https://docs.conda.io/en/latest/miniconda.html${NC}"
        echo -e "${CYAN}  2. Run: bash Miniconda3-latest-*.sh${NC}"
        echo -e "${CYAN}  3. Restart your terminal${NC}"
        echo -e "${CYAN}  4. Re-run this bootstrap script${NC}"
        echo ""
        exit 1
    fi

    echo ""
    if ! install_miniconda; then
        exit 1
    fi

    conda_path=$(find_conda 2>/dev/null || echo "")
    if [ -z "$conda_path" ]; then
        echo -e "${RED}✗ Could not find conda after installation${NC}"
        echo -e "  Please restart your terminal and try again"
        exit 1
    fi

    initialize_conda_shell "$conda_path"
else
    echo -e "${GREEN}✓ Conda found: $conda_path${NC}"
fi

# Verify conda works
conda_version=$("$conda_path" --version 2>&1 || echo "unknown")
echo -e "${GREEN}✓ $conda_version${NC}"
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

print_banner "Launching Installation Wizard"

# Get conda base directory and find python
conda_base=$(dirname $(dirname "$conda_path"))
python_path="$conda_base/bin/python"

if [ -x "$python_path" ]; then
    "$python_path" scripts/install_wizard.py "$@"
else
    # Fall back to conda run
    "$conda_path" run -n base python scripts/install_wizard.py "$@"
fi

wizard_exit=$?

if [ $wizard_exit -ne 0 ]; then
    print_banner "Installation Failed!" "$RED"
    echo -e "${YELLOW}Check the error messages above and try again.${NC}"
    exit $wizard_exit
fi

print_banner "Installation Complete!" "$GREEN"
echo "Next steps:"
echo "  1. Restart your terminal (if conda was just installed)"
echo "  2. Activate environment: conda activate vfx-pipeline"
echo "  3. Or use: source $INSTALL_DIR/.vfx_pipeline/activate.sh"
echo "  4. Run: python scripts/run_pipeline.py --help"
echo ""
