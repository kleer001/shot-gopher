#!/bin/bash
# VFX Pipeline Bootstrap Script - Docker Edition (with tests)
# Downloads and runs the Docker installation wizard with test pipeline
#
# Usage (Linux or WSL2):
#   curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/scripts/bootstrap_docker_test.sh | bash
#   or
#   wget -qO- https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/scripts/bootstrap_docker_test.sh | bash

set -e

REPO_URL="https://github.com/kleer001/comfyui_ingest.git"
INSTALL_DIR="$(pwd)/comfyui_ingest"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

is_wsl2() {
    if [ -f /proc/version ]; then
        grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null && return 0
    fi
    return 1
}

check_nvidia_driver() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi &> /dev/null && return 0
    fi
    return 1
}

check_docker_installed() {
    if ! command -v docker &> /dev/null; then
        return 1
    fi
    docker info &> /dev/null && return 0
    return 1
}

print_header "VFX Pipeline - Docker Automated Installer (with tests)"

print_info "Checking basic prerequisites..."

if ! command -v git &> /dev/null; then
    print_error "git is not installed"
    echo "   Install with: sudo apt install git"
    exit 1
fi
print_success "git is installed"

if ! command -v python3 &> /dev/null; then
    print_error "python3 is not installed"
    echo "   Install with: sudo apt install python3"
    exit 1
fi
print_success "python3 is installed"

if ! command -v curl &> /dev/null; then
    print_error "curl is not installed"
    echo "   Install with: sudo apt install curl"
    exit 1
fi
print_success "curl is installed"

print_info "Checking NVIDIA driver..."
if check_nvidia_driver; then
    print_success "NVIDIA driver is working"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 | while read line; do
        echo "   GPU: $line"
    done
else
    print_error "NVIDIA driver not found or not working"
    echo ""
    if is_wsl2; then
        echo "   For WSL2, install NVIDIA driver on Windows:"
    else
        echo "   Install NVIDIA driver for your GPU:"
    fi
    echo "   https://www.nvidia.com/Download/index.aspx"
    echo ""
    exit 1
fi

print_info "Checking Docker..."
if check_docker_installed; then
    print_success "Docker is installed and running"
else
    print_error "Docker is not installed or not running"
    echo ""
    if is_wsl2; then
        echo "   For WSL2, install Docker Desktop for Windows:"
        echo "   https://www.docker.com/products/docker-desktop"
        echo ""
        echo "   Enable WSL2 backend in Docker Desktop settings"
    else
        echo "   Install Docker:"
        echo "   curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "   sudo sh get-docker.sh"
        echo "   sudo usermod -aG docker \$USER"
        echo "   newgrp docker"
    fi
    echo ""
    exit 1
fi

echo ""

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR already exists."
    read -p "Update existing installation? (y/N): " -n 1 -r < /dev/tty
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Updating repository..."
        cd "$INSTALL_DIR"
        git pull
    else
        print_info "Using existing installation at $INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
else
    print_info "Cloning repository to $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

print_header "Launching Docker Installation Wizard (with tests)"

python3 scripts/install_wizard_docker.py --test "$@"

print_header "Installation Complete!"

echo "Next steps:"
echo "  1. Copy your video: cp video.mp4 ~/VFX-Projects/"
echo "  2. Run pipeline: ./scripts/run_docker.sh --input /workspace/projects/video.mp4 --name MyProject"
echo "  3. Read: $INSTALL_DIR/QUICKSTART.md"
echo ""
