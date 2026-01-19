# VFX Pipeline Bootstrap Script - Docker Edition (Windows)
# Sets up WSL2 and launches the Docker installation wizard
#
# Usage (PowerShell as Administrator):
#   irm https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.ps1 | iex
#   or
#   Invoke-WebRequest -Uri https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.ps1 -UseBasicParsing | Invoke-Expression

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  VFX Pipeline - Docker Automated Installer (Windows)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠️  This script requires Administrator privileges" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please right-click PowerShell and select 'Run as Administrator', then run:" -ForegroundColor Yellow
    Write-Host "  irm https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.ps1 | iex" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Running as Administrator" -ForegroundColor Green
Write-Host ""

# Check WSL2
Write-Host "Checking WSL2..." -ForegroundColor Cyan

$wslInstalled = $false
try {
    $wslVersion = wsl --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $wslInstalled = $true
        Write-Host "✓ WSL is installed" -ForegroundColor Green
    }
} catch {
    $wslInstalled = $false
}

if (-not $wslInstalled) {
    Write-Host "⚠️  WSL2 is not installed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Installing WSL2..." -ForegroundColor Cyan

    try {
        wsl --install -d Ubuntu-22.04
        Write-Host ""
        Write-Host "✓ WSL2 installation initiated" -ForegroundColor Green
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  RESTART REQUIRED" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Please restart your computer, then:" -ForegroundColor Yellow
        Write-Host "  1. Open Ubuntu-22.04 from Start menu" -ForegroundColor White
        Write-Host "  2. Create username and password" -ForegroundColor White
        Write-Host "  3. Run in Ubuntu terminal:" -ForegroundColor White
        Write-Host "     curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.sh | bash" -ForegroundColor Green
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 0
    } catch {
        Write-Host "❌ Failed to install WSL2" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install manually:" -ForegroundColor Yellow
        Write-Host "  1. Open PowerShell as Administrator" -ForegroundColor White
        Write-Host "  2. Run: wsl --install -d Ubuntu-22.04" -ForegroundColor White
        Write-Host "  3. Restart computer" -ForegroundColor White
        Write-Host "  4. Open Ubuntu and complete setup" -ForegroundColor White
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if Ubuntu is installed
Write-Host "Checking for Ubuntu distribution..." -ForegroundColor Cyan

$ubuntuInstalled = $false
try {
    $wslList = wsl --list --quiet
    if ($wslList -match "Ubuntu") {
        $ubuntuInstalled = $true
        Write-Host "✓ Ubuntu is installed in WSL" -ForegroundColor Green
    }
} catch {
    $ubuntuInstalled = $false
}

if (-not $ubuntuInstalled) {
    Write-Host "⚠️  Ubuntu is not installed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Installing Ubuntu 22.04..." -ForegroundColor Cyan

    try {
        wsl --install -d Ubuntu-22.04
        Write-Host ""
        Write-Host "✓ Ubuntu installation initiated" -ForegroundColor Green
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  COMPLETE UBUNTU SETUP" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Ubuntu will open automatically. Please:" -ForegroundColor Yellow
        Write-Host "  1. Create a username and password" -ForegroundColor White
        Write-Host "  2. Wait for setup to complete" -ForegroundColor White
        Write-Host "  3. Then run in Ubuntu terminal:" -ForegroundColor White
        Write-Host "     curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.sh | bash" -ForegroundColor Green
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 0
    } catch {
        Write-Host "❌ Failed to install Ubuntu" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install manually from Microsoft Store:" -ForegroundColor Yellow
        Write-Host "  https://www.microsoft.com/store/productId/9PN20MSR04DW" -ForegroundColor White
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check Docker Desktop
Write-Host "Checking Docker Desktop..." -ForegroundColor Cyan

$dockerInstalled = $false
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $dockerInstalled = $true
    Write-Host "✓ Docker Desktop is installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Docker Desktop is not installed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install Docker Desktop for Windows:" -ForegroundColor Yellow
    Write-Host "  1. Download: https://www.docker.com/products/docker-desktop" -ForegroundColor White
    Write-Host "  2. Install with WSL2 backend enabled" -ForegroundColor White
    Write-Host "  3. In Docker Desktop settings:" -ForegroundColor White
    Write-Host "     - Enable 'Use the WSL 2 based engine'" -ForegroundColor White
    Write-Host "     - Under 'Resources > WSL Integration', enable Ubuntu" -ForegroundColor White
    Write-Host "  4. Restart Docker Desktop" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to continue after installing Docker Desktop"
}

# Check NVIDIA CUDA on WSL
Write-Host ""
Write-Host "Checking NVIDIA CUDA on WSL..." -ForegroundColor Cyan

try {
    $nvidiaCheck = wsl nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NVIDIA CUDA is available in WSL" -ForegroundColor Green
    } else {
        Write-Host "⚠️  NVIDIA CUDA not detected in WSL" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Please install NVIDIA CUDA on WSL2:" -ForegroundColor Yellow
        Write-Host "  https://docs.nvidia.com/cuda/wsl-user-guide/index.html" -ForegroundColor White
        Write-Host ""
        Write-Host "Key steps:" -ForegroundColor Yellow
        Write-Host "  1. Install NVIDIA Driver on Windows (version 450.80.02 or later)" -ForegroundColor White
        Write-Host "  2. No need to install CUDA toolkit in WSL - uses Windows driver" -ForegroundColor White
        Write-Host "  3. Verify in WSL with: nvidia-smi" -ForegroundColor White
        Write-Host ""
    }
} catch {
    Write-Host "⚠️  Could not check NVIDIA CUDA (WSL may not be running)" -ForegroundColor Yellow
    Write-Host ""
}

# All prerequisites ready - launch wizard in WSL
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Launching Installation in WSL Ubuntu" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting WSL Ubuntu terminal..." -ForegroundColor Cyan
Write-Host ""

# Launch the bootstrap script in WSL
$bootstrapCmd = "curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.sh | bash"

try {
    wsl -d Ubuntu-22.04 bash -c $bootstrapCmd

    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Access WSL: wsl -d Ubuntu-22.04" -ForegroundColor White
    Write-Host "  2. Navigate to: cd ~/comfyui_ingest" -ForegroundColor White
    Write-Host "  3. Run pipeline: ./scripts/run_docker.sh --name MyProject /workspace/projects/video.mp4" -ForegroundColor White
    Write-Host ""
    Write-Host "Projects are saved in WSL at: ~/VFX-Projects/" -ForegroundColor Cyan
    Write-Host "Access from Windows at: \\wsl$\Ubuntu-22.04\home\<username>\VFX-Projects\" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "❌ Installation failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "You can run the installer manually in WSL Ubuntu terminal:" -ForegroundColor Yellow
    Write-Host "  1. Open Ubuntu-22.04 from Start menu" -ForegroundColor White
    Write-Host "  2. Run: curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap-docker.sh | bash" -ForegroundColor Green
    Write-Host ""
}

Read-Host "Press Enter to exit"
