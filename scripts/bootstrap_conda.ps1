# VFX Pipeline Bootstrap Script - Conda Edition (Windows)
# Downloads and runs the installation wizard
#
# Usage:
#   irm https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.ps1 | iex

$ErrorActionPreference = "Stop"

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$REPO_URL = "https://github.com/kleer001/shot-gopher.git"
$INSTALL_DIR = Join-Path (Get-Location) "shot-gopher"
$MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$MINICONDA_INSTALLER = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"
$GIT_INSTALLER = Join-Path $env:TEMP "Git-installer.exe"

function Write-Banner {
    param([string]$Text, [string]$Color = "Cyan")
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor $Color
    Write-Host "  $Text" -ForegroundColor $Color
    Write-Host ("=" * 60) -ForegroundColor $Color
    Write-Host ""
}

function Install-Git {
    $installed = $false

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Winget detected. Attempting to install Git via winget..." -ForegroundColor Yellow
        try {
            $process = Start-Process -FilePath "winget" -ArgumentList @(
                "install", "--id", "Git.Git", "-e", "--source", "winget", "--accept-package-agreements", "--accept-source-agreements"
            ) -Wait -PassThru -NoNewWindow

            if ($process.ExitCode -eq 0 -or $process.ExitCode -eq -1978335189) {
                Write-Host "OK Git installed successfully via winget" -ForegroundColor Green
                $installed = $true
            } else {
                Write-Host "! Winget returned exit code $($process.ExitCode), trying direct download..." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "! Winget failed: $_, trying direct download..." -ForegroundColor Yellow
        }
    }

    if (-not $installed) {
        Write-Host "Fetching latest Git release info..." -ForegroundColor Yellow
        $gitUrl = $null

        $arch = if ([Environment]::Is64BitOperatingSystem) { "64-bit" } else { "32-bit" }

        try {
            $release = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest" -UseBasicParsing
            $asset = $release.assets | Where-Object { $_.name -match "^Git-.*-$arch\.exe$" } | Select-Object -First 1
            if ($asset) {
                $gitUrl = $asset.browser_download_url
                Write-Host "  Found: $($asset.name)" -ForegroundColor Gray
            }
        } catch {
            Write-Host "! Could not fetch release info: $_" -ForegroundColor Yellow
        }

        if (-not $gitUrl) {
            Write-Host "X Could not determine Git download URL" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install manually from: https://git-scm.com/download/win" -ForegroundColor Yellow
            return $false
        }

        Write-Host "Downloading Git installer..." -ForegroundColor Yellow

        try {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $gitUrl -OutFile $GIT_INSTALLER -UseBasicParsing
            $ProgressPreference = 'Continue'
        } catch {
            Write-Host "X Failed to download Git installer" -ForegroundColor Red
            Write-Host "  Error: $_" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install manually from: https://git-scm.com/download/win" -ForegroundColor Yellow
            return $false
        }

        if (-not (Test-Path $GIT_INSTALLER) -or (Get-Item $GIT_INSTALLER).Length -lt 1000000) {
            Write-Host "X Downloaded file appears invalid" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install manually from: https://git-scm.com/download/win" -ForegroundColor Yellow
            Remove-Item $GIT_INSTALLER -Force -ErrorAction SilentlyContinue
            return $false
        }

        Write-Host "Installing Git (this may take a minute)..." -ForegroundColor Yellow
        Write-Host "  Installation will be silent - please wait..." -ForegroundColor Gray

        try {
            $process = Start-Process -FilePath $GIT_INSTALLER -ArgumentList @(
                "/VERYSILENT",
                "/NORESTART",
                "/NOCANCEL",
                "/SP-",
                "/CLOSEAPPLICATIONS",
                "/RESTARTAPPLICATIONS",
                "/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh"
            ) -Wait -PassThru -NoNewWindow

            if ($process.ExitCode -ne 0) {
                throw "Installer exited with code $($process.ExitCode)"
            }
        } catch {
            Write-Host "X Git installation failed" -ForegroundColor Red
            Write-Host "  Error: $_" -ForegroundColor Red
            Remove-Item $GIT_INSTALLER -Force -ErrorAction SilentlyContinue
            return $false
        }

        Remove-Item $GIT_INSTALLER -Force -ErrorAction SilentlyContinue
        Write-Host "OK Git installed successfully" -ForegroundColor Green
    }

    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    $gitPaths = @(
        "C:\Program Files\Git\cmd",
        "C:\Program Files\Git\bin",
        "C:\Program Files (x86)\Git\cmd",
        "C:\Program Files (x86)\Git\bin"
    )
    foreach ($gitPath in $gitPaths) {
        if ((Test-Path $gitPath) -and ($env:Path -notlike "*$gitPath*")) {
            $env:Path = "$gitPath;$env:Path"
        }
    }

    return $true
}

function Test-CondaInstalled {
    # Check multiple locations for conda
    $condaLocations = @(
        (Get-Command conda -ErrorAction SilentlyContinue),
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\AppData\Local\miniconda3\Scripts\conda.exe"
    )

    foreach ($loc in $condaLocations) {
        if ($loc -and (Test-Path $loc -ErrorAction SilentlyContinue)) {
            return $loc
        }
    }
    return $null
}

function Get-CondaPath {
    $conda = Test-CondaInstalled
    if ($conda) {
        if ($conda -is [System.Management.Automation.CommandInfo]) {
            return $conda.Source
        }
        return $conda
    }
    return $null
}

function Install-Miniconda {
    Write-Host "Downloading Miniconda installer..." -ForegroundColor Yellow

    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $MINICONDA_URL -OutFile $MINICONDA_INSTALLER -UseBasicParsing
        $ProgressPreference = 'Continue'
    } catch {
        Write-Host "X Failed to download Miniconda" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install manually from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
        return $false
    }

    Write-Host "Installing Miniconda (this may take a few minutes)..." -ForegroundColor Yellow
    Write-Host "  Installation will be silent - please wait..." -ForegroundColor Gray

    $installPath = "$env:USERPROFILE\miniconda3"

    try {
        $process = Start-Process -FilePath $MINICONDA_INSTALLER -ArgumentList @(
            "/InstallationType=JustMe",
            "/AddToPath=1",
            "/RegisterPython=1",
            "/S",
            "/D=$installPath"
        ) -Wait -PassThru -NoNewWindow

        if ($process.ExitCode -ne 0) {
            throw "Installer exited with code $($process.ExitCode)"
        }
    } catch {
        Write-Host "X Miniconda installation failed" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
        return $false
    }

    # Clean up installer
    Remove-Item $MINICONDA_INSTALLER -Force -ErrorAction SilentlyContinue

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    # Also add conda to current session path
    $env:Path = "$installPath;$installPath\Scripts;$installPath\Library\bin;$env:Path"

    Write-Host "OK Miniconda installed successfully" -ForegroundColor Green
    return $true
}

function Initialize-CondaShell {
    param([string]$CondaPath)

    $condaDir = Split-Path (Split-Path $CondaPath -Parent) -Parent

    Write-Host "Initializing conda for PowerShell..." -ForegroundColor Yellow

    try {
        & $CondaPath init powershell 2>&1 | Out-Null
        Write-Host "OK Conda initialized for PowerShell" -ForegroundColor Green
        Write-Host ""
        Write-Host "NOTE: You may need to restart PowerShell for conda to work properly." -ForegroundColor Yellow
    } catch {
        Write-Host "! Could not auto-initialize conda" -ForegroundColor Yellow
        Write-Host "  Run manually: conda init powershell" -ForegroundColor Gray
    }
}

function Install-VFXPipeline {
    Write-Banner "VFX Pipeline - Automated Installer"

    # Check for git
    Write-Host "Checking prerequisites..."

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "X Git is not installed" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Git is required for the VFX Pipeline." -ForegroundColor White
        Write-Host "It is used to download and update the pipeline code." -ForegroundColor Gray
        Write-Host ""

        $response = Read-Host "Would you like to install Git automatically? (Y/n)"

        if ($response -match "^[Nn]$") {
            Write-Host ""
            Write-Host "Manual installation required:" -ForegroundColor Yellow
            Write-Host "  1. Run: winget install Git.Git" -ForegroundColor Cyan
            Write-Host "  2. Or download from: https://git-scm.com/download/win" -ForegroundColor Cyan
            Write-Host "  3. Restart PowerShell after installation" -ForegroundColor Cyan
            Write-Host "  4. Re-run this bootstrap script" -ForegroundColor Cyan
            Write-Host ""
            return 1
        }

        Write-Host ""
        if (-not (Install-Git)) {
            return 1
        }

        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Write-Host "X Could not find git after installation" -ForegroundColor Red
            Write-Host "  Please restart PowerShell and try again" -ForegroundColor Yellow
            return 1
        }
    }
    Write-Host "OK Git found" -ForegroundColor Green

    # Check for conda
    $condaPath = Get-CondaPath

    if (-not $condaPath) {
        Write-Host "X Conda/Miniconda not found" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Miniconda is required for the VFX Pipeline." -ForegroundColor White
        Write-Host "It provides isolated Python environments with GPU support." -ForegroundColor Gray
        Write-Host ""

        $response = Read-Host "Would you like to install Miniconda automatically? (Y/n)"

        if ($response -match "^[Nn]$") {
            Write-Host ""
            Write-Host "Manual installation required:" -ForegroundColor Yellow
            Write-Host "  1. Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Cyan
            Write-Host "  2. Run the installer (use defaults)" -ForegroundColor Cyan
            Write-Host "  3. Restart PowerShell" -ForegroundColor Cyan
            Write-Host "  4. Re-run this bootstrap script" -ForegroundColor Cyan
            Write-Host ""
            return 1
        }

        Write-Host ""
        if (-not (Install-Miniconda)) {
            return 1
        }

        $condaPath = Get-CondaPath
        if (-not $condaPath) {
            Write-Host "X Could not find conda after installation" -ForegroundColor Red
            Write-Host "  Please restart PowerShell and try again" -ForegroundColor Yellow
            return 1
        }

        Initialize-CondaShell -CondaPath $condaPath
    } else {
        Write-Host "OK Conda found: $condaPath" -ForegroundColor Green
    }

    # Verify conda works
    try {
        $condaVersion = & $condaPath --version 2>&1
        Write-Host "OK $condaVersion" -ForegroundColor Green
    } catch {
        Write-Host "! Could not verify conda version" -ForegroundColor Yellow
    }

    Write-Host ""

    # Clone or update repository
    if (Test-Path $INSTALL_DIR) {
        Write-Host "Directory $INSTALL_DIR already exists."
        $response = Read-Host "Update existing installation? (y/N)"
        if ($response -match "^[Yy]$") {
            Write-Host "Updating repository..."
            Push-Location $INSTALL_DIR
            git pull
        } else {
            Write-Host "Using existing installation at $INSTALL_DIR"
            Push-Location $INSTALL_DIR
        }
    } else {
        Write-Host "Cloning repository to $INSTALL_DIR..."
        git clone $REPO_URL $INSTALL_DIR
        Push-Location $INSTALL_DIR
    }

    Write-Banner "Launching Installation Wizard"

    # Run the wizard using the conda we found/installed
    $pythonPath = Join-Path (Split-Path (Split-Path $condaPath -Parent) -Parent) "python.exe"

    if (Test-Path $pythonPath) {
        & $pythonPath scripts/install_wizard.py $args
    } else {
        # Fall back to conda run
        & $condaPath run -n base python scripts/install_wizard.py $args
    }

    $wizardExitCode = $LASTEXITCODE

    Pop-Location

    if ($wizardExitCode -ne 0) {
        Write-Banner "Installation Failed!" "Red"
        Write-Host "Check the error messages above and try again." -ForegroundColor Yellow
        return $wizardExitCode
    }

    Write-Banner "Installation Complete!" "Green"
    Write-Host "Next steps:"
    Write-Host "  1. Restart PowerShell (if conda was just installed)"
    Write-Host "  2. Activate environment: conda activate vfx-pipeline"
    Write-Host "  3. Or use: . $INSTALL_DIR\.vfx_pipeline\activate.ps1"
    Write-Host "  4. Run: python scripts/run_pipeline.py --help"
    Write-Host ""

    return 0
}

$script:exitCode = 0
try {
    $script:exitCode = Install-VFXPipeline
} catch {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
    Write-Host "  Unexpected Error!" -ForegroundColor Red
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    $script:exitCode = 1
}

Write-Host ""
Read-Host "Press Enter to close"
