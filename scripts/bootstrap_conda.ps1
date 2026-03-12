# VFX Pipeline Bootstrap Script - Conda Edition (Windows)
# Downloads and runs the installation wizard
#
# Usage:
#   irm https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_conda.ps1 | iex

# NOTE: We intentionally do NOT set $ErrorActionPreference = "Stop" globally.
# Native commands (git, conda) write progress to stderr, which PowerShell
# converts to non-terminating errors. With "Stop", these become terminating
# errors that abort the script on perfectly normal output like
# "Cloning into 'shot-gopher'...". Instead, we use explicit error checking
# ($LASTEXITCODE, try/catch) where needed.

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$REPO_URL = "https://github.com/kleer001/shot-gopher.git"

# Capture starting directory FIRST - this is where the user wants to install
$STARTING_DIR = (Get-Location).Path
$INSTALL_DIR = Join-Path $STARTING_DIR "shot-gopher"

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

function Read-UserInput {
    <#
    .SYNOPSIS
    Read user input, working correctly even when stdin is piped (irm | iex).
    Falls back to [Console]::ReadLine() when stdin is not interactive.
    Returns empty string if no console is available (fully non-interactive).
    #>
    param([string]$Prompt, [string]$Default = "")

    Write-Host $Prompt -NoNewline

    try {
        if ([Console]::IsInputRedirected) {
            $result = [Console]::ReadLine()
            if ($null -eq $result) { return $Default }
            return $result
        }
    } catch {
        # [Console]::IsInputRedirected may not be available on PS 5.1
    }

    try {
        $result = Read-Host
        return $result
    } catch {
        return $Default
    }
}

function Invoke-NativeCommand {
    <#
    .SYNOPSIS
    Run a native command (git, conda, etc.) without PowerShell's stderr
    interference. Returns $true if exit code is 0, $false otherwise.
    Output is displayed to the console normally.
    #>
    param(
        [string]$Command,
        [string[]]$Arguments,
        [switch]$Silent
    )

    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        if ($Silent) {
            & $Command @Arguments 2>&1 | Out-Null
        } else {
            & $Command @Arguments 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    Write-Host $_.ToString()
                } else {
                    Write-Host $_
                }
            }
        }
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $savedEAP
    }
}

function Install-Git {
    $installed = $false

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Winget detected. Attempting to install Git via winget..." -ForegroundColor Yellow

        # Save current directory before winget (some package managers change working dir)
        $savedDir = (Get-Location).Path

        try {
            # Pre-update winget sources to avoid stale source errors on fresh installs
            Start-Process -FilePath "winget" -ArgumentList @("source", "update") -Wait -NoNewWindow 2>$null

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
        } finally {
            # Restore directory in case winget changed it
            Set-Location $savedDir
        }
    }

    if (-not $installed) {
        Write-Host "Fetching latest Git release info..." -ForegroundColor Yellow
        $gitUrl = $null

        $arch = if ([Environment]::Is64BitOperatingSystem) { "64-bit" } else { "32-bit" }

        try {
            $release = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest"
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

        # Save current directory before installer (some installers change working dir)
        $savedDir = (Get-Location).Path

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
        } finally {
            # Restore directory in case installer changed it
            Set-Location $savedDir
        }

        Remove-Item $GIT_INSTALLER -Force -ErrorAction SilentlyContinue
        Write-Host "OK Git installed successfully" -ForegroundColor Green
    }

    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($machinePath, $userPath) | Where-Object { $_ }) -join ";"

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
    # Check if conda is available (returns path or command name)
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        return $condaCmd.Source
    }

    # Check common installation locations
    $condaLocations = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )

    foreach ($loc in $condaLocations) {
        if (Test-Path $loc -ErrorAction SilentlyContinue) {
            return $loc
        }
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

    # Save current directory before installer (some installers change working dir)
    $savedDir = (Get-Location).Path

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
    } finally {
        # Restore directory in case installer changed it
        Set-Location $savedDir
    }

    # Clean up installer
    Remove-Item $MINICONDA_INSTALLER -Force -ErrorAction SilentlyContinue

    # Refresh PATH
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($machinePath, $userPath) | Where-Object { $_ }) -join ";"

    # Also add conda to current session path
    $env:Path = "$installPath;$installPath\Scripts;$installPath\Library\bin;$env:Path"

    Write-Host "OK Miniconda installed successfully" -ForegroundColor Green
    return $true
}

function Initialize-CondaShell {
    param([string]$CondaPath)

    # conda init powershell creates a profile.ps1 that requires script execution.
    # If the execution policy is Restricted (Windows default), that profile will
    # error on every PowerShell session. Fix this BEFORE running conda init.
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    if ($policy -eq "Restricted" -or $policy -eq "Undefined") {
        Write-Host "Setting PowerShell execution policy to RemoteSigned (CurrentUser scope)..." -ForegroundColor Yellow
        try {
            Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
            Write-Host "OK Execution policy set to RemoteSigned" -ForegroundColor Green
        } catch {
            Write-Host "! Could not set execution policy automatically" -ForegroundColor Yellow
            Write-Host "  Run manually (as Administrator):" -ForegroundColor Gray
            Write-Host "    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  Without this, PowerShell will show an error on every startup" -ForegroundColor Gray
            Write-Host "  after conda init creates a profile script." -ForegroundColor Gray
        }
    }

    Write-Host "Initializing conda for PowerShell..." -ForegroundColor Yellow

    Invoke-NativeCommand -Command $CondaPath -Arguments @("init", "powershell") -Silent

    Write-Host "OK Conda initialized for PowerShell" -ForegroundColor Green
    Write-Host ""
    Write-Host "NOTE: You may need to restart PowerShell for conda to work properly." -ForegroundColor Yellow
}

function Install-VFXPipeline {
    Write-Banner "VFX Pipeline - Automated Installer"

    # Verify we can write to the install directory before proceeding
    $testFile = Join-Path $STARTING_DIR ".bootstrap_write_test"
    try {
        [IO.File]::WriteAllText($testFile, "test")
        Remove-Item $testFile -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Host ""
        Write-Host "X Cannot install here: $STARTING_DIR" -ForegroundColor Red
        Write-Host "  No write permission. Run PowerShell from a directory you own." -ForegroundColor Gray
        Write-Host ""
        return 1
    }

    # Check for git
    Write-Host "Checking prerequisites..."

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "X Git is not installed" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Git is required for the VFX Pipeline." -ForegroundColor White
        Write-Host "It is used to download and update the pipeline code." -ForegroundColor Gray
        Write-Host ""

        $response = Read-UserInput "Would you like to install Git automatically? (Y/n): "

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
    $condaPath = Test-CondaInstalled

    if (-not $condaPath) {
        Write-Host "X Conda/Miniconda not found" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Miniconda is required for the VFX Pipeline." -ForegroundColor White
        Write-Host "It provides isolated Python environments with GPU support." -ForegroundColor Gray
        Write-Host ""

        $response = Read-UserInput "Would you like to install Miniconda automatically? (Y/n): "

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

        $condaPath = Test-CondaInstalled
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
    $condaVersion = & $condaPath --version 2>&1 | Out-String
    if ($condaVersion) {
        Write-Host "OK $($condaVersion.Trim())" -ForegroundColor Green
    } else {
        Write-Host "! Could not verify conda version" -ForegroundColor Yellow
    }

    Write-Host ""

    # Clone or update repository
    $isValidRepo = (Test-Path $INSTALL_DIR) -and (Test-Path (Join-Path $INSTALL_DIR ".git"))

    if ($isValidRepo) {
        Write-Host "Directory $INSTALL_DIR already exists (valid git repo)."
        $response = Read-UserInput "Update existing installation? (y/N): "
        if ($response -match "^[Yy]$") {
            Write-Host "Updating repository..."
            Invoke-NativeCommand -Command "git" -Arguments @("-C", $INSTALL_DIR, "pull")
            if (-not $?) {
                Write-Host "! Git pull failed, continuing with existing code..." -ForegroundColor Yellow
            }
        } else {
            Write-Host "Using existing installation at $INSTALL_DIR"
        }
    } else {
        if (Test-Path $INSTALL_DIR) {
            Write-Host "Directory $INSTALL_DIR exists but is not a valid git repository." -ForegroundColor Yellow
            Write-Host "Removing and re-cloning..." -ForegroundColor Yellow
            Remove-Item $INSTALL_DIR -Recurse -Force -ErrorAction SilentlyContinue
            # Windows filesystem needs time to release directory handles
            $retries = 0
            while ((Test-Path $INSTALL_DIR) -and $retries -lt 10) {
                Start-Sleep -Milliseconds 500
                Remove-Item $INSTALL_DIR -Recurse -Force -ErrorAction SilentlyContinue
                $retries++
            }
            if (Test-Path $INSTALL_DIR) {
                Write-Host "X Could not remove $INSTALL_DIR" -ForegroundColor Red
                Write-Host "  Close any programs using that folder, then try again." -ForegroundColor Yellow
                return 1
            }
        }
        Write-Host "Cloning repository to $INSTALL_DIR..."
        $cloneOk = Invoke-NativeCommand -Command "git" -Arguments @("clone", $REPO_URL, $INSTALL_DIR)
        if (-not $cloneOk) {
            Write-Host "X Failed to clone repository" -ForegroundColor Red
            return 1
        }
    }

    Write-Banner "Launching Installation Wizard"

    # Always use --yolo: `conda run` buffers all stdout/stderr until the
    # subprocess exits, so interactive prompts never reach the user's
    # terminal even when stdin appears interactive.  The IsInputRedirected
    # check cannot detect this because conda run does not redirect stdin —
    # it just swallows the output.  Non-interactive mode is the only
    # reliable path through `conda run`.
    $wizardArgs = @("run", "--no-capture-output", "-n", "base", "python", "scripts/install_wizard.py", "--yolo")

    Push-Location $INSTALL_DIR
    Invoke-NativeCommand -Command $condaPath -Arguments $wizardArgs
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
    Write-Host ("=" * 60) -ForegroundColor Red
    Write-Host "  Unexpected Error!" -ForegroundColor Red
    Write-Host ("=" * 60) -ForegroundColor Red
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    $script:exitCode = 1
}

# Only prompt "Press Enter" when running interactively.
# When piped via irm|iex or run over SSH, there is no console to read from.
$isInteractive = $false
try {
    $isInteractive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
} catch {
    # PS 5.1 may not support IsInputRedirected; fall back to checking stdin
    try {
        $isInteractive = [Console]::In -ne $null -and [Environment]::UserInteractive
    } catch {
        $isInteractive = $false
    }
}

if ($isInteractive) {
    Write-Host ""
    Read-Host "Press Enter to close"
}
