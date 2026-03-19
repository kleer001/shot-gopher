<#
.SYNOPSIS
    VFX Pipeline Conda Environment Activation Script (Windows)

.DESCRIPTION
    Ensures the vfx-pipeline conda environment is active in the current shell.
    Must be dot-sourced to affect the calling session.

.PARAMETER Check
    Check if the environment is active without activating it.

.PARAMETER Quiet
    Suppress informational output.

.EXAMPLE
    . .\activate_env.ps1

.EXAMPLE
    . .\activate_env.ps1 -Check

.EXAMPLE
    # In other scripts (located one level deep, e.g. src\):
    . (Join-Path $PSScriptRoot "..\activate_env.ps1")
#>

[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Quiet
)

# Configuration - single source of truth
$REPO_ROOT     = $PSScriptRoot
$VFX_ENV_PREFIX = Join-Path $REPO_ROOT ".vfx_pipeline\envs\vfx-pipeline"
$VFX_ENV_NAME  = "vfx-pipeline"

# Must be dot-sourced so conda activate modifies the calling shell's environment
$DotSourced = $MyInvocation.InvocationName -eq '.'
if (-not $DotSourced) {
    Write-Host "This script must be dot-sourced, not executed directly." -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  . .\activate_env.ps1"
    Write-Host ""
    Write-Host "Or add to your PowerShell profile:"
    Write-Host "  echo `". $($MyInvocation.MyCommand.Path)`" >> `$PROFILE"
    exit 1
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-VfxInfo  { param([string]$Msg) if (-not $Quiet) { Write-Host "[vfx-env] $Msg" -ForegroundColor Green  } }
function Write-VfxWarn  { param([string]$Msg) Write-Host "[vfx-env] $Msg" -ForegroundColor Yellow }
function Write-VfxError { param([string]$Msg) Write-Host "[vfx-env] $Msg" -ForegroundColor Red    }

function Show-WrongEnvWarning {
    param([string]$CurrentEnv)
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "  ║              WRONG CONDA ENVIRONMENT ACTIVE                  ║" -ForegroundColor Red
    Write-Host "  ╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    if ($CurrentEnv) {
        Write-Host "    Currently active: " -NoNewline; Write-Host "'$CurrentEnv'" -ForegroundColor Yellow
        Write-Host "    Required:         " -NoNewline; Write-Host "'$VFX_ENV_NAME'" -ForegroundColor Green
    } else {
        Write-Host "    No conda environment is currently active."
        Write-Host "    Required: " -NoNewline; Write-Host "'$VFX_ENV_NAME'" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "    To fix this, run:"
    Write-Host "      conda activate `"$VFX_ENV_PREFIX`"" -ForegroundColor Cyan
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Check current environment
# ---------------------------------------------------------------------------

function Test-InVfxEnv {
    $prefix = $env:CONDA_PREFIX
    if (-not $prefix) { return $false }
    # Normalize both sides before comparing (trailing backslash, case)
    $a = $prefix.TrimEnd('\').ToLower()
    $b = $VFX_ENV_PREFIX.TrimEnd('\').ToLower()
    return $a -eq $b
}

# ---------------------------------------------------------------------------
# Locate conda and initialize hooks in the current session if needed
# ---------------------------------------------------------------------------

function Find-CondaExe {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) { return $condaCmd.Source }

    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\mambaforge\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniforge3\Scripts\conda.exe"
    )

    foreach ($path in $candidates) {
        if (Test-Path $path -ErrorAction SilentlyContinue) { return $path }
    }

    return $null
}

function Initialize-CondaHooks {
    param([string]$CondaExe)

    # Derive conda root from the Scripts\conda.exe path
    $condaRoot = Split-Path (Split-Path $CondaExe -Parent) -Parent
    $hookScript = Join-Path $condaRoot "shell\condabin\conda-hook.ps1"

    if (-not (Test-Path $hookScript -ErrorAction SilentlyContinue)) {
        return $false
    }

    # Dot-source the hook so 'conda' function is available in this session
    . $hookScript
    return ($null -ne (Get-Command conda -ErrorAction SilentlyContinue))
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if (Test-InVfxEnv) {
    Write-VfxInfo "Environment '$VFX_ENV_NAME' is already active"
    return
}

if ($Check) {
    Show-WrongEnvWarning -CurrentEnv $env:CONDA_DEFAULT_ENV
    return
}

# Ensure conda is available in this session
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $condaExe = Find-CondaExe
    if (-not $condaExe) {
        Write-VfxError "Could not find conda. Please install Miniconda first."
        Write-VfxError "Visit: https://docs.conda.io/en/latest/miniconda.html"
        return
    }

    if (-not (Initialize-CondaHooks -CondaExe $condaExe)) {
        Write-VfxError "Found conda at '$condaExe' but could not initialize shell hooks."
        Write-VfxError "Run 'conda init powershell' once, then restart PowerShell."
        return
    }
}

# Verify the environment directory exists
if (-not (Test-Path (Join-Path $VFX_ENV_PREFIX "conda-meta") -ErrorAction SilentlyContinue)) {
    Write-VfxError "Environment '$VFX_ENV_NAME' does not exist."
    Write-VfxError "Please run the installation wizard first:"
    Write-VfxError "  python scripts\install_wizard.py"
    return
}

Write-VfxInfo "Activating '$VFX_ENV_NAME' environment..."
conda activate "$VFX_ENV_PREFIX"

if (Test-InVfxEnv) {
    Write-VfxInfo "Environment activated successfully"
} else {
    Write-VfxError "Failed to activate environment '$VFX_ENV_NAME'"
    Show-WrongEnvWarning -CurrentEnv $env:CONDA_DEFAULT_ENV
}
