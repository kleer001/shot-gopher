@echo off
REM =============================================================================
REM Launch Interactive Segmentation - Auto-activating wrapper (Windows)
REM =============================================================================
REM Activates the vfx-pipeline conda environment and launches interactive
REM segmentation workflow in ComfyUI
REM
REM Usage:
REM   src\launch-interactive-segmentation.bat <project_dir>
REM
REM Example:
REM   src\launch-interactive-segmentation.bat C:\path\to\projects\My_Shot
REM =============================================================================

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Conda not found in PATH
    echo.
    echo Please install Miniconda or Anaconda and add it to your PATH.
    exit /b 1
)

call conda activate vfx-pipeline
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate vfx-pipeline environment
    echo.
    echo Please run the install wizard first:
    echo   python scripts\install_wizard.py
    exit /b 1
)

python "%REPO_ROOT%\scripts\launch_interactive_segmentation.py" %*
exit /b %ERRORLEVEL%
