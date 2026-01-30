@echo off
REM =============================================================================
REM Run COLMAP - Auto-activating wrapper (Windows)
REM =============================================================================
REM Activates the vfx-pipeline conda environment and runs run_colmap.py
REM
REM Usage:
REM   src\run-colmap.bat <project_dir> [options]
REM
REM Example:
REM   src\run-colmap.bat C:\path\to\projects\My_Shot --dense --mesh
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
    call "%REPO_ROOT%\scripts\show_conda_error.bat"
    exit /b 1
)

python "%REPO_ROOT%\scripts\run_colmap.py" %*
exit /b %ERRORLEVEL%
