@echo off
REM =============================================================================
REM Run Pipeline - Auto-activating wrapper (Windows)
REM =============================================================================
REM Activates the vfx-pipeline conda environment and runs run_pipeline.py
REM
REM Usage:
REM   src\run-pipeline.bat <input_movie> [options]
REM
REM Example:
REM   src\run-pipeline.bat C:\path\to\footage.mp4 --name "My_Shot" --stages all
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

python "%REPO_ROOT%\scripts\run_pipeline.py" %*
exit /b %ERRORLEVEL%
