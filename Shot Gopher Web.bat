@echo off
REM Shot Gopher Web GUI Launcher for Windows
REM Double-click this file to launch the web interface

setlocal EnableDelayedExpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo  ====================================
echo   Shot Gopher - Web GUI Launcher
echo  ====================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Conda not found in PATH
    echo.
    echo Please install Miniconda or Anaconda and add it to your PATH.
    echo.
    pause
    exit /b 1
)

REM Activate the vfx-pipeline environment
echo Activating vfx-pipeline environment...
call conda activate vfx-pipeline
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate vfx-pipeline environment
    echo.
    echo Please run the install wizard first:
    echo   python scripts/install_wizard.py
    echo.
    pause
    exit /b 1
)

REM Launch the web GUI
echo Starting Shot Gopher Web GUI...
echo.
python scripts/launch_web_gui.py

REM Keep window open if there was an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
)
