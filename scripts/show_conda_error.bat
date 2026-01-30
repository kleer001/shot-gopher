@echo off
REM =============================================================================
REM Shared error message for conda environment activation failure
REM Called by wrapper scripts in src/
REM =============================================================================

REM Environment name - single source of truth for Windows scripts
set "VFX_ENV_NAME=vfx-pipeline"

echo.
echo !!! ============================================================== !!!
echo !!!                                                                !!!
echo !!!          FAILED TO ACTIVATE CONDA ENVIRONMENT                  !!!
echo !!!                                                                !!!
echo !!! ============================================================== !!!
echo.
echo     The '%VFX_ENV_NAME%' conda environment could not be activated.
echo.
echo     +----------------------------------------------------------+
echo     ^|  To fix this, first run the install wizard:             ^|
echo     ^|                                                          ^|
echo     ^|      python scripts\install_wizard.py                    ^|
echo     ^|                                                          ^|
echo     ^|  Then activate the environment:                          ^|
echo     ^|                                                          ^|
echo     ^|      conda activate %VFX_ENV_NAME%                         ^|
echo     ^|                                                          ^|
echo     +----------------------------------------------------------+
echo.
