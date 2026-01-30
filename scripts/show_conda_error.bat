@echo off
REM =============================================================================
REM Shared error message for conda environment activation failure
REM Called by wrapper scripts in src/
REM =============================================================================

echo.
echo [91m!!! ============================================================== !!![0m
echo [91m!!!                                                                !!![0m
echo [91m!!!          FAILED TO ACTIVATE CONDA ENVIRONMENT                  !!![0m
echo [91m!!!                                                                !!![0m
echo [91m!!! ============================================================== !!![0m
echo.
echo     The 'vfx-pipeline' conda environment could not be activated.
echo.
echo     [93m+----------------------------------------------------------+[0m
echo     [93m^|  To fix this, first run the install wizard:             ^|[0m
echo     [93m^|                                                          ^|[0m
echo     [93m^|      python scripts\install_wizard.py                    ^|[0m
echo     [93m^|                                                          ^|[0m
echo     [93m^|  Then activate the environment:                          ^|[0m
echo     [93m^|                                                          ^|[0m
echo     [93m^|      conda activate vfx-pipeline                         ^|[0m
echo     [93m^|                                                          ^|[0m
echo     [93m+----------------------------------------------------------+[0m
echo.
