@echo off
REM Shot Gopher - Windows launcher
REM Runs the Shot Gopher TUI

python "%~dp0shot-gopher" %*
if errorlevel 1 pause
