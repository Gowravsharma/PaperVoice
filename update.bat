@echo off
rem Double-click to update PaperVoice to the newest version (runs install.ps1 -Update next to this file).
rem "update.bat -Force" reinstalls even when this is already the newest version.
title Updating PaperVoice
if not exist "%~dp0install.ps1" (
  echo.
  echo   install.ps1 is not next to this file: run update.bat from inside the PaperVoice folder.
  echo.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Update %*
if errorlevel 1 (
  echo.
  echo ==============================================================================
  echo   The update did NOT finish. The reason is in the red text above.
  echo   Run update.bat again: it skips what is already done and continues.
  echo   If it fails again, send everything printed in this window (Ctrl+A, then Enter, copies it).
  echo ==============================================================================
)
echo.
pause
