@echo off
rem Double-click to update PaperVoice to the newest version (runs install.ps1 -Update next to this file).
rem "update.bat -Force" reinstalls even when this is already the newest version.
title Updating PaperVoice
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Update %*
if errorlevel 1 (
  echo.
  echo ==============================================================================
  echo   The update did NOT finish. The reason is in the red text above.
  echo   Run update.bat again: it skips what is already done and continues.
  echo ==============================================================================
)
echo.
pause
