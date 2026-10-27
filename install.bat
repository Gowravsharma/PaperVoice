@echo off
rem Double-click to install PaperVoice (runs install.ps1 next to this file; no admin needed).
title Installing PaperVoice
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo ==============================================================================
  echo   PaperVoice was NOT installed. The reason is in the red text above.
  echo   Run install.bat again: it skips what is already done and continues.
  echo   If it fails again, send that red text so it can be fixed.
  echo ==============================================================================
) else (
  echo.
  echo   PaperVoice is installed. Finish by adding the extension, as described above.
)
echo.
pause
