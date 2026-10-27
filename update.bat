@echo off
rem Double-click to update PaperVoice to the newest version (runs install.ps1 -Update next to this file).
title Updating PaperVoice
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Update
if errorlevel 1 (
  echo.
  echo ==============================================================================
  echo   The update did NOT finish. The reason is in the red text above.
  echo   Run update.bat again: it skips what is already done and continues.
  echo ==============================================================================
) else (
  echo.
  echo   Updated. Now reload the extension on the extensions page, as described above.
)
echo.
pause
