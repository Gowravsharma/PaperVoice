@echo off
rem Double-click to install PaperVoice (runs install.ps1 next to this file; no admin needed).
title Installing PaperVoice
if not exist "%~dp0install.ps1" (
  echo.
  echo   install.ps1 is not next to this file.
  echo   If you opened install.bat from inside the downloaded ZIP, first extract the whole ZIP:
  echo   right-click the ZIP file, choose "Extract All...", then open the new folder and run install.bat there.
  echo.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo ==============================================================================
  echo   PaperVoice was NOT installed. The reason is in the red text above.
  echo   Run install.bat again: it skips what is already done and continues.
  echo   If it fails again, send everything printed in this window: click the window,
  echo   press Ctrl+A then Enter to copy it, and paste it into a message.
  echo   (A line saying a library "CRASHED" is the computer check - it tells what to try.)
  echo ==============================================================================
) else (
  echo.
  echo   Finished. The last step - adding the extension to the browser - is described above.
)
echo.
pause
