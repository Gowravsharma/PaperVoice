@echo off
rem Double-click to remove what PaperVoice registered outside its folder. After this, delete the folder itself.
title Uninstalling PaperVoice
echo Stopping the reading engine if it is running...
powershell -NoProfile -Command "try { Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/api/shutdown -TimeoutSec 3 | Out-Null } catch { }"
if exist "%~dp0.venv\Scripts\papervoice-setup.exe" (
  "%~dp0.venv\Scripts\papervoice-setup.exe" --remove
) else (
  echo The helper was never set up from this folder: nothing to remove there.
)
echo.
echo ==============================================================================
echo   Done. PaperVoice kept everything inside its own folder, so nothing else is left:
echo   now delete this folder  %~dp0
echo   and remove the extension on the browser's extensions page.
echo ==============================================================================
echo.
pause
