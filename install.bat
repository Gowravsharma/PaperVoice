@echo off
rem Double-click to install PaperVoice (runs install.ps1 next to this file; no admin needed).
title Installing PaperVoice
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
