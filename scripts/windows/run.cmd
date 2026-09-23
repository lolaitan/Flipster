@echo off
rem Double-click to start Flipster after setup.cmd has run once.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
echo.
pause
