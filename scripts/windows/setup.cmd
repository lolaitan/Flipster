@echo off
rem Double-click to set up Flipster (Python venv, native engine, web app, tests) and start it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
