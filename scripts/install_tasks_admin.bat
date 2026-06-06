@echo off
REM Klik kanan -> Run as administrator
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0register_windows_tasks.ps1"
echo.
pause
