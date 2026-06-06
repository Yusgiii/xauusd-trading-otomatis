@echo off
REM Dipanggil Task Scheduler / VBS — tanpa pause.
set "PROJECT_DIR=%~dp0.."
cd /d "%PROJECT_DIR%"

set "PYTHON="
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
  "%PROJECT_DIR%\.venv\Scripts\python.exe" -c "import requests" 2>nul && set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
)
if not defined PYTHON (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYTHON (
      "%%P" -c "import requests" 2>nul && set "PYTHON=%%P"
    )
  )
)
if not defined PYTHON set "PYTHON=python"

set "LOG_DIR=%PROJECT_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] Stage9 service start PYTHON=%PYTHON%>> "%LOG_DIR%\stage9_service.log"
"%PYTHON%" -u "%PROJECT_DIR%\scripts\stage9_service.py" --latest-run >> "%LOG_DIR%\stage9_service.log" 2>&1
