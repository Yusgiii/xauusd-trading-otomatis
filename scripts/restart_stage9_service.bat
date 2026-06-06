@echo off
setlocal

:: Path project
set "PROJECT_DIR=%~dp0.."
set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%\scripts\stage9_service.py"

cd /d "%PROJECT_DIR%"

:: Pastikan python ada
if not exist "%PYTHON%" (
    echo ERROR: Python tidak ditemukan di %PYTHON%
    pause
    exit /b 1
)

:: Stop instance lama
echo Menghentikan instance lama...
taskkill /F /FI "WINDOWTITLE eq XAUUSD*" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*stage9_service*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 3 /nobreak >nul

:: Hapus lock file jika ada
if exist "%PROJECT_DIR%\logs\stage9_service.lock" (
    del "%PROJECT_DIR%\logs\stage9_service.lock"
    echo Lock file dihapus
)

:: Start service baru
echo Menjalankan service...
start "XAUUSD Stage9 Bot" /D "%PROJECT_DIR%" "%PYTHON%" "%SCRIPT%" --latest-run

echo Selesai. Cek Telegram: /status
pause
