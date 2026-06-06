@echo off
setlocal

:: Startup script untuk Task Scheduler — dipanggil saat login Windows

set "PROJECT_DIR=%~dp0.."
set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%\scripts\stage9_service.py"
set "LOG=%PROJECT_DIR%\logs\startup.log"

if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

echo [%DATE% %TIME%] Startup script dipanggil >> "%LOG%"

:: Tunggu 60 detik agar MT5 dan network siap
echo Menunggu 60 detik untuk sistem siap...
timeout /t 60 /nobreak >nul

:: Cek python ada
if not exist "%PYTHON%" (
    echo [%DATE% %TIME%] ERROR: Python tidak ditemukan >> "%LOG%"
    exit /b 1
)

:: Hapus lock file stale
if exist "%PROJECT_DIR%\logs\stage9_service.lock" (
    del "%PROJECT_DIR%\logs\stage9_service.lock"
    echo [%DATE% %TIME%] Lock file dihapus >> "%LOG%"
)

:: Start service
echo [%DATE% %TIME%] Menjalankan service... >> "%LOG%"
start "XAUUSD Stage9 Bot" /D "%PROJECT_DIR%" "%PYTHON%" "%SCRIPT%" --latest-run

echo [%DATE% %TIME%] Service distart >> "%LOG%"
