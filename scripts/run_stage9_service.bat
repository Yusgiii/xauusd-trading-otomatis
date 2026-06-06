@echo off
title XAUUSD H1 Stage9 Bot
cd /d "%~dp0.."
echo Menjalankan bot Telegram XAUUSD H1 (on-demand) ...
echo Ketik /analisa di bot Telegram Anda
echo.
echo Task Scheduler: jalankan scripts\register_windows_tasks.ps1 sebagai Admin
echo Log: logs\stage9_service.log
echo Tekan Ctrl+C untuk berhenti.
call "%~dp0run_stage9_service_task.bat"
pause
